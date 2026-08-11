"""
RunPod Serverless handler — GLM-OCR, drop-in for Z.ai's layout_parsing response.

    input : {"image": "<data:image/png;base64,...>" or bare base64}
    output: {"md_results", "layout_details", "data_info", "usage"}  (Z.ai shape)

Three things here are not obvious. Each exists because of something measured on the
validation pod, written up in PHASE1-RESULTS.md / PHASE2-DESIGN.md:

1. bbox_2d must be RESCALED. Self-hosted glmocr emits 0-1000 normalized coordinates
   (glmocr/layout/base.py:42); the Z.ai cloud emits absolute pixels (glmocr/api.py:353).
   Verified element-by-element against the golden sample: x*W/1000 lands within 1-4px.
   Ship it unscaled and ReportStitchOcr's seam, HeaderRecovery and BboxAspect all
   misplace silently while every field still looks correct.

2. EMPTY markdown is ambiguous and the two causes need opposite handling. A layout-stage
   CUDA OOM makes glmocr skip the batch and STILL answer HTTP 200 with empty output
   ("Layout detection failed for pages [0], skipping batch: CUDA out of memory").
   A genuinely blank page looks identical over HTTP. We are in the same container as
   glmocr, so we read its log instead of guessing: OOM -> fail the job loudly;
   blank page -> return the empty envelope. A false success in audit data is invisible;
   a false failure is not.

3. The semaphore is an ASSERTION, not a queue. RunPod's queue endpoint already bounds
   in-flight jobs to concurrency_modifier's return value, so this should never block.
   If it ever does, the platform contract broke and we want a log line, because the
   consequence of exceeding the GPU is (1)'s silent empty output.
"""
import asyncio
import base64
import io
import logging
import os
import sys
import time

import httpx
import runpod
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("glmocr-handler")

GLMOCR_URL = "http://127.0.0.1:5002/glmocr/parse"
GLMOCR_LOG = "/var/log/glmocr.log"
SELFTEST_PAGE = "/app/selftest_page.png"

# MUST be re-calibrated with runpod_test.py's ladder whenever GPU_MEMORY_UTILIZATION,
# the GPU tier, or the process count changes. The pod's "6" was measured at util=0.55
# with 3 processes and does NOT carry over to util=0.70 with 1. Default is deliberately
# conservative until the ladder speaks.
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
SELFTEST_MIN_MD = int(os.environ.get("SELFTEST_MIN_MD", "800"))
PARSE_TIMEOUT = float(os.environ.get("PARSE_TIMEOUT", "300"))

_gate = asyncio.Semaphore(MAX_CONCURRENCY)


# ------------------------------------------------------------------ glmocr log probe
def _log_offset():
    try:
        return os.path.getsize(GLMOCR_LOG)
    except OSError:
        return 0


def _oom_since(offset):
    """Did the layout stage skip a batch since `offset`? Deterministic, not a heuristic."""
    try:
        with open(GLMOCR_LOG, "rb") as f:
            f.seek(offset)
            tail = f.read()
        return b"skipping batch" in tail or b"out of memory" in tail
    except OSError:
        return False        # can't read the log -> never CLAIM an OOM we didn't see


# ------------------------------------------------------------------ call + adapt
def _as_data_uri(raw):
    s = raw.strip()
    return s if s.startswith("data:") else "data:image/png;base64," + s


def _decode(raw):
    s = raw.strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def _flat(body):
    ld = (body or {}).get("layout_details") or (body or {}).get("json_result") or []
    return ld[0] if (ld and isinstance(ld[0], list)) else ld


async def _parse(client, uri):
    r = await client.post(GLMOCR_URL, json={"images": [uri]})
    r.raise_for_status()
    return r.json()


def _adapt(body, width, height):
    """glmocr self-hosted envelope -> the exact Z.ai layout_parsing shape."""
    md = (body or {}).get("md_results") or (body or {}).get("markdown_result") or ""

    elems = []
    for e in _flat(body):
        if not isinstance(e, dict):
            continue
        d = dict(e)
        if "native_label" not in d and "label" in d:
            d["native_label"] = d["label"]
        d.pop("polygon", None)                       # cloud has no such field

        b = d.get("bbox_2d")
        if isinstance(b, list) and len(b) == 4:
            d["bbox_2d"] = [
                int(round(b[0] * width / 1000.0)), int(round(b[1] * height / 1000.0)),
                int(round(b[2] * width / 1000.0)), int(round(b[3] * height / 1000.0)),
            ]
        d["height"] = height                         # cloud carries these per element
        d["width"] = width
        elems.append(d)

    usage = dict((body or {}).get("usage") or {})
    usage.setdefault("prompt_tokens", 0)             # self-hosted isn't token-billed
    usage.setdefault("completion_tokens", 0)

    return {
        "md_results": md,
        "layout_details": elems,
        "data_info": {"num_pages": 1, "pages": [{"height": height, "width": width}]},
        "usage": usage,
    }


# ------------------------------------------------------------------ boot self-test
def _selftest():
    """A REAL page must come back with real markdown, or this worker must not serve."""
    with open(SELFTEST_PAGE, "rb") as f:
        raw = f.read()
    with Image.open(io.BytesIO(raw)) as im:
        w, h = im.size
    uri = "data:image/png;base64," + base64.b64encode(raw).decode()

    t0 = time.perf_counter()
    with httpx.Client(timeout=PARSE_TIMEOUT) as c:
        r = c.post(GLMOCR_URL, json={"images": [uri]})
        r.raise_for_status()
        body = r.json()
    out = _adapt(body, w, h)
    md, n = out["md_results"], len(out["layout_details"])
    log.info("selftest: %.1fs  md=%d  elements=%d  page=%dx%d",
             time.perf_counter() - t0, len(md), n, w, h)

    if len(md) < SELFTEST_MIN_MD:
        raise RuntimeError(
            "self-test FAILED: md=%d < %d. Empty/short output on a known-good page is the "
            "layout CUDA-OOM signature. Check /var/log/glmocr.log and lower "
            "GPU_MEMORY_UTILIZATION (currently %s)."
            % (len(md), SELFTEST_MIN_MD, os.environ.get("GPU_MEMORY_UTILIZATION"))
        )
    # A boot where bbox came back unscaled would silently misplace every letterhead.
    xs = [c for e in out["layout_details"] if isinstance(e.get("bbox_2d"), list)
          for c in e["bbox_2d"][0::2]]
    if xs and max(xs) <= 1000 < w:
        raise RuntimeError("self-test FAILED: bbox still <=1000 after scaling — adapter broken")


# ------------------------------------------------------------------ handler
async def handler(job):
    inp = job.get("input") or {}
    raw = inp.get("image")
    if not raw:
        return {"error": "input.image is required (data URI or bare base64 PNG/JPG)"}

    try:
        img_bytes = _decode(raw)
        with Image.open(io.BytesIO(img_bytes)) as im:
            width, height = im.size
    except Exception as ex:
        return {"error": "could not decode input.image: %s" % ex}

    uri = _as_data_uri(raw)

    if _gate.locked():
        log.error("in-flight exceeded MAX_CONCURRENCY=%d — platform concurrency contract "
                  "violated; this line should never appear", MAX_CONCURRENCY)

    async with _gate:
        async with httpx.AsyncClient(timeout=PARSE_TIMEOUT) as client:
            for attempt in (1, 2):
                mark = _log_offset()
                body = await _parse(client, uri)
                out = _adapt(body, width, height)

                if out["md_results"].strip():
                    return out

                if _oom_since(mark):
                    # Loud failure on purpose: RunPod retries the job, and a visible
                    # error beats audit data that is silently wrong.
                    raise RuntimeError(
                        "layout stage hit CUDA OOM and skipped the batch (attempt %d). "
                        "Lower MAX_CONCURRENCY (now %d) or GPU_MEMORY_UTILIZATION (now %s), "
                        "then re-run the ladder in runpod_test.py to re-calibrate."
                        % (attempt, MAX_CONCURRENCY, os.environ.get("GPU_MEMORY_UTILIZATION"))
                    )
                if attempt == 1:
                    log.warning("empty markdown, no OOM in log — retrying once")
                    await asyncio.sleep(1.5)

    # Empty twice, no OOM either time: treat as a genuinely blank page. Blank pages are
    # common in scans and failing them would stall the whole pipeline.
    out["warning"] = "empty markdown after 2 attempts; no OOM in glmocr log — treated as a blank page"
    log.warning(out["warning"])
    return out


if __name__ == "__main__":
    try:
        _selftest()
    except Exception as ex:
        log.error("%s", ex)
        sys.exit(1)          # worker refuses to serve rather than return silent garbage
    log.info("ready — MAX_CONCURRENCY=%d", MAX_CONCURRENCY)
    runpod.serverless.start({
        "handler": handler,
        # A CONSTANT, not the docs' traffic-reactive example. Our ceiling is VRAM, which
        # doesn't change with request rate; a dynamic modifier would just walk into the
        # OOM cliff the ladder found.
        "concurrency_modifier": lambda current: MAX_CONCURRENCY,
    })
