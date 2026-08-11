"""
GLM-OCR self-hosted test bench — LADDER + STRESS in one file.

    python runpod_test.py          (edit the CONFIG block; no argv, by design)

WHY ONE FILE
    The Replicate side had ladder_test.py and stress_test.py sharing ~90% of their
    code and both wrapped around Replicate's prediction API. Self-hosted glmocr is a
    plain HTTP endpoint, so the API plumbing is gone and the two tests are just two
    drivers over the same call.

WHAT THE LADDER IS ACTUALLY FOR NOW
    Empty md_results is NOT a random glitch — it is the signature of the layout
    stage hitting CUDA OOM and SILENTLY SKIPPING THE BATCH while still answering
    HTTP 200 (proven on the pod: "Layout detection failed for pages [0], skipping
    batch: CUDA out of memory"). So the level at which the first empty appears IS
    the OOM cliff, and this ladder is the tool for tuning --gpu-memory-utilization.
    Green all the way up = your VRAM budget has headroom.

TRANSPORTS — set ENDPOINTS to whichever you are measuring:
    on the pod          ["http://localhost:5002"]                        pure GPU throughput
    from Windows        ["http://localhost:5002"]  + an SSH tunnel:      GPU + MY->RO network
        ssh -N -L 5002:localhost:5002 -p 11053 -i $env:USERPROFILE\\.ssh\\id_ed25519 root@<POD_IP>
    RunPod HTTP proxy   ["https://<podid>-5002.proxy.runpod.net"]        needs the port exposed
    several backends    [...5002, ...5003]                               round-robined
"""
import base64
import glob
import json
import os
import statistics
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ======================= CONFIG (edit here) =======================
ENDPOINTS = os.environ.get("ENDPOINTS", "http://localhost:5002").split(",")

IMAGE_DIR = os.environ.get(
    "IMAGE_DIR",
    r"\\ABOT-TEST-03\MBRSUploadTraining\YE2026\3C876CBE-23C6-4DA5-B082-23483B4063BA\SplitImage")

MODE = os.environ.get("MODE", "both")        # ladder | stress | both

LADDER_LEVELS = [1, 2, 3, 4, 6, 8, 12]       # simultaneous calls per round
LADDER_STOP_ON_EMPTY = True                  # first empty = the OOM cliff -> stop
PAUSE_BETWEEN_LEVELS = 3

STRESS_TOTAL = int(os.environ.get("STRESS_TOTAL", "30"))   # production-shaped run
STRESS_CONC = int(os.environ.get("STRESS_CONC", "6"))      # in-flight cap

TIMEOUT = 180
RETRIES = 2          # connection resets / 5xx only. Self-hosted has no 429.
# ==================================================================

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ helpers
def png_size(path):
    """(width, height) from the PNG IHDR — no Pillow dependency."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return (0, 0)
    return struct.unpack(">II", head[16:24])


def as_uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def md_of(body):
    if not isinstance(body, dict):
        return ""
    return body.get("md_results") or body.get("markdown_result") or ""


def flat_elems(body):
    ld = (body or {}).get("layout_details") or (body or {}).get("json_result") or []
    return ld[0] if (ld and isinstance(ld[0], list)) else ld


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return s[k]


def call(endpoint, path, label):
    """One page through one endpoint. ok == non-empty markdown, not just HTTP 200."""
    uri = as_uri(path)
    t0 = time.perf_counter()
    err = ""
    for attempt in range(RETRIES + 1):
        try:
            r = requests.post(endpoint.rstrip("/") + "/glmocr/parse",
                              json={"images": [uri]}, timeout=TIMEOUT)
            if r.status_code >= 500:
                err = "HTTP %d: %s" % (r.status_code, r.text[:120])
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code != 200:
                return _res(label, path, t0, None, "HTTP %d: %s" % (r.status_code, r.text[:160]), attempt)
            return _res(label, path, t0, r.json(), "", attempt)
        except Exception as ex:
            err = "%s: %s" % (type(ex).__name__, str(ex)[:120])
            time.sleep(1.0 * (attempt + 1))
    return _res(label, path, t0, None, err, RETRIES)


def _res(label, path, t0, body, err, retries):
    md = md_of(body)
    elems = flat_elems(body)
    return {
        "label": label,
        "img": os.path.basename(path),
        "wall": time.perf_counter() - t0,
        "md_len": len(md),
        "tables": md.count("<table"),
        "elems": len(elems),
        "ok": bool(md.strip()) and not err,
        "retries": retries,
        "error": err,
        "_body": body,
    }


def show(r):
    print("  %s %-9s %-14s wall=%5.1fs md=%-5d elems=%-3d tables=%d%s %s" % (
        "OK  " if r["ok"] else "FAIL", r["label"], r["img"], r["wall"],
        r["md_len"], r["elems"], r["tables"],
        (" retry=%d" % r["retries"]) if r["retries"] else "", r["error"]))


def diagnose_empty(n_empty, level):
    print("")
    print("  >>> %d/%d calls returned EMPTY markdown at concurrency %d." % (n_empty, level, level))
    print("  >>> This is the LAYOUT-STAGE CUDA OOM signature: the batch is skipped and")
    print("  >>> the server still answers HTTP 200. Confirm on the pod with:")
    print("  >>>     grep -c 'out of memory' /workspace/logs/glmocr_50*.log")
    print("  >>> Fix: lower --gpu-memory-utilization (each glmocr process needs ~1.9 GiB")
    print("  >>> of VRAM on top of whatever vLLM reserves), or run fewer processes.")


# ------------------------------------------------------------------ contract check
def contract_check(body, path):
    """One-off parity check against the Z.ai cloud contract (see PHASE1-RESULTS.md)."""
    print("\n=== CONTRACT CHECK (vs Z.ai layout_parsing) ===")
    if not isinstance(body, dict):
        print("  response is not a JSON object — nothing to check")
        return
    md = md_of(body)
    elems = flat_elems(body)
    print("  md_results        : %d chars, %d table(s)" % (len(md), md.count("<table")))
    print("  layout_details    : %d elements" % len(elems))
    if not elems:
        print("  !! no elements — stitching / title logic would go blind")
        return

    need = ("bbox_2d", "content", "index", "native_label")
    missing = [k for k in need if k not in elems[0]]
    print("  element fields    : %s" % ("OK" if not missing else "MISSING " + str(missing)))

    vocab = sorted({e.get("native_label") for e in elems if isinstance(e, dict)})
    print("  native_label      : %s" % vocab)
    if vocab == ["text"]:
        print("  !! only 'text' — label_task_mapping lost (config merge failed)")

    xs = [c for e in elems if isinstance(e.get("bbox_2d"), list) for c in e["bbox_2d"][0::2]]
    ys = [c for e in elems if isinstance(e.get("bbox_2d"), list) for c in e["bbox_2d"][1::2]]
    iw, ih = png_size(path)
    if xs and ys:
        mx, my = max(xs), max(ys)
        print("  bbox max x/y      : %d/%d   (page %dx%d)" % (mx, my, iw, ih))
        if max(mx, my) <= 1000 < max(iw, ih):
            # verified on the pod: 903/1000*3166 = 2859 px vs the Z.ai golden's 2861
            print("  -> NORMALIZED 0-1000 (self-hosted). The adapter MUST scale:")
            print("       x_px = round(x * %d / 1000)   y_px = round(y * %d / 1000)" % (iw, ih))
            print("     Without it ReportStitchOcr's seam, HeaderRecovery and BboxAspect")
            print("     silently misplace while every field still looks correct.")
        else:
            print("  -> looks like PIXELS already (cloud-style); no scaling needed")
    if "polygon" in elems[0]:
        print("  note: 'polygon' present — the adapter drops it (cloud has no such field)")
    if "height" not in elems[0]:
        print("  note: per-element height/width absent — cloud has them; adapter fills in")


# ------------------------------------------------------------------ ladder
def run_ladder(images):
    print("\n" + "=" * 74)
    print("LADDER — climbing concurrency until the first EMPTY (= the OOM cliff)")
    print("=" * 74)
    idx = 0
    rows = []
    for k in LADDER_LEVELS:
        print("\n=== LEVEL %d: %d simultaneous calls ===" % (k, k))
        batch = []
        for i in range(k):
            batch.append((ENDPOINTS[i % len(ENDPOINTS)], images[idx % len(images)], "L%d-%d" % (k, i + 1)))
            idx += 1

        t0 = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=k) as ex:
            futs = [ex.submit(call, e, p, lb) for e, p, lb in batch]
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                show(r)
        wall = time.perf_counter() - t0
        bad = [r for r in results if not r["ok"]]
        # Count only SUCCESSFUL pages. An OOM-skipped call returns empty in ~2s, so
        # counting it would make the level that just broke look like the fastest one.
        thr = (k - len(bad)) / wall
        rows.append((k, wall, thr, len(bad)))
        print("  LEVEL %d wall: %.1fs   throughput: %.2f pages/s (ok only)   empty/fail: %d/%d"
              % (k, wall, thr, len(bad), k))

        if bad and LADDER_STOP_ON_EMPTY:
            diagnose_empty(len(bad), k)
            break
        time.sleep(PAUSE_BETWEEN_LEVELS)

    print("\n--- LADDER SUMMARY ---")
    print("  %-6s %8s %10s %8s" % ("N", "wall", "pages/s", "fail"))
    for k, wall, thr, bad in rows:
        print("  %-6d %7.1fs %10.2f %8d" % (k, wall, thr, bad))
    clean = [r for r in rows if r[3] == 0]
    if clean:
        best = max(clean, key=lambda r: r[2])
        print("  PEAK (clean): %.2f pages/s at N=%d" % (best[2], best[0]))
        print("  -> a 28-page report ~ %.0fs of GPU time" % (28 / best[2]))
    return rows


# ------------------------------------------------------------------ stress
def run_stress(images):
    print("\n" + "=" * 74)
    print("STRESS — %d calls at %d in-flight (distinct pages, no cache inflation)"
          % (STRESS_TOTAL, STRESS_CONC))
    print("=" * 74)
    tasks = [(ENDPOINTS[i % len(ENDPOINTS)], images[i % len(images)], "s%d" % (i + 1))
             for i in range(STRESS_TOTAL)]
    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=STRESS_CONC) as ex:
        futs = [ex.submit(call, e, p, lb) for e, p, lb in tasks]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print("  [%3d/%d]" % (len(results), STRESS_TOTAL), end="")
            show(r)
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    walls = [r["wall"] for r in ok]

    print("\n=== STRESS SUMMARY ===")
    print("  calls        : %d   ok: %d   FAIL/EMPTY: %d" % (STRESS_TOTAL, len(ok), len(bad)))
    print("  burst wall   : %.1fs   throughput: %.2f pages/s  (%.0f pages/min)"
          % (wall, len(ok) / wall, len(ok) / wall * 60))
    if walls:
        print("  latency  p50 : %.1fs   p95: %.1fs   max: %.1fs   mean: %.1fs"
              % (pct(walls, 50), pct(walls, 95), max(walls), statistics.mean(walls)))
    if ok:
        print("  md length    : min=%d  median=%d  max=%d"
              % (min(r["md_len"] for r in ok),
                 int(statistics.median(r["md_len"] for r in ok)),
                 max(r["md_len"] for r in ok)))
    if bad:
        empties = [r for r in bad if not r["error"]]
        if empties:
            diagnose_empty(len(empties), STRESS_CONC)
        for r in bad[:8]:
            print("    FAIL %s %s: %s" % (r["label"], r["img"], r["error"] or "EMPTY markdown"))

    out = os.path.join(HERE, "runpod_stress_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "_body"} for r in results],
                  f, indent=2, ensure_ascii=False)
    print("  saved -> %s" % out)
    return results


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "Page_*.png")))
    if not images:
        sys.exit("no Page_*.png found in %s" % IMAGE_DIR)
    print("endpoints : %s" % ENDPOINTS)
    print("corpus    : %d pages from %s" % (len(images), IMAGE_DIR))
    print("mode      : %s" % MODE)

    # WARMUP every endpoint. Each glmocr process loads its own layout model on ITS
    # first request; an unwarmed backend dumps that one-off cost into the first
    # measured level and fakes a low parallelism number.
    print("\n=== WARMUP (all endpoints — first call per process loads the layout model) ===")
    first = None
    for e in ENDPOINTS:
        r = call(e, images[0], "warm")
        print("  %-45s wall=%.1fs md=%d %s" % (e, r["wall"], r["md_len"], r["error"]))
        if first is None:
            first = r
    if not first["ok"]:
        print("\n!! WARMUP FAILED — nothing below would mean anything.")
        if not first["error"]:
            diagnose_empty(1, 1)
        sys.exit(1)

    contract_check(first["_body"], images[0])

    if MODE in ("ladder", "both"):
        run_ladder(images)
    if MODE in ("stress", "both"):
        run_stress(images)
