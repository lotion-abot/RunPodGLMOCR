"""
RunPod POD validator — runs on WINDOWS, hits the pod over its HTTPS proxy.

    python pod_test.py

Two things we need out of this, and we have never had either:
  A. CORRECTNESS — a REAL 3166x4096 audit page comes back with non-empty md_results
     and a proper native_label vocabulary. (The Replicate build passed a synthetic
     self-test image and still returned md=0 on every real page.)
  B. THE REAL CONCURRENCY NUMBER — a 1/2/3/6 ladder measured against a stack that
     is actually producing output. The old "parallelism ~1.3" came from the md=0
     run: it timed the pipeline short-circuiting, not OCR. It is void.

Pod side must already be up:  bash /workspace/pod_setup.sh
and ports 5002..5004 exposed as HTTP ports on the pod.

NOTE: the RunPod HTTP proxy is Cloudflare-fronted with a hard 100s connection cap
(524 beyond that). Single-page OCR is seconds, so this only matters if something
is badly wrong.
"""
import base64
import glob
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# ======================= CONFIG (edit here) =======================
POD_ID    = os.environ.get("POD_ID", "7flxvv3va0w9aj")
PORTS     = [int(p) for p in os.environ.get("PORTS", "5002,5003,5004").split(",")]
IMAGE_DIR = os.environ.get(
    "IMAGE_DIR",
    r"\\ABOT-TEST-03\MBRSUploadTraining\YE2026\3C876CBE-23C6-4DA5-B082-23483B4063BA\SplitImage")

LADDER    = [1, 2, 3, 6]             # concurrency levels to measure

# LOCAL=1 runs this ON the pod against localhost. Strongly preferred for the
# ladder: the pod is in EU-RO-1 and a Malaysia->Romania round trip would swamp a
# 2-3s OCR call and make the concurrency numbers meaningless.
LOCAL     = os.environ.get("LOCAL", "0") == "1"
# ==================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 95                          # stay under the Cloudflare 100s cap

TOP_KEYS = ["md_results", "layout_details", "data_info", "usage"]
ELEM_KEYS = ["bbox_2d", "native_label", "content", "index"]


def url_for(port):
    if LOCAL:
        return f"http://localhost:{port}/glmocr/parse"
    return f"https://{POD_ID}-{port}.proxy.runpod.net/glmocr/parse"


def as_uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def png_size(path):
    """(width, height) straight out of the PNG IHDR — no Pillow needed."""
    import struct
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return (0, 0)
    return struct.unpack(">II", head[16:24])


def parse(port, uri, path):
    """One page through one backend. Returns (elapsed, body_or_None, error_str)."""
    t0 = time.perf_counter()
    try:
        r = requests.post(url_for(port), json={"images": [uri]}, timeout=TIMEOUT)
        el = time.perf_counter() - t0
        if r.status_code != 200:
            return el, None, f"HTTP {r.status_code}: {r.text[:200]}"
        return el, r.json(), ""
    except Exception as ex:
        return time.perf_counter() - t0, None, f"{type(ex).__name__}: {ex}"


def md_of(body):
    if not isinstance(body, dict):
        return ""
    return body.get("md_results") or body.get("markdown_result") or ""


# ------------------------------------------------------------------ A: correctness
def correctness(images):
    print("=== A. CORRECTNESS — one REAL page ===")
    path = images[0]
    el, body, err = parse(PORTS[0], as_uri(path), path)
    print(f"  {os.path.basename(path)} on :{PORTS[0]}  ->  {el:.1f}s")
    if err:
        sys.exit(f"  FAILED: {err}\n  (pod side: tail -f /workspace/logs/glmocr_{PORTS[0]}.log)")

    md = md_of(body)
    print(f"  md length         : {len(md)}")
    if not md.strip():
        print("  !! EMPTY MARKDOWN — this is the exact Replicate bug. STOP and read the pod log:")
        print(f"     tail -100 /workspace/logs/glmocr_{PORTS[0]}.log")
        print(f"     tail -100 /workspace/logs/vllm.log")
        sys.exit(1)
    print(f"  md preview        : {md[:160].replace(chr(10), ' / ')}")

    ld = body.get("layout_details") or body.get("json_result") or []
    if ld and isinstance(ld[0], list):
        ld = ld[0]
    print(f"  layout_details    : {len(ld)} elements")
    if ld:
        first = ld[0]
        missing = [k for k in ELEM_KEYS if k not in first and not (k == "native_label" and "label" in first)]
        print(f"  element fields    : {'OK' if not missing else 'MISSING ' + str(missing)}")
        vocab = sorted({(e.get("native_label") or e.get("label")) for e in ld if isinstance(e, dict)})
        print(f"  native_label vocab: {vocab}")
        if vocab == ["text"]:
            print("  !! ONLY 'text' — label_task_mapping was lost (runner.py merge failed).")
            print("     C# title detection / dedup / header recovery all go blind on this.")

        # bbox coordinate space. ReportStitchOcr (seam), HeaderRecovery and BboxAspect all
        # assume bbox == PIXELS of the input page. If this build emits NORMALIZED coords,
        # every field above still looks perfect and letterhead stitching silently misplaces.
        xs = [c for e in ld if isinstance(e.get("bbox_2d"), list) for c in e["bbox_2d"][0::2]]
        ys = [c for e in ld if isinstance(e.get("bbox_2d"), list) for c in e["bbox_2d"][1::2]]
        if xs and ys:
            iw, ih = png_size(path)
            print(f"  bbox max x/y      : {max(xs)}/{max(ys)}   (image is {iw}x{ih})")
            if max(max(xs), max(ys)) <= 1000 and max(iw, ih) > 1000:
                print("  !! bboxes look NORMALIZED (<=1000), not pixel — stitching / header")
                print("     recovery / aspect checks will silently misplace. Needs a scale step.")
        else:
            print("  !! no bbox_2d on any element — stitching cannot work")
    else:
        print("  !! layout_details EMPTY — drop-in would BREAK stitching + title logic")

    with open(os.path.join(HERE, "pod_output.json"), "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)
    print(f"  saved -> {os.path.join(HERE, 'pod_output.json')}")
    print(f"  tables in md      : {md.count('<table')}   headings: {md.count(chr(10) + '#')}")
    return el


# ------------------------------------------------------------------ B: concurrency
def ladder(images, solo_baseline):
    print("\n=== B. CONCURRENCY LADDER ===")
    print(f"  backends: {len(PORTS)} glmocr processes over 1 GPU")
    print(f"  solo baseline: {solo_baseline:.2f}s/page\n")
    print(f"  {'N':>3} {'wall':>8} {'pages/s':>9} {'avg':>7} {'p95':>7} {'parallel':>9}  ok")

    rows = []
    for n in LADDER:
        if len(images) < n + 1:
            print(f"  skip N={n}: need {n} distinct pages in IMAGE_DIR")
            continue
        # distinct pages so nothing can be served from a cache; round-robin the ports
        batch = [(PORTS[i % len(PORTS)], images[1 + (i % (len(images) - 1))]) for i in range(n)]
        payloads = [(p, as_uri(f), f) for p, f in batch]

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(lambda a: parse(*a), payloads))
        wall = time.perf_counter() - t0

        times = [r[0] for r in res]
        ok = sum(1 for r in res if md_of(r[1]).strip())
        avg = statistics.mean(times)
        p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
        par = n * solo_baseline / wall
        rows.append((n, wall, n / wall, par, ok))
        print(f"  {n:>3} {wall:>7.1f}s {n/wall:>9.2f} {avg:>6.1f}s {p95:>6.1f}s {par:>8.2f}x  {ok}/{n}")
        for r in res:
            if r[2]:
                print(f"        err: {r[2][:160]}")

    if rows:
        best = max(rows, key=lambda r: r[2])
        print(f"\n  PEAK THROUGHPUT: {best[2]:.2f} pages/s at N={best[0]}  "
              f"(effective parallelism {best[3]:.2f}x)")
        print(f"  -> a 6-page report takes ~{6 / best[2]:.1f}s of GPU time at that level")
        if best[3] < 1.5 and len(PORTS) >= 3:
            print("  -> GPU-bound, not process-bound: more glmocr processes will NOT help.")
            print("     Only a bigger GPU (24GB L4 / 48GB A6000) or more workers (= more GPUs) will.")
    return rows


if __name__ == "__main__":
    if POD_ID == "PUT_POD_ID_HERE" and not LOCAL:
        sys.exit("edit POD_ID at the top of this file first")
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "Page_*.png")))
    if len(images) < 2:
        sys.exit(f"no pages found in {IMAGE_DIR}")
    print(f"corpus: {len(images)} pages from {IMAGE_DIR}\n")

    # Warm EVERY backend, not just PORTS[0]. Each glmocr process loads its own copy of
    # the layout model on ITS first request. If 5003/5004 are still cold when the ladder
    # hits N=2/N=3, that one-time load lands inside `wall` and understates parallelism —
    # indistinguishable from a real GPU ceiling. That is exactly how the void 1.3 happened.
    print("=== WARMUP (all backends; first call per process loads the layout model) ===")
    uri0 = as_uri(images[0])
    for p in PORTS:
        el, body, err = parse(p, uri0, images[0])
        print(f"  :{p} {el:.1f}s  md={len(md_of(body))}  {err}")
    print()

    solo = correctness(images)
    ladder(images, solo)
