"""
GLM-OCR test bench — LADDER + STRESS + COLDSTART, two transports, one file.

    python runpod_test.py          (edit the CONFIG block; no argv, by design)

TRANSPORTS
    direct      POST straight to a glmocr server (the dev pod, or an SSH tunnel).
                Measures the GPU with no platform in the way.
    serverless  POST /v2/{id}/run and poll /status. Measures what production sees,
                including queue time and cold boots.

MODES
    ladder      climb concurrency until the first EMPTY result. Empty markdown is NOT
                a random glitch — it is the signature of the layout stage hitting CUDA
                OOM and SILENTLY SKIPPING THE BATCH while still answering HTTP 200
                (proven on the pod: "Layout detection failed for pages [0], skipping
                batch: CUDA out of memory"). So the level where the first empty appears
                IS the OOM cliff, and this is the tool for calibrating MAX_CONCURRENCY.
    stress      sustained burst at fixed concurrency; p50/p95 latency.
    coldstart   THE FLASHBOOT QUESTION. Fire a call, wait past the idle timeout so the
                worker scales down, fire again — repeat. RunPod's delayTime shows what
                the revive actually cost. FlashBoot is a CRIU-style process snapshot,
                so in principle it restores vLLM *after* its 301s compile warm-up; but
                there is a known "very slow cold starts even with flashboot" report
                against vLLM workers specifically. Measure, never assume.

SSH tunnel for `direct` from Windows:
    ssh -N -L 5002:localhost:5002 -p <PORT> -i $env:USERPROFILE\\.ssh\\id_ed25519 root@<POD_IP>
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

# ======================= CONFIG (edit here, then just `py runpod_test.py`) =======================
# serverless: set ENDPOINT_ID. direct (pod / SSH tunnel): set it to "" and use ENDPOINTS.
ENDPOINT_ID = "5v72jxyxmp31jv"

ENDPOINTS = ["http://localhost:5002"]        # only used when ENDPOINT_ID == ""

IMAGE_DIR = r"\\ABOT-TEST-03\MBRSUploadTraining\YE2026\3C876CBE-23C6-4DA5-B082-23483B4063BA\SplitImage"

MODE = "stress"                              # ladder | stress | coldstart | all

# SAME_IMAGE: use ONE page for every call, so per-level numbers differ only by
# concurrency, not by which pages the rotation happened to deal (the N=6/8 dips in
# the first ladder were exactly that - those levels drew the heavy full-text pages).
#   ""              off: rotate distinct pages (realistic, no cache help)
#   "1"             use the first page of the corpus
#   "Page_012.png"  use that specific page
# CAVEAT: vLLM has prefix caching enabled, so repeating one page may score FASTER
# than real mixed traffic. Same-image mode is for COMPARING levels fairly, not for
# quoting absolute throughput - quote absolute numbers from distinct-page runs.
# Phase 3: distinct pages — production traffic is distinct pages, and the prefix
# cache must NOT be allowed to flatter the final number we quote.
SAME_IMAGE = ""

# Calibration-campaign ladder: fine steps, ascending, STOP at first fail — after an
# OOM the worker is refreshed (by design), so any level measured after a failure would
# run on a cold replacement and be garbage. One cliff per boot is the protocol.
LADDER_LEVELS = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
LADDER_STOP_ON_EMPTY = True
PAUSE_BETWEEN_LEVELS = 3

# The REAL peak shape (Lotion): ~40 concurrent queues x 4 requests each = 160 jobs
# submitted SIMULTANEOUSLY, not fed through a window. STRESS_CONC == STRESS_TOTAL
# means every request is posted at once; RunPod's queue absorbs the pile and the
# worker drains it 16 at a time (the calibrated gate).
STRESS_TOTAL = 480
STRESS_CONC = 480

# Gaps (seconds) between coldstart probes. Each must EXCEED the endpoint's idle timeout
# or the worker never scales down and the probe measures nothing.
COLDSTART_GAPS = [700, 700]

TIMEOUT = 900
RETRIES = 0
POLL_S = 2.0

# The API key is NOT hardcoded on purpose - this file lives in the git repo, and a
# pasted key would leak the way the GitHub token pasted into chat did. It is read from
# the file `flash login` writes; nothing to set, no env var needed.
def _load_key():
    try:
        import re
        cfg = open(os.path.join(os.path.expanduser("~"), ".runpod", "config.toml")).read()
        m = re.search(r'(?im)^\s*(?:api_key|apikey|key)\s*=\s*"?([^"\r\n]+)"?', cfg)
        return m.group(1).strip() if m else ""
    except OSError:
        return ""

API_KEY = _load_key()
# ==================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
SERVERLESS = bool(ENDPOINT_ID)
RP = "https://api.runpod.ai/v2"


# ------------------------------------------------------------------ helpers
def png_size(path):
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
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


# ------------------------------------------------------------------ transports
def _call_direct(endpoint, uri):
    r = requests.post(endpoint.rstrip("/") + "/glmocr/parse",
                      json={"images": [uri]}, timeout=TIMEOUT)
    if r.status_code != 200:
        return None, "HTTP %d: %s" % (r.status_code, r.text[:160]), {}
    return r.json(), "", {}


def _call_serverless(_endpoint, uri):
    """POST /run then poll /status. /runsync is unusable: its result window is 1 min
    (5 max) and a cold boot is 6-7 min."""
    hdr = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}
    r = requests.post("%s/%s/run" % (RP, ENDPOINT_ID), headers=hdr,
                      json={"input": {"image": uri}}, timeout=120)
    if r.status_code != 200:
        return None, "submit HTTP %d: %s" % (r.status_code, r.text[:160]), {}
    job = r.json()
    jid = job.get("id")
    if not jid:
        return None, "no job id: %s" % str(job)[:160], {}

    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        g = requests.get("%s/%s/status/%s" % (RP, ENDPOINT_ID, jid), headers=hdr, timeout=60)
        if g.status_code != 200:
            return None, "status HTTP %d: %s" % (g.status_code, g.text[:160]), {}
        job = g.json()
        st = job.get("status")
        if st in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(POLL_S)

    # delayTime = queue + cold boot. This is the number the FlashBoot question lives in.
    meta = {"delay_ms": job.get("delayTime"), "exec_ms": job.get("executionTime"),
            "status": job.get("status"), "job_id": jid}
    if job.get("status") != "COMPLETED":
        return None, "job %s: %s" % (job.get("status"), str(job.get("error"))[:160]), meta
    return job.get("output"), "", meta


def call(endpoint, path, label):
    """One page. ok == non-empty markdown, not merely HTTP 200."""
    uri = as_uri(path)
    t0 = time.perf_counter()
    err, meta = "", {}
    for attempt in range(RETRIES + 1):
        try:
            body, err, meta = (_call_serverless if SERVERLESS else _call_direct)(endpoint, uri)
            if not err:
                return _res(label, path, t0, body, "", attempt, meta)
            if "submit HTTP 4" in err or "job FAILED" in err:
                break                                   # not transient
        except Exception as ex:
            err = "%s: %s" % (type(ex).__name__, str(ex)[:120])
        time.sleep(1.0 * (attempt + 1))
    return _res(label, path, t0, None, err, RETRIES, meta)


def _res(label, path, t0, body, err, retries, meta):
    md = md_of(body)
    return {
        "label": label, "img": os.path.basename(path),
        "wall": time.perf_counter() - t0,
        "md_len": len(md), "tables": md.count("<table"), "elems": len(flat_elems(body)),
        "ok": bool(md.strip()) and not err,
        "retries": retries, "error": err,
        "delay_s": (meta.get("delay_ms") or 0) / 1000.0 if meta.get("delay_ms") else None,
        "exec_s": (meta.get("exec_ms") or 0) / 1000.0 if meta.get("exec_ms") else None,
        "warning": (body or {}).get("warning") if isinstance(body, dict) else None,
        "_body": body,
    }


def show(r, prefix=""):
    extra = ""
    if r["delay_s"] is not None:
        extra = " delay=%5.1fs exec=%5.1fs" % (r["delay_s"], r["exec_s"] or 0)
    print("%s%s %-9s %-14s wall=%6.1fs%s md=%-5d elems=%-3d tables=%d %s" % (
        prefix, "OK  " if r["ok"] else "FAIL", r["label"], r["img"], r["wall"], extra,
        r["md_len"], r["elems"], r["tables"], r["error"] or (r["warning"] or "")))


def diagnose_empty(n_empty, level):
    print("")
    print("  >>> %d/%d calls returned EMPTY markdown at concurrency %d." % (n_empty, level, level))
    print("  >>> Two known causes, both now HARD-FAILED by the handler:")
    print("  >>>   a) layout-stage CUDA OOM (skipped batch, HTTP 200) -> job fails loudly")
    print("  >>>   b) vLLM engine death (CUDA assert, connection refused) -> job fails +")
    print("  >>>      worker replaced (refresh_worker)")
    print("  >>> So an EMPTY that still reaches this client should only be a genuinely")
    print("  >>> blank page (carries a 'warning' field). Anything else = handler regression.")
    print("  >>> Worker forensics: grep -inE 'assert|out of memory' /var/log/vllm.log")
    print("  >>>                   grep -c 'skipping batch' /var/log/glmocr.log")


# ------------------------------------------------------------------ contract check
def contract_check(body, path):
    print("\n=== CONTRACT CHECK (vs Z.ai layout_parsing) ===")
    if not isinstance(body, dict):
        print("  response is not a JSON object — nothing to check")
        return
    md, elems = md_of(body), flat_elems(body)
    print("  md_results        : %d chars, %d table(s)" % (len(md), md.count("<table")))
    print("  layout_details    : %d elements" % len(elems))
    if not elems:
        print("  !! no elements — stitching / title logic would go blind")
        return

    need = ("bbox_2d", "content", "index", "native_label")
    missing = [k for k in need if k not in elems[0]]
    print("  element fields    : %s" % ("OK" if not missing else "MISSING " + str(missing)))
    print("  native_label      : %s" % sorted({e.get("native_label") for e in elems if isinstance(e, dict)}))

    xs = [c for e in elems if isinstance(e.get("bbox_2d"), list) for c in e["bbox_2d"][0::2]]
    ys = [c for e in elems if isinstance(e.get("bbox_2d"), list) for c in e["bbox_2d"][1::2]]
    iw, ih = png_size(path)
    if xs and ys:
        mx, my = max(xs), max(ys)
        print("  bbox max x/y      : %d/%d   (page %dx%d)" % (mx, my, iw, ih))
        if max(mx, my) <= 1000 < max(iw, ih):
            print("  !! still 0-1000 NORMALIZED. Through the serverless handler this means the")
            print("     adapter is broken; ReportStitchOcr / HeaderRecovery / BboxAspect would")
            print("     misplace silently. Expected: x_px = round(x * %d / 1000)." % iw)
        else:
            print("  -> PIXELS, as the cloud contract requires")
    if "polygon" in elems[0]:
        print("  !! 'polygon' leaked through — the adapter should drop it")
    if "height" not in elems[0]:
        print("  !! per-element height/width missing — the cloud has them")


# ------------------------------------------------------------------ ladder
def run_ladder(images):
    print("\n" + "=" * 74)
    print("LADDER — climbing concurrency until the first EMPTY (= the OOM cliff)")
    print("=" * 74)
    idx, rows = 0, []
    for k in LADDER_LEVELS:
        print("\n=== LEVEL %d ===" % k)
        batch = []
        for i in range(k):
            batch.append((ENDPOINTS[i % len(ENDPOINTS)], images[idx % len(images)], "L%d-%d" % (k, i + 1)))
            idx += 1
        t0, results = time.perf_counter(), []
        with ThreadPoolExecutor(max_workers=k) as ex:
            for f in as_completed([ex.submit(call, e, p, lb) for e, p, lb in batch]):
                r = f.result(); results.append(r); show(r, "  ")
        wall = time.perf_counter() - t0
        bad = [r for r in results if not r["ok"]]
        # Count only SUCCESSFUL pages: an OOM-skipped call returns empty in ~2s, so
        # counting it would make the level that just broke look like the fastest one.
        thr = (k - len(bad)) / wall
        rows.append((k, wall, thr, len(bad)))
        print("  LEVEL %d wall %.1fs  %.2f pages/s (ok only)  empty/fail %d/%d"
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
        print("  -> set MAX_CONCURRENCY to at most %d (leave a margin below the cliff)"
              % max(1, best[0]))
        print("  -> a 28-page report ~ %.0fs of GPU time" % (28 / best[2]))
    return rows


# ------------------------------------------------------------------ stress
def run_stress(images):
    print("\n" + "=" * 74)
    print("STRESS — %d calls at %d in-flight (distinct pages)" % (STRESS_TOTAL, STRESS_CONC))
    print("=" * 74)
    tasks = [(ENDPOINTS[i % len(ENDPOINTS)], images[i % len(images)], "s%d" % (i + 1))
             for i in range(STRESS_TOTAL)]
    results, t0 = [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=STRESS_CONC) as ex:
        for f in as_completed([ex.submit(call, e, p, lb) for e, p, lb in tasks]):
            r = f.result(); results.append(r)
            show(r, "  [%3d/%d] " % (len(results), STRESS_TOTAL))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    walls = [r["wall"] for r in ok]
    print("\n=== STRESS SUMMARY ===")
    print("  calls %d   ok %d   FAIL/EMPTY %d" % (STRESS_TOTAL, len(ok), len(bad)))
    print("  burst wall %.1fs   throughput %.2f pages/s (%.0f pages/min)"
          % (wall, len(ok) / wall, len(ok) / wall * 60))
    if walls:
        print("  latency p50 %.1fs   p95 %.1fs   max %.1fs   mean %.1fs"
              % (pct(walls, 50), pct(walls, 95), max(walls), statistics.mean(walls)))
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


# ------------------------------------------------------------------ coldstart / FlashBoot
def run_coldstart(images):
    print("\n" + "=" * 74)
    print("COLDSTART — does FlashBoot actually revive this worker cheaply?")
    print("=" * 74)
    if not SERVERLESS:
        print("  SKIPPED: only meaningful against a serverless endpoint (set RUNPOD_ENDPOINT_ID)")
        return []
    print("  Each gap must EXCEED the endpoint's Idle Timeout, or the worker never scales")
    print("  down and this measures nothing. Gaps: %s s\n" % COLDSTART_GAPS)

    rows = []
    r = call(ENDPOINTS[0], images[0], "boot-1")
    show(r, "  ")
    rows.append(("boot-1 (first ever)", r))
    for i, gap in enumerate(COLDSTART_GAPS, start=2):
        print("  ... sleeping %ds so the worker scales down ..." % gap)
        time.sleep(gap)
        r = call(ENDPOINTS[0], images[i % len(images)], "boot-%d" % i)
        show(r, "  ")
        rows.append(("boot-%d (after %ds idle)" % (i, gap), r))

    print("\n--- COLDSTART SUMMARY (delay = queue + boot) ---")
    for name, r in rows:
        d = "%.1fs" % r["delay_s"] if r["delay_s"] is not None else "n/a"
        print("  %-26s delay=%8s  exec=%5.1fs  ok=%s"
              % (name, d, r["exec_s"] or 0, r["ok"]))
    delays = [r["delay_s"] for _, r in rows[1:] if r["delay_s"] is not None]
    if delays:
        worst = max(delays)
        print("\n  worst revive delay: %.1fs" % worst)
        if worst < 60:
            print("  -> FlashBoot IS working. Idle Timeout can drop to ~60s and the endpoint")
            print("     can sit at zero between bursts — cheapest possible configuration.")
        elif worst < 180:
            print("  -> FlashBoot helps but is not free. Keep Idle Timeout around 600s.")
        else:
            print("  -> FlashBoot is NOT saving us (a full boot is ~400s). Options: raise Idle")
            print("     Timeout to cover the workday, or revive warm_schedule.py with an")
            print("     endpoint-write API key. Do NOT ship a 6-minute first-page wait.")
    return rows


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    if SERVERLESS and not API_KEY:
        sys.exit("RUNPOD_ENDPOINT_ID is set but RUNPOD_API_KEY is not")
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "Page_*.png")))
    if not images:
        sys.exit("no Page_*.png found in %s" % IMAGE_DIR)

    if SAME_IMAGE:
        if SAME_IMAGE == "1":
            target = images[0]
        else:
            match = [p for p in images if os.path.basename(p).lower() == SAME_IMAGE.lower()]
            if not match:
                sys.exit("SAME_IMAGE=%r not found in %s" % (SAME_IMAGE, IMAGE_DIR))
            target = match[0]
        # a single-element corpus: every rotation index maps to the same page
        images = [target]

    print("transport : %s" % ("serverless " + ENDPOINT_ID if SERVERLESS else "direct " + str(ENDPOINTS)))
    if SAME_IMAGE:
        print("corpus    : SAME-IMAGE mode -> %s for every call" % os.path.basename(images[0]))
        print("            (fair level-to-level comparison; prefix cache may flatter")
        print("             absolute numbers - quote those from distinct-page runs)")
    else:
        print("corpus    : %d pages from %s" % (len(images), IMAGE_DIR))
    print("mode      : %s" % MODE)

    if MODE == "coldstart":
        run_coldstart(images)
        sys.exit(0)

    # Warm every endpoint. Each glmocr process loads its layout model on ITS first
    # request; an unwarmed backend dumps that one-off cost into the first measured level
    # and fakes a low parallelism number.
    # print("\n=== WARMUP ===")
    # first = None
    # for e in ENDPOINTS:
    #     r = call(e, images[0], "warm")
    #     show(r, "  ")
    #     first = first or r
    # if not first["ok"]:
    #     print("\n!! WARMUP FAILED — nothing below would mean anything.")
    #     if not first["error"]:
    #         diagnose_empty(1, 1)
    #     sys.exit(1)

    # contract_check(first["_body"], images[0])

    if MODE in ("ladder", "all"):
        run_ladder(images)
    if MODE in ("stress", "all"):
        run_stress(images)
    if MODE == "all":
        run_coldstart(images)
