"""
RunPod endpoint warm scheduler — ONE fixed GPU, warm in working hours, cold at night.

    set RUNPOD_API_KEY=...        (never hardcode it; never commit it)
    python warm_schedule.py

RunPod has NO built-in schedule feature, so we flip `workersMin` ourselves:

    workersMax = 1        fixed at one GPU, forever. Never autoscales.
    workersMin = 1        during working hours  -> worker stays up (Active rate)
    workersMin = 0        outside               -> scales to zero, night requests cold-boot

SELF-HEALING BY DESIGN
    This script computes the DESIRED state from the clock and reconciles. Run it every
    5 minutes from Task Scheduler / cron with a single trigger. A missed run fixes
    itself on the next tick — no state file, no paired on/off jobs that can desync.

CHECK-THEN-PATCH
    It GETs the current value and only PATCHes on a real change. On Replicate a PATCH
    with an unchanged value triggered a release rollout that REPLACED the running
    instance — a pointless cold boot. Whether RunPod does the same is UNVERIFIED, so
    we simply never issue a no-op write.

WARM-UP LEAD TIME
    Cold start is 6-7 minutes (measured), so WARM_FROM is set EARLIER than the hour
    people actually start. 07:50 -> warm by ~07:57.
"""
import datetime
import os
import sys

import requests

# ======================= CONFIG (edit here) =======================
ENDPOINT_ID = "PUT_ENDPOINT_ID_HERE"

WARM_FROM = datetime.time(7, 50)     # start booting BEFORE the workday (6-7 min cold start)
WARM_UNTIL = datetime.time(19, 0)    # 7pm
WARM_ON_WEEKENDS = False             # True if the firm works Saturdays — see cost note below

MAX_WORKERS = 1                      # ONE card. Never more. Bursts queue instead of scaling.
# ==================================================================

API = "https://rest.runpod.io/v1"
KEY = os.environ.get("RUNPOD_API_KEY", "")


def want_warm(now):
    if not WARM_ON_WEEKENDS and now.weekday() >= 5:      # 5=Sat 6=Sun
        return False
    return WARM_FROM <= now.time() < WARM_UNTIL


def get_endpoint(hdr):
    r = requests.get(f"{API}/endpoints/{ENDPOINT_ID}", headers=hdr, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    if not KEY:
        sys.exit("RUNPOD_API_KEY not set in the environment")
    if ENDPOINT_ID == "PUT_ENDPOINT_ID_HERE":
        sys.exit("edit ENDPOINT_ID at the top of this file")

    hdr = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    now = datetime.datetime.now()                        # LOCAL time (MYT on our servers)
    want_min = 1 if want_warm(now) else 0

    ep = get_endpoint(hdr)
    cur_min = ep.get("workersMin")
    cur_max = ep.get("workersMax")

    patch = {}
    if cur_min != want_min:
        patch["workersMin"] = want_min
    if cur_max != MAX_WORKERS:
        # keeps the "one fixed card" guarantee even if someone changes it in the console
        patch["workersMax"] = MAX_WORKERS

    stamp = now.strftime("%Y-%m-%d %H:%M")
    if not patch:
        print(f"{stamp}  min={cur_min} max={cur_max}  already correct — no PATCH")
        return

    r = requests.patch(f"{API}/endpoints/{ENDPOINT_ID}", headers=hdr, json=patch, timeout=30)
    r.raise_for_status()
    print(f"{stamp}  min {cur_min}->{patch.get('workersMin', cur_min)} "
          f"max {cur_max}->{patch.get('workersMax', cur_max)}  (HTTP {r.status_code})")


if __name__ == "__main__":
    main()
