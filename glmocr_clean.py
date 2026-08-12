"""glmocr_clean.py - launch glmocr.server with in-place memory cleaning.

Lotion's ruling (2026-08-13): NO proactive kill-and-restart of the glmocr process.
Instead, after every parse request completes, hand memory back on the spot:

    gc.collect()          free unreferenced Python objects
    libc.malloc_trim(0)   return freed heap pages from glibc's arenas to the OS

Best-effort by nature: memory glmocr's pipeline still holds a reference to cannot be
freed by anyone. If RSS climbs to the container cap regardless, RunPod OOM-kills the
whole container (full worker reboot) - that trade was accepted when the RSS fence
was removed.

Why patch Flask's class instead of the app: glmocr builds its Flask app inside a
function at startup (no importable module-level app object), so the reliable hook is
the class method every request passes through, patched BEFORE glmocr constructs the
app. glmocr.server then runs exactly as `python -m glmocr.server` would, same argv.

Cleaning is gated to the parse endpoint only - a gc.collect() on every health probe
would be pure overhead.
"""
import ctypes
import gc
import runpy
import sys

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:          # non-glibc platform (never the case in our image)
    _libc = None

import flask

_orig_dispatch = flask.Flask.full_dispatch_request
_PARSE_PATH = "/glmocr/parse"


def _cleaning_dispatch(self, *args, **kwargs):
    try:
        return _orig_dispatch(self, *args, **kwargs)
    finally:
        try:
            if flask.request.path == _PARSE_PATH:
                gc.collect()
                if _libc is not None:
                    _libc.malloc_trim(0)
        except Exception:
            pass            # cleaning must never break a response


flask.Flask.full_dispatch_request = _cleaning_dispatch

if __name__ == "__main__":
    sys.argv[0] = "glmocr.server"
    runpy.run_module("glmocr.server", run_name="__main__", alter_sys=True)
