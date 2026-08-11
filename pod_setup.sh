#!/usr/bin/env bash
# =============================================================================
# GLM-OCR self-hosted stack — RUNPOD POD bring-up (PHASE 1: manual validation)
#
#   bash /workspace/pod_setup.sh
#
# Boots: 1x vLLM (serves zai-org/GLM-OCR)  +  N x glmocr Flask servers.
# glmocr's server is single-threaded Flask app.run() -> ONE parse at a time per
# process, so N processes = N concurrent pages. All N share the one vLLM/GPU.
#
# EVERY VERSION IS PINNED. The Replicate build drifted between pushes and started
# returning empty markdown on real pages with an unchanged config — never again.
#
# Re-runnable: kills the previous stack first.
# =============================================================================
set -euo pipefail

# ----------------------------- CONFIG ---------------------------------------
VLLM_VER="0.19.1"          # PyPI latest 2026-04-18; glmocr[selfhosted] wants >=0.17.0
GLMOCR_VER="0.1.5"         # PyPI latest 2026-04-08
TRANSFORMERS_SPEC=">=5.3.0"  # glmocr[selfhosted] requirement for the GLM-V arch

N_BACKENDS="${N_BACKENDS:-3}"     # glmocr server processes = max concurrent pages
BASE_PORT=5002                    # -> 5002, 5003, 5004
VLLM_PORT=8080

USE_MTP="${USE_MTP:-1}"           # Multi-Token Prediction speculative decoding.
                                  # Official vLLM recipe for GLM-OCR recommends it
                                  # (the model ships MTP layers). Set 0 to disable
                                  # if vLLM rejects the flag on this version.
# ROOT CAUSE OF THE REPLICATE "md=0" BUG, found in this pod's glmocr log:
#   Layout detection failed for pages [0], skipping batch: CUDA out of memory.
#   vLLM had 16.69 GiB of 22.03 GiB; each of the 3 layout processes wants ~1.80 GiB;
#   255 MiB left.
# On OOM the layout stage SKIPS THE BATCH and the server still answers HTTP 200 with
# EMPTY markdown - fast, silent, and indistinguishable from success. It was never
# version drift. Budget: vLLM gets util*VRAM, the N layout processes need ~1.8 GiB
# EACH on top, plus headroom. 0.55 on a 24GB L4 leaves ~10 GiB for 3 of them.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
export PYTORCH_ALLOC_CONF=expandable_segments:True   # cuts fragmentation in the layout procs
MAX_MODEL_LEN=32768               # glmocr's page loader asks 8192 OUTPUT tokens;
                                  # serving 8192 total => HTTP 400. Hard-won.

# MEASURED on this pod: /workspace is a NETWORK filesystem (mfs#euro.runpod.net),
# not a local disk. A venv is tens of thousands of tiny files and imports would
# crawl over it, so the venv lives on the local container disk. The models are big
# sequential reads, which the network volume handles fine — and keeping them there
# means a pod Stop/Start doesn't re-download 2GB+.
VENV="${VENV:-/opt/venv}"         # local container disk (20GB) — fast imports
LOGS=/workspace/logs
CFGS=/workspace/cfg
export HF_HOME=/workspace/hf      # persistent volume -> re-runs skip download
# -----------------------------------------------------------------------------

mkdir -p "$LOGS" "$CFGS" "$HF_HOME"

echo "=== 0/6  stopping any previous stack ==="
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "glmocr.server"    2>/dev/null || true
sleep 3

if [ ! -x "$VENV/bin/python" ]; then
  echo "=== 1/6  creating venv (isolated from the pod template's torch) ==="
  # DETECT the interpreter instead of assuming one — the template's Python differs by
  # base image (ubuntu2204 -> 3.10/3.11, ubuntu2404 -> 3.12) and glmocr needs >=3.10.
  # Prefer 3.12 > 3.11 > 3.10 (all have vllm wheels); refuse anything else loudly.
  PY=""
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      V=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
      case "$V" in
        3.10|3.11|3.12) PY="$cand"; echo "    using $cand (Python $V)"; break ;;
        *) echo "    skipping $cand (Python $V — outside 3.10-3.12)" ;;
      esac
    fi
  done
  [ -n "$PY" ] || { echo "NO USABLE PYTHON (need 3.10-3.12). Pick a different pod template."; exit 1; }

  # Debian/Ubuntu split venv into its own package; install on demand rather than failing.
  if ! "$PY" -m venv "$VENV" 2>/dev/null; then
    echo "    venv module missing — apt installing ${PY}-venv"
    apt-get update -qq && apt-get install -y -qq "${PY}-venv"
    "$PY" -m venv "$VENV"
  fi
  "$VENV/bin/pip" install --no-cache-dir --upgrade pip
else
  echo "=== 1/6  venv exists — skip  ($("$VENV/bin/python" -V)) ==="
fi

echo "=== 2/6  installing PINNED deps (first run pulls ~5GB of torch/vllm) ==="
"$VENV/bin/pip" install --no-cache-dir \
    "vllm==${VLLM_VER}" \
    "glmocr[selfhosted,server]==${GLMOCR_VER}" \
    "transformers${TRANSFORMERS_SPEC}"
echo "--- installed versions (RECORD THESE) ---"
"$VENV/bin/pip" list 2>/dev/null | grep -Ei "^(vllm|glmocr|transformers|torch) " || true

echo "=== 3/6  pre-downloading BOTH models into $HF_HOME ==="
# Boot-time anonymous HF download proved flaky on Replicate (one boot came up with
# an empty layout model -> silent empty markdown for a whole session). Fetch both
# up front, onto the persistent volume, and fail loudly here if either is missing.
"$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("zai-org/GLM-OCR", "PaddlePaddle/PP-DocLayoutV3_safetensors"):
    p = snapshot_download(repo)
    print(f"  OK {repo} -> {p}")
PY

echo "=== 4/6  starting vLLM on :$VLLM_PORT ==="
SPEC=()
if [ "$USE_MTP" = "1" ]; then
  SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
fi
# --served-model-name BOTH names: glmocr sends 'glm-ocr' with the package config and
# 'default' with a minimal one. Serving both removes a whole class of 404s.
nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model zai-org/GLM-OCR \
    --port "$VLLM_PORT" \
    --served-model-name default glm-ocr \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    ${SPEC[@]+"${SPEC[@]}"} \
    > "$LOGS/vllm.log" 2>&1 &

echo -n "    waiting for vLLM /health "
for i in $(seq 1 450); do
  if curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then echo " UP"; break; fi
  if [ "$i" = "450" ]; then echo " TIMEOUT"; tail -50 "$LOGS/vllm.log"; exit 1; fi
  sleep 2; echo -n "."
done

echo "=== 5/6  starting $N_BACKENDS glmocr servers ==="
for i in $(seq 0 $((N_BACKENDS - 1))); do
  PORT=$((BASE_PORT + i))
  cat > "$CFGS/overrides_$PORT.yaml" <<EOF
pipeline:
  maas:
    enabled: false
  layout:
    model_dir: PaddlePaddle/PP-DocLayoutV3_safetensors
  ocr_api:
    api_host: localhost
    api_port: $VLLM_PORT
server:
  host: 0.0.0.0
  port: $PORT
  debug: false
EOF
  # runner.py merges the glmocr PACKAGE config as the base. A hand-written minimal
  # config loses label_task_mapping -> every native_label degrades to 'text' and the
  # C# title/dedup logic goes blind. Do not "simplify" this away.
  "$VENV/bin/python" /workspace/runner.py "$CFGS/overrides_$PORT.yaml" "$CFGS/merged_$PORT.yaml"
  nohup "$VENV/bin/python" -m glmocr.server --config "$CFGS/merged_$PORT.yaml" \
      > "$LOGS/glmocr_$PORT.log" 2>&1 &
  echo "    launched glmocr :$PORT"
done

for i in $(seq 0 $((N_BACKENDS - 1))); do
  PORT=$((BASE_PORT + i))
  echo -n "    waiting :$PORT "
  for j in $(seq 1 300); do
    if curl -s -o /dev/null "http://localhost:$PORT/" 2>/dev/null; then echo " UP"; break; fi
    if [ "$j" = "300" ]; then echo " TIMEOUT"; tail -50 "$LOGS/glmocr_$PORT.log"; exit 1; fi
    sleep 2; echo -n "."
  done
done

# PARITY NOTE for the Phase 2 adapter (verified on this pod against the Z.ai golden):
#   glmocr/layout/base.py:42  "bbox_2d: normalized coordinates [x1,y1,x2,y2] (0-1000)"
#   glmocr/api.py:353         "The MaaS API returns bbox_2d in absolute pixel coordinates"
# So SELF-HOSTED gives 0-1000, the Z.ai CLOUD gives pixels. Check: Page_005 max x = 903
# -> 903/1000*3166 = 2859 px, and the Z.ai sample's max x is 2861. The adapter MUST do
#   x_px = round(x * W / 1000)   y_px = round(y * H / 1000)
# or ReportStitchOcr's seam, HeaderRecovery and BboxAspect all silently misplace.
echo "=== 6/6  READY ==="
echo "  vLLM     : http://localhost:$VLLM_PORT"
for i in $(seq 0 $((N_BACKENDS - 1))); do
  echo "  glmocr   : http://localhost:$((BASE_PORT + i))/glmocr/parse"
done
echo
echo "  logs     : tail -f $LOGS/vllm.log   |   tail -f $LOGS/glmocr_$BASE_PORT.log"
echo "  external : https://<POD_ID>-<PORT>.proxy.runpod.net    (expose 5002..$((BASE_PORT + N_BACKENDS - 1)) as HTTP ports)"
