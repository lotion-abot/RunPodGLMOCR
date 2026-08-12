#!/usr/bin/env bash
# Container entrypoint: vLLM -> ONE glmocr server -> the RunPod handler.
#
# ONE glmocr process, not three. Measured on the pod: 1/2/3 processes give
# 0.92/0.95/0.96 pages/s — the extra two buy 4% and cost 3.8 GiB of VRAM. The premise
# for running three was wrong anyway: predict.py claimed Flask's app.run() is
# single-threaded, but Flask has defaulted to threaded=True since 1.0.
set -euo pipefail

VLLM_PORT=8080
GLMOCR_PORT=5002
GLMOCR_LOG=/var/log/glmocr.log

echo "=== 1/3  vLLM on :$VLLM_PORT ==="
# --served-model-name BOTH names: glmocr sends 'glm-ocr' with the package config and
# 'default' with a minimal one. Serving both removes a whole class of 404s.
# --max-model-len 32768: glmocr asks for 8192 OUTPUT tokens; serving 8192 total => 400.
#
# MTP speculative decoding is OFF by default. 2026-08-12, worker 2c0f197ad821, L4,
# ~7 min into a concurrency ladder: EngineCore died with
#   vectorized_gather_kernel ... Assertion `ind >= 0 && ind < ind_dim_size` failed
# (an index-out-of-bounds in a gather kernel - not OOM), and every request after that
# was Connection refused. The draft-token gathers of MTP are the prime suspect for
# exactly this class of assert. It is an optional speed-up; stability wins. Re-enable
# only as a controlled experiment via USE_MTP=1.
USE_MTP="${USE_MTP:-0}"
SPEC_ARGS=()
if [ "$USE_MTP" = "1" ]; then
  SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
fi
python3 -m vllm.entrypoints.openai.api_server \
    --model zai-org/GLM-OCR \
    --port "$VLLM_PORT" \
    --served-model-name default glm-ocr \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
    > /var/log/vllm.log 2>&1 &
VLLM_PID=$!

echo -n "    waiting for /health "
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1; then echo " UP"; break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo " DIED"; tail -60 /var/log/vllm.log; exit 1; fi
  if [ "$i" = "600" ]; then echo " TIMEOUT"; tail -60 /var/log/vllm.log; exit 1; fi
  sleep 2
done

echo "=== 2/3  glmocr on :$GLMOCR_PORT ==="
# Config was merged at BUILD time (/app/glmocr.yaml) — nothing to generate here.
python3 -m glmocr.server --config /app/glmocr.yaml > "$GLMOCR_LOG" 2>&1 &
GLMOCR_PID=$!

echo -n "    waiting "
for i in $(seq 1 300); do
  # '/' legitimately 404s — any HTTP answer means the server is listening.
  if curl -s -o /dev/null "http://127.0.0.1:$GLMOCR_PORT/" 2>/dev/null; then echo " UP"; break; fi
  if ! kill -0 "$GLMOCR_PID" 2>/dev/null; then echo " DIED"; tail -60 "$GLMOCR_LOG"; exit 1; fi
  if [ "$i" = "300" ]; then echo " TIMEOUT"; tail -60 "$GLMOCR_LOG"; exit 1; fi
  sleep 2
done

echo "=== 3/3  handler (self-test runs at import) ==="
python3 /app/handler.py &
HANDLER_PID=$!

# WATCHDOG - the zombie lesson. When EngineCore died mid-session, the vLLM process
# exited but Flask and the handler stayed up, so the worker kept answering HTTP 200
# with empty markdown for every job it took. A worker whose OCR brain is gone must
# DIE, not serve: exiting here makes RunPod replace it (a clean cold boot) instead of
# letting it silently blank pages until someone notices.
while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "WATCHDOG: vLLM process died - tail of /var/log/vllm.log follows; exiting so RunPod replaces this worker"
    tail -40 /var/log/vllm.log
    kill "$HANDLER_PID" "$GLMOCR_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$GLMOCR_PID" 2>/dev/null; then
    echo "WATCHDOG: glmocr server died - tail of $GLMOCR_LOG follows; exiting"
    tail -40 "$GLMOCR_LOG"
    kill "$HANDLER_PID" "$VLLM_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$HANDLER_PID" 2>/dev/null; then
    wait "$HANDLER_PID"; rc=$?
    echo "WATCHDOG: handler exited rc=$rc - exiting"
    kill "$VLLM_PID" "$GLMOCR_PID" 2>/dev/null || true
    exit "$rc"
  fi
  sleep 10
done
