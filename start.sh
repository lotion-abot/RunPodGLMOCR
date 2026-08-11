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
# --speculative-config mtp: the model ships MTP layers; the official vLLM recipe
#   recommends it and 0.19.1 accepts it (verified on the pod).
python3 -m vllm.entrypoints.openai.api_server \
    --model zai-org/GLM-OCR \
    --port "$VLLM_PORT" \
    --served-model-name default glm-ocr \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
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
exec python3 /app/handler.py
