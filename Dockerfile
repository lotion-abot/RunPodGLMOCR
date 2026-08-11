# GLM-OCR self-hosted — RunPod Serverless worker
#
# Base is the OFFICIAL vLLM image at the exact version Phase 1 validated. That gives us
# vLLM + torch + CUDA already matched to each other, so the build cannot drift the way
# the Replicate/cog build did (same config, different behaviour, silent empty markdown).
#
# Everything Phase 1 measured is pinned here. Do not "upgrade for cleanliness".
FROM vllm/vllm-openai:v0.19.1

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# glmocr on top of the vLLM image. transformers is pinned to what Phase 1 actually ran.
# The assert is the point: if pip's resolver ever moves vllm or torch out from under us,
# the BUILD fails loudly instead of shipping a worker that returns empty markdown.
RUN pip install --no-cache-dir \
        "glmocr[selfhosted,server]==0.1.5" \
        "transformers==5.15.0" \
        "runpod" \
 && python -c "\
import vllm, torch, transformers, glmocr; \
assert vllm.__version__ == '0.19.1', 'vllm drifted: ' + vllm.__version__; \
assert torch.__version__.startswith('2.10.0'), 'torch drifted: ' + torch.__version__; \
assert transformers.__version__ == '5.15.0', 'transformers drifted: ' + transformers.__version__; \
print('PINNED OK  vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"

# Bake BOTH models into the image. On Replicate, downloading them at boot was flaky —
# one boot came up with a broken layout model and served empty markdown for a whole
# session. Baked = deterministic, and no HF dependency at runtime.
RUN python -c "\
from huggingface_hub import snapshot_download; \
[print('baked', r, '->', snapshot_download(r)) for r in ('zai-org/GLM-OCR', 'PaddlePaddle/PP-DocLayoutV3_safetensors')]"

# Freeze the merged glmocr config at BUILD time (it never changes at runtime).
# runner.py deep-merges our overrides onto the glmocr PACKAGE default config. Skipping
# that merge loses label_task_mapping, every native_label degrades to 'text', and the
# C# title detection / dedup / header recovery all go blind — silently.
COPY runner.py /app/runner.py
RUN printf '%s\n' \
      'pipeline:' \
      '  maas:' \
      '    enabled: false' \
      '  layout:' \
      '    model_dir: PaddlePaddle/PP-DocLayoutV3_safetensors' \
      '  ocr_api:' \
      '    api_host: 127.0.0.1' \
      '    api_port: 8080' \
      'server:' \
      '  host: 127.0.0.1' \
      '  port: 5002' \
      '  debug: false' \
      > /app/overrides.yaml \
 && python /app/runner.py /app/overrides.yaml /app/glmocr.yaml \
 && grep -q label_task_mapping /app/glmocr.yaml \
 && echo "merged config OK (label_task_mapping present)"

# A REAL audit page for the boot self-test. The Replicate build self-tested with a
# synthetic 400x120 image: it passed on every boot while real 3166x4096 pages came back
# empty. A self-test that cannot fail is not a self-test.
COPY selftest_page.png /app/selftest_page.png

COPY handler.py /app/handler.py
COPY start.sh    /app/start.sh
RUN chmod +x /app/start.sh

# Layout needs ~1.9 GiB of VRAM ON TOP of whatever vLLM reserves. Exceeding the card is
# what produced the silent empty results. See PHASE2-DESIGN.md before changing these.
ENV GPU_MEMORY_UTILIZATION=0.70 \
    MAX_MODEL_LEN=32768 \
    MAX_CONCURRENCY=4 \
    SELFTEST_MIN_MD=800 \
    PYTORCH_ALLOC_CONF=expandable_segments:True

ENTRYPOINT []
CMD ["/app/start.sh"]
