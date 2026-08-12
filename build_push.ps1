# =============================================================================
# Local build + push of the GLM-OCR worker image.
#
#   .\build_push.ps1              build, verify, push
#   .\build_push.ps1 -NoPush      build + verify only (the fast inner loop)
#   .\build_push.ps1 -Latest      also move the :latest tag
#
# WHY LOCAL
#   CI is ~10 min per attempt. Locally, Docker's layer cache makes a re-build after
#   a handler.py change take seconds. The two failures that cost two CI rounds today
#   (an apt-managed blinker that pip refused to replace, and `python` not existing in
#   the vLLM base) would each have surfaced in under a minute here.
#
# WHY IT STILL HURTS ONCE
#   The first push uploads every layer (~8-12 GB compressed) from here. After that
#   only changed layers move, and the Dockerfile is ordered so the expensive steps
#   (pip install, model bake) sit ABOVE the COPY of handler.py/start.sh - so a code
#   change re-uploads kilobytes, not gigabytes.
#
# AUTH (one time)
#   Pushing needs a GitHub PAT with write:packages, even for a public package:
#     $env:CR_PAT = "ghp_..."
#     $env:CR_PAT | docker login ghcr.io -u lotion-abot --password-stdin
# =============================================================================
param(
    [switch]$NoPush,
    [switch]$Latest
)
$ErrorActionPreference = "Stop"

$IMAGE = "ghcr.io/lotion-abot/runpodglmocr"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE

# Tag with the commit, never :latest as the deployable ref. "Which image is actually
# running" must never be a guess - a deployment silently pinned to a stale version
# already cost three wasted test runs on the previous platform.
$sha = (git rev-parse HEAD).Trim()
$dirty = (git status --porcelain)
if ($dirty) {
    Write-Warning "working tree is DIRTY - the image will NOT match commit $($sha.Substring(0,8)):"
    $dirty | ForEach-Object { Write-Warning "    $_" }
    Write-Warning "commit first if this image is going to be deployed."
}
$ref = "${IMAGE}:${sha}"
Write-Host "`n=== BUILD  $ref ===" -ForegroundColor Cyan

# --platform linux/amd64 is not optional: Runpod hosts are x86_64 and an arm64 image
# dies at start with "exec format error", the single most common deploy failure.
$tags = @("-t", $ref)
if ($Latest) { $tags += @("-t", "${IMAGE}:latest") }
& docker build --platform linux/amd64 @tags .
if ($LASTEXITCODE -ne 0) { throw "docker build FAILED" }

# ---------------------------------------------------------------- pre-push gate
# Cheap checks that need no GPU. The point is to never upload 8-12 GB of broken:
# every one of these has a matching failure mode we have actually hit.
Write-Host "`n=== VERIFY (inside the image, no GPU needed) ===" -ForegroundColor Cyan
$probe = @'
import sys, os
from PIL import Image
import vllm, torch, transformers, glmocr
print("versions   :", vllm.__version__, torch.__version__, transformers.__version__)
assert vllm.__version__ == "0.19.1"
assert transformers.__version__ == "5.15.0"

cfg = open("/app/glmocr.yaml").read()
assert "label_task_mapping" in cfg, "merged config lost label_task_mapping"
print("config     : label_task_mapping present")

with Image.open("/app/selftest_page.png") as im:
    assert im.size == (3166, 4096), im.size
print("selftest   : page %sx%s" % im.size)

for p in ("/app/handler.py", "/app/start.sh", "/app/runner.py"):
    assert os.path.exists(p), p
assert os.access("/app/start.sh", os.X_OK), "start.sh not executable"
print("files      : all present, start.sh executable")

hf = os.environ.get("HF_HOME", "")
n = sum(len(f) for _, _, f in os.walk(hf)) if hf and os.path.isdir(hf) else 0
assert n > 0, "no baked model files under HF_HOME=%s" % hf
print("models     : %d files baked under %s" % (n, hf))
print("IMAGE OK")
'@
$probe | Out-File -FilePath (Join-Path $env:TEMP "_probe.py") -Encoding ascii
& docker run --rm --platform linux/amd64 -v "$env:TEMP\_probe.py:/tmp/_probe.py:ro" --entrypoint python3 $ref /tmp/_probe.py
if ($LASTEXITCODE -ne 0) { throw "IMAGE VERIFY FAILED - not pushing" }

if ($NoPush) {
    Write-Host "`n-NoPush set: built and verified, nothing uploaded." -ForegroundColor Yellow
    Write-Host "  local ref: $ref"
    exit 0
}

Write-Host "`n=== PUSH ===" -ForegroundColor Cyan
& docker push $ref
if ($LASTEXITCODE -ne 0) { throw "docker push FAILED (logged in to ghcr.io?)" }
if ($Latest) { & docker push "${IMAGE}:latest" }

Write-Host "`nDEPLOY THIS REF:" -ForegroundColor Green
Write-Host "  $ref"
