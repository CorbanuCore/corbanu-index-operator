#!/usr/bin/env bash
set -euo pipefail

run_root=/workspace/postfiat-sec-index-qwen38-fp8-fixed32
model_id=Qwen/Qwen3.8-27B-FP8
model_revision=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
runtime_image='lmsysorg/sglang:nightly-dev-cu13-20260817-d91c3682@sha256:fa8774dd128600a09fd6d46670b06fb69a55dac8a3881e50ccf0916a45eb39af'

mkdir -p "$run_root/runtime" "$run_root/hf"
exec > >(tee -a "$run_root/runtime/bootstrap.log") 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_root/runtime/bootstrap-started-at.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap \
  --format=csv,noheader > "$run_root/runtime/gpu.csv"
python3 --version > "$run_root/runtime/python-version.txt" 2>&1
python3 - <<'PY' > "$run_root/runtime/package-versions.txt"
import importlib.metadata

for package in ("sglang", "torch", "transformers"):
    print(package, importlib.metadata.version(package))
PY

export HF_HOME="$run_root/hf"
export HF_HUB_CACHE="$run_root/hf/hub"
export HF_XET_HIGH_PERFORMANCE=1
export SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648

model_path="$(python3 - <<'PY'
from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id="Qwen/Qwen3.8-27B-FP8",
        revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    )
)
PY
)"
printf '%s\n' "$model_path" > "$run_root/runtime/model-path.txt"

cat > "$run_root/runtime/strict-launch-profile.txt" <<EOF
model_id=$model_id
model_revision=$model_revision
model_path=$model_path
runtime_image=$runtime_image
tensor_parallel=1
context_length=32768
mem_fraction_static=0.75
chunked_prefill_size=4096
max_running_requests=32
attention_backend=triton
linear_attention_backend=triton_runtime_default_verify_in_log
deterministic_inference=true
radix_cache=false
overlap_schedule=false
prefill_cuda_graph=false
decode_cuda_graph=false
thinking=false_at_request
random_seed=438916795
EOF
sha256sum "$run_root/runtime/strict-launch-profile.txt" \
  > "$run_root/runtime/strict-launch-profile.sha256"

exec python3 -m sglang.launch_server \
  --model-path "$model_path" \
  --served-model-name "$model_id" \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --tp 1 \
  --context-length 32768 \
  --mem-fraction-static 0.75 \
  --chunked-prefill-size 4096 \
  --max-running-requests 32 \
  --reasoning-parser qwen3 \
  --attention-backend triton \
  --enable-deterministic-inference \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --disable-prefill-cuda-graph \
  --disable-decode-cuda-graph \
  --random-seed 438916795 \
  --enable-metrics \
  2>&1 | tee -a "$run_root/runtime/strict-server.log"
