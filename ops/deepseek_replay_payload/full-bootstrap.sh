#!/bin/bash
set -euo pipefail
cd /workspace/trial
mkdir -p evidence
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > evidence/worker-exit.json' EXIT
python3 - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('.')
for x in json.loads((r/'payload-manifest.json').read_text()):
 p=r/x['path']; assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x['path']
PY
python3 hardware_monitor.py > evidence/hardware-monitor.log 2>&1 &
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3.12-venv zstd libdw-dev libnuma-dev git ninja-build rsync > evidence/apt.log 2>&1
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > evidence/gpus.csv
nvidia-smi topo -m > evidence/topology.txt
python3.12 -m venv /workspace/download-venv
/workspace/download-venv/bin/pip install -q huggingface_hub==0.36.2 hf_xet
/workspace/download-venv/bin/python download.py model > evidence/download-model.log 2>&1 &
model_download_pid=$!
/workspace/download-venv/bin/python download.py prebuilt > evidence/download-prebuilt.log 2>&1
bash /workspace/prebuilt/install.sh /workspace/di-venv /workspace > evidence/install.log 2>&1
/workspace/di-venv/bin/pip install cryptography==41.0.7 >> evidence/install.log 2>&1
zstd -dc /workspace/prebuilt/jit-cache/jit-cache-4xb200-ep4.tar.zst | tar -x -C /workspace
tar -xzf replay-jit-cache.tar.gz -C /workspace
wait "$model_download_pid"
/workspace/di-venv/bin/python patch_runtime.py /workspace/DeepInfer
/workspace/di-venv/bin/pip freeze > evidence/pip-freeze.txt
export DEEPINFER_ACCELERATOR=nvidia
export DEEPINFER_BATCH_INVARIANT=1
export DEEPINFER_ENABLE_MTP_VARLEN=0
export DEEPINFER_USE_NEXTN_DECODE=0
export DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=0
export DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL=0
export DEEPINFER_ENGRAM_LOCAL=1
export DEEPINFER_NCCL_NVLS_ENABLE=0
export DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=off
export DEEPINFER_LOGGING_INTERVAL_STEP=100
export DG_JIT_CACHE_DIR=/workspace/jit-cache/deep_gemm
export TILELANG_CACHE_DIR=/workspace/jit-cache/tilelang
export PYTHONPATH=/workspace/DeepInfer
export RANK=0
export WORLD_SIZE=1
export PYTHONUNBUFFERED=1
/workspace/di-venv/bin/python run_index.py --round full --mode full --copies 1 > evidence/full.log 2>&1
/workspace/di-venv/bin/python verify_native.py full evidence > evidence/full-verification.log 2>&1
