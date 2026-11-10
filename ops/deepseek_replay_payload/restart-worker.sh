#!/bin/bash
set -euo pipefail
cd /workspace/trial
trap 'rc=$?; printf "{\"exit_code\":%s,\"restart\":true}\n" "$rc" > evidence/worker-exit.json' EXIT
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
/workspace/di-venv/bin/python run_questions.py --round boot-smoke --mode canary --copies 1 --resume >> evidence/ready-engine-restart.log 2>&1
