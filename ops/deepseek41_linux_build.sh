#!/usr/bin/env bash
# Run inside the pinned CUDA 13.0.1 image. No model download or inference.
set -euo pipefail
source_root="${1:?Pass the pinned llama.cpp checkout}"
build_root="${2:?Pass the build directory on the allocated disk}"
evidence_root="${3:?Pass the evidence directory}"
operator_root="${4:?Pass the corbanu-index-operator directory}"
mkdir -p "$build_root" "$evidence_root"
python3 "$operator_root/ops/patch_deepseek_runtime.py" "$source_root"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader > "$evidence_root/gpu.csv"
nvcc --version > "$evidence_root/nvcc.txt"
python3 - "$evidence_root" <<'PY'
import csv,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=list(csv.reader((root/'gpu.csv').read_text().splitlines()))
assert len(rows)==1 and 'RTX 5090' in rows[0][0], 'Requires exactly one RTX 5090'
assert 'release 13.0' in (root/'nvcc.txt').read_text(), 'Requires CUDA toolkit 13.0'
PY
cmake -S "$source_root" -B "$build_root" \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON > "$evidence_root/configure.log" 2>&1
timeout 1800 cmake --build "$build_root" \
  --target llama-server llama-gguf-split test-backend-ops test-grammar-parser -j 8 \
  > "$evidence_root/build.log" 2>&1
"$build_root/bin/llama-server" --help > "$evidence_root/server-help.txt"
timeout 300 "$build_root/bin/test-backend-ops" test -b CUDA0 -o MUL_MAT_ID \
  > "$evidence_root/kernel-test.log" 2>&1
"$build_root/bin/test-grammar-parser" > "$evidence_root/grammar-test.log" 2>&1
python3 - "$build_root" "$evidence_root" <<'PY'
import hashlib,json,platform,sys
from pathlib import Path
build,root=map(Path,sys.argv[1:])
files=[p for p in (build/'bin').iterdir() if p.is_file() and ('.so' in p.name or p.name=='llama-server')]
manifest={'status':'built_and_kernel_tested','os':platform.platform(),
          'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
          'gpu_inference_qualified':False,'publication_allowed':False}
(root/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
