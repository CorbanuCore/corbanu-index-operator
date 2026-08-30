# Corbanu H200 deterministic-replay runbook

This is the mandatory bring-up procedure for the two Qwen verifiers used by
Corbanu Indexes. Do not assemble an SGLang command from memory. Copy the
committed `ops/vast_qwen38_fp8_fixed32_strict_onstart.sh` to both machines and
use it unchanged.

## Admitted profile

The admitted fast path is exactly 32 simultaneous requests per H200. It was
selected because fixed 32 passed prior byte-replay testing. A lower ad hoc
concurrency is not the remedy for a misconfigured runtime.

The following controls are one indivisible profile:

| Control | Required value |
| --- | --- |
| GPU | NVIDIA H200 |
| Model | `Qwen/Qwen3.8-27B-FP8` |
| Model revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Maximum running requests | `32` |
| Request batch size per host | `32` |
| Attention backend | `triton` |
| Linear-attention backend | effective runtime value `triton` |
| Deterministic inference | enabled |
| Radix cache | disabled |
| Overlap schedule | disabled |
| Prefill CUDA graph | disabled |
| Decode CUDA graph | disabled |
| Random seed | `438916795` |
| Request temperature | `0` |
| Request `top_p` | `1` |
| Thinking | disabled |

The pinned SGLang image used by this operator does not accept the obsolete
`--linear-attention-backend` spelling. Its effective linear-attention backend
must still be reported as `triton` in the startup log. Never remove another
strict flag to compensate for an invalid CLI option.

The following is **not** an admitted profile, even if responses happen to
match in a small sample:

- FlashAttention 3 (`fa3`) selected as the default attention backend;
- overlap scheduling enabled;
- prefill or decode CUDA graphs enabled;
- only `--enable-deterministic-inference` and `--disable-radix-cache` supplied;
- different model revisions, request schemas, seeds, batch membership, or
  batch sizes across the two machines.

## Provision both verifiers

1. Start two independent H200 machines with the pinned container image recorded
   by the bootstrap script.
2. Copy the same committed bootstrap script to each host.
3. Record its digest before execution:

   ```bash
   sha256sum ops/vast_qwen38_fp8_fixed32_strict_onstart.sh
   ```

4. Run that script on each host. It safely replaces at most one existing
   SGLang server; it refuses to proceed if multiple server processes exist. It
   writes the model path, GPU information,
   package versions, strict launch profile, and server log under
   `/workspace/postfiat-sec-index-qwen38-fp8-fixed32/runtime/`.
5. Forward each host's loopback port 8000 to separate local ports. The operator
   defaults are primary `127.0.0.1:18001` and replay `127.0.0.1:18002`.

## Fail-closed runtime preflight

Run these checks on **each** H200 before starting the Corbanu operator:

```bash
run_root=/workspace/postfiat-sec-index-qwen38-fp8-fixed32
sha256sum "$run_root/runtime/strict-launch-profile.txt"
ps -eo args= | grep '[s]glang.launch_server'
grep -E 'attention_backend|linear_attn_backend|disable_overlap_schedule|cuda_graph|max_running_requests' \
  "$run_root/runtime/strict-server.log" | tail -40
curl -fsS http://127.0.0.1:8000/v1/models
```

The process command must contain all of these flags:

```text
--max-running-requests 32
--attention-backend triton
--enable-deterministic-inference
--disable-radix-cache
--disable-overlap-schedule
--disable-prefill-cuda-graph
--disable-decode-cuda-graph
--random-seed 438916795
```

The effective startup configuration must show Triton attention, Triton linear
attention, overlap scheduling disabled, CUDA graphs disabled, and maximum
running requests 32. If any value is absent, different, or ambiguous, stop.
Do not start the public operator and do not call that host a replay verifier.

Also record and compare on both machines:

```bash
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
python3 - <<'PY'
import importlib.metadata
for package in ("sglang", "torch", "transformers"):
    print(package, importlib.metadata.version(package))
PY
```

Driver differences belong in the receipt even when the byte canary passes.
Model revision, container digest, SGLang, PyTorch, Transformers, launch-profile
digest, and request profile must match exactly.

## Mandatory byte-replay gate

The service is not ready merely because both `/v1/models` endpoints respond.
Before accepting a public job:

1. Use the frozen production universe and transcript snapshot.
2. Send the same 32 transcript-grounded scoring requests to each host as one
   fixed batch. Batch membership and order must match.
3. Repeat that fixed batch on each host.
4. Compare raw UTF-8 response bytes—not parsed scores—across hosts and repeats.
5. Require 32/32 exact matches for every comparison. One mismatch fails the
   entire bring-up.
6. Preserve request hashes, response hashes, launch-profile hashes, runtime
   versions, GPU/driver identities, and the canary result in the run receipt.

Only after that gate passes may the operator start with its locked concurrency:

```bash
python3 corbanu_index_operator.py \
  --host 127.0.0.1 \
  --port 8788 \
  --workers-per-host 32
```

Do not silently fall back to a slower configuration or accept parsed-score
equality. A completed index is valid only when every admitted raw model output
has a byte-identical replay partner.

## Incident rule

If a batch produces mismatches:

1. Mark the run invalid; do not publish its weights.
2. Capture both process commands and both effective startup configurations.
3. Compare them with the admitted-profile table above before investigating the
   prompts or transcripts.
4. Specifically check for `fa3`, enabled overlap scheduling, and enabled CUDA
   graphs. These were the configuration regression that caused the 2026-08-30
   mismatch incident.
5. Relaunch both hosts from the committed script and rerun the 32-request gate.

The serial recovery path in the application is defensive evidence handling,
not permission to operate a misconfigured server. A large recovery count is an
incident signal and requires checking the launch profile.
