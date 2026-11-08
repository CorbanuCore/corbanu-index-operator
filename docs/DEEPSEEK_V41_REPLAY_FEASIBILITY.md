# DeepSeek V4.1 Flash: H200 and replay feasibility

For the subsequent no-H200 implementation specification and comparison of streaming recipes, see [DEEPSEEK_V41_STREAMING_REPLAY_SPEC.md](DEEPSEEK_V41_STREAMING_REPLAY_SPEC.md).

Checked 2026-09-11. **H200 serving is documented; Corbanu byte-replay admission is not established.** No GPUs were rented, weights downloaded, or indexes published during this investigation.

Update: a [single-RTX-5090 V4.1 report](https://github.com/JigSawPT/deepseek-v41-flash-on-5090) documents expert streaming through RAM/NVMe and claims repeatability with one I/O worker. This provides a smaller experimental candidate. Eight H200s is a documented serving configuration, not a minimum. Independent replay remains untested here.

## Hardware and immutable inputs

| Item | Finding |
| --- | --- |
| Weights | `deepseek-ai/DeepSeek-V4.1-Flash`, MIT license |
| Frozen model revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Model dimensions | 552B backbone parameters plus 196B Engram memory; 8B/16B active per token during prefill/decode |
| Checkpoint | 48 safetensors shards, 510,296,708,312 bytes including shard headers; about 510.3 GB / 475.3 GiB |
| Stored precision | Mixed precision: FP8 dense weights with 32×32 block scales, packed FP4 routed experts, other tensors in their released types |
| Documented H200 serving configuration | One node, 8×H200, TP8/EP8; each H200 has 141 GB HBM3e |
| Corbanu independent verification configuration | Two such replicas: 16 H200s if operated simultaneously; sequential independent rentals can reduce concurrent capacity, not total GPU-hours |
| Alternative documented serving configuration | 4×B200 per replica; this is not a determinism qualification |
| Storage planning estimate | 2 TB local NVMe per node for weights, container, temporary load/conversion files and evidence; not a measured minimum |
| Host RAM planning estimate | Prefer 1 TB per node for loading headroom; not an upstream minimum or measured requirement |

The checkpoint size is not the model's peak serving memory. Four H200s have more aggregate HBM than the stored checkpoint, but that arithmetic does not qualify a four-card configuration. The documented H200 recipe uses eight. Host offload also needs its own memory and throughput measurements.

Sources: [released model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277), [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json), [SGLang deployment matrix](https://github.com/sgl-project/sglang/blob/4309c7ce19dc42fb42cc9e7d883691c8dd8bda10/docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx), [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/).

## Runtime boundary

SGLang's V4.1 support is a preview. The inspected image is:

```text
lmsysorg/sglang:dev-dsv41@sha256:e56358a68b06427362283c8c8a9d7d706448082ae53098ea1131b5d51aa1fd62
image source revision: e087e662ba1ac4ef7747537e2a9141085efd4561
documentation revision: 4309c7ce19dc42fb42cc9e7d883691c8dd8bda10
```

The image manifest/config were fetched without downloading its 14.86 GB compressed layers. The image itself was not run. Source was inspected at the revision declared by its labels; those labels are not hardware attestation.

The [V4.1 cookbook](https://github.com/sgl-project/sglang/blob/4309c7ce19dc42fb42cc9e7d883691c8dd8bda10/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx) explicitly identifies batch-dependent output and rejects the general deterministic mode on the required `dsv4` backend. A CPU-only check extracted and executed the unmodified configuration-guard function from the image's declared source revision. `dsv4` with deterministic mode enabled raises `ValueError`; the negative control without that mode passes this guard. This verifies a configuration boundary, not GPU behavior.

vLLM has V4.1 model implementation code at `9dd969da096e37256ee37e24f6a4689d860f39ce`. Its [batch-invariance tested-model list](https://github.com/vllm-project/vllm/blob/9dd969da096e37256ee37e24f6a4689d860f39ce/docs/features/batch_invariance.md) does not include V4.1. That is missing qualification evidence, not proof it cannot work.

Fixed-workload reproducibility remains a legitimate experiment. Identical submitted HTTP batches alone do not ensure identical scheduler batches or collective reduction order. Do not remove the upstream guard and call the result deterministic. The existing Qwen profile and its fixed-32 admission remain unchanged.

## Actual AOC inputs prepared

Local evidence directory:

```text
/mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-h200-feasibility-20260911/
```

`freeze-aoc.mts` reconstructed all 452 original first-pass request bodies and matched all 452 against the prior API receipt hashes. `aoc-frozen-inputs.json` binds the current AOC source, mandate, scoring profile, universe and request bodies. Its SHA-256 is `f577e7e4a873e0d4de96088e5e564db1bd43b463162a1cdfc113e5e744eac3b9`.

Preserve the current methodology: executive briefs from company name/ticker and mandate; cutoff 70; market-cap rank weights; explicitly disclosed missing-cap exclusions. The current usable result has 29 holdings after excluding LIT and REMX. Transcript selections are retained as source metadata but transcripts do not enter this scoring profile. Do not inherit the Qwen operator's rubric compiler, transcript prompts, fundamental weighting, or holding caps.

Existing API generation used temperature 0 and seed 438916795, with token ceilings of 8,192 / 60,000 / 4,096 across its three stages. The main reasoning budget was provider-default, not numerically pinned. Self-hosted encoding, reasoning budget and JSON decoding must become explicit runtime parameters before a new run. Match the intended thinking behavior; do not accidentally inherit SGLang's thinking-off default. Existing API scores cannot be retroactively declared verified. A new self-hosted run may change scores and holdings.

## Qualification required before a full run

1. Freeze the runtime image, model and tokenizer files, encoded prompt token IDs, generation settings and actual execution schedule. Preserve all shard content hashes from `model-metadata.json`. Record drivers, CUDA/NCCL versions, GPU identities and NVLink topology.
2. Use the prepared diagnostic sample: ENPH (ordinary success), INTC (inclusion boundary), INRO and NVT (previous long-output/retry cases). This selection tests execution; it does not change index eligibility. Produce usable complete briefs, including overflow/finalization when required, before expanding compute.
3. Repeat the exact workload on the first replica, then on an independent replica. Compare raw generated token IDs, reasoning and final-output bytes as well as parsed scores. Any derived finalizer request must itself match. Record failures instead of retrying until a matching answer appears.
4. A fixed workload can be qualified without arbitrary-batch invariance, but its schedule must be enforced and included in the replay contract. Test the production schedule, restarts and partial/recovery batches before admitting that profile. Small-sample equality alone does not admit all index outputs.
5. Only after qualification, regenerate all 452 names on both replicas, require an exact replay partner for every accepted output, and reconstruct weights with the frozen current methodology. Publish a new artifact with its measured evidence. Keep the previous API result as a comparison.

No spend estimate is claimed: there is no captured provider quote or measured V4.1 index throughput yet. At a rental rate of `p` dollars per H200-hour, two eight-card replicas cost `16 × p × hours`, plus storage/transfer and any startup time billed. Start with the bounded sample and a resource deadline before committing to the universe.

This workflow establishes independent computational replay. TEE attestation requires a separate supported host and measured workload; an H200 listing alone does not provide it. Neither replay nor attestation proves the economic judgment in the scoring prompt is correct.

## Checks completed

- All 452 reconstructed first-pass request hashes match the original API receipts.
- Pinned SGLang configuration guard rejects deterministic `dsv4`, with a passing negative control.
- Model metadata, shard hashes, runtime image manifest, configuration, relevant upstream source and documentation saved locally.
- No inference endpoint was listening on this host's expected local replay ports during inspection.
- GPU inference, cross-host byte equality, TEE attestation and full replay-index generation: **not executed / not qualified**.

Re-run the two offline checks with:

```bash
/home/pfrpc/repos/CorbanuAPI/node_modules/.bin/tsx /mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-h200-feasibility-20260911/freeze-aoc.mts
python3 /mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-h200-feasibility-20260911/check-runtime-boundary.py
```
