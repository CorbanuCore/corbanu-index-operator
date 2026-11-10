# Executable contract and implementation handoff

Owner: the implementing Codex agent. Date: 2026-09-11. This directory implements the offline contract and initial raw adapter for the [streaming specification](../docs/DEEPSEEK_V41_STREAMING_REPLAY_SPEC.md). It does not contain a qualified GPU launcher or publish indexes. Consult the [execution ledger](../docs/DEEPSEEK_V41_EXECUTION_STATUS.md) for current implementation/test status; the tasks below describe the complete delivery sequence. The remaining work belongs to the implementing agent.

## Run the implemented acceptance gate

From `/home/pfrpc/repos/corbanu-index-operator`, with Python 3.12 and the existing CorbanuAPI Node/tsx installation:

```bash
python3 -m unittest discover -s replay_contract -t . -p 'test_*.py' -v
python3 -m replay_contract.offline --output-dir /mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-offline-check
```

Choose a new output directory on each execution. Existing evidence is never overwritten. No key, network connection, model download or GPU is needed. The second command exits unsuccessfully on any failed assertion and writes `report.json` only after every offline check passes. Its historical inputs are deliberately fixed in `offline.py`; this is the AOC experiment, not a general index builder.

The command verifies the exact frozen input and source packet hashes; rebuilds all 452 first-pass bodies; reconstructs all 460 historical stage requests and decisions; compares all accepted scores and briefs; checks Unicode/finalizer/JSON behavior against the pinned TypeScript implementation; and reproduces the complete existing 29-holding construction. It also rejects missing/duplicate score coverage and confirms `needs_data` for empty eligible or capitalization universes. Historical receipt text is evidence of prior API behavior, not a new inference run.

`source-lock.json` identifies the original implementations used for comparison. A changed original file fails the cross-language check instead of silently changing the contract. The copied weighting implementation is isolated from future API helper changes. Every call explicitly passes cutoff `70`, `market_cap_rank`, and missing-cap policy `exclude`. No model score can be excluded because inference failed.

## Individual commands

```bash
python3 -m replay_contract digest
python3 -m replay_contract prepare --inputs /mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-h200-feasibility-20260911/aoc-frozen-inputs.json --output /mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-preparation.json
python3 -m replay_contract host-preflight --output /mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-host.json
```

The host probe checks OS, GPU identity and CUDA 13.0 compiler availability only. It exits `2` when blocked. A zero exit does **not** establish RAM capacity, model readiness or runtime qualification. Those remain explicit hardware tasks below.

Create a UTF-8 JSON context file with exactly `question`, `ticker`, and `company_name`, copied from the frozen job, then run:

```bash
python3 -m replay_contract request --context /absolute/context.json --stage first_pass --output /absolute/request.json
python3 -m replay_contract request --context /absolute/context.json --stage capped_finalizer --notes /absolute/previous-notes.txt --output /absolute/finalizer.json
python3 -m replay_contract compare /absolute/primary.json /absolute/replay.json --output /absolute/comparison.json
/home/pfrpc/repos/CorbanuAPI/node_modules/.bin/tsx replay_contract/weights.mts /absolute/consumption-index.json /absolute/scores.json /absolute/weights.json
```

The request command emits the legacy request, its digest, and the candidate's rendered prompt and digest. The legacy body records `json_object`; translation into a thinking-compatible local decoder is still an implementation gate. Do not send this OpenAI-shaped body directly to the raw completion endpoint.

The weights command performs arithmetic and score-coverage checks, not source authentication or inference verification. The offline acceptance command checks the frozen source hash. A production coordinator must also verify the publisher signature and full replay evidence before calling weights.

## Normative encodings and stage policy

`core.py` is normative. Canonical JSON has sorted ASCII object keys, compact separators, no trailing newline, finite integral numbers within JavaScript's safe-integer range, and ASCII-escaped Unicode string values using UTF-16 surrogate pairs. Numeric array-index object keys are rejected to prevent JavaScript ordering drift. Arbitrary JSON floats and Unicode object keys are outside this contract. This restriction applies to hash manifests, not the preserved raw model output.

Text hashes are SHA-256 over exact UTF-8 bytes. No trimming, newline rewriting or Unicode normalization precedes comparison. Token encoding is a uint64 little-endian count followed by that many uint32 little-endian IDs. Boolean token IDs fail. Model tokenization and termination-token inclusion still require qualification; the offline encoder only specifies storage bytes.

| Current stage | Output ceiling | Decision |
| --- | ---: | --- |
| `first_pass` | 8,192 | Accept on `stop` plus valid brief. Otherwise, `stop` with invalid brief or `length`, with nonempty notes, advances to `overflow`. |
| `overflow` | 60,000 | Same acceptance rule; otherwise advance to `capped_finalizer` when notes are nonempty. |
| `capped_finalizer` | 4,096 | Accept on `stop` plus valid brief; every other outcome fails the security. |

An abnormal finish reason fails immediately. Notes are exactly the current stage's reasoning followed by its content, with no separator. Overflow repeats the original two-message prompt with the larger budget; **it does not append prior notes**. The finalizer uses only the immediately preceding stage's notes. Up to 48,000 Unicode code points are retained whole; longer notes become the first 16,000 plus `\n\n[DETERMINISTIC MIDDLE OMISSION]\n\n` plus the last 32,000. These are code points, not UTF-8 bytes or UTF-16 units.

A brief has exactly `score` and `brief`; score is mathematically integral in 1–100, and the text contains at least 250 words under ECMAScript whitespace rules. Only the accepted parsed brief normalizes whitespace. Raw channels remain intact. Empty company names already present in the frozen inputs stay empty; they do not become an eligibility filter.

`candidate.json` sets local thinking effort to the pinned official encoder's default `high` = **75**, with thinking disabled for finalization. This declares a candidate local execution variant; it does not assert what numeric effort the hosted provider previously used. No scoring prompt, threshold, weighting or stage budget changes.

## Stage observation schema

The `compare` command accepts an exact-key object; extra or missing keys fail:

| Field | Type and constraint |
| --- | --- |
| `schema` | Literal `corbanu.streaming-observation.v1` |
| `fixture` | Boolean; synthetic observations must be `true` |
| `contract_sha256` | Current `digest` result |
| `profile_sha256` | 64 lowercase hexadecimal characters; must match across pair |
| `host` | Exactly `machine_id`, `gpu_uuid`, `runtime_manifest_sha256`; nonempty identities and a 64-character lowercase hex digest |
| `security_id` | Nonempty string |
| `stage` | One of the three stages above |
| `request_sha256` | 64 lowercase hexadecimal characters |
| `parent_output_sha256` | `null` for first pass, otherwise a 64-character lowercase hex digest |
| `input_token_ids`, `output_token_ids` | Nonempty arrays of uint32 integers |
| `reasoning`, `content` | UTF-8-encodable strings |
| `finish_reason` | `stop` or `length` |

All non-host fields must match exactly. Host machine and GPU identities must both differ, while runtime manifest digests match. Equal malformed model output still fails the scoring transition even if byte comparison succeeds. Every comparison returns `publication_allowed:false`.

This is a **diagnostic stage comparator**, not an admission verifier: it does not authenticate operator-provided host IDs, prove token/text correspondence, reconstruct the expected request, validate parent chains, or prove that a worker performed inference. These checks belong to the coordinator and independent execution evidence below. Do not turn its `byte_equal` label into a public replay claim.

## Remaining implementation, in execution order

Each task ends in a concrete artifact and test. A failure retains evidence and blocks dependent tasks; it never substitutes another model, changes the score methodology, or retries until outputs happen to match.

1. **Build and hardware qualification.** Start with Windows/CUDA 13.0 because it is the reported working path. Use the source revision in `candidate.json` and the pinned upstream `README-dsv41.md` build commands. Capture compiler/driver/OS versions, command arguments, executable/shared-library hashes, GPU UUID, physical machine identity, RAM and local NVMe capacity. Run `--help` and a small streaming/kernel fixture before downloading 502 GB. If only Linux is available, compile and run that fixture first; a Linux pass creates a distinct candidate profile and still requires every GPU gate. A compiler failure produces `build-failure.json`; do not rent a second host or download weights while unresolved.
2. **Conversion audit.** Write `audit_model.py`: enumerate every tensor and shard from the pinned snapshots; reject missing or duplicate tensors, missing Engram tables/scales, wrong shapes or file hashes. Stream through *every* losslessly repacked expert block and Engram row and compare decoded values/bytes as applicable. For intentionally changed dense/attention precision, record exact conversion algorithm, type, count and error statistics; do not apply an invented numerical-fidelity threshold or claim native equivalence. Missing source lineage requires a fresh conversion from the pinned native snapshot, followed by the same audit. Deliver `model-manifest.json`, `conversion-audit.json`, and tampered-shard/omitted-Engram regression tests.
3. **Raw runtime adapter.** Implement `runtime.py` against the pinned fork's `/completion` path with `return_tokens`; avoid its generic chat parser. Keep the two-message text rendering in `core.py`. Capture and verify tokenizer IDs, generated IDs, raw decoded bytes including special-token termination, reasoning/content split, `stop_type`, `stopping_word`, `truncated`, and effective generation settings. Disable speculation, prompt cache and context shifting; require fresh process, one active request, one I/O worker, 18 GiB GPU expert cache and 72 GiB host expert cache. A text-only, typed token parser must pass split-UTF-8 and special-token fixtures without regex. If the endpoint omits terminal IDs, instrument the sampling boundary. Do not infer token IDs by retokenizing final text.
4. **Decoder and capacity conformance.** Resolve JSON decoding while retaining unrestricted reasoning. A grammar may begin only at the final-answer boundary; do not constrain the reasoning channel to JSON or silently drop `json_object`. Demonstrate valid first-pass and finalizer outputs using actual AOC prompts. Capture all sampler/grammar/stop defaults as explicit manifest values. Tokenize every stage input and require input tokens plus that stage's full output ceiling to fit the context, with truncation forbidden. Fail rather than shrink 60,000 or clip the finalizer excerpt. Any necessary context-size change creates a new candidate digest and repeats capacity qualification.
5. **Coordinator and full verifier.** Implement `execute.py` and `verify_job.py`. The coordinator verifies source release signature/hash, enumerates the exact 452 securities, generates requests from the contract, and seals stage evidence. Define the parent digest as canonical SHA-256 of `{stage,input_token_ids,output_token_ids,reasoning,content,finish_reason,request_sha256}` from the previous stage. The verifier rebuilds every request and dependency, checks ordered stage coverage, token/text correspondence and complete model/runtime manifests, then compares independently built scores and weights. Tests must reject omitted/duplicate securities, a forged request hash, missing/changed parent, copied primary evidence, altered output text, wrong model/profile and interruption causing duplicate work. No `verified` status exists until these tests and G1–G5 pass.
6. **Bounded hardware execution.** Before each phase, write an execution manifest with provider quote, rate, resource IDs, spend ceiling, elapsed ceiling, token ceiling and expiry. The watchdog starts before inference and shuts down only owned resources at expiry or failure. A transport/process failure permits at most one fresh-process retry with a new attempt ID and both attempts retained; mismatched completed outputs permit zero recovery retries. Complete ENPH twice before the four-name canary, then independent-host and stress gates. Seal second-host outputs before exposing primary outputs to its worker. Decide on full-index execution from measured useful throughput, reset time and the actual quote.

The four-RTX-PRO-6000 Linux recipe becomes the next candidate only after the single-card build fails reproducibly, its fixed-profile outputs diverge, or its measured forecast exceeds the phase's recorded limits. It needs its own profile, native-variant audit and all gates. It does not automatically inherit the 5090 result. DGX Spark remains research context rather than another parallel implementation project.

## Vocabulary and completion

The current `job.py` command produces an authenticated portable job, and `execute.py` supplies stage orchestration with an engine callback. `engine.py` implements that callback using a fresh local server process for each stage, with process cleanup tested on success, failure and expiry. CUDA/model integration remains unqualified. `verify_job.py --help` lists the recorded-job verification CLI. It requires both complete runs, the signed frozen source, pinned native tokenizer, compiled tokenizer probe and GGUF vocabulary; it independently reconstructs both holdings. It does not yet admit the runtime. See the execution ledger for outstanding GPU qualification and conversion audit.

**Engram** is the model's large learned lookup memory; **MoE experts** are the routed neural-network blocks streamed through caches; **GGUF** is the converted weight container; **KV cache** stores attention state; **DSpark/speculation** predicts draft tokens; **TP/EP** distribute tensor/expert work across GPUs. A **profile** is a hashed execution contract. A **receipt** records observed execution; it is not a hardware attestation. A **TEE** is a trusted execution environment and is not claimed here.

Done means two physical machines regenerate every contributing stage for all 452 names, independent construction produces identical holdings, the offline full-job verifier passes, and actual elapsed time/cost and numerical-variant limitations are reported. Passing the implemented offline suite completes only the request/weighting portion of G0.
