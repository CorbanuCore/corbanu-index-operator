# DeepSeek V4.1 streaming replay execution

**Update 2026-11-08: DeepSeek V4.1 Flash repeated byte-for-byte on one host, but replay is not proven or admitted.** On 2026-09-12 a single RTX 5090 running the converted MXFP4 GGUF on the streaming llama.cpp fork generated the 4,263-token ENPH first-pass brief twice in fresh processes with identical token IDs, raw bytes, reasoning, content, finish reason and score (85); a 384-token short probe also matched. The raw evidence (`/workspace/g1-evidence` on the instance) was not copied off before the instance was destroyed, so the record is the agent session log and `tensorcash/docs/frontier_moe_work_function_note_20260912.md`. Native V4.1 Flash on 8× H200 under vLLM (2026-09-14) failed: two runs of the same request diverged at token 137. A second host (G3), the four-name canary (G2) and the full index (G5) have never run; the admitted profile for published indexes is still Qwen3.8-27B-FP8 on H200 ([runbook](H200_REPLAY_RUNBOOK.md)).

## GPU phases since 2026-09-11

| Phase | Hardware and runtime | Outcome | Evidence |
| --- | --- | --- | --- |
| G0 build and kernel test, then G1 (model download, ENPH brief, fresh-process repeat) | 1× RTX 5090, 128 GB RAM, Vast; converted MXFP4 GGUF, JigSawPT llama.cpp fork, one I/O thread, temperature 0, fixed seed; $15 ceiling | **G0 passed** (`build-manifest.json`: `built_and_kernel_tested`). **G1 repeat passed on this host:** 384-token probe and full 4,263-token brief identical across two fresh processes, score 85. Decode 4.1–4.3 tok/s. Same-host repeatability only. Instance destroyed at the deadline before the evidence directory was retrieved. | `deepseek41-execution-20260911/vast-build-g0-attempt3/` (build and kernel logs only); session log 2026-09-12; `tensorcash/docs/frontier_moe_work_function_note_20260912.md` |
| Q1 native SGLang repeatability | 4× RTX PRO 6000, Vast; $30 ceiling, 3 hours | Instance destroyed at the deadline; no repeatability results recorded. The watchdog logged a Vast HTTP 429. | `deepseek41-native-pro6000-20260912/vast-native-q1/` |
| Native replay on H200 | 8× H200, native `deepseek-ai/DeepSeek-V4.1-Flash` (`dba1be0a…`), vLLM | **Failed** (`final-comparison.json`: `verified_replay_failure`). With fixed NCCL ring/simple and custom reductions disabled, two runs of the same request in the same process diverged at token 137. Batch-invariant mode would not start (no MXFP4 MoE backend for the configuration). `report.json`: `production_deterministic_admission: false`. | `deepseek-h200-20260914/` |

All three phases have cleanup records confirming no instance remains. `replay_contract/candidate.json` remains `"admitted": false`.


Updated 2026-09-11. Goal is blocked on Vast funding after the same billing blocker was confirmed across three consecutive execution turns. **No replay-qualified GPU inference or new index has been completed.** Vast rentals are authorized by the user.

Evidence root: `/mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-execution-20260911/`.

## Current external blocker

Vast account `535381` returned `insufficient_credit` when creating the bounded G0 rental. The account query reported a balance of `-0.05150192554901878` USD. Both locally available Vast credentials resolve to this account. The user has been asked to fund it; no new permission to rent is needed.

The selected offer was one RTX 5090, approximately 128 GB RAM and 2 TB allocated storage, quoted at approximately $1.356/hour including storage. The initial phase was limited to one hour, $5 maximum, zero generated tokens and no model download. An independent local watchdog was started before creation. Both failed attempts have `cleanup.json` records confirming that their exact ownership labels have no instances. No rental remains running.

## Requirement evidence

| Requirement | State and authoritative evidence |
| --- | --- |
| Exact AOC inputs, historical stages and existing weights | Passed offline: `offline-acceptance/report.json`; 452 requests, 460 stages, 29 holdings, existing LIT/REMX exclusions. |
| Linux source build | CPU build passed after adding `<cmath>` to the upstream streaming file: `cpu-build-fixed.log`, `cpu-build-patched.log`, `cpu-build-joyai.log`. CUDA build remains unexecuted. |
| Regex-free DeepSeek pretokenization | Implemented typed scanner in `ops/deepseek3_split.cpp.inc`. Routes all three upstream aliases sharing these rules, including the actual GGUF's `joyai-llm`. 17 explicit segmentation fixtures and 10,000 Unicode preservation samples passed. No claim of exhaustive upstream-regex parity. |
| Actual request tokenization and context | `tokenizer-qualification.json`: all 452 rendered prompts round-trip; 248–268 input tokens, so each plus 60,000 output tokens fits 65,536. Finalizer capacity must be checked on each actual stage. |
| Native/converted tokenizer metadata | `native-converted-tokenizer-parity.json`: 129,280 vocabulary entries, 127,741 merges and all 452 actual prompt token sequences agree. The real converted first shard was downloaded and SHA-256 verified; it is only 5,769,184 bytes. |
| Thinking-compatible JSON decoding | Actual native grammar sampler test passed in `token-grammar-test.log`: unrestricted reasoning, activation on the single `</think>` token, immediate grammar for finalizer, valid JSON and EOS acceptance. This is a decoder fixture, not model generation. |
| Raw runtime adapter | `replay_contract/runtime.py` implemented and unit-tested. Uses `/completion` and `/tokenize` with supplied IDs to recover exact pieces. Rejects missing token coverage, changed effective controls, unexpected stop types and truncation. GPU endpoint behavior still requires qualification. |
| Snapshot file integrity | Pinned inventories for 11 GGUF and 48 native shards written. `audit_model.py` verifies every file size/hash against the candidate's pinned inventory. Full large-weight verification has not run. |
| Conversion lineage and tensor fidelity | Incomplete. No exhaustive expert/Engram audit yet; tokenizer agreement is not weight fidelity. |
| CUDA kernel and streaming fixture | Incomplete; requires funded GPU rental. `ops/deepseek41_linux_build.sh` is prepared and passes shell syntax validation. |
| G1 useful ENPH inference and repeat | Not started. |
| G2 four-name canary and stage boundaries | Not started. |
| G3 two physical hosts, two executions each | Not started. |
| G4 long-context/cache/recovery stress | Not started. |
| Authenticated portable job | `portable-job/identity.json`: publisher signature verified, all 452 frozen request identities pinned, historical scores excluded. |
| Stage orchestration and resume | `execute.py` implemented with atomic, non-overwriting receipt seals, exclusive worker lock, verified resume, one recovery attempt, and no retries for completed invalid output. |
| Fresh-process engine | `engine.py` supplies the coordinator callback, pinned launch controls, build/model checks, raw response capture and independent process expiry. Real subprocess lifecycle tests pass; CUDA/model integration remains unqualified. |
| Recorded job verifier | `verify_job.py` checks complete security coverage, requests, ordered parent chains, wire responses, native-vocabulary token pieces, independent input tokenization and separately rebuilt holdings. It always reports unadmitted status; hardware evidence admission is still missing. |
| G5 all 452 local scores and independent holdings | Not started. |
| Admitted profile registry / public verifier rollout | Not started; existing service unchanged. |
| Measured GPU performance, total cost and cleanup | No GPU timings; no rental created. Cleanup absence confirmed. Full-job cost remains unmeasured. |

## Implementation and next action

Source checkout: `/mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-execution-20260911/llama.cpp`, pinned at `3c95ac82c19e496348fe57d06f083880a7d86e95` with explicit local patches. Build: `/mnt/HC_Volume_101713660/pfrpc/build-cache/deepseek41-jigsaw-cpu-3c95ac82`. Patch application is reproducible with `ops/patch_deepseek_runtime.py`; source snippets are hashed in `candidate.json`. This is a distinct, **unadmitted Linux candidate**; the upstream reported Windows recipe remains the reference.

35 Python contract/adapter/integrity/job/resume/process tests pass. Three Vast lifecycle tests previously passed. The Vast tests reject repeated launch attempts and spending above the phase limit, and isolate cleanup to an exact owned label. Process tests prove child cleanup on success, caller failure and deadline expiry. Runtime and orchestration tests are fixtures and do not count as GPU replay evidence. `independent-piece-decoder.json` verifies all 452 actual prompt byte sequences using a Python vocabulary decoder independently of llama.cpp detokenization. `verifier-construction-check.json` records two separate constructions matching the historical 29-holding basket. The current contract also passed offline acceptance in `offline-engine-integration/report.json`.

The stage coordinator, process engine and recorded-job verifier now exist, but their fixture results do not satisfy hardware gates. Conversion audit implementation and qualification, CUDA/model integration, and hardware-evidence admission remain unfinished. The latest billing/instance check is `funding-recheck-third-goal-turn.json`; no instance exists. Once funding is available, refresh the quote, create a **new** phase directory/label, and run the CUDA build plus bounded kernel fixture. Do not replay a launch-intent file or assume an old offer still exists. Inspect the measured filesystem capacity and memory allocation before large downloads; allocate sufficient storage to retain the required working headroom.

Then verify the complete model files and conversion audit before ENPH; complete a useful scored brief and fresh-process repeat before expanding. All scoring, stage-budget and weighting rules remain frozen. No full-universe run or publication may skip a gate because the local parser tests pass.
