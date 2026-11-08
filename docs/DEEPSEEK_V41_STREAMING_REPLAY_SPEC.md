# Specification: replayable DeepSeek V4.1 indexes without H200s

Date: 2026-09-11. Status: **execution underway; offline contract and CPU build tested; GPU qualification pending**. See the [live execution ledger](DEEPSEEK_V41_EXECUTION_STATUS.md) for requirement-by-requirement evidence and remaining work. Vast rentals are now user-authorized; the first rental was rejected for insufficient account credit.

Execution owner: the implementing Codex agent. Start with [the runnable contract and implementation handoff](../replay_contract/README.md). That document defines commands, schemas, exact stage transitions, tests and remaining build tasks; the research and hardware gates below explain why those tasks are required. Passing offline checks does not complete GPU qualification.

## 1. Objective and decision

Generate an index using locally executed DeepSeek V4.1 Flash, then independently reproduce every contributing inference output and the resulting holdings. Use consumer or workstation hardware and local storage. H200s are not required by this specification.

Start qualification with **one RTX 5090, 128 GB system RAM and local NVMe**, using the pinned JigSawPT streaming fork. This is the smallest concrete V4.1 recipe found in the searches below. Its Windows build is the reference starting point; Linux support must be established before choosing a Linux rental. Hardware choice is a proposed execution configuration, not a claim that it passes Corbanu replay today.

Keep the four-RTX-PRO-6000 recipe as the next candidate if the single-card path cannot meet correctness or measured completion-time requirements. The four-DGX-Spark recipe is another documented alternative. Do not provision all candidates or switch to a different quantization automatically.

The deliverable is a new, explicitly identified execution profile and an index artifact with complete replay evidence. The current API-generated AOC index remains a comparison artifact. Its scores cannot be retroactively converted into verified inference receipts.

## 2. Recipe research and evidence limits

Web searches covered V4.1 with streaming, offload, RTX 5090, RTX PRO 6000, DGX Spark, llama.cpp, SGLang, vLLM, KTransformers and EXL3. Search discovery was followed by the authors' repositories, launchers, model cards and selected result files. These are independent authors' reports, **not experiments reproduced on this host**. V4/V4-0731 recipes do not establish V4.1 compatibility.

| Candidate | Published setup and approach | Evidence and qualification gap |
| --- | --- | --- |
| [JigSawPT single-card report](https://github.com/JigSawPT/deepseek-v41-flash-on-5090/tree/7be408844471a9639a509f571775cd2ece9f912d) | RTX 5090, approximately 128 GB RAM; experts move through an 18 GiB GPU cache and 72 GiB RAM tier; Engram remains disk-backed. | Reports about 5.1 tokens/s for new content and 21.3 for repeated resident content. Reports repeatability with one I/O worker at about 3.6 tokens/s. No independent-host index replay demonstrated. |
| [JigSawPT runtime instructions](https://github.com/JigSawPT/llama.cpp/blob/3c95ac82c19e496348fe57d06f083880a7d86e95/README-dsv41.md) | Custom llama.cpp branch, Windows/CUDA 13.0; matching GGUF format. | Linux is explicitly untested. Dense/attention representation differs from the native FP8 execution. The author's numerical comparison is not proof of identical official-model outputs. |
| [0xSero workstation recipe](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000/tree/45f538a18569420721e00e37353f9a7e1af7e5da) | Linux, four 96 GB RTX PRO 6000 Blackwell GPUs, 128 GB RAM, local NVMe; native checkpoint with Engram offload and TP4/EP4. | Real requests and concurrency tests are reported. The performance configuration enables DSpark and bounded decoder replay. Final soak/rollback qualification remains pending; no end-to-end independent byte-replay claim. |
| [tonyd2wild DGX Spark recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/458fadec6106e6fd84f7eda36dd6e0a8fa219ef6) | Four DGX Sparks, vLLM TP4; rank-local Engram data on disk, patched kernels, DSpark and CUDA graphs. | Reports roughly 24–74 tokens/s across several single-stream content categories. Tests generally use short outputs and thinking off. Rank-offset and hardware slow-state issues required fixes. Index-length reasoning and byte replay need new tests. |
| [MixedQ2 conversion](https://huggingface.co/apetersson/DeepSeek-V4.1-Flash-MixedQ2-GGUF) | Additional routed-expert requantization. | Not the baseline: changes numerical execution and requires separate quality and runtime qualification. Conversion completion does not demonstrate a usable replay engine. |

Inspect result scope, not filenames: 0xSero's [checkpoint parity receipt](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000/blob/45f538a18569420721e00e37353f9a7e1af7e5da/results/b12x-uring-checkpoint-parity.json) checks 8,192 sampled Engram rows; its [graph replay receipt](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000/blob/45f538a18569420721e00e37353f9a7e1af7e5da/results/b12x-uring-graph-replay.json) concerns a synthetic adapter fixture. Neither proves repeatability of full model generation.

The [official request-encoding toolkit](https://github.com/deepseek-ai/deepseek-recipe) supplies conversation rendering and parsing, not an inference engine. The [llama.cpp conversion PR](https://github.com/ggml-org/llama.cpp/pull/28696) is not interchangeable with the selected fork's Engram format. Generic VRAM calculators and API streaming demonstrations are not hardware qualification evidence.

## 3. Frozen source and build identities

| Component | Inspected immutable revision |
| --- | --- |
| Native model: `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Candidate model: `JigSawPT/DeepSeek-V4.1-Flash-GGUF` | `61ca37086ef36f20a407da7630cfc91273822dcb` |
| JigSawPT runtime | `3c95ac82c19e496348fe57d06f083880a7d86e95` |
| JigSawPT report | `7be408844471a9639a509f571775cd2ece9f912d` |
| 0xSero launcher | `45f538a18569420721e00e37353f9a7e1af7e5da` |
| DGX Spark launcher | `458fadec6106e6fd84f7eda36dd6e0a8fa219ef6` |

Native and converted snapshots contain 510,296,708,312 and 501,809,981,408 weight-file bytes respectively. The converted snapshot has 11 GGUF shards. These are metadata measurements, not downloaded-weight validation. Model cards: [native](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277), [converted](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-GGUF/tree/61ca37086ef36f20a407da7630cfc91273822dcb).

Before admission, establish the converted files' exact source checkpoint and conversion provenance; do not infer that the two current snapshot revisions share a conversion lineage. If this cannot be established, convert from the pinned native snapshot with the pinned converter. Record tensor inventory, original/converted types, converter arguments and every file's SHA-256. Retain both Engram tables and their scales. Validate all repacked expert blocks and Engram rows with a deterministic exhaustive streaming check, rather than treating sampled parity as complete verification.

Record the compiler, CUDA toolkit, driver, operating-system image, dependencies, build flags, kernel architecture targets and executable/library hashes. Fetch by immutable revisions and verify file content hashes. Floating branches or container tags cannot enter an admitted manifest. A source hash alone does not identify the executable.

## 4. Single-card candidate contract

These are **proposed qualification settings**. A launcher must verify the effective values against the running binary; unsupported fields are errors.

| Control | Candidate requirement |
| --- | --- |
| GPU and host | One RTX 5090 32 GB, 128 GB RAM; dedicated inference worker |
| Storage | Local NVMe; plan for 2 TB available working capacity including conversion/download headroom; measure actual available bytes |
| OS | First reproduce the Windows/CUDA 13.0 recipe; Linux is a separate port/build qualification |
| Model | Full Engram-bearing GGUF at the locked revision; no expert pruning or added low-bit conversion |
| Expert cache | Initially 18 GiB GPU and 72 GiB pinned host tier, matching the published starting point |
| I/O workers | Exactly one; cache assignment and accumulation order must be deterministic |
| Concurrent requests | One active request; reject or queue others outside the inference process |
| Speculation | Disabled initially, including DSpark and draft models |
| Prompt/KV reuse | Disabled initially; no cross-request prefix or session reuse |
| Execution state | Fresh inference process per contributing stage for the baseline; measure reload cost. A faster explicit reset is a new profile until equivalence is qualified |
| Context | Candidate 65,536 tokens; verify encoded input plus each stage's output budget fits, with no truncation. Increase only through a recorded profile change |
| Sampling | Temperature 0, seed 438916795; explicitly pin all sampler, grammar, stop-token and token-limit settings |
| Thinking | Enabled for first pass and overflow, disabled for finalizer, preserving the existing stage intent |
| Precision/kernels | Record all effective settings and automatic dispatch decisions; prevent unrecorded changes between replicas |

The reference recipe's 8K context and thinking-off chat example cannot be copied unchanged for this index. Likewise, do not assume that one I/O worker alone makes all reductions, caches and long-context kernels deterministic. Inspect the expert-slot allocation/reduction boundary and test it directly. Fix ordering there if needed; do not repair generated text after the fact.

The local candidate now explicitly uses numeric reasoning effort **75**, the `high` default in the [pinned official encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/encoding.py). The prior hosted API default remains unverified. Label this a local execution variant rather than asserting hosted equivalence. First pass and overflow retain thinking; finalization disables it. The executable renderer supports only the frozen two-message text format. JSON-decoder translation remains a required adapter test, not an assumed capability.

Before large downloads or rentals, inspect build compatibility and storage capacity, verify metadata, and compile the smallest relevant kernel/streaming fixture. Before a full run, execute a complete real scoring request. Benchmark non-destructive reads against the actual model layout; sequential SSD specifications are insufficient. Preserve OS page-cache headroom and verify that pinned allocations do not cause swap thrashing. Do not infer that more RAM cache or more I/O threads necessarily improves useful throughput.

## 5. Definition of replayability

There are three separate properties:

1. **Repeatability:** the same admitted configuration regenerates the same token sequence and output bytes on repeat execution.
2. **Independent replay:** another physical machine independently loads the locked inputs and regenerates those outputs. Different ports, processes or containers on one machine are insufficient.
3. **Model fidelity:** conversion and runtime implement the declared numerical variant. Similar logits or matching prose are not proof of native FP8 equivalence.

Admission requires the first two and an audited declaration of the third. No requirement exists to match the previous hosted API's text. A converted runtime may be independently replayable while producing different scores from native or hosted execution; its artifact must say which numerical variant ran.

Initially both verifiers use the same GPU model, runtime binary, OS/toolkit/driver versions and effective execution contract. Different GPU architectures or OS versions require a separate compatibility qualification. Record physical identities separately from matching profile fields. Two machines may run sequentially; independent recomputation remains mandatory.

Replay receipts are not a TEE attestation or a cryptographic proof of every GPU operation. Consumer hardware is not assumed to support a suitable TEE. The receipt describes the evidence and operator trust involved; neither replay nor attestation validates the economic judgment in the prompt.

## 6. Request and evidence protocol

Prepare a content-addressed request bundle before either verifier runs. Keep expected primary outputs unavailable to the second verifier's inference process until its outputs are sealed. Compare afterward in the coordinator.

Each contributing stage produces an immutable receipt containing:

| Group | Required contents |
| --- | --- |
| Identity | Run, security, stage and attempt IDs; profile digest; host identity |
| Inputs | Source packet digest; exact message bytes; canonical request digest; rendered prompt bytes and input token IDs; generation/grammar/stop settings |
| Dependencies | Hashes of prior reasoning and finalizer-input excerpts; ordered parent-stage receipt IDs |
| Execution | Build/model/file manifests; effective settings; reset state; token limits; start/end times; peak RAM/VRAM; I/O counters |
| Outputs | Full generated token IDs, raw generated bytes, reasoning bytes, final-content bytes, finish reason and usage |
| Outcome | Completed, expected token-limit transition, transport failure, parse failure, or replay mismatch; structured diagnostic data |

The implemented token encoding is a uint64 little-endian count followed by uint32 little-endian token IDs, without a terminator; hash these exact bytes. Hash generated output before JSON parsing or whitespace normalization. Compare the full contributing sequence, including reasoning and EOS/termination behavior. HTTP IDs, timestamps and SSE packet boundaries are transport metadata, not model output. Decode streaming UTF-8 incrementally; do not count chunks as tokens.

A pair passes only if profile, inputs, dependencies, token sequences, generated bytes and termination outcomes match exactly. Matching parsed scores alone fails. If the server cannot expose raw token IDs, add a narrow instrumentation hook or use an offline decode driver before admission; do not silently drop that requirement.

Both replicas independently execute the same stage policy. A token-limit transition may be expected, but its contributing reasoning must match before constructing an overflow/finalizer input. Do not pass one verifier's reasoning into the other as a replacement for recomputation.

Checkpoint completed receipts atomically. Resume only from verified stage boundaries. A transport failure receives a distinct recorded attempt and a clean deterministic restart; never splice a partial stream into another generation. An output mismatch stops qualification or publication for that profile. Do not retry until two answers happen to agree. Preserve all attempts and the first divergence location. Retry and resource limits are finite and recorded before execution.

Use typed structures, explicit schemas and ordinary protocol parsing throughout. No regex may inspect, route, repair, score or gate LLM inputs, outputs or provider responses. If an imported parser violates that rule, replace that call-path dependency before adoption.

## 7. Qualification gates

Each phase has a manifest with an explicit monetary ceiling, elapsed-time ceiling, output-token ceiling, expiry and owned resource IDs. Unset budgets do not launch paid work. Derive the ceilings from an actual provider quote and measured pilot behavior; this document is not a purchase or rental instruction. An external watchdog enforces expiry, persists evidence and removes only resources created for that phase.

| Gate | Work and acceptance condition |
| --- | --- |
| G0: offline preparation | Verify model/build provenance, complete tensors, storage needs, exact AOC request hashes and current weighting reconstruction. Compile/check the target OS path before full-model execution. |
| G1: useful inference | On one machine, complete ENPH using the actual first-pass prompt and settings; validate the JSON and complete brief. Repeat from a fresh process. Exact raw/token equality and usable output are both required. |
| G2: execution boundaries | Run ENPH, INTC, INRO and NVT, covering ordinary output, cutoff boundary and previous difficult/long-output cases. Execute first/overflow/finalizer paths as needed. If finalizer is not naturally reached, add a separately labelled control-flow fixture without changing production limits. |
| G3: independent replay | Repeat the frozen G2 workload twice on each of two independently provisioned physical machines. Every contributing stage and its derived inputs must agree. |
| G4: stress and recovery | Exercise context above 1,024/1,401 tokens, longest admitted inputs/outputs, repeated and cold reads, page-cache pressure, restart, cancellation and resume. Test Unicode, SSE chunking, incomplete JSON and token-limit transitions. Keep inference settings fixed; do not evict system caches on unrelated workloads. |
| G5: full index | Independently execute all 452 names with the frozen profile. Every contributing stage must have a replay partner. Reconstruct the score vector and holdings independently and compare their canonical hashes. |

G1 is the cheapest useful hardware test; arithmetic-only output is not sufficient. G2 examples are diagnostic selections, not portfolio eligibility rules. Cold and warm measurements must refer to actual content/state, not merely discarding the first request. Report how many outputs and tokens were tested; a short matching canary never proves arbitrary future generations.

If G2–G4 fail, diagnose the first divergent token and relevant cache/routing/reduction state. Fix the general execution boundary, assign a new profile digest, and re-run its qualification gates. If a coherent long reasoning pass fails, the candidate fails even if short chat text is repeatable.

## 8. Preserve the AOC methodology

The immutable inputs prepared previously are at:

```text
/mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-h200-feasibility-20260911/aoc-frozen-inputs.json
SHA-256 f577e7e4a873e0d4de96088e5e564db1bd43b463162a1cdfc113e5e744eac3b9
```

All 452 reconstructed first-pass request hashes matched prior receipts. The effective scoring contract remains:

- Frozen Felix universe and source packet `f83902666fbdfa667e7d1abb252400fd261e638ad46cf5a290cd867dd78bb5f2`.
- Existing AOC mandate and executive-brief prompt; stock-specific inputs are company name and ticker. No transcript injection, retrieval or rubric compilation.
- Score 1–100; target brief 400–600 words, current validator minimum 250 words. Preserve the existing typed validator's exact behavior.
- Stage output ceilings 8,192 / 60,000 / 4,096. Preserve stage-transition rules and the existing 16,000-head/32,000-tail finalizer excerpt behavior.
- Inclusion cutoff 70 and market-cap rank weighting. Retain every score; disclose qualifying missing-cap exclusions and normalize using the already authorized rule.
- No added holding caps, sector filters, profitability rules, hedges, rebalance decisions or orders.

The current comparison basket has 29 holdings, with LIT and REMX excluded for missing capitalization. New inference may change which names qualify; do not force the old holdings. An inference failure cannot be disguised as a capitalization exclusion. All 452 scores are required for this job's completion. All-missing capitalization or no qualifiers must retain the existing `needs_data` behavior rather than publish an empty completed index.

## 9. Runtime and cost sizing

Local inspection of the prior successful-stage receipts found **460 stages and 1,479,269 output tokens**, including reasoning and overflow. Failed HTTP attempts are excluded. This is an actual historical workload count, not a prediction of the local runtime's token count.

Using 3.6 tokens/s solely as a scenario:

```text
one pass:  1,479,269 / 3.6 / 3,600 = 114.14 compute-hours
two passes:                         228.28 compute-hours
```

This excludes prefill, loading, conversion, downloading, process resets, canaries and failures. Parallel independent hosts shorten wall-clock time but do not halve billed compute. The published speed comes from different prompts and settings, so it is not a completion-time forecast. Do not use the 21-token/s repeated-content result to price unseen index requests.

After G1/G2, forecast using measured stage times and observed stage frequencies, report uncertainty for rare long generations, and include startup/reset costs. Compare total quoted cost and wall time with the workstation candidate before G5. No full-universe job starts on the strength of model fit alone. The specification preserves reasoning limits; removing reasoning to fit a budget is a methodology change.

## 10. Minimal implementation and rollout

The implemented experimental contract lives in `replay_contract/`:

| File | Implemented responsibility |
| --- | --- |
| `core.py`, `__main__.py` | Request construction, candidate prompt rendering, exact stage/brief validation, canonical hashing, token storage, diagnostic pair comparison and host probe |
| `legacy-profile.json`, `candidate.json`, `source-lock.json` | Frozen scoring settings, unadmitted hardware candidate and original source identities |
| `weights.ts`, `weights.mts` | Isolated existing weighting algorithm with explicit parameters and score validation |
| `offline.py`, `cross-check.mts` | Actual-input acceptance against 452 requests, 460 historical stages and the existing 29-holding basket |
| `test_contract.py` | Protocol, Unicode, stage, identity and replay-mismatch regression tests |
| `README.md` | Exact commands, schemas, ordered remaining implementation tasks and acceptance artifacts |

Run from the repository root:

```bash
python3 -m unittest replay_contract.test_contract -v
python3 -m replay_contract.offline --output-dir /mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-offline-check
```

Use a fresh output directory. The suite uses local pinned historical artifacts and the existing CorbanuAPI `tsx`; it makes no provider calls. A changed source lock, request, accepted historical brief, stage transition or construction fails. The host probe records absence of suitable GPU/compiler separately from offline success.

The raw runtime adapter, snapshot file verifier and bounded rental lifecycle now have implementations and focused tests; the Linux CUDA build script is prepared. Their hardware qualification, exhaustive conversion auditor, scoring coordinator and full-job verifier remain incomplete. Current evidence is tracked in the execution ledger. The `compare` command is an operator-recorded stage diagnostic; it never authorizes publication or authenticates inference. The final verifier must bind requests, parent chains, tokenizer output and source signatures before G5 can pass.

The adapter owns inference settings and instrumentation; the coordinator owns stage transitions, deadlines and immutable evidence; the offline verifier owns acceptance. Reuse canonical hashing and the existing authorized capitalization construction where their contracts match. Do not inherit the Qwen operator's rubric, transcript path, fixed-32 scheduling or fundamental weighting defaults.

Use a profile digest in artifact/cache keys so Qwen, hosted DeepSeek and converted local DeepSeek cannot share inference caches. Keep the current public service unchanged until the new adapter passes G0–G5. Then replace model-specific hardcoding with a small explicit admitted-profile registry; do not weaken the existing Qwen allowlist into accepting arbitrary model names. Make the public verifier select and enforce the artifact's full profile contract.

Regression coverage must reject changed model files, omitted Engram, changed prompts/settings, unknown profile IDs, missing dependency stages, duplicate security coverage, score-only equality, normalized-output equality, reused primary outputs, and missing replica identity. Test interruption/resume without double-counting. Verify the authorized weighting reconstruction with current exclusions and prove that unrelated helper defaults cannot add portfolio constraints.

## 11. Completion criteria and unresolved decisions

Complete means: independently reproducible builds/weights, a measured admitted execution profile, complete two-host stage receipts for the index, verified 452-score coverage, independently identical holdings, an offline verification command, and a measured cost/time report. Hardware owners, build identities and numerical conversion differences must be visible in the recipe.

Outstanding before GPU execution: suitable hardware and reference-OS build; exact conversion lineage; thinking-compatible JSON decoding; tokenizer/raw token instrumentation; long-context memory; independent-host determinism; actual hardware quote and phase budgets. Numeric local effort and the offline scoring/weighting contract are now pinned. The current Linux host reports no `nvidia-smi` and no CUDA compiler. These are explicit qualification tasks, not assertions that the project is impossible. No H200 requirement or full GPU residency requirement is introduced.

## 12. Research artifacts and validation performed

Research cache:

```text
/mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-streaming-spec-20260911/
```

It contains pinned repository metadata/readmes, selected launcher/result files, model metadata, `source-pins.json`, `workload-sizing.json`, and `evidence-manifest.json`. Large weights and third-party binaries were not downloaded or executed. Source commits were resolved through GitHub's commits API and fetched README bytes checked against the inspected snapshots. Selected result JSON was parsed, and historical token totals were recomputed from local receipts.

This specification supersedes the H200-centered feasibility note for the current no-H200 research task. It does not change the existing admitted Qwen runbook or claim a newly verified index.


## 13. Execution status and next action

The offline acceptance run reconstructed all 452 requests and 460 historical stages, reproduced all saved scores/briefs, and matched the entire 29-holding construction with LIT/REMX exclusions. This validates the ported contract against actual inputs. An initial overstrict empty-company-name check was removed after the actual-input test exposed it, and a regression test preserves that behavior.

The next execution step is the reference-OS build and small kernel/streaming fixture in the handoff, followed by the raw runtime adapter and one useful ENPH request. Start with Windows/CUDA 13.0; Linux-only access requires a separate build/fixture qualification. Do not turn lack of local CUDA into a full-model CPU run. Hardware and paid phase limits must be concrete before provisioning; no paid job is running from this specification.

Use the four-RTX-PRO-6000 Linux candidate only after a reproducible build failure, fixed-profile divergence, or measured forecast exceeding the recorded phase limits. It receives a new profile and all gates. This is an ordered fallback, not simultaneous exploration of three hardware stacks.
