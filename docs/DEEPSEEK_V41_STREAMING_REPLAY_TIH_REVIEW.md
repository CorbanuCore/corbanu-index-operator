# TIH assessment: DeepSeek V4.1 streaming replay specification

Date: 2026-09-11. Reviewed document: [streaming replay specification](DEEPSEEK_V41_STREAMING_REPLAY_SPEC.md).

Document SHA-256: `0e8fd6f1663b3b70b73848cad4e7f33164fd1dd16d53342683509c94bcce8e22`.

**Overall: 87.7/100**, the arithmetic mean of 15 real TIH ratings, five per model. The specification was not rewritten during this assessment.

| Judge | Individual scores | Mean | Range |
| --- | --- | ---: | --- |
| GPT-6 Astra | 90, 90, 90, 91, 91 | 90.4 | 90–91 |
| Claude Fable 5.1 | 84, 85, 84, 84, 86 | 84.6 | 84–86 |
| GLM 5.3 | 88, 88, 88, 88, 88 | 88.0 | 88 |

TIH used its standard prompt: “Rate this document on a scale of 1-100. Output the score and your reasoning.” Each response also supplied the best feature, worst feature and highest-value next edit. These are document-quality judgments, not an independent source audit, hardware test, admission decision or probability of success. The prompt is general; it does not define a fixed weighted engineering rubric. Repeated model ratings are not independent empirical measurements of feasibility.

## What the judges agreed was strong

The specification consistently distinguishes external reports, proposed settings and measurements actually made. Its separation of repeatability, independent replay and model fidelity is useful. It also preserves the existing AOC methodology and avoids claiming that matching scores, sampled tensor checks, or successful short chat establish verified inference.

The judges particularly valued preserving failures, refusing selective retries, comparing raw outputs, binding execution profiles into cache identities, and including reasoning tokens in workload sizing.

## Highest-value changes

1. **Make the execution contract machine-readable.** All five Astra ratings ask for versioned request/profile/receipt schemas, exact serialization and hashing rules, a stage-transition state machine, and immutable identities for inherited validator and weighting implementations. Add a small golden fixture with expected input, receipt, score-vector and holdings hashes. Two implementers should not have to resolve those rules independently from prose.
2. **Make the document understandable outside the current conversation.** All five Fable ratings request definitions for project terms, ticker examples and unexplained constants. Define AOC, Felix, Engram, DSpark and `needs_data`; explain why the long-context boundary probes and finalizer excerpt lengths exist.
3. **Turn unresolved decisions into an ordered preflight table.** State who resolves each item, what test resolves it, which gate it blocks and what happens on failure. Define the Windows/Linux decision path and how a measured cost/time envelope triggers consideration of the four-GPU candidate. Do not invent a spending ceiling or scoring parameter merely to fill a table.
4. **Define the model-fidelity audit.** Explain exactly which tensor/conversion/runtime properties must be checked, what numerical variant is declared, and what evidence permits admission. Keep same-runtime replay equality separate from fidelity to native FP8 execution. A vague “audited declaration” is weaker than the other testable gates.

The most useful next revision is an execution-contract appendix plus a short glossary and decision table. Additional prose about rigor would not resolve these gaps.

## Corrections to judge suggestions

The raw reviews are preserved, including errors. GLM runs 3 and 4 describe ENPH, INTC, INRO and NVT as pipeline stages. They are **company tickers** used as diagnostic cases; the actual stages are first pass, overflow and capped finalizer. This misreading supports clearer definitions, not renaming the pipeline around tickers.

GLM run 4 speculates that the 60,000-token overflow cannot fit a 65,536-token context. That has not been established: the actual encoded prompt size must be measured. Do not reduce the authorized output ceiling on the basis of that suggestion.

GLM run 1 suggests an H200 or hosted API reference for fidelity testing. H200s are outside this specification's scope, and a mutable hosted API does not establish an exact pinned native-runtime reference. Any reference test must use an identified implementation and checkpoint consistent with the no-H200 constraint.

## Reproducibility record

Run group: `deepseek41-streaming-v1-20260911`.

Evidence directory:

```text
/mnt/HC_Volume_101713660/pfrpc/scratch/deepseek41-streaming-tih-20260911/
```

Contents include the frozen document, `harness.sqlite3` with prompts and raw judge text, `results.json` with all 15 reviews, `run-metadata.json`, scoring logs and this report.

The harness checkout is based on commit `441c3473c8842fb8c1f7ba5468f6d952aa591676` with pre-existing local changes. Exact Python source hashes are recorded in run metadata; the commit alone does not identify the harness that ran. Credentials were resolved through its existing key mechanism and were not written into the result database.

The first complete three-model assessment validated the scoring path, then four additional calls per model completed the five-run assessment. All 15 ratings succeeded on their first attempt. The full scoring command was used without automatic rewriting or modality selection. No promotion comparison was performed, and no source-document bytes changed.
