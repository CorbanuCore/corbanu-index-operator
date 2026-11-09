# Corbanu Index Operator

Felix packet mode now supports capitalization weighting from canonical
NavStrategies packets. It consumes complete cached transcript bodies and keeps
the existing Qwen revision, runtime and two-verifier replay contract. See
[`docs/FELIX_PACKETS.md`](docs/FELIX_PACKETS.md) for packet ingestion, the offline
weight command, and configuration of an inference service against a pinned
packet. The revenue-universe mode described below is the legacy mode.

Local execution service for Corbanu Indexes. It binds only to
localhost and is exposed through a narrow HTTPS tunnel. A job compiles the
mandate, scores the canonical top-1,000 SEC-revenue universe on two independent
H200s in identical fixed 32-request batches under the locked replay profile, verifies every raw response byte for byte, constructs exact integer
weights, and publishes a content-addressed artifact.

The downloadable artifact includes the exact compiled rubric, all 1,000 parsed
scores and reasoning blocks, and the primary/replay content hashes for every
company so the aggregate replay claim can be inspected company by company.

Every score request also binds one record from a frozen transcript snapshot.
The record contains the accepted EarningsCall.biz transcript, the FMP fallback
for the same selected period, or an explicit unavailable body. Missing
transcripts remain eligible for scoring from the admitted model's training
knowledge. SEC 10-K business sections are not transcript substitutes.

Licensed transcript text is stored under `/home/postfiat/var`, not committed to
the repository. Build and validate the snapshot before starting the service:

```bash
set -a
. /home/postfiat/.config/navstrategies/credentials.env
set +a
PYTHONPATH=/home/postfiat/repos/navstrategies:. \
  /home/postfiat/repos/navstrategies/.venv/bin/python build_transcript_snapshot.py \
  --universe data/sec_v81_top1000_universe_20260828.json \
  --output /home/postfiat/var/corbanu-indexes/transcripts/sec_v81_top1000_transcripts_20260830.json
```

The canonical universe is selected exclusively from frozen SEC registry,
submissions, and Company Facts artifacts. Membership and revenue use the frozen
`sec-fundamentals-proxy-v77` cohort; routed profitability uses the audited v80/v81
FCF component repairs.
Membership is the validated top 1,000 by SEC-derived trailing-four-quarter
revenue. SEC SIC codes route financials and utilities to trailing net income;
all other companies use trailing free cash flow. Licensed vendor data is not an
input, fallback, or membership gate.

The legacy revenue-universe mode enables the fundamental method. Cap weighting
requires packet mode and a request binding its Felix provider, immutable packet
hash and explicit capitalization basis. It does not accept the legacy holding
cap or fundamental overlays. A source change does not by itself update the
separately deployed public service or its website configuration.

Run:

```bash
python3 corbanu_index_operator.py --host 127.0.0.1 --port 8788
```

Run the bounded two-verifier sample without starting the API:

```bash
python3 run_transcript_sample.py
```

Before provisioning or restarting either H200, follow
[`docs/H200_REPLAY_RUNBOOK.md`](docs/H200_REPLAY_RUNBOOK.md). The server is
admitted only under the committed strict fixed-32 launch profile and after its
two-host byte-replay canary passes. `--enable-deterministic-inference` by itself
is not an admitted configuration.

Public evidence endpoints:

- `GET /v1/indexes` lists completed artifacts after verifying their content
  hashes.
- `GET /v1/indexes/{job_id}/artifact` returns the complete index artifact,
  including scores, reasoning, holdings, and replay comparisons.
- `GET /v1/indexes/{job_id}/recipe` returns a smaller
  `corbanu.index-replay-recipe.v1` document with the frozen request, execution
  contract, rubric, input hashes, all per-company receipts, expected holdings,
  and replay procedure.

Replay recipes do not contain licensed transcript bodies. A verifier must
obtain those inputs separately and verify every body against the transcript
hash committed by the recipe before inference.
