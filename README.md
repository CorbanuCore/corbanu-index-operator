# Corbanu Index Operator

Local execution service for Corbanu Indexes. It binds only to
localhost and is exposed through a narrow HTTPS tunnel. A job compiles the
mandate, scores the canonical top-1,000 SEC-revenue universe on two independent
H200s in separate 32-request passes under the locked replay profile, verifies every raw response byte for byte, constructs exact integer
weights, and publishes a content-addressed artifact.

The downloadable artifact includes the exact compiled rubric, all 1,000 parsed
scores and reasoning blocks, and the primary/replay content hashes for every
company so the aggregate replay claim can be inspected company by company.

The canonical universe is selected exclusively from frozen SEC registry,
submissions, and Company Facts artifacts. Membership and revenue use the frozen
`sec-fundamentals-proxy-v77` cohort; routed profitability uses the audited v80/v81
FCF component repairs.
Membership is the validated top 1,000 by SEC-derived trailing-four-quarter
revenue. SEC SIC codes route financials and utilities to trailing net income;
all other companies use trailing free cash flow. Licensed vendor data is not an
input, fallback, or membership gate.

This first end-to-end operator enables the recommended fundamental method. The
cap-weighted option remains visible in the product contract but is rejected by
the public operator until a suitable market-data redistribution arrangement is
in place.

Run:

```bash
python3 corbanu_index_operator.py --host 127.0.0.1 --port 8788
```
