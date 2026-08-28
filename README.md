# Corbanu Index Operator

Local execution service for the public Corbanu Indexes demo. It binds only to
localhost and is exposed through a narrow HTTPS tunnel. A job compiles the
mandate, scores the 108-company public demonstration universe on two independent
H200s under the strict one-request-per-host replay profile, verifies every raw response byte for byte, constructs exact integer
weights, and publishes a content-addressed artifact.

This first end-to-end operator enables the recommended fundamental method. The
cap-weighted option remains visible in the product contract but is rejected by
the public operator until a suitable market-data redistribution arrangement is
in place.

Run:

```bash
python3 corbanu_index_operator.py --host 127.0.0.1 --port 8788
```
