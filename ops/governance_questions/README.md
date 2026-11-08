# Post Fiat governance question log (MVP)

Two owned Vast hosts (4x B200 each) run the qualified DeepSeek V4.1 Flash-1017 / DeepInfer profile.
Every question is answered by all 8 GPU ranks across both machines; the log records the answer,
each rank's output-token hash, whether all 8 match, and a SHA-256 chain over prior entries.

Run dir layout (scratch): controller.py, ask.py, extend.py, payload/ = the 2026-11-08 warm payload
plus payload-overlay/, with payload-manifest.json regenerated.

    python3 controller.py 120      # rent the qualified pair under a $120 ceiling; boot ~11 min
    python3 ask.py "question"      # ask both hosts, append to ../postfiat-governance-question-log/log.jsonl
    python3 ask.py --show          # print the log and verify the hash chain
    python3 extend.py 30           # extend the lease if account credit covers it
    touch stop                     # release both rentals

Lifecycle: rentals are destroyed only at the lease deadline, on the stop file, or by the account spend
guard. A worker exit triggers a reload from local disk (restart_worker.py, max 3 per host); a bad or
failing question writes evidence/command-errors/ and never stops the engine. This fixes the
2026-11-08 failure where replacing the worker for a free-text question destroyed both hosts.
