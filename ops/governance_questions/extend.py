"""Extend the rental lease by N minutes if account credit covers it. usage: extend.py MINUTES"""
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, '/home/pfrpc/repos/corbanu-index-operator/ops')
from vast_replay_phase import api, write  # noqa: E402

ROOT = Path(__file__).resolve().parent
minutes = float(sys.argv[1])
plan = json.loads((ROOT / 'execution-plan.json').read_text())
lease = json.loads((ROOT / 'lease.json').read_text())
hourly = plan['combined_hourly_compute_storage_usd']
a = api('GET', 'users/current/')
credit = float(a.get('credit', 0)) + float(a.get('balance', 0))
remaining_cost = max(0, lease['deadline'] - time.time()) / 3600 * hourly
added = minutes / 60 * hourly
if credit < remaining_cost + added + 8:
    raise SystemExit(f'Refused: credit ${credit:.2f} does not cover ${remaining_cost + added:.2f} plus an $8 buffer')
new = {'deadline': lease['deadline'] + minutes * 60, 'ceiling_usd': lease['ceiling_usd'] + added,
       'written_at': time.time(), 'extended_by_minutes': minutes}
write(ROOT / 'lease.json', new)
print(json.dumps({'new_deadline_utc': time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime(new['deadline'])),
                  'ceiling_usd': round(new['ceiling_usd'], 2), 'credit_usd': round(credit, 2)}))
