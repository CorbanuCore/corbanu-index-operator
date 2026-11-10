"""G1 probe on the qualified G0 host: shard verification, then fresh-process
repeatability of the real ENPH first-pass request.

Round A (short): two fresh llama-server processes, ENPH first_pass with
max_tokens capped to --short-tokens, compare generated token IDs and bytes.
Round B (full): the frozen 8,192-token first pass, once or twice as time allows.

This is qualification evidence for the candidate profile; it does not admit
the profile or publish anything.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_contract import core as c  # noqa: E402
from replay_contract import engine as e  # noqa: E402
from replay_contract import runtime as r  # noqa: E402
from replay_contract.audit_model import verify_files  # noqa: E402

SPECIAL = {'bos': 0, 'eos': 1, 'think_end': 128822}


def sha_tokens(ids):
    return c.sha(c.token_bytes(ids))


def one_round(binary, model, context, stage, out, remaining, label):
    out.mkdir(parents=True, exist_ok=True)
    port = e.free_port()
    endpoint = 'http://127.0.0.1:' + str(port)
    command = e.arguments(binary, model, port)
    c.save(out / 'launch.json', {'argv': command, 'label': label})
    started = time.time()
    with e.server(command, endpoint, out / 'runtime.log', remaining) as (process, available):
        ready = time.time() - started
        obs = r.observe(endpoint, context, stage, '', SPECIAL, timeout=available, evidence_dir=out)
    ids = obs.get('output_token_ids')
    summary = {
        'label': label,
        'startup_seconds': round(ready, 1),
        'total_seconds': round(time.time() - started, 1),
        'output_tokens': len(ids) if ids else None,
        'output_token_sha256': sha_tokens(ids) if ids else None,
        'finish_reason': obs.get('finish_reason'),
        'transition': obs.get('transition'),
        'raw_output_sha256': c.sha(obs['raw_output_base64'].encode()) if obs.get('raw_output_base64') else None,
        'content_sha256': c.sha(obs['content'].encode()) if isinstance(obs.get('content'), str) else None,
        'reasoning_sha256': c.sha(obs['reasoning'].encode()) if isinstance(obs.get('reasoning'), str) else None,
        'observation_keys': sorted(obs),
    }
    if ids and summary['total_seconds'] > summary['startup_seconds']:
        summary['tokens_per_second'] = round(len(ids) / (summary['total_seconds'] - summary['startup_seconds']), 2)
    c.save(out / 'observation.json', obs)
    c.save(out / 'summary.json', summary)
    return summary


def compare(a, b):
    keys = ('output_tokens', 'output_token_sha256', 'raw_output_sha256', 'content_sha256', 'reasoning_sha256', 'finish_reason', 'transition')
    return {k: (a.get(k) == b.get(k)) for k in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--security', default='felix:ENPHon')
    p.add_argument('--short-tokens', type=int, default=384)
    p.add_argument('--full-rounds', type=int, default=1)
    p.add_argument('--deadline-epoch', type=float, required=True)
    p.add_argument('--skip-verify', action='store_true')
    p.add_argument('--only-full', action='store_true', help='skip the short rounds; run full rounds only')
    p.add_argument('--full-start', type=int, default=1)
    a = p.parse_args()
    ev = Path(a.evidence)
    ev.mkdir(parents=True, exist_ok=True)
    inventory = c.load(a.inventory)['inventory'] if 'inventory' in c.load(a.inventory) else c.load(a.inventory)
    if not a.skip_verify:
        t = time.time()
        verified = verify_files(inventory, Path(a.model_dir))
        c.save(ev / 'model-integrity.json', {'seconds': round(time.time() - t, 1), **verified} if isinstance(verified, dict) else {'seconds': round(time.time() - t, 1), 'result': verified})
        print('model integrity:', json.dumps(verified)[:300], flush=True)
    job = c.load(a.job)
    context = next(s['context'] for s in job['securities'] if s['security_id'] == a.security)
    model = sorted(Path(a.model_dir) / f['name'] for f in inventory['files'])[0]
    stage = 'first_pass'
    report = {'security': a.security, 'short_tokens': a.short_tokens, 'rounds': [], 'short_match': None, 'full_match': None}

    def remaining():
        return a.deadline_epoch - time.time() - 60

    if not a.only_full:
        with patch.dict(c.P['stages'][c.STAGES.index(stage)], {'max_tokens': a.short_tokens}):
            s1 = one_round(a.binary, model, context, stage, ev / 'short-1', remaining(), 'short-1')
            print('short-1', json.dumps(s1), flush=True)
            s2 = one_round(a.binary, model, context, stage, ev / 'short-2', remaining(), 'short-2')
            print('short-2', json.dumps(s2), flush=True)
        report['rounds'] += [s1, s2]
        report['short_match'] = compare(s1, s2)
        print('SHORT MATCH', json.dumps(report['short_match']), flush=True)

    # Reload any full rounds already sealed on disk (evidence files are write-once).
    fulls = []
    for existing in sorted(ev.glob('full-*/summary.json')):
        fulls.append(c.load(existing))
    for i in range(a.full_start, a.full_start + a.full_rounds):
        if remaining() < 900:
            break
        if (ev / f'full-{i}').exists():
            continue
        f = one_round(a.binary, model, context, stage, ev / f'full-{i}', remaining(), f'full-{i}')
        print(f'full-{i}', json.dumps(f), flush=True)
        fulls.append(f)
    fulls.sort(key=lambda x: x['label'])
    report['rounds'] += fulls
    if len(fulls) >= 2:
        report['full_match'] = compare(fulls[0], fulls[1])
        print('FULL MATCH', json.dumps(report['full_match']), flush=True)
    c.save(ev / f'report-{int(time.time())}.json', report)


if __name__ == '__main__':
    main()
