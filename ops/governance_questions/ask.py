"""Ask the two rented DeepSeek hosts a question and append the replayed answer to the governance log.

usage: ask.py "question text" [--timeout SECONDS]
       ask.py --show            print the log

Each question is sent to both physical hosts; every GPU rank (4 per host) answers it with the pinned
profile. The log entry records the answer, every rank's output-token hash, whether all 8 agree, and
a SHA-256 chain over previous entries so the log cannot be silently edited.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / 'postfiat-governance-question-log'
LOG = LOG_DIR / 'log.jsonl'
ROLES = ('primary', 'replay')


def read(p):
    return json.loads(p.read_text())


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def entries():
    return [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()] if LOG.exists() else []


def ask(question, timeout):
    status = read(ROOT / 'status.json')
    if status.get('state') not in ('ready', 'busy_or_restarting'):
        raise SystemExit('Hosts are not ready: ' + json.dumps(status))
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    round_id = 'q' + stamp + '-' + hashlib.sha256(question.encode()).hexdigest()[:8]
    command = json.dumps({'round': round_id, 'mode': 'question', 'question': question})
    writer = ("import sys,pathlib;d=pathlib.Path('/workspace/trial/commands');d.mkdir(exist_ok=True);"
              f"t=d/'{round_id}.tmp';t.write_text(sys.stdin.read());t.replace(d/'{round_id}.json')")
    hosts = {}
    for role in ROLES:
        conn = read(ROOT / role / 'connection.json')
        r = subprocess.run(conn['ssh'] + [conn['remote'], 'python3 -c ' + shlex.quote(writer)],
                           input=command, text=True, capture_output=True, timeout=60)
        if r.returncode:
            raise SystemExit(role + ' command write failed: ' + r.stderr[-500:])
        hosts[role] = conn
    asked_at = time.time()
    print(f'asked {round_id} on both hosts; waiting for 8 ranks...', file=sys.stderr, flush=True)
    receipts, errors = {}, {}
    while time.time() - asked_at < timeout:
        for role, conn in hosts.items():
            local = ROOT / role / 'retained'
            subprocess.run(['rsync', '-a', '-e', shlex.join(conn['ssh']),
                            conn['remote'] + ':/workspace/trial/evidence/questions/' + round_id,
                            conn['remote'] + ':/workspace/trial/evidence/command-errors',
                            str(local) + '/questions-sync/'], capture_output=True, timeout=60)
            for rank in range(4):
                p = local / 'questions-sync' / round_id / f'rank{rank}.json'
                e = local / 'questions-sync/command-errors' / f'{round_id}-rank{rank}.json'
                if p.exists():
                    receipts[(role, rank)] = read(p)
                elif e.exists():
                    errors[(role, rank)] = read(e)
        if len(receipts) + len(errors) == 8:
            break
        time.sleep(5)
    if errors and not receipts:
        raise SystemExit('All ranks rejected the question: ' + json.dumps({f'{k[0]}-{k[1]}': v['error'] for k, v in errors.items()}))
    token_hashes = {f'{role}-rank{rank}': r['output_token_ids_sha256'] for (role, rank), r in sorted(receipts.items())}
    ref = receipts[sorted(receipts)[0]] if receipts else None
    agree = len(receipts) == 8 and len(set(token_hashes.values())) == 1 \
        and len({r['rendered_prompt_sha256'] for r in receipts.values()}) == 1
    machines = {role: read(ROOT / role / 'host-identity.json') for role in ROLES}
    prior = entries()
    entry = {
        'schema': 'postfiat.governance-question-log.v1',
        'seq': len(prior) + 1,
        'round': round_id,
        'asked_at_utc': stamp,
        'question': question,
        'model': ref['request']['model'] if ref else None,
        'model_revision': ref['model_revision'] if ref else None,
        'native_profile_sha256': ref['base_native_profile_sha256'] if ref else None,
        'system_prompt': ref['request']['messages'][0]['content'] if ref else None,
        'sampling': ref['effective_sampling'] if ref else None,
        'rendered_prompt_sha256': ref['rendered_prompt_sha256'] if ref else None,
        'answer': ref['content'] if ref else None,
        'reasoning_sha256': hashlib.sha256((ref['reasoning'] or '').encode()).hexdigest() if ref else None,
        'finish_reason': ref['finish_reason'] if ref else None,
        'output_tokens': len(ref['output_token_ids']) if ref else None,
        'raw_output_sha256': ref['raw_sha256'] if ref else None,
        'rank_output_token_hashes': token_hashes,
        'ranks_answered': len(receipts),
        'rank_errors': {f'{k[0]}-rank{k[1]}': v['error'] for k, v in sorted(errors.items())},
        'all_8_ranks_identical_across_2_hosts': agree,
        'hosts': {role: {'machine_id': m['machine_id'], 'instance_id': m['instance_id'],
                         'gpu_uuids': m['gpu_uuids']} for role, m in machines.items()},
        'seconds_to_answer': round(time.time() - asked_at, 1),
        'prev_entry_sha256': prior[-1]['entry_sha256'] if prior else '0' * 64,
    }
    entry['entry_sha256'] = hashlib.sha256(canon(entry)).hexdigest()
    LOG_DIR.mkdir(exist_ok=True)
    rdir = LOG_DIR / 'receipts' / round_id
    rdir.mkdir(parents=True, exist_ok=True)
    for (role, rank), r in receipts.items():
        (rdir / f'{role}-rank{rank}.json').write_text(json.dumps(r, indent=1) + '\n')
    with LOG.open('a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry


def show():
    prev = '0' * 64
    for e in entries():
        body = {k: v for k, v in e.items() if k != 'entry_sha256'}
        ok = e['prev_entry_sha256'] == prev and hashlib.sha256(canon(body)).hexdigest() == e['entry_sha256']
        prev = e['entry_sha256']
        print(f"#{e['seq']} {e['asked_at_utc']}  replay={'MATCH' if e['all_8_ranks_identical_across_2_hosts'] else 'MISMATCH'}  chain={'ok' if ok else 'BROKEN'}")
        print('Q:', e['question'])
        print('A:', (e['answer'] or '')[:2000])
        print()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('question', nargs='?')
    p.add_argument('--timeout', type=float, default=1500)
    p.add_argument('--show', action='store_true')
    a = p.parse_args()
    if a.show or not a.question:
        show()
    else:
        e = ask(a.question, a.timeout)
        print(json.dumps({k: e[k] for k in ('seq', 'round', 'all_8_ranks_identical_across_2_hosts', 'ranks_answered',
                                            'output_tokens', 'finish_reason', 'seconds_to_answer', 'entry_sha256')}, indent=1))
        print('\nANSWER:\n' + (e['answer'] or '(no answer channel; see receipts)'))
