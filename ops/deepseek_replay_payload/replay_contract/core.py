"""Normative request/stage/receipt checks for the streaming qualification profile.

No model calls, rentals, regex parsing, or publication are performed here.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'legacy-profile.json').read_text())
CANDIDATE = json.loads((ROOT / 'candidate.json').read_text())
STAGES = tuple(x['name'] for x in P['stages'])


def fail(message):
    raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    """Sorted ASCII keys, integral safe numbers, ASCII-escaped JSON, no newline.

    Restriction avoids JS/Python float and non-ASCII object-key ordering drift.
    Unicode string values use JS UTF-16 escapes; no output-text normalization.
    """
    def check(x):
        if x is None or type(x) in (str, bool):
            return
        if type(x) is int and abs(x) <= 9007199254740991:
            return
        if type(x) is list:
            for item in x:
                check(item)
            return
        if type(x) is dict and all(type(k) is str and k.isascii() and
                                  not (k.isdigit() and str(int(k)) == k and int(k) < 4294967295) for k in x):
            for item in x.values():
                check(item)
            return
        fail('Unsupported canonical value: use integral numbers and ASCII object keys')
    check(value)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    result = []
    for c in text:
        n = ord(c)
        if n < 128:
            result.append(c)
        elif n <= 65535:
            result.append('\\u%04x' % n)
        else:
            n -= 65536
            result.append('\\u%04x\\u%04x' % (55296 + (n >> 10), 56320 + (n & 1023)))
    return ''.join(result).encode('ascii')


def digest(value):
    return sha(canonical(value))


def load(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                fail('Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: fail('Non-finite JSON number'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: qualification evidence is never silently replaced.
    with path.open('x') as out:
        out.write(json.dumps(value, indent=2, ensure_ascii=True) + '\n')


def exact(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        fail(label + ': unexpected/missing fields')


def token_bytes(ids):
    if type(ids) is not list or any(type(x) is not int or x < 0 or x > 4294967295 for x in ids):
        fail('Expected uint32 token IDs')
    return struct.pack('<Q', len(ids)) + b''.join(struct.pack('<I', x) for x in ids)


def messages(context, stage, notes=''):
    exact(context, ['question', 'ticker', 'company_name'], 'context')
    if any(type(x) is not str for x in context.values()):
        fail('Context values must be strings')
    if stage not in STAGES:
        fail('Unknown stage')
    stock = '{"ticker":' + canonical(context['ticker']).decode() + ',"company_name":' + canonical(context['company_name']).decode() + '}'
    if stage != 'capped_finalizer':
        return [{'role': 'system', 'content': P['system_prompt']},
                {'role': 'user', 'content': 'Question: "' + context['question'] + '".\n\nStock: ' + stock}]
    if not notes:
        fail('No notes to finalize')
    if len(notes) > P['excerpt_head_chars'] + P['excerpt_tail_chars']:
        notes = notes[:P['excerpt_head_chars']] + '\n\n[DETERMINISTIC MIDDLE OMISSION]\n\n' + notes[-P['excerpt_tail_chars']:]
    return [{'role': 'system', 'content': P['finalizer_system']}, {'role': 'user', 'content':
            'Original question: ' + context['question'] + '\n\nStock: ' + stock +
            '\n\nPrior reasoning notes follow. Finalize the requested score and brief immediately.\n\n' + notes}]


def request(context, stage, notes=''):
    setting = P['stages'][STAGES.index(stage)]
    body = {'model': 'corbanu/deepseek-v4.1-flash', 'messages': messages(context, stage, notes),
            'temperature': P['temperature'], 'seed': P['seed'], 'max_tokens': setting['max_tokens'],
            'response_format': {'type': 'json_object'}}
    if not setting['thinking']:
        body['reasoning_effort'] = 'none'
    return body


def brief(raw):
    # Match the pinned JS validator's Number.isInteger semantics.
    value = json.loads(raw)
    exact(value, ['score', 'brief'], 'brief')
    score = value['score']
    if type(score) not in (int, float) or not 1 <= score <= 100 or int(score) != score or type(value['brief']) is not str:
        fail('Invalid score or brief')
    # ECMAScript trim whitespace; Python str.split has additional separators.
    whitespace = '\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff'
    words, word = [], []
    for char in value['brief']:
        if char in whitespace:
            if word:
                words.append(''.join(word))
                word = []
        else:
            word.append(char)
    if word:
        words.append(''.join(word))
    if len(words) < P['minimum_words']:
        fail('Brief below minimum word count')
    return {'score': int(score), 'brief': ' '.join(words)}


def transition(stage, finish, reasoning, content):
    if stage not in STAGES or finish not in ('stop', 'length'):
        return {'action': 'failed', 'reason': 'abnormal_stage_or_finish'}
    if type(reasoning) is not str or type(content) is not str:
        fail('Output channels must be strings')
    if finish == 'stop':
        try:
            return {'action': 'accepted', 'parsed': brief(content)}
        except (ValueError, TypeError, OverflowError):
            pass
    notes = reasoning + content
    if stage == 'capped_finalizer' or not notes:
        return {'action': 'failed', 'reason': 'invalid_finalizer_or_empty_notes'}
    return {'action': 'next', 'stage': STAGES[STAGES.index(stage) + 1], 'notes': notes}


def contract_digest():
    return digest({p.name: sha(p.read_bytes()) for p in sorted(ROOT.iterdir())
                   if p.suffix in ('.py', '.json', '.ts', '.mts', '.gbnf')})


def render_prompt(body):
    """Pinned official system+user text-only rendering; rejects other shapes."""
    turns = body['messages']
    if len(turns) != 2 or [x.get('role') for x in turns] != ['system', 'user']:
        fail('Only the frozen two-message scoring format is supported')
    for turn in turns:
        exact(turn, ['role', 'content'], 'turn')
    thinking = body.get('reasoning_effort') != 'none'
    effort = ('Reasoning Effort: ' + str(CANDIDATE['thinking_numeric_effort']) +
              ' (range 1-100, the higher the value, the more thorough the reasoning)\n\n') if thinking else ''
    return ('<｜begin▁of▁sentence｜><｜System｜>' + effort + turns[0]['content'] +
            '<｜User｜>' + turns[1]['content'] + '<｜Assistant｜>' + ('<think>' if thinking else '</think>'))


def host_preflight():
    import platform
    import shutil
    import subprocess
    problems = []
    gpu = None
    if shutil.which('nvidia-smi'):
        try:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version',
                                           '--format=csv,noheader'], text=True, timeout=15).strip()
            if len(gpu.splitlines()) != 1 or 'RTX 5090' not in gpu:
                problems.append('candidate_requires_one_RTX_5090')
        except (subprocess.SubprocessError, OSError):
            problems.append('gpu_query_failed')
    else:
        problems.append('nvidia_smi_unavailable')
    if not shutil.which('nvcc'):
        problems.append('cuda_compiler_unavailable')
    else:
        try:
            version = subprocess.check_output(['nvcc', '--version'], text=True, timeout=15)
            if 'release 13.0' not in version:
                problems.append('candidate_requires_cuda_13_0')
        except (subprocess.SubprocessError, OSError):
            problems.append('cuda_query_failed')
    return {'status': 'hardware_available_unqualified' if not problems else 'blocked',
            'os': platform.system(), 'gpu': gpu, 'blockers': problems,
            'linux_runtime_qualified': False, 'publication_allowed': False}


def validate_observation(o):
    exact(o, ['schema', 'fixture', 'contract_sha256', 'profile_sha256', 'host', 'security_id',
              'stage', 'request_sha256', 'parent_output_sha256', 'input_token_ids',
              'output_token_ids', 'reasoning', 'content', 'finish_reason'], 'observation')
    if o['schema'] != 'corbanu.streaming-observation.v1' or type(o['fixture']) is not bool:
        fail('Invalid observation schema')
    if o['contract_sha256'] != contract_digest():
        fail('Contract drift')
    exact(o['host'], ['machine_id', 'gpu_uuid', 'runtime_manifest_sha256'], 'host')
    for key in ['contract_sha256', 'profile_sha256', 'request_sha256']:
        if type(o[key]) is not str or len(o[key]) != 64 or any(c not in '0123456789abcdef' for c in o[key]):
            fail('Invalid hash ' + key)
    if any(type(x) is not str or not x for x in o['host'].values()):
        fail('Missing host identity')
    runtime_hash = o['host']['runtime_manifest_sha256']
    if len(runtime_hash) != 64 or any(c not in '0123456789abcdef' for c in runtime_hash):
        fail('Invalid runtime manifest hash')
    if o['stage'] not in STAGES or o['finish_reason'] not in ('stop', 'length'):
        fail('Unfinished/failed observation')
    if o['parent_output_sha256'] is not None:
        parent = o['parent_output_sha256']
        if type(parent) is not str or len(parent) != 64 or any(c not in '0123456789abcdef' for c in parent):
            fail('Invalid parent hash')
    if (o['stage'] == 'first_pass') != (o['parent_output_sha256'] is None):
        fail('Parent hash must be absent only for first pass')
    if type(o['security_id']) is not str or not o['security_id']:
        fail('Missing security identity')
    for key in ['reasoning', 'content']:
        if type(o[key]) is not str:
            fail('Invalid output channel')
        o[key].encode('utf-8', errors='strict')
    if not o['input_token_ids'] or not o['output_token_ids']:
        fail('Missing tokens')
    token_bytes(o['input_token_ids'])
    token_bytes(o['output_token_ids'])


def compare(a, b):
    for o in [a, b]:
        validate_observation(o)
    if a['host']['machine_id'] == b['host']['machine_id'] or a['host']['gpu_uuid'] == b['host']['gpu_uuid']:
        fail('Independent machine and GPU identities required')
    if a['host']['runtime_manifest_sha256'] != b['host']['runtime_manifest_sha256']:
        fail('Runtime mismatch')
    compared = [x for x in a if x != 'host']
    for field in compared:
        if a[field] != b[field]:
            fail('Replay mismatch: ' + field)
    result = transition(a['stage'], a['finish_reason'], a['reasoning'], a['content'])
    return {'status': 'byte_equal', 'fixture': a['fixture'], 'publication_allowed': False,
            'scope': 'one operator-recorded stage pair; not runtime admission or full-index verification',
            'output_token_sha256': sha(token_bytes(a['output_token_ids'])),
            'reasoning_sha256': sha(a['reasoning'].encode()), 'content_sha256': sha(a['content'].encode()),
            'transition': result}


def prepare(source, out):
    raw = Path(source).read_bytes()
    if sha(raw) != 'f577e7e4a873e0d4de96088e5e564db1bd43b463162a1cdfc113e5e744eac3b9':
        fail('Wrong frozen AOC input file')
    bundle = load(source)
    rows = bundle['requests']
    if len(rows) != 452 or len({x['security_id'] for x in rows}) != 452:
        fail('Universe coverage')
    for row in rows:
        if digest(row['body']) != row['request_sha256']:
            fail('Original request hash mismatch')
    if bundle['scoring_profile'] != P:
        fail('Scoring profile drift')
    save(out, {'schema': 'corbanu.streaming-preparation.v1', 'status': 'offline_ready',
               'contract_sha256': contract_digest(), 'source_file_sha256': sha(raw),
               'source_packet_sha256': bundle['source']['index_sha256'],
               'universe_count': 452, 'first_pass_request_hashes': {x['security_id']: x['request_sha256'] for x in rows},
               'canary': [x for x in rows if x['ticker'] in ('ENPH', 'INTC', 'INRO', 'NVT')],
               'publication_allowed': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='command', required=True)
    x = s.add_parser('prepare'); x.add_argument('--inputs', required=True); x.add_argument('--output', required=True)
    x = s.add_parser('request'); x.add_argument('--context', required=True); x.add_argument('--stage', choices=STAGES, required=True); x.add_argument('--notes'); x.add_argument('--output', required=True)
    x = s.add_parser('compare'); x.add_argument('primary'); x.add_argument('replay'); x.add_argument('--output', required=True)
    s.add_parser('digest')
    x = s.add_parser('host-preflight'); x.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.inputs, args.output)
        elif args.command == 'request':
            notes = Path(args.notes).read_bytes().decode('utf-8') if args.notes else ''
            body = request(load(args.context), args.stage, notes)
            prompt = render_prompt(body)
            save(args.output, {'request': body, 'request_sha256': digest(body), 'contract_sha256': contract_digest(),
                               'rendered_prompt': prompt, 'rendered_prompt_sha256': sha(prompt.encode('utf-8'))})
        elif args.command == 'compare':
            save(args.output, compare(load(args.primary), load(args.replay)))
        elif args.command == 'host-preflight':
            result = host_preflight()
            save(args.output, result)
            print(json.dumps(result))
            raise SystemExit(2 if result['blockers'] else 0)
        else:
            print(contract_digest())
        if args.command != 'digest':
            print(json.dumps({'status': 'passed', 'output': args.output, 'publication_allowed': False}))
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(json.dumps({'status': 'failed', 'error': str(error), 'publication_allowed': False}))
        raise SystemExit(2)
