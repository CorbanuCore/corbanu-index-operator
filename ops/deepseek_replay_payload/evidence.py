"""Strict token evidence and comparison; no prompt or output regex processing."""
import hashlib
import json
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_bytes(ids):
    return len(ids).to_bytes(8, 'little') + b''.join(int(i).to_bytes(4, 'little') for i in ids)


def decode_raw(ids, tokenizer_file):
    tokenizer = json.loads(Path(tokenizer_file).read_text())
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_map = {chr(b): b for b in direct}
    offset = 0
    for b in range(256):
        if b not in direct:
            byte_map[chr(256 + offset)] = b
            offset += 1
    vocabulary = {v: k for k, v in tokenizer['model']['vocab'].items()}
    added = {x['id']: x['content'].encode() for x in tokenizer['added_tokens']}
    return b''.join(added[i] if i in added else bytes(byte_map[c] for c in vocabulary[i]) for i in ids)


def compare(a, b):
    keys = ['input_token_ids', 'output_token_ids', 'raw_sha256', 'finish_reason']
    different = [key for key in keys if a[key] != b[key]]
    x, y = a['output_token_ids'], b['output_token_ids']
    first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    if first is None and len(x) != len(y):
        first = min(len(x), len(y))
    return {'match': not different, 'different_fields': different, 'first_divergent_token_zero_based': first}


def validate_brief(ids, raw, tokenizer_file, finish_reason, minimum_words):
    vocab = json.loads(Path(tokenizer_file).read_text())
    added = {x['content']: x['id'] for x in vocab['added_tokens']}
    eos = added['<｜end▁of▁sentence｜>']
    boundary = added['</think>']
    if finish_reason != 'stop' or not ids or ids[-1] != eos or ids.count(boundary) != 1:
        raise ValueError('Missing normal stop, EOS, or unique reasoning boundary')
    cut = ids.index(boundary)
    reasoning = decode_raw(ids[:cut], tokenizer_file)
    content = decode_raw(ids[cut + 1:-1], tokenizer_file)
    value = json.loads(content)
    if set(value) != {'score', 'brief'} or type(value['score']) not in (int, float) or not 1 <= value['score'] <= 100 or int(value['score']) != value['score']:
        raise ValueError('Invalid score object')
    if not isinstance(value['brief'], str):
        raise ValueError('Brief must be a string')
    whitespace = '\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff'
    count, in_word = 0, False
    for char in value['brief']:
        if char in whitespace:
            in_word = False
        elif not in_word:
            count += 1
            in_word = True
    if count < minimum_words:
        raise ValueError('Brief below minimum word count')
    return {'score': value['score'], 'brief_words': count, 'reasoning_sha256': sha(reasoning), 'content_sha256': sha(content)}
