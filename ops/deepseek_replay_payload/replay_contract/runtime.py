"""Raw llama.cpp adapter; structured protocol only, no chat-output repair.

Runtime qualification remains separate from collecting an observation.
"""
import base64
import json
from pathlib import Path
import time
import urllib.request
from . import core as c


def post(endpoint, path, payload, timeout):
    if not endpoint.startswith('http://127.0.0.1:'):
        c.fail('Use a loopback server or authenticated SSH tunnel')
    request=urllib.request.Request(endpoint+path,data=json.dumps(payload).encode('utf-8'),
                                   headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        raw=response.read()
    value=json.loads(raw)
    if type(value) is not dict or 'error' in value:
        c.fail('Runtime returned an error')
    return value,raw


def make_body(legacy, prompt_ids, special):
    c.token_bytes(prompt_ids)
    if not prompt_ids or prompt_ids[0]!=special['bos']:
        c.fail('Encoded prompt must begin with exactly the explicit BOS')
    if len(prompt_ids)+legacy['max_tokens']>c.CANDIDATE['context_tokens']:
        c.fail('Full stage budget does not fit; truncation is forbidden')
    thinking=legacy.get('reasoning_effort')!='none'
    grammar=(c.ROOT/'json-object.gbnf').read_text()
    if thinking:
        # Lazy token activation consumes the trigger itself, so include it in root.
        grammar=grammar.replace('root ::= ws object ws','root ::= "</think>" ws object ws',1)
    return {'prompt':prompt_ids,'n_predict':legacy['max_tokens'],'seed':legacy['seed'],
            'temperature':legacy['temperature'],'stream':False,'return_tokens':True,
            'cache_prompt':False,'n_keep':0,'stop':[],'ignore_eos':False,
            'samplers':['temperature'],'top_k':0,'top_p':1,'min_p':0,'typical_p':1,
            'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,
            'dry_multiplier':0,'mirostat':0,'dynatemp_range':0,
            'grammar':grammar,'grammar_lazy':thinking,
            'grammar_triggers':[{'type':0,'value':'</think>','token':special['think_end']}] if thinking else [],
            'preserved_tokens':['</think>'] if thinking else [],'reasoning_budget_tokens':-1,
            'reasoning_control':False}


def channels(ids, pieces, thinking, finish, special):
    c.token_bytes(ids)
    if len(ids)!=len(pieces) or not ids:
        c.fail('Missing raw output token pieces')
    if any(type(x) is not bytes for x in pieces):
        c.fail('Token pieces must be raw bytes')
    eos=[i for i,token in enumerate(ids) if token==special['eos']]
    if finish=='stop':
        if eos!=[len(ids)-1]: c.fail('Normal stop must include exactly one terminal EOS')
        end=len(ids)-1
    elif finish=='length':
        if eos: c.fail('Length finish cannot include EOS')
        end=len(ids)
    else:
        c.fail('Abnormal termination')
    boundaries=[i for i,token in enumerate(ids[:end]) if token==special['think_end']]
    if thinking:
        if len(boundaries)>1: c.fail('Repeated reasoning boundary')
        if not boundaries:
            if finish!='length': c.fail('Stopped without final-answer boundary')
            reasoning=b''.join(pieces[:end]);content=b''
        else:
            at=boundaries[0]
            if pieces[at]!=b'</think>': c.fail('Reasoning token does not match vocabulary')
            reasoning=b''.join(pieces[:at]);content=b''.join(pieces[at+1:end])
    else:
        if boundaries: c.fail('Finalizer unexpectedly generated a reasoning boundary')
        reasoning=b'';content=b''.join(pieces[:end])
    # A length cutoff may end inside a UTF-8 codepoint. Keep all raw bytes and fail
    # the stage explicitly until the exact legacy channel behavior is qualified.
    return reasoning.decode('utf-8'),content.decode('utf-8')


def validate_response(value, expected_ids, body):
    required={'tokens','content','stop','stop_type','stopping_word','truncated',
              'tokens_predicted','tokens_evaluated','generation_settings'}
    if not required.issubset(value):c.fail('Incomplete completion response')
    if value['stop'] is not True or value['truncated'] is not False:
        c.fail('Incomplete or truncated generation')
    c.token_bytes(value['tokens'])
    if len(value['tokens'])!=value['tokens_predicted']:
        c.fail('Runtime token count does not include every generated token')
    if value['tokens_evaluated']!=len(expected_ids):
        c.fail('Runtime did not evaluate the exact prompt')
    if len(value['tokens'])>body['n_predict']:c.fail('Output budget exceeded')
    if value['stopping_word']!='':c.fail('Unrequested stop string')
    finish={'eos':'stop','limit':'length'}.get(value['stop_type'])
    if finish is None:c.fail('Unexpected stop type')
    effective=value['generation_settings']
    # The full effective settings are retained; these fields are decisive gates.
    for key in ('seed','temperature','n_predict','grammar','grammar_lazy','grammar_triggers',
                'top_k','top_p','min_p','typical_p','repeat_penalty',
                'presence_penalty','frequency_penalty','dry_multiplier','mirostat',
                'dynatemp_range','stop','ignore_eos','stream'):
        if effective.get(key)!=body[key]:c.fail('Effective setting mismatch: '+key)
    expected_preserved=[x['token'] for x in body['grammar_triggers']]
    if effective.get('preserved_tokens')!=expected_preserved:
        c.fail('Effective preserved token IDs differ')
    return finish


def observe(endpoint, context, stage, notes, special, timeout=3600,evidence_dir=None):
    legacy=c.request(context,stage,notes);prompt=c.render_prompt(legacy)
    tokens,_=post(endpoint,'/tokenize',{'content':prompt,'add_special':False,'parse_special':True},30)
    ids=tokens['tokens'];body=make_body(legacy,ids,special)
    if evidence_dir is not None:c.save(Path(evidence_dir)/'local-request.json',body)
    started=time.time()
    response,wire=post(endpoint,'/completion',body,timeout)
    if evidence_dir is not None:
        with (Path(evidence_dir)/'response-wire.json').open('xb') as stream:stream.write(wire)
    finish=validate_response(response,ids,body)
    # The pinned /tokenize endpoint accepts IDs directly and returns each token's
    # vocabulary piece. This bypasses text tokenization and preserves special IDs.
    decoded,_=post(endpoint,'/tokenize',{'content':response['tokens'],'with_pieces':True,
                                      'add_special':False,'parse_special':True},30)
    c.exact(decoded,['tokens'],'token piece response')
    if [x['id'] for x in decoded['tokens']]!=response['tokens']:c.fail('Token identity changed during decoding')
    pieces=[]
    for token in decoded['tokens']:
        piece=token['piece']
        if type(piece) is str: pieces.append(piece.encode('utf-8'))
        elif type(piece) is list and all(type(x) is int and 0<=x<=255 for x in piece): pieces.append(bytes(piece))
        else:c.fail('Invalid token piece encoding')
    reasoning,content=channels(response['tokens'],pieces,stage!='capped_finalizer',finish,special)
    return {'request':legacy,'request_sha256':c.digest(legacy),'rendered_prompt':prompt,
            'rendered_prompt_sha256':c.sha(prompt.encode()),'input_token_ids':ids,
            'local_request':body,'response':response,'response_wire_sha256':c.sha(wire),
            'response_wire_base64':base64.b64encode(wire).decode(),
            'output_token_ids':response['tokens'],'raw_output_base64':base64.b64encode(b''.join(pieces)).decode(),
            'pieces_base64':[base64.b64encode(x).decode() for x in pieces],'reasoning':reasoning,'content':content,
            'finish_reason':finish,'elapsed_seconds':time.time()-started,
            'transition':c.transition(stage,finish,reasoning,content),'publication_allowed':False}
