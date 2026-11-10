"""Verify complete recorded stage chains and independent byte equality.

This verifier does not admit hardware or attest that an operator performed inference.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile
from . import core as c
from . import runtime as r
from .job import verify_source

SPECIAL={'bos':0,'eos':1,'think_end':128822}


class Vocabulary:
    def __init__(self,path):
        raw=Path(path).read_bytes()
        if c.sha(raw)!=c.CANDIDATE['native_tokenizer_sha256']:c.fail('Wrong native tokenizer bytes')
        model=json.loads(raw)
        self.tokens={i:s for s,i in model['model']['vocab'].items()}
        # Added control and user-defined tokens both decode as literal text;
        # only base BPE vocabulary entries use the byte-level alphabet.
        self.special={x['id']:x['content'] for x in model['added_tokens']}
        self.tokens.update({x['id']:x['content'] for x in model['added_tokens']})
        byte_values=list(range(33,127))+list(range(161,173))+list(range(174,256))
        codepoints=byte_values.copy();extra=0
        for value in range(256):
            if value not in byte_values:
                byte_values.append(value);codepoints.append(256+extra);extra+=1
        self.decode_bytes={chr(cp):b for cp,b in zip(codepoints,byte_values)}

    def piece(self,token):
        if type(token) is not int or token not in self.tokens:c.fail('Unknown vocabulary ID')
        if token in self.special:return self.special[token].encode('utf-8')
        return bytes(self.decode_bytes[ch] for ch in self.tokens[token])


def commitment(stage,observation):
    return c.digest({'stage':stage,**{key:observation[key] for key in (
        'input_token_ids','output_token_ids','reasoning','content','finish_reason','request_sha256')}})


def verify_stage(context, stage, notes, observation, vocab):
    legacy=c.request(context,stage,notes);prompt=c.render_prompt(legacy)
    if observation['request']!=legacy or observation['request_sha256']!=c.digest(legacy):
        c.fail('Request binding mismatch')
    if observation['rendered_prompt']!=prompt or observation['rendered_prompt_sha256']!=c.sha(prompt.encode()):
        c.fail('Rendered prompt binding mismatch')
    c.token_bytes(observation['input_token_ids'])
    if b''.join(vocab.piece(t) for t in observation['input_token_ids'])!=prompt.encode():
        c.fail('Input token/text mismatch')
    body=r.make_body(legacy,observation['input_token_ids'],SPECIAL)
    if observation['local_request']!=body:c.fail('Local decoder request changed')
    wire=base64.b64decode(observation['response_wire_base64'],validate=True)
    if c.sha(wire)!=observation['response_wire_sha256'] or json.loads(wire)!=observation['response']:
        c.fail('Response wire binding mismatch')
    finish=r.validate_response(observation['response'],observation['input_token_ids'],body)
    ids=observation['output_token_ids'];c.token_bytes(ids)
    if ids!=observation['response']['tokens']:c.fail('Generated token identity mismatch')
    pieces=[vocab.piece(t) for t in ids]
    if pieces!=[base64.b64decode(x,validate=True) for x in observation['pieces_base64']]:
        c.fail('Generated token pieces differ from native vocabulary')
    if b''.join(pieces)!=base64.b64decode(observation['raw_output_base64'],validate=True):
        c.fail('Raw output bytes mismatch')
    reasoning,content=r.channels(ids,pieces,stage!='capped_finalizer',finish,SPECIAL)
    if (reasoning,content,finish)!=(observation['reasoning'],observation['content'],observation['finish_reason']):
        c.fail('Output channels or termination changed')
    decision=c.transition(stage,finish,reasoning,content)
    if decision!=observation['transition']:c.fail('Stage transition changed')
    return decision,prompt


def verify_run(job,run,vocab,tokenizer=None):
    if c.digest(job)!=c.CANDIDATE['job_sha256']:c.fail('Job differs from the frozen execution mandate')
    if run['schema']!='corbanu.streaming-run.v1' or run['job_sha256']!=c.digest(job):
        c.fail('Wrong job binding')
    if run['contract_sha256']!=c.contract_digest() or run['candidate_sha256']!=c.digest(c.CANDIDATE):
        c.fail('Execution contract changed')
    rows=run['securities'];expected={x['security_id']:x for x in job['securities']}
    if len(rows)!=len(expected) or {x['security_id'] for x in rows}!=set(expected):
        c.fail('Incomplete or duplicate security coverage')
    scores=[];chain=[];prompts=[];tokens=[]
    for row in rows:
        context=expected[row['security_id']]['context'];stage='first_pass';notes='';parent=None;decision=None
        for record in row['stages']:
            if stage is None or record['stage']!=stage or record['parent_output_sha256']!=parent:
                c.fail('Stage ordering or dependency mismatch')
            observation=record['observation']
            decision,prompt=verify_stage(context,stage,notes,observation,vocab)
            parent=commitment(stage,observation)
            chain.append({'security_id':row['security_id'],'stage':stage,'commitment':parent,
                          'raw_sha256':c.sha(base64.b64decode(observation['raw_output_base64'],validate=True))})
            prompts.append(prompt);tokens.append(observation['input_token_ids'])
            if decision['action']=='next':stage=decision['stage'];notes=decision['notes']
            elif decision['action']=='accepted':stage=None
            else:c.fail('Unsuccessful scoring stage')
        if decision is None or decision['action']!='accepted':c.fail('Security lacks an accepted score')
        scores.append({'security_id':row['security_id'],**decision['parsed']})
    if tokenizer is not None:
        binary,vocab_file=tokenizer
        request=''.join(json.dumps({'text':text})+'\n' for text in prompts).encode()
        result=subprocess.run([str(binary),str(vocab_file)],input=request,stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,check=True,timeout=120)
        rebuilt=[json.loads(line) for line in result.stdout.splitlines()]
        if len(rebuilt)!=len(tokens) or any(row['ids']!=ids for row,ids in zip(rebuilt,tokens)):
            c.fail('Input IDs differ from independently reconstructed tokenization')
    return {'scores':sorted(scores,key=lambda x:x['security_id']),
            'stages':sorted(chain,key=lambda x:(x['security_id'],c.STAGES.index(x['stage']))),
            'input_tokenization_rebuilt':tokenizer is not None}


def compare_runs(job,primary,replay,vocab,tokenizer=None,weight_builder=None):
    for key in ('run_id',):
        if not primary.get(key) or not replay.get(key) or primary[key]==replay[key]:
            c.fail('Distinct execution identities required')
    for key in ('machine_id','gpu_uuid'):
        if not primary['host'].get(key) or not replay['host'].get(key) or primary['host'][key]==replay['host'][key]:
            c.fail('Independent physical machine and GPU required')
    if not primary.get('runtime_manifest_sha256') or primary['runtime_manifest_sha256']!=replay['runtime_manifest_sha256']:
        c.fail('Runtime manifests differ')
    a=verify_run(job,primary,vocab,tokenizer);b=verify_run(job,replay,vocab,tokenizer)
    if a!=b:c.fail('Independent stage chains or scores differ')
    result={'status':'complete_recorded_job_equal','security_count':len(a['scores']),
            'stage_count':len(a['stages']),'score_sha256':c.digest(a['scores']),
            'stage_chain_sha256':c.digest(a['stages']),'input_tokenization_rebuilt':tokenizer is not None,
            'runtime_admitted':False,'publication_allowed':False,'scores':a['scores']}
    if weight_builder is not None:
        left=weight_builder(a['scores']);right=weight_builder(b['scores'])
        if left!=right:c.fail('Independent holdings reconstruction differs')
        if left.get('status')!='ready' or not left.get('weights'):c.fail('Construction needs data; no completed index')
        result.update({'construction':left,'construction_sha256':c.digest(left)})
    return result


def build_weights(source,scores,tsx):
    with tempfile.TemporaryDirectory(prefix='corbanu-replay-weights-') as name:
        directory=Path(name);c.save(directory/'scores.json',scores)
        subprocess.run([str(tsx),str(c.ROOT/'weights.mts'),str(source),str(directory/'scores.json'),
                        str(directory/'construction.json')],check=True,capture_output=True,timeout=60)
        return c.load(directory/'construction.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('job-dir','primary','replay','native-tokenizer','tokenizer-probe','vocab-gguf','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--tsx',type=Path,default=Path('/home/pfrpc/repos/CorbanuAPI/node_modules/.bin/tsx'))
    args=parser.parse_args();job=c.load(args.job_dir/'job.json')
    verify_source(job['source'],(args.job_dir/'source.json').read_bytes())
    if job['required_security_count']!=452 or len(job['securities'])!=452:c.fail('Not the complete AOC universe')
    result=compare_runs(job,c.load(args.primary),c.load(args.replay),Vocabulary(args.native_tokenizer),
                        (args.tokenizer_probe,args.vocab_gguf),
                        lambda scores:build_weights(args.job_dir/'source.json',scores,args.tsx))
    c.save(args.output,result)
