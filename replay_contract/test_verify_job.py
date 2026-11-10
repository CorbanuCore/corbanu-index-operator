import base64
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import core as c
from . import runtime as r
from . import verify_job as v


class FixtureVocabulary:
    def __init__(self,pieces):self.pieces=pieces
    def piece(self,token):return self.pieces[token]


class JobVerificationTests(unittest.TestCase):
    def fixture(self):
        context={'question':'Frozen fixture premise','ticker':'TEST','company_name':'Fixture'}
        legacy=c.request(context,'first_pass');prompt=c.render_prompt(legacy)
        content=json.dumps({'score':70,'brief':' '.join(['evidence']*250)})
        body=r.make_body(legacy,[0,10],v.SPECIAL)
        effective=body.copy();effective['preserved_tokens']=[128822]
        response={'tokens':[20,128822,30,1],'content':'fixture','stop':True,'stop_type':'eos',
                  'stopping_word':'','truncated':False,'tokens_predicted':4,'tokens_evaluated':2,
                  'generation_settings':effective}
        wire=json.dumps(response).encode()
        pieces={0:'<｜begin▁of▁sentence｜>'.encode(),
                10:prompt[len('<｜begin▁of▁sentence｜>'):].encode(),20:b'notes',128822:b'</think>',
                30:content.encode(),1:b'<eos>'}
        obs={'request':legacy,'request_sha256':c.digest(legacy),'rendered_prompt':prompt,
             'rendered_prompt_sha256':c.sha(prompt.encode()),'input_token_ids':[0,10],
             'local_request':body,'response':response,'response_wire_sha256':c.sha(wire),
             'response_wire_base64':base64.b64encode(wire).decode(),'output_token_ids':response['tokens'],
             'raw_output_base64':base64.b64encode(b''.join(pieces[t] for t in response['tokens'])).decode(),
             'pieces_base64':[base64.b64encode(pieces[t]).decode() for t in response['tokens']],
             'reasoning':'notes','content':content,'finish_reason':'stop',
             'transition':c.transition('first_pass','stop','notes',content)}
        job={'securities':[{'security_id':'fixture:TEST','context':context}]}
        run={'schema':'corbanu.streaming-run.v1','run_id':'fixture-a','job_sha256':c.digest(job),
             'contract_sha256':c.contract_digest(),'candidate_sha256':c.digest(c.CANDIDATE),
             'host':{'machine_id':'fixture-machine-a','gpu_uuid':'fixture-gpu-a'},'runtime_manifest_sha256':'a'*64,
             'securities':[{'security_id':'fixture:TEST','stages':[{'stage':'first_pass',
                 'parent_output_sha256':None,'observation':obs}]}]}
        return job,run,FixtureVocabulary(pieces)

    def checked(self,job,run,vocab):
        with patch.dict(c.CANDIDATE,{'job_sha256':c.digest(job)}):
            run['candidate_sha256']=c.digest(c.CANDIDATE)
            return v.verify_run(job,run,vocab)

    def test_valid_record_is_not_admission(self):
        job,a,vocab=self.fixture();b=copy.deepcopy(a)
        b['run_id']='fixture-b';b['host']={'machine_id':'fixture-machine-b','gpu_uuid':'fixture-gpu-b'}
        with patch.dict(c.CANDIDATE,{'job_sha256':c.digest(job)}):
            a['candidate_sha256']=b['candidate_sha256']=c.digest(c.CANDIDATE)
            result=v.compare_runs(job,a,b,vocab)
        self.assertEqual(result['security_count'],1);self.assertFalse(result['runtime_admitted'])
        self.assertFalse(result['publication_allowed']);self.assertFalse(result['input_tokenization_rebuilt'])

    def test_missing_duplicate_and_extra_stages_rejected(self):
        for mutation in ['missing','duplicate','extra_stage','wrong_parent']:
            job,run,vocab=self.fixture()
            if mutation=='missing':run['securities']=[]
            if mutation=='duplicate':run['securities']*=2
            if mutation=='extra_stage':run['securities'][0]['stages']*=2
            if mutation=='wrong_parent':run['securities'][0]['stages'][0]['parent_output_sha256']='f'*64
            with self.assertRaises(ValueError):self.checked(job,run,vocab)

    def test_tampered_request_wire_tokens_and_channels_rejected(self):
        for field,value in [('request_sha256','f'*64),('response_wire_sha256','f'*64),
                            ('output_token_ids',[30,1]),('content','different'),
                            ('raw_output_base64',base64.b64encode(b'changed').decode())]:
            job,run,vocab=self.fixture();run['securities'][0]['stages'][0]['observation'][field]=value
            with self.assertRaises(ValueError):self.checked(job,run,vocab)

    def test_copied_execution_or_physical_host_rejected(self):
        job,a,vocab=self.fixture();b=copy.deepcopy(a)
        with self.assertRaises(ValueError):v.compare_runs(job,a,b,vocab)
        b['run_id']='fixture-b'
        with self.assertRaises(ValueError):v.compare_runs(job,a,b,vocab)

    def test_unpinned_job_rejected(self):
        job,run,vocab=self.fixture()
        with self.assertRaises(ValueError):v.verify_run(job,run,vocab)

    def test_user_defined_added_tokens_are_literal(self):
        model={'model':{'vocab':{'Ġx':0}},'added_tokens':[{'id':1,'content':'<｜User｜>','special':False}]}
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'tokenizer.json';path.write_text(json.dumps(model))
            with patch.dict(c.CANDIDATE,{'native_tokenizer_sha256':c.sha(path.read_bytes())}):
                vocab=v.Vocabulary(path)
                self.assertEqual(vocab.piece(0),b' x')
                self.assertEqual(vocab.piece(1),'<｜User｜>'.encode())


if __name__=='__main__':unittest.main()
