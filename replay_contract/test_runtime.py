import unittest
from . import core as c
from . import runtime as r


class RuntimeTests(unittest.TestCase):
    special={'bos':0,'eos':1,'think_end':128822}

    def body(self,stage='first_pass'):
        legacy=c.request({'question':'AOC premise','ticker':'ENPH','company_name':'Enphase'},stage,'notes')
        return r.make_body(legacy,[0,3],self.special)

    def response(self):
        body=self.body()
        effective=body.copy();effective['preserved_tokens']=[128822]
        return {'tokens':[3,1],'content':'text','stop':True,'stop_type':'eos','stopping_word':'',
                'truncated':False,'tokens_predicted':2,'tokens_evaluated':2,'generation_settings':effective}

    def test_grammar_begins_at_exact_token_boundary(self):
        body=self.body()
        self.assertTrue(body['grammar_lazy'])
        self.assertEqual(body['grammar_triggers'],[{'type':0,'value':'</think>','token':128822}])
        self.assertTrue(body['grammar'].startswith('root ::= "</think>"'))
        final=self.body('capped_finalizer')
        self.assertFalse(final['grammar_lazy']);self.assertEqual(final['grammar_triggers'],[])
        self.assertNotIn('</think>',final['grammar'])

    def test_overflow_budget_is_not_clipped(self):
        legacy=c.request({'question':'Q','ticker':'T','company_name':''},'overflow')
        self.assertEqual(r.make_body(legacy,[0],self.special)['n_predict'],60000)
        with self.assertRaises(ValueError):r.make_body(legacy,[0]*6000,self.special)

    def test_split_utf8_pieces_join_before_decoding(self):
        ids=[3,4,128822,5,1]
        pieces=[b'\xf0\x9f',b'\x90\x8b',b'</think>',b'{"score":70}',b'<EOS>']
        self.assertEqual(r.channels(ids,pieces,True,'stop',self.special),('🐋','{"score":70}'))

    def test_length_reasoning_can_advance(self):
        self.assertEqual(r.channels([3],[b'notes'],True,'length',self.special),('notes',''))

    def test_eos_and_reasoning_boundaries_are_mandatory(self):
        for ids,pieces,finish in [([3],[b'notes'],'stop'),([1,3],[b'eos',b'notes'],'stop'),
                                 ([1],[b'eos'],'length'),([128822,128822,1],[b'</think>',b'</think>',b'eos'],'stop')]:
            with self.assertRaises(ValueError):r.channels(ids,pieces,True,finish,self.special)

    def test_protocol_and_effective_drift_fail(self):
        self.assertEqual(r.validate_response(self.response(),[0,3],self.body()),'stop')
        for key,value in [('truncated',True),('stop',False),('tokens_predicted',1),('tokens_evaluated',1),('stop_type','word')]:
            response=self.response();response[key]=value
            with self.assertRaises(ValueError):r.validate_response(response,[0,3],self.body())
        for key in ['grammar_triggers','temperature','top_k']:
            response=self.response();del response['generation_settings'][key]
            with self.assertRaises(ValueError):r.validate_response(response,[0,3],self.body())


if __name__=='__main__':unittest.main()
