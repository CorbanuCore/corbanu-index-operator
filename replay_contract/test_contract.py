import ast
import copy
import json
import struct
import unittest
from . import core as c


class ContractTests(unittest.TestCase):
    def context(self):
        return {'question': 'AOC premise', 'ticker': 'EXAMPLE', 'company_name': 'Example 🐋 & Co'}

    def output(self):
        return json.dumps({'score': 70, 'brief': ' '.join(['evidence'] * 250)})

    def observation(self):
        return {'schema': 'corbanu.streaming-observation.v1', 'fixture': True,
                'contract_sha256': c.contract_digest(), 'profile_sha256': '1' * 64,
                'host': {'machine_id': 'fixture-a', 'gpu_uuid': 'fixture-gpu-a', 'runtime_manifest_sha256': '2' * 64},
                'security_id': 'fixture:EXAMPLE', 'stage': 'first_pass', 'request_sha256': '3' * 64,
                'parent_output_sha256': None, 'input_token_ids': [1, 2], 'output_token_ids': [3, 4],
                'reasoning': 'fixture reasoning', 'content': self.output(), 'finish_reason': 'stop'}

    def pair(self):
        a = self.observation(); b = copy.deepcopy(a)
        b['host']['machine_id'] = 'fixture-b'; b['host']['gpu_uuid'] = 'fixture-gpu-b'
        return a, b

    def test_canonical_golden(self):
        self.assertEqual(c.canonical({'z': '🐋\u007f', 'a': 70}), b'{"a":70,"z":"\\ud83d\\udc0b\x7f"}')
        for invalid in [0.5, {'2': 'numeric object key'}, float('nan'), {'é': 1}]:
            with self.assertRaises(ValueError): c.canonical(invalid)

    def test_token_encoding(self):
        self.assertEqual(c.token_bytes([1, 4294967295]), struct.pack('<QII', 2, 1, 4294967295))
        for invalid in [[True], [-1], [4294967296], ['1']]:
            with self.assertRaises(ValueError): c.token_bytes(invalid)

    def test_stages(self):
        self.assertEqual(c.transition('first_pass', 'stop', '', self.output())['action'], 'accepted')
        a = c.transition('first_pass', 'length', 'notes', '')
        self.assertEqual(a, {'action': 'next', 'stage': 'overflow', 'notes': 'notes'})
        a = c.transition('overflow', 'stop', 'notes', '{}')
        self.assertEqual(a['stage'], 'capped_finalizer')
        self.assertEqual(a['notes'], 'notes{}')
        for stage, finish, notes, raw in [('first_pass', 'stop', '', ''), ('overflow', 'error', 'n', ''), ('capped_finalizer', 'length', 'n', '')]:
            self.assertEqual(c.transition(stage, finish, notes, raw)['action'], 'failed')

    def test_missing_company_name_is_preserved(self):
        context = self.context(); context['company_name'] = ''
        self.assertIn('"company_name":""', c.request(context, 'first_pass')['messages'][1]['content'])

    def test_validator_exactness(self):
        raw = json.dumps({'score': 70.0, 'brief': '\ufeff'.join(['word'] * 250)})
        self.assertEqual(c.brief(raw)['score'], 70)
        for value in [{'score': True, 'brief': 'word '*300}, {'score': 101, 'brief': 'word '*300}, {'score': 70, 'brief': 'word '*249}, {'score': 70, 'brief': 'word '*300, 'extra': 1}]:
            with self.assertRaises(ValueError): c.brief(json.dumps(value))

    def test_finalizer_unicode_boundaries(self):
        notes = '🐋' * 16000 + 'discard' * 10000 + 'Ω' * 32000
        body = c.request(self.context(), 'capped_finalizer', notes)
        text = body['messages'][1]['content']
        self.assertIn('🐋' * 16000 + '\n\n[DETERMINISTIC MIDDLE OMISSION]\n\n' + 'Ω' * 32000, text)
        self.assertNotIn('discard', text)
        self.assertEqual(body['reasoning_effort'], 'none')
        self.assertEqual(body['max_tokens'], 4096)

    def test_rendering_is_explicit(self):
        body = c.request(self.context(), 'first_pass')
        self.assertIn('Reasoning Effort: 75 ', c.render_prompt(body))
        self.assertTrue(c.render_prompt(body).endswith('<think>'))
        final = c.request(self.context(), 'capped_finalizer', 'notes')
        self.assertNotIn('Reasoning Effort:', c.render_prompt(final))
        self.assertTrue(c.render_prompt(final).endswith('</think>'))
        body['messages'] = [{'role':'system','content':'sys'},{'role':'user','content':'usr'}]
        self.assertEqual(c.render_prompt(body), '<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\nsys<｜User｜>usr<｜Assistant｜><think>')

    def test_equal_fixture_is_never_publication(self):
        result = c.compare(*self.pair())
        self.assertEqual(result['status'], 'byte_equal')
        self.assertTrue(result['fixture'])
        self.assertFalse(result['publication_allowed'])

    def test_same_score_different_bytes_rejected(self):
        a, b = self.pair(); b['content'] += '\n'
        with self.assertRaises(ValueError) as error: c.compare(a, b)
        self.assertIn('content', str(error.exception))

    def test_drift_and_missing_fields_rejected(self):
        for field, changed in [('output_token_ids', [4, 3]), ('input_token_ids', [2, 1]), ('profile_sha256', '9'*64), ('reasoning', 'other'), ('finish_reason', 'length')]:
            a, b = self.pair(); b[field] = changed
            with self.assertRaises(ValueError): c.compare(a, b)
        a, b = self.pair(); del b['output_token_ids']
        with self.assertRaises(ValueError): c.compare(a, b)

    def test_same_machine_rejected(self):
        a, b = self.pair(); b['host']['machine_id'] = a['host']['machine_id']
        with self.assertRaises(ValueError): c.compare(a, b)

    def test_parent_and_runtime_identity_validation(self):
        a, b = self.pair(); a['stage'] = b['stage'] = 'overflow'
        with self.assertRaises(ValueError): c.compare(a, b)
        a, b = self.pair(); a['host']['runtime_manifest_sha256'] = b['host']['runtime_manifest_sha256'] = 'unhashed'
        with self.assertRaises(ValueError): c.compare(a, b)
        a, b = self.pair(); b['host']['gpu_uuid'] = a['host']['gpu_uuid']
        with self.assertRaises(ValueError): c.compare(a, b)

    def test_no_regex_on_contract_path(self):
        for path in c.ROOT.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import): self.assertFalse(any(x.name in ('re', 'regex') for x in node.names))
                if isinstance(node, ast.ImportFrom): self.assertNotIn(node.module, ('re', 'regex'))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ('assertRaisesRegex', 'assertRegex'))


if __name__ == '__main__':
    unittest.main()
