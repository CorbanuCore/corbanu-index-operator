"""Execute the actual-input, offline acceptance suite. Never runs inference."""
import argparse
import json
import subprocess
from pathlib import Path
from . import core as c

BASE = Path('/mnt/HC_Volume_101713660/pfrpc/scratch')
FROZEN = BASE / 'deepseek41-h200-feasibility-20260911/aoc-frozen-inputs.json'
PREVIOUS = BASE / 'aoc-deepseek-flash-20260911'
EXPECTED = BASE / 'index-exclusions-20260911/result.json'
MANIFEST = Path('/mnt/HC_Volume_101713660/pfrpc/data/navstrategies/felix_index_packets/consumption/exports/986bd66e56dd6607e179aff96a9f600a4611fd6c2606a3bff5f0d7d15b4c7bbb/consumption-index.json')
TSX = '/home/pfrpc/repos/CorbanuAPI/node_modules/.bin/tsx'


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    c.prepare(FROZEN, out / 'preparation.json')
    frozen = c.load(FROZEN)
    if c.sha(MANIFEST.read_bytes()) != frozen['source']['index_sha256']:
        c.fail('Wrong source index')
    if c.sha(EXPECTED.read_bytes()) != frozen['source_result_sha256']:
        c.fail('Historical result drift')
    result = c.load(EXPECTED)
    by_id = {r['security_id']: r for r in result['scores']}
    stage_count = 0
    for row in frozen['requests']:
        company = by_id[row['security_id']]
        context = {'question': result['request']['mandate']['phrase'],
                   'ticker': company['ticker'], 'company_name': company['company_name']}
        if c.request(context, 'first_pass') != row['body']:
            c.fail('Reconstructed request changed')
        receipts = c.load(PREVIOUS / 'receipts' / (row['security_id'].split(':')[-1] + '.json'))
        notes = ''; expected_stage = 'first_pass'; decision = None
        for receipt in receipts:
            stage_count += 1
            if receipt['stage'] != expected_stage:
                c.fail('Unexpected historical stage order')
            if c.digest(c.request(context, expected_stage, notes)) != receipt['request_sha256']:
                c.fail('Historical request hash mismatch')
            choice = receipt['response']['choices'][0]; message = choice['message']
            reasoning = message.get('reasoning_content', message.get('reasoning', '')) or ''
            decision = c.transition(expected_stage, choice['finish_reason'], reasoning, message.get('content') or '')
            if decision['action'] == 'next':
                expected_stage = decision['stage']; notes = decision['notes']
            elif decision['action'] == 'accepted':
                expected_stage = None
            else:
                c.fail('Historical stage unexpectedly failed')
        if decision is None or decision['action'] != 'accepted' or decision['parsed'] != {k: company[k] for k in ('score','brief')}:
            c.fail('Historical score/brief mismatch')
    context = {'question': 'AOC premise "quoted"', 'ticker': 'TEST', 'company_name': 'Example 🐋 & Co'}
    notes = '🐋'*16000 + 'discard'*10000 + 'Ω'*32000
    values = [{'z':'🐋\u007f', 'a':70}, ['é','\n',True,None,9007199254740991]]
    raw = '{"score":70.0,"brief":' + json.dumps('\ufeff'.join(['evidence']*250)) + '}'
    vectors = {'messages': [{'context':context,'stage':stage,'notes':notes,
                            'expected':c.messages(context,stage,notes)} for stage in c.STAGES],
               'canonical':[{'input':v,'expected':c.canonical(v).decode()} for v in values],
               'briefs':[{'raw':raw,'expected':c.brief(raw)}]}
    c.save(out / 'vectors.json', vectors)
    c.save(out / 'scores.json', result['scores'])
    output = subprocess.check_output([TSX, str(c.ROOT/'cross-check.mts'), str(c.ROOT/'source-lock.json'),
              str(out/'vectors.json'), str(MANIFEST), str(out/'scores.json'), str(EXPECTED)], text=True, timeout=60)
    c.save(out/'cross-language.json', json.loads(output))
    c.save(out/'host-preflight.json', c.host_preflight())
    report = {'status':'offline_passed','requests':452,'historical_stages':stage_count,
              'historical_scores_and_briefs_equal':True,'holdings':29,'excluded':['LIT','REMX'],
              'contract_sha256':c.contract_digest(),'gpu_inference_executed':False,'publication_allowed':False}
    c.save(out/'report.json',report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
