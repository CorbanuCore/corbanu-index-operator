"""Authenticated, portable AOC job inputs. Historical scores are not copied."""
import argparse
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from . import core as c

FROZEN_HASH='f577e7e4a873e0d4de96088e5e564db1bd43b463162a1cdfc113e5e744eac3b9'
PUBLISHER='a9a49f5a1384033bcd3310ac307e00dbe38db7bf44f09540cace730e4556b26e'
SOURCE_HASH='f83902666fbdfa667e7d1abb252400fd261e638ad46cf5a290cd867dd78bb5f2'


def verify_source(source, raw):
    release=source['release']
    if release['public_key']!=PUBLISHER:c.fail('Untrusted source publisher')
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(PUBLISHER)).verify(
        bytes.fromhex(release['signature']),c.canonical(release['payload']))
    if c.sha(raw)!=SOURCE_HASH or source['index_sha256']!=SOURCE_HASH:
        c.fail('Frozen source packet changed')
    if release['payload']['index_sha256']!=SOURCE_HASH or release['payload']['cid']!=source['cid']:
        c.fail('Signed release/source mismatch')


def prepare(inputs, source_path, output):
    if c.sha(inputs.read_bytes())!=FROZEN_HASH:c.fail('Wrong frozen input bundle')
    frozen=c.load(inputs)
    original=c.load(frozen['source_result_path'])
    if c.sha(Path(frozen['source_result_path']).read_bytes())!=frozen['source_result_sha256']:
        c.fail('Historical identity source changed')
    raw=source_path.read_bytes();verify_source(frozen['source'],raw)
    companies={row['security_id']:row for row in original['scores']}
    contexts=[]
    for row in frozen['requests']:
        company=companies[row['security_id']]
        context={'question':frozen['index_request']['mandate']['phrase'],
                 'ticker':company['ticker'],'company_name':company['company_name']}
        body=c.request(context,'first_pass')
        if body!=row['body'] or c.digest(body)!=row['request_sha256']:c.fail('Request reconstruction mismatch')
        contexts.append({'security_id':row['security_id'],'context':context,
                         'first_pass_request_sha256':row['request_sha256']})
    if len(contexts)!=452 or len({r['security_id'] for r in contexts})!=452:c.fail('Wrong universe')
    job={'schema':'corbanu.streaming-job.v1','frozen_inputs_sha256':FROZEN_HASH,'source':frozen['source'],
         'scoring_profile':c.P,'securities':contexts,'cutoff':70,'weighting':'market_cap_rank',
         'missing_capitalization':'exclude','required_security_count':452}
    output.mkdir(parents=True,exist_ok=False)
    c.save(output/'job.json',job)
    with (output/'source.json').open('xb') as target:target.write(raw)
    c.save(output/'identity.json',{'job_sha256':c.digest(job),'source_sha256':SOURCE_HASH,
                                 'historical_scores_included':False,'publication_allowed':False})
    return job


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();prepare(args.inputs,args.source,args.output_dir)
