"""Verify every pinned weight-file byte before allowing model loading.

This establishes snapshot integrity, not native-to-GGUF numerical fidelity.
The conversion audit and replay gates remain separate admission requirements.
"""
import argparse
import hashlib
from pathlib import Path
from . import core as c


def inventory(snapshot, kind):
    revision=c.CANDIDATE['gguf_revision' if kind=='gguf' else 'native_revision']
    if snapshot.get('sha')!=revision:c.fail('Unpinned model revision')
    suffix='.gguf' if kind=='gguf' else '.safetensors'
    files=[]
    for row in snapshot['siblings']:
        name=row['rfilename']
        if not name.endswith(suffix):continue
        if Path(name).name!=name:c.fail('Unexpected nested weight path')
        size=row.get('size');lfs=row.get('lfs',{})
        digest=lfs.get('sha256','')
        if type(size) is not int or size<=0 or lfs.get('size')!=size:
            c.fail('Missing or inconsistent weight size')
        if len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):
            c.fail('Missing weight content hash')
        files.append({'name':name,'bytes':size,'sha256':digest})
    expected=11 if kind=='gguf' else 48
    if len(files)!=expected or len({x['name'] for x in files})!=expected:
        c.fail('Wrong shard coverage')
    return {'schema':'corbanu.model-file-inventory.v1','revision':revision,'kind':kind,
            'total_bytes':sum(x['bytes'] for x in files),'files':sorted(files,key=lambda x:x['name'])}


def verify_files(manifest, directory):
    pinned=c.CANDIDATE['weight_inventory_sha256'].get(manifest['kind'])
    if c.digest(manifest)!=pinned:c.fail('Weight inventory differs from the admitted candidate input')
    suffix='.gguf' if manifest['kind']=='gguf' else '.safetensors'
    expected={x['name'] for x in manifest['files']}
    actual={p.name for p in directory.iterdir() if p.name.endswith(suffix)}
    if actual!=expected:c.fail('Missing or extra weight shards')
    for row in manifest['files']:
        path=directory/row['name']
        if path.stat().st_size!=row['bytes']:c.fail('Weight size mismatch: '+row['name'])
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(8*1024*1024):digest.update(block)
        if digest.hexdigest()!=row['sha256']:c.fail('Weight checksum mismatch: '+row['name'])
    return {'status':'all_weight_file_bytes_verified','inventory':manifest,
            'inventory_sha256':c.digest(manifest),'conversion_fidelity_audited':False,
            'publication_allowed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True);parser.add_argument('--kind',choices=['gguf','native'],required=True)
    parser.add_argument('--directory',type=Path);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    manifest=inventory(c.load(args.snapshot),args.kind)
    result=verify_files(manifest,args.directory) if args.directory else {
        'status':'metadata_only','inventory':manifest,'inventory_sha256':c.digest(manifest),
        'weight_bytes_verified':False,'publication_allowed':False}
    c.save(args.output,result)
