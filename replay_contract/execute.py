"""Immutable stage orchestration and verified resume for a prepared job.

The engine callback must own one fresh runtime process per call. Hardware setup
and its bounded phase watchdog are required before supplying that callback.
"""
import json
import fcntl
import os
from pathlib import Path
import time
import urllib.error
import uuid
from . import core as c
from .verify_job import commitment, verify_stage


def seal(path,value):
    """Publish a completed receipt atomically, without replacing prior evidence."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write((json.dumps(value,ensure_ascii=True,indent=2)+'\n').encode())
            stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def identity(job,host,runtime_manifest_sha256,run_id):
    if c.digest(job)!=c.CANDIDATE['job_sha256']:c.fail('Wrong job')
    if any(not host.get(k) for k in ('machine_id','gpu_uuid')):c.fail('Missing physical identities')
    if len(runtime_manifest_sha256)!=64 or any(ch not in '0123456789abcdef' for ch in runtime_manifest_sha256):
        c.fail('Invalid runtime manifest hash')
    return {'schema':'corbanu.streaming-run.v1','run_id':run_id,'job_sha256':c.digest(job),
            'contract_sha256':c.contract_digest(),'candidate_sha256':c.digest(c.CANDIDATE),
            'host':host,'runtime_manifest_sha256':runtime_manifest_sha256}


def execute(job,run_identity,directory,vocab,engine,deadline,security_ids=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'execution.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:c.fail('Another worker owns this run')
        return _execute(job,run_identity,directory,vocab,engine,deadline,security_ids)


def _execute(job,run_identity,directory,vocab,engine,deadline,security_ids=None):
    """Resume only sealed, revalidated stages; never combine interrupted streams.

engine(context,stage,notes,attempt_directory,remaining_seconds) -> observation.
Exceptions stop the run. A caller may resume one interrupted attempt; a third
attempt at a stage is prohibited. Completed mismatching observations never retry.
"""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    expected=identity(job,run_identity['host'],run_identity['runtime_manifest_sha256'],run_identity['run_id'])
    if expected!=run_identity:c.fail('Run identity does not match current contract')
    if (directory/'identity.json').exists():
        if c.load(directory/'identity.json')!=run_identity:c.fail('Resume identity drift')
    else:seal(directory/'identity.json',run_identity)
    by_id={row['security_id']:row for row in job['securities']}
    chosen=list(by_id) if security_ids is None else list(security_ids)
    if not chosen or len(chosen)!=len(set(chosen)) or not set(chosen).issubset(by_id):
        c.fail('Invalid diagnostic subset')
    scope={'security_ids':chosen,'deadline_epoch_seconds':int(deadline)}
    if (directory/'scope.json').exists():
        if c.load(directory/'scope.json')!=scope:c.fail('Resume scope or deadline drift')
    else:seal(directory/'scope.json',scope)
    rows=[]
    for security_id in chosen:
        context=by_id[security_id]['context'];parent=None;notes='';stage='first_pass';records=[]
        root=directory/'securities'/c.sha(security_id.encode())
        while stage is not None:
            stage_root=root/stage;stage_root.mkdir(parents=True,exist_ok=True)
            path=stage_root/'receipt.json'
            if path.exists():
                record=c.load(path)
            else:
                if (stage_root/'invalid.json').exists():c.fail('Completed invalid output requires investigation, not retry')
                attempts=list(stage_root.glob('attempt-*'))
                if len(attempts)>=2:c.fail('Stage recovery limit exhausted')
                remaining=deadline-time.time()
                if remaining<=0:c.fail('Phase deadline reached')
                attempt=stage_root/('attempt-'+str(len(attempts)+1));attempt.mkdir()
                seal(attempt/'intent.json',{'security_id':security_id,'stage':stage,'started_at_ns':time.time_ns()})
                try:
                    observation=engine(context,stage,notes,attempt,remaining)
                except Exception as error:
                    seal(attempt/'error.json',{'type':type(error).__name__,'message':str(error),'at_ns':time.time_ns()})
                    retryable=isinstance(error,(TimeoutError,ConnectionError,urllib.error.URLError)) and not isinstance(error,urllib.error.HTTPError)
                    if not retryable:
                        seal(stage_root/'invalid.json',{'attempt':attempt.name,'type':type(error).__name__,'message':str(error)})
                    raise
                seal(attempt/'observation.json',observation)
                try:
                    if time.time()>deadline:c.fail('Observation arrived after phase expiry')
                    decision,_=verify_stage(context,stage,notes,observation,vocab)
                    if decision['action']=='failed':c.fail('Scoring output failed the stage policy')
                except Exception as error:
                    seal(stage_root/'invalid.json',{'attempt':attempt.name,'type':type(error).__name__,'message':str(error)})
                    raise
                record={'stage':stage,'parent_output_sha256':parent,'observation':observation}
                seal(path,record)
            if record['stage']!=stage or record['parent_output_sha256']!=parent:c.fail('Resume parent chain changed')
            observation=record['observation']
            decision,_=verify_stage(context,stage,notes,observation,vocab)
            records.append(record);parent=commitment(stage,observation)
            if decision['action']=='accepted':stage=None
            elif decision['action']=='next':stage=decision['stage'];notes=decision['notes']
            else:c.fail('Invalid sealed receipt')
        rows.append({'security_id':security_id,'stages':records})
    result={**run_identity,'securities':rows}
    if (directory/'run.json').exists():
        if c.load(directory/'run.json')!=result:c.fail('Completed run changed')
    else:seal(directory/'run.json',result)
    return result
