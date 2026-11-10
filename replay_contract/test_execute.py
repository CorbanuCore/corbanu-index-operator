import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from . import core as c
from . import execute as e
from . import test_verify_job as fixtures


class ResumeTests(unittest.TestCase):
    def fixture(self):
        return fixtures.JobVerificationTests().fixture()

    def test_verified_completed_stage_is_not_generated_twice(self):
        job,run,vocab=self.fixture()
        observation=run['securities'][0]['stages'][0]['observation']
        with tempfile.TemporaryDirectory() as root,patch.dict(c.CANDIDATE,{'job_sha256':c.digest(job)}):
            ident=e.identity(job,run['host'],run['runtime_manifest_sha256'],'fixture-resume')
            deadline=int(time.time())+60
            calls=[]
            def engine(*args):calls.append(1);return observation
            first=e.execute(job,ident,root,vocab,engine,deadline)
            second=e.execute(job,ident,root,vocab,engine,deadline)
            self.assertEqual(first,second);self.assertEqual(len(calls),1)

    def test_interruption_restarts_once_and_cannot_loop(self):
        job,run,vocab=self.fixture()
        with tempfile.TemporaryDirectory() as root,patch.dict(c.CANDIDATE,{'job_sha256':c.digest(job)}):
            ident=e.identity(job,run['host'],run['runtime_manifest_sha256'],'fixture-interruption')
            deadline=int(time.time())+60;calls=[]
            def engine(*args):calls.append(1);raise TimeoutError('interrupted fixture')
            for _ in range(2):
                with self.assertRaises(TimeoutError):e.execute(job,ident,root,vocab,engine,deadline)
            with self.assertRaises(ValueError):e.execute(job,ident,root,vocab,engine,deadline)
            self.assertEqual(len(calls),2)

    def test_completed_invalid_output_never_retries(self):
        job,run,vocab=self.fixture();observation=run['securities'][0]['stages'][0]['observation']
        observation['request_sha256']='0'*64
        with tempfile.TemporaryDirectory() as root,patch.dict(c.CANDIDATE,{'job_sha256':c.digest(job)}):
            ident=e.identity(job,run['host'],run['runtime_manifest_sha256'],'fixture-invalid')
            deadline=int(time.time())+60;calls=[]
            def engine(*args):calls.append(1);return observation
            for _ in range(2):
                with self.assertRaises(ValueError):e.execute(job,ident,root,vocab,engine,deadline)
            self.assertEqual(len(calls),1)

    def test_seal_does_not_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'receipt.json';e.seal(path,{'first':True})
            with self.assertRaises(FileExistsError):e.seal(path,{'second':True})
            self.assertEqual(c.load(path),{'first':True})

    def test_adapter_validation_failure_never_retries(self):
        job,run,vocab=self.fixture()
        with tempfile.TemporaryDirectory() as root,patch.dict(c.CANDIDATE,{'job_sha256':c.digest(job)}):
            ident=e.identity(job,run['host'],run['runtime_manifest_sha256'],'fixture-adapter-invalid')
            deadline=int(time.time())+60;calls=[]
            def engine(*args):calls.append(1);raise ValueError('invalid completed provider output')
            for _ in range(2):
                with self.assertRaises(ValueError):e.execute(job,ident,root,vocab,engine,deadline)
            self.assertEqual(len(calls),1)

    def test_concurrent_run_is_rejected_before_engine(self):
        job,run,vocab=self.fixture()
        with tempfile.TemporaryDirectory() as root:
            with (Path(root)/'execution.lock').open('a') as lock:
                e.fcntl.flock(lock,e.fcntl.LOCK_EX|e.fcntl.LOCK_NB)
                with self.assertRaises(ValueError):e.execute(job,run,root,vocab,None,time.time()+60)


if __name__=='__main__':unittest.main()
