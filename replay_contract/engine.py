"""Fresh-process execution for a qualified local llama.cpp installation.

Call only after the controller has qualified the CUDA build, conversion audit,
model snapshot and bounded rental phase. This module does not admit hardware.
"""
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from . import core as c
from . import runtime as r
from .audit_model import verify_files


def terminate(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=5)


@contextmanager
def server(command,endpoint,log_path,remaining,startup_limit=900):
    if remaining<=0:c.fail('Expired runtime phase')
    start=time.monotonic();expired=threading.Event()
    with Path(log_path).open('xb') as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        def cutoff():
            expired.set();terminate(process)
        timer=threading.Timer(remaining,cutoff);timer.daemon=True;timer.start()
        try:
            while True:
                if expired.is_set():raise TimeoutError('Runtime phase expired')
                if process.poll() is not None:c.fail('Runtime exited during startup; inspect '+str(log_path))
                elapsed=time.monotonic()-start
                if elapsed>=min(startup_limit,remaining):raise TimeoutError('Runtime startup deadline reached')
                try:
                    with urllib.request.urlopen(endpoint+'/health',timeout=min(2,remaining-elapsed)) as response:
                        healthy=json.load(response).get('status')=='ok'
                    if healthy and process.poll() is None:break
                except (urllib.error.URLError,TimeoutError,ConnectionError):pass
                time.sleep(0.2)
            yield process,remaining-(time.monotonic()-start)
            if expired.is_set():raise TimeoutError('Runtime phase expired')
        finally:
            timer.cancel();terminate(process);timer.join(timeout=11)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def arguments(binary,model,port):
    profile=c.CANDIDATE
    return [str(binary),'-m',str(model),'-ngl','99','-c',str(profile['context_tokens']),
            '-b','512','-ub','512','--parallel','1','--cache-ram','0','--no-context-shift',
            '--spec-type','none','--no-warmup','--special','--moe-stream',
            '--moe-stream-cache',str(profile['expert_gpu_cache_gib']),
            '--moe-stream-l2',str(profile['expert_host_cache_gib']),
            '--moe-stream-io-threads',str(profile['io_threads']),
            '--host','127.0.0.1','--port',str(port)]


class ProcessEngine:
    def __init__(self,binary,model_dir,model_inventory,build_manifest,evidence_dir):
        self.binary=Path(binary);self.model_dir=Path(model_dir)
        self.build=c.load(build_manifest)
        if self.build.get('status')!='built_and_kernel_tested':c.fail('GPU build/kernel evidence missing')
        if self.build.get('gpu_inference_qualified') is not False:c.fail('Unexpected build evidence contract')
        self.binary_files={self.binary.parent/name:digest for name,digest in self.build['files'].items()}
        if self.binary.name not in self.build['files']:c.fail('Server binary is not in build manifest')
        for path,digest in self.binary_files.items():
            if path.parent!=self.binary.parent or c.sha(path.read_bytes())!=digest:c.fail('Runtime binary/library changed')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True,timeout=15)
        rows=list(csv.reader(raw.splitlines()))
        if len(rows)!=1 or 'RTX 5090' not in rows[0][0]:c.fail('Requires exactly one RTX 5090')
        self.gpu={'name':rows[0][0].strip(),'uuid':rows[0][1].strip(),'driver':rows[0][2].strip()}
        c.save(Path(evidence_dir)/'loaded-gpu.json',self.gpu)
        # Full byte verification happens once at run start, not once per stage.
        verified=verify_files(model_inventory,self.model_dir)
        c.save(Path(evidence_dir)/'loaded-model-integrity.json',verified)
        self.model_files=[self.model_dir/x['name'] for x in model_inventory['files']]
        self.model=sorted(self.model_files)[0]
        self.stamps={p:(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino) for p in self.model_files}

    def __call__(self,context,stage,notes,attempt,remaining):
        for path,stamp in self.stamps.items():
            current=path.stat()
            if (current.st_size,current.st_mtime_ns,current.st_ino)!=stamp:c.fail('Model changed between stages')
        for path,digest in self.binary_files.items():
            if c.sha(path.read_bytes())!=digest:c.fail('Runtime changed between stages')
        port=free_port();endpoint='http://127.0.0.1:'+str(port)
        command=arguments(self.binary,self.model,port)
        c.save(Path(attempt)/'launch.json',{'argv':command,'gpu':self.gpu,'fresh_process':True})
        with server(command,endpoint,Path(attempt)/'runtime.log',remaining) as (process,available):
            observation=r.observe(endpoint,context,stage,notes,{'bos':0,'eos':1,'think_end':128822},
                                  timeout=available,evidence_dir=attempt)
            observation['runtime_process']={'pid':process.pid,'fresh_process':True,'gpu_uuid':self.gpu['uuid']}
            return observation
