import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent / "evidence"
root.mkdir(exist_ok=True)
(root / "hardware-host.json").write_text(json.dumps({
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "cpu_model": sorted(set(x for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))),
}))
with (root / "hardware-samples.jsonl").open("a", buffering=1) as stream:
    while True:
        host = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemFree', 'MemAvailable', 'Cached'):
                host[key] = value.strip()
        for key in ('memory.current', 'memory.peak', 'memory.max'):
            p = Path('/sys/fs/cgroup') / key
            if p.exists(): host[key] = p.read_text().strip()
        try:
            gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=15)
            stream.write(json.dumps({'at': time.time(), 'host': host, 'gpu_csv': gpus}) + '\n')
        except Exception as e:
            stream.write(json.dumps({'at': time.time(), 'error': str(e)}) + '\n')
        time.sleep(10)
