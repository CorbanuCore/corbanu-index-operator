"""Relaunch a dead question worker from local disk. Refuses if any worker process is alive."""
import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
ev = root / 'evidence'
pid_file = ev / 'worker.pid'
if pid_file.exists():
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
    if alive:
        print(json.dumps({'status': 'refused_worker_alive', 'pid': pid}))
        raise SystemExit(0)
alive_engines = subprocess.run(['pgrep', '-f', 'run_questions.py'], capture_output=True, text=True).stdout.split()
if alive_engines:
    print(json.dumps({'status': 'refused_engine_processes_alive', 'pids': alive_engines}))
    raise SystemExit(0)
archive = ev / 'prior-exits' / str(int(time.time()))
archive.mkdir(parents=True, exist_ok=True)
for name in ['worker-exit.json', 'worker.pid'] + [f'ready-rank{i}.json' for i in range(4)]:
    f = ev / name
    if f.exists():
        f.rename(archive / name)
with (ev / 'restart.log').open('ab') as log:
    process = subprocess.Popen(['bash', str(root / 'restart-worker.sh')], cwd=root, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=log, start_new_session=True, close_fds=True)
pid_file.write_text(str(process.pid) + '\n')
print(json.dumps({'status': 'restarted', 'pid': process.pid, 'archived_to': str(archive)}))
