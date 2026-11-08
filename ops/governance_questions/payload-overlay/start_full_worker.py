import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
(root / 'evidence').mkdir(parents=True, exist_ok=True)
pid_file = root / 'evidence/worker.pid'
if pid_file.exists():
    pid = int(pid_file.read_text())
    print(json.dumps({'pid': pid, 'already_started': True, 'alive': Path('/proc', str(pid)).exists()}))
else:
    with (root / 'evidence/bootstrap.log').open('ab') as log:
        process = subprocess.Popen(['bash', str(root / 'question-bootstrap.sh')], cwd=root,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True, close_fds=True)
    pid_file.write_text(str(process.pid) + '\n')
    print(json.dumps({'pid': process.pid, 'already_started': False}))
