"""Bounded Vast lifecycle for a single, explicitly owned replay qualification phase."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

KEY = Path('/home/pfrpc/.config/vastai/vast_api_key')
BASE = 'https://console.vast.ai/api/v0/'


def api(method, path, payload=None):
    request = urllib.request.Request(BASE+path, method=method,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+KEY.read_text().strip(), 'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        raw=error.read(65536)
        try:
            body=json.loads(raw)
            detail={k:body[k] for k in ('error','msg','message','detail') if k in body}
        except ValueError:
            detail={'message':'Non-JSON error response'}
        raise ValueError('Vast HTTP '+str(error.code)+': '+json.dumps(detail)) from None
    if isinstance(value, dict) and value.get('success') is False:
        raise ValueError('Vast rejected request: '+str(value.get('error'))+' '+str(value.get('msg')))
    return value


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def owned(phase):
    value = api('GET','instances/')
    if not isinstance(value,dict) or not isinstance(value.get('instances'),list):
        raise ValueError('Invalid instance inventory')
    return [r for r in value['instances'] if r.get('label') == phase['label']]


def watch(root):
    while True:
        phase = json.loads((root/'phase.json').read_text())
        try:
            rows = owned(phase)
            fields = ('id','machine_id','label','actual_status','intended_status','dph_total',
                      'gpu_name','num_gpus','cpu_ram','disk_space','ssh_host','ssh_port','public_ipaddr','ports')
            write(root/'inventory.json', {'observed_at':time.time(), 'instances':[
                {k:r.get(k) for k in fields} for r in rows]})
            write(root/'watchdog.json', {'pid':os.getpid(),'heartbeat':time.time(),'deadline':phase['deadline']})
            expired = time.time() >= phase['deadline'] or (root/'stop').exists()
            if expired:
                for row in rows:
                    result = api('DELETE','instances/'+str(row['id'])+'/')
                    write(root/('destroy-'+str(row['id'])+'.json'),result)
                if not rows:
                    write(root/'cleanup.json', {'status':'confirmed_absent','observed_at':time.time(),
                                              'label':phase['label']})
                    return
        except Exception as error:
            # Unknown inventory state never authorizes another rental or counts as cleanup.
            write(root/'watchdog-error.json', {'at':time.time(),'type':type(error).__name__,'error':str(error)})
        time.sleep(20)


def launch(root):
    phase = json.loads((root/'phase.json').read_text())
    if (root/'launch-intent.json').exists():
        raise ValueError('Launch already attempted; inspect inventory rather than reissuing')
    if phase['deadline'] <= time.time() or phase['hours'] <= 0:
        raise ValueError('Invalid phase expiry')
    expected = phase['quote']['dph_total']*phase['hours'] + phase['network_reserve_usd']
    if expected > phase['spend_ceiling_usd']:
        raise ValueError('Quote exceeds phase ceiling')
    if owned(phase):
        raise ValueError('Phase label already owns an instance')
    with (root/'watchdog.log').open('ab') as log:
        process = subprocess.Popen([sys.executable,__file__,'watch',str(root)],stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=log,start_new_session=True)
    for _ in range(20):
        if (root/'watchdog.json').exists(): break
        if process.poll() is not None: raise ValueError('Watchdog failed to start')
        time.sleep(0.5)
    else: raise ValueError('No watchdog heartbeat')
    write(root/'launch-intent.json',{'at':time.time(),'offer_id':phase['quote']['id'],'watchdog_pid':process.pid})
    try:
        result = api('PUT','asks/'+str(phase['quote']['id'])+'/',phase['create'])
        write(root/'created.json',result)
        print(json.dumps({k:result.get(k) for k in ("success","new_contract")}))
    except Exception as error:
        write(root/'create-error.json',{'type':type(error).__name__,'error':str(error)})
        # Retain watchdog until deadline: a timed-out creation can still complete remotely.
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['launch','watch','stop','status'])
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    if args.command=='watch': watch(args.root)
    elif args.command=='launch': launch(args.root)
    elif args.command=='stop': (args.root/'stop').touch()
    else:
        print(json.dumps({'instances':[{k:r.get(k) for k in ('id','label','actual_status','ssh_host','ssh_port')}
                            for r in owned(json.loads((args.root/'phase.json').read_text()))]}))
