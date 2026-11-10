import json,subprocess,time,shlex,os
R='/mnt/HC_Volume_101713660/pfrpc/scratch/deepinfer-question-log-20261109b'
c=json.load(open(R+'/replay/connection-direct.json'))
os.replace(R+'/replay/connection.json',R+'/replay/connection-proxy.json') if os.path.exists(R+'/replay/connection.json') and json.load(open(R+'/replay/connection.json'))['remote']!=c['remote'] else None
json.dump(c,open(R+'/replay/connection.json','w'))
while not os.path.exists(R+'/stop'):
    subprocess.run(['rsync','-a','-e',shlex.join(c['ssh']),c['remote']+':/workspace/trial/evidence/',R+'/replay/retained/'],capture_output=True,timeout=60)
    time.sleep(8)
