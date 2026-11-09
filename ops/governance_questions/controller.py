"""Post Fiat governance question log: two owned Vast 4xB200 hosts running the qualified DeepSeek runtime.

Lifecycle rules (the Nov 8 failure was a worker exit that destroyed both rentals):
  * Rentals are destroyed ONLY at the lease deadline, on ROOT/stop, or when the account spend guard trips.
  * A worker exit triggers a reload from local disk (restart_worker.py), up to MAX_RESTARTS per host.
    If a host exhausts its restarts the pair is held for the operator; nothing is destroyed early.
  * The lease can be extended with extend.py; the provider watchdogs re-read phase.json every 20 s.
"""
from pathlib import Path
import importlib.util
import json
import os
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parent
QUAL = ROOT.parent / 'deepinfer-replay-boot-20261108'
OLD = ROOT.parent / 'deepinfer-full-index-20261108/full-run'
sys.path.insert(0, '/home/pfrpc/repos/corbanu-index-operator/ops')
from vast_replay_phase import api, launch, write  # noqa: E402

spec = importlib.util.spec_from_file_location('qualified_controller', OLD / 'controller.py')
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)
qc.ROOT = ROOT
spec = importlib.util.spec_from_file_location('warm_controller', QUAL / 'controller.py')
wc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wc)
readiness = wc.readiness  # boot smoke must match the prior 452-company two-host index

QUALIFIED_MACHINES = (156164, 156229)
MAX_HOST_DPH = 43.0
MAX_RESTARTS = 3


def read(p):
    return json.loads(p.read_text())


def credit_now():
    a = api('GET', 'users/current/')
    return float(a.get('credit', 0)) + float(a.get('balance', 0))


def push_lease(host, deadline):
    write(host['root'] / 'lease.json', {'deadline': deadline, 'written_at': time.time()})
    if 'ssh' in host:
        qc.run(['rsync', '-a', '-e', shlex.join(host['ssh']), str(host['root'] / 'lease.json'),
                host['remote'] + ':/workspace/trial/lease.json'], timeout=30)


def connect(host):
    """Try the Vast SSH proxy first, then the instance's direct SSH port (the proxy dropped on 2026-11-09)."""
    if qc.connect(host):
        return True
    inv = host['root'] / 'inventory.json'
    if not inv.exists():
        return False
    data = read(inv)
    rows = data.get('instances') or []
    if len(rows) != 1:
        return False
    row = dict(rows[0])
    try:
        port = row['ports']['22/tcp'][0]['HostPort']
    except (KeyError, IndexError, TypeError):
        return False
    if not row.get('public_ipaddr') or (row.get('ssh_host'), str(row.get('ssh_port'))) == (row['public_ipaddr'], str(port)):
        return False
    row['ssh_host'], row['ssh_port'] = row['public_ipaddr'], int(port)
    write(inv, {**data, 'instances': [row]})
    return qc.connect(host)


def main(ceiling):
    uncapped = ceiling is None
    assert os.path.ismount('/mnt/HC_Volume_101713660')
    assert not (ROOT / 'execution-plan.json').exists(), 'Already launched; inspect instead of retrying'
    assert read(OLD / 'report.json')['status'] == 'full_452_two_host_index_passed'
    query = {'gpu_name': {'eq': 'B200'}, 'num_gpus': {'eq': 4}, 'rentable': {'eq': True},
             'rented': {'eq': False}, 'disk_space': {'gte': 1200}, 'allocated_storage': 1200,
             'type': 'ondemand', 'order': [['dph_total', 'asc']], 'limit': 40}
    response = api('PUT', 'search/asks/', {'q': query})
    write(ROOT / 'launch-offers.json', response)
    candidates = {}
    for row in response.get('offers', []):
        if row['machine_id'] not in QUALIFIED_MACHINES:
            continue
        if row.get('cpu_ram', 0) < 900000 or row.get('inet_down', 0) < 8000:
            continue
        if row.get('cuda_max_good', 0) < 13.2 or row.get('inet_down_cost', 1e9) > .004:
            continue
        quote = dict(row)
        quote['dph_total'] = float(row.get('dph_base', row['dph_total'])) + float(row['storage_cost']) * 1200 / 720
        if quote['dph_total'] <= MAX_HOST_DPH:
            candidates.setdefault(row['machine_id'], quote)
    assert len(candidates) == 2, 'Both qualified hosts must be available'
    quotes = sorted(candidates.values(), key=lambda r: r['dph_total'])
    credit = credit_now()
    began = time.time()
    hourly = sum(q['dph_total'] for q in quotes)
    if uncapped:
        ceiling = 1e9
        deadline = began + 30 * 86400
        assert credit >= 30, 'Credit does not cover boot'
    else:
        assert credit >= ceiling + 2, 'Insufficient credit for this ceiling'
        deadline = began + (ceiling - 10) / hourly * 3600 - 60
        assert deadline - began >= 2400, 'Less than 40 minutes would remain in the lease'
    hosts = []
    for role, quote in zip(('primary', 'replay'), quotes):
        root = ROOT / role
        root.mkdir(mode=0o700)
        hours = (deadline - began + 60) / 3600
        phase = {'schema': 'corbanu.vast-phase.v1', 'phase': 'postfiat-governance-question-log',
                 'label': 'postfiat-govq-' + role + '-' + str(int(began)),
                 'created_at': began, 'deadline': deadline, 'hours': hours,
                 'spend_ceiling_usd': quote['dph_total'] * hours + 5,
                 'network_reserve_usd': 5, 'combined_run_ceiling_usd': ceiling,
                 'quote': quote, 'create': {'client_id': 'me',
                     'image': 'nvidia/cuda@sha256:d266e59b88c295bc5fa0e4cef9064eaff84939381b09e2c3d76a5532a303e42d',
                     'disk': 1200, 'runtype': 'ssh_direct', 'cancel_unavail': True,
                     'onstart': 'mkdir -p /workspace/trial', 'env': {}}}
        phase['create']['label'] = phase['label']
        write(root / 'phase.json', phase)
        hosts.append({'name': role, 'root': root, 'phase': phase, 'restarts': 0})
    write(ROOT / 'execution-plan.json', {'started_at': began, 'deadline': deadline,
          'combined_ceiling_usd': ceiling, 'combined_hourly_compute_storage_usd': hourly,
          'account_credit_start_usd': credit, 'mode': 'boot_smoke_then_question_log',
          'phases': [h['phase']['label'] for h in hosts]})
    write(ROOT / 'lease.json', {'deadline': deadline, 'ceiling_usd': ceiling, 'written_at': time.time()})
    ready = False
    last_credit = 0
    try:
        for host in hosts:
            qc.log('Launching ' + host['name'] + ' at $%.4f/hour' % host['phase']['quote']['dph_total'])
            launch(host['root'])
        while not (ROOT / 'stop').exists():
            lease = read(ROOT / 'lease.json')
            if lease['deadline'] != deadline:  # operator extension via extend.py
                deadline, ceiling = lease['deadline'], lease['ceiling_usd']
                for host in hosts:
                    phase = read(host['root'] / 'phase.json')
                    phase['deadline'] = deadline
                    write(host['root'] / 'phase.json', phase)
                    push_lease(host, deadline)
                qc.log('Lease extended to %s, ceiling $%.2f' % (time.strftime('%H:%M:%SZ', time.gmtime(deadline)), ceiling))
            if time.time() >= deadline - 30:
                qc.log('Lease deadline reached')
                break
            if time.time() - last_credit > 60:
                try:
                    now_credit = credit_now()
                    write(ROOT / 'credit-monitor.json', {'at': time.time(), 'available_credit_usd': now_credit,
                          'observed_account_decrease_usd': credit - now_credit, 'spend_ceiling_usd': ceiling})
                    if ceiling < 1e8 and now_credit < max(5, credit - ceiling + 3):
                        raise RuntimeError('Account spend guard reached')  # skipped when the operator removes the cap
                except ValueError as error:
                    qc.log('Credit refresh unavailable: ' + str(error))
                last_credit = time.time()
            for host in hosts:
                if 'ssh' not in host:
                    if time.time() > began + 900 and not uncapped:
                        raise TimeoutError('Host SSH was not ready within 15 minutes')
                    if connect(host):
                        push_lease(host, deadline)
                if 'ssh' in host:
                    qc.sync(host)
                    exit_file = host['root'] / 'retained/worker-exit.json'
                    if exit_file.exists():
                        if host['restarts'] >= MAX_RESTARTS:
                            if not host.get('held'):
                                qc.log(host['name'] + ' worker exhausted restarts; holding pair for operator')
                                host['held'] = True
                        else:
                            host['restarts'] += 1
                            r = qc.run(host['ssh'] + [host['remote'], 'python3 /workspace/trial/restart_worker.py'], timeout=60)
                            write(host['root'] / f'restart-{host["restarts"]}.json',
                                  {'at': time.time(), 'returncode': r.returncode, 'stdout': r.stdout[-2000:], 'stderr': r.stderr[-2000:]})
                            exit_file.rename(host['root'] / f'retained-worker-exit-{host["restarts"]}.json')
                            qc.log(host['name'] + ' worker exited; reload %d/%d: %s' % (host['restarts'], MAX_RESTARTS, r.stdout.strip()[-200:]))
            both_ready = False
            if all('identity' in h for h in hosts):
                assert hosts[0]['identity']['machine_id'] != hosts[1]['identity']['machine_id']
                states = [readiness(h) for h in hosts]
                both_ready = all(states)
                if both_ready and not ready:
                    assert states[0]['commitments'] == states[1]['commitments']
                    write(ROOT / 'ready.json', {'status': 'both_models_loaded_and_boot_replay_matched',
                          'observed_at': time.time(), 'deadline': deadline, 'hourly_usd': hourly,
                          'spend_ceiling_usd': ceiling, 'matching_prior_qualified_index': True, 'hosts': states})
                    ready = True
                    qc.log('READY: both models resident; boot smoke matches each other and the prior full index')
            if not ready and time.time() > began + 2700 and not uncapped:
                raise TimeoutError('Boot and smoke check exceeded 45 minutes')
            write(ROOT / 'status.json', {'at': time.time(),
                  'state': ('ready' if both_ready else 'busy_or_restarting') if ready else 'booting',
                  'deadline': deadline, 'hourly_usd': hourly, 'ceiling_usd': ceiling,
                  'restarts': {h['name']: h['restarts'] for h in hosts}})
            time.sleep(10)
    except Exception as error:
        write(ROOT / 'controller-error.json', {'at': time.time(), 'type': type(error).__name__, 'error': str(error)})
        qc.log(type(error).__name__ + ': ' + str(error))
    finally:
        if uncapped and not (ROOT / 'stop').exists() and time.time() < deadline - 30:
            # Uncapped: a controller failure never releases the rentals; only the stop file or Vast credit does.
            write(ROOT / 'status.json', {'at': time.time(), 'state': 'controller_exited_hosts_kept', 'deadline': deadline})
            qc.log('Controller exiting; uncapped rentals left running (touch stop to release)')
            return
        for host in hosts:
            (host['root'] / 'stop').touch()
        for _ in range(18):
            if all((h['root'] / 'cleanup.json').exists() for h in hosts):
                break
            time.sleep(10)
        clean = all((h['root'] / 'cleanup.json').exists() for h in hosts)
        write(ROOT / 'cleanup-summary.json', {'observed_at': time.time(), 'all_owned_hosts_absent': clean,
              'ready_was_reached': ready, 'independent_watchdogs_remain_responsible_until_absence': not clean})
        write(ROOT / 'status.json', {'at': time.time(), 'state': 'closed' if clean else 'cleanup_pending',
              'deadline': deadline, 'ready_was_reached': ready})


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else '120'
    main(None if arg == 'uncapped' else float(arg))
