"""Finite paired holdout evaluation; one intent per job, no automatic retries."""
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
from compare_candidate import paired, summary, validate

spec = importlib.util.spec_from_file_location('holdout_transport', PARENT / 'paired_fixed20_step5000/controller.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
transport.ROOT = ROOT
load, save, observe = transport.load, transport.save, transport.observe
TERMINAL = transport.TERMINAL


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_local(plan):
    hashes = {p.name: sha(p.read_bytes()) for p in (ROOT / 'frozen').glob('*.py')}
    if hashes != load(ROOT / 'frozen_sha256.json'):
        raise ValueError('Frozen worker changed')
    proof = load(ROOT / 'worker_validation.json')
    if not proof['all_passed'] or proof['worker_sha256'] != hashes['worker.py']:
        raise ValueError('Worker behavior not validated')
    assembly = load(ROOT / 'assembly_validation.json')
    if (not assembly['all_passed'] or assembly['members'] != 40 or
            assembly['assembler_sha256'] != sha((ROOT / 'assemble.py').read_bytes()) or
            assembly['frozen_manifest_sha256'] != sha((ROOT / 'frozen_sha256.json').read_bytes())):
        raise ValueError('Assembly validation changed')
    protocol = load(ROOT / 'protocol.json')
    if sha(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()) != plan['protocol_sha256']:
        raise ValueError('Protocol changed')
    if plan['pairs'] != protocol['pairs'] or plan['native_task_budgets'] != protocol['native_task_budgets']:
        raise ValueError('Pair or frame budget changed')
    if not load(PARENT / 'dual_evaluation_status.json')['complete']:
        raise ValueError('Prior comparison incomplete')
    preflight = load(PARENT / 'heldout20_preflight/result.json')
    if (not preflight['all_instance_states_found'] or not preflight['train_resolution_exact'] or
            protocol['asset_proof_sha256'] != sha((PARENT / 'heldout20_preflight/result.json').read_bytes())):
        raise ValueError('CPU asset proof changed')
    assignments = [(m['role'], m['index']) for group in plan['groups'] for m in group['members']]
    if sorted(assignments) != sorted((role, i) for role in ('baseline', 'candidate') for i in range(20)):
        raise ValueError('Each model must cover each pair once')
    if len(plan['groups']) != 10 or any(len(g['members']) != 4 for g in plan['groups']):
        raise ValueError('Expected ten four-episode jobs')
    if plan['job_walltime_seconds'] > 128 * 3600 or plan['job_walltime_seconds'] < 4 * plan['member_walltime_seconds']:
        raise ValueError('Invalid finite job budget')


def native_create(intent, command, cpu=False):
    args = [transport.CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
            '--copy-job-name=' + ('b1k-qbal5k-failed-metadata-1016' if cpu else 'b1k-expert1k-1016-5x4-batch-03'),
            '--name=' + intent['name'], '--job-name=' + intent['name'],
            '--worker-spec=' + ('n6ls.iu.i40.8c32g' if cpu else 'n12lp.nn.i10a.1'),
            '--worker-nodes=1', '--quota-type=spot', '--retry-times=0',
            '--container-image-url=' + intent['image'],
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs', '--command=' + command]
    if not cpu:
        args.append('--env=NVIDIA_DRIVER_CAPABILITIES:all')
    if len(command.encode()) >= 100000:
        raise ValueError('Startup exceeds safe argument size')
    intent.update(creation_attempted=True, creation_started_at=time.time(), state='creation_unknown')
    path = ROOT / 'intents' / (intent['id'] + '.json')
    save(path, intent)
    try:
        result = subprocess.run(args, env=transport.environment(), capture_output=True, text=True, timeout=90)
        intent['cli_exit_code'] = result.returncode
    except subprocess.TimeoutExpired:
        intent['cli_outcome'] = 'timeout_unknown'
    save(path, intent)
    state = observe(intent)
    intent.update(uid=state['uid'], state=state['state'])
    save(path, intent)


def signed(client, key):
    return client.generate_presigned_url('put_object', Params={'Bucket': '260010165', 'Key': key}, ExpiresIn=604800)


def make_startups(client, plan):
    from assemble import assemble
    urls = {}
    for group in plan['groups']:
        urls[group['id']] = {'report': signed(client, plan['object_prefix'] + '/reports/' + group['id'] + '.json')}
        for item in group['members']:
            identity = item['role'] + '-' + f"{item['index']:02d}"
            urls[identity] = {'report': signed(client, plan['object_prefix'] + '/reports/' + identity + '.json'),
                              'heartbeat': signed(client, plan['object_prefix'] + '/telemetry/' + identity + '/heartbeat.json'),
                              'diagnostics': signed(client, plan['object_prefix'] + '/telemetry/' + identity + '/diagnostics.json')}
    data = {'plan': plan, 'files': {p.name: p.read_text() for p in sorted((ROOT / 'frozen').glob('*.py'))},
            'bundle_hash': sha((ROOT / 'frozen_sha256.json').read_bytes()), 'urls': urls}
    return data, assemble(data)


def prepare_stage(client, plan):
    if (ROOT / 'intents/stage.json').exists():
        return
    if not load(ROOT / 'prior_terminal.json')['all_terminal']:
        raise ValueError('Prior writers still active')
    data, startups = make_startups(client, plan)
    expected = {}
    for identity, source in startups.items():
        path = ROOT / 'private' / (identity + '.py')
        path.write_text(source)
        os.chmod(path, 0o600)
        expected[identity] = sha(source.encode())
    save(ROOT / 'startup_hashes.json', expected)
    package = {'data': data, 'expected': expected, 'assembler': (ROOT / 'assemble.py').read_text()}
    packed = base64.b64encode(zlib.compress(json.dumps(package).encode(), 9)).decode()
    source = """import base64,hashlib,json,os,zlib
from pathlib import Path
package=json.loads(zlib.decompress(base64.b64decode(PACKED)))
exec(compile(package['assembler'],'assemble.py','exec'))
files=assemble(package['data'])
root=Path(DEST);root.mkdir(parents=True,exist_ok=True)
hashes={}
for identity,source in files.items():
 p=root/(identity+'.py');raw=source.encode()
 if p.exists():
  assert p.read_bytes()==raw,'Existing startup differs'
 else:
  with p.open('xb') as f:f.write(raw)
 p.chmod(0o600)
 hashes[identity]=hashlib.sha256(p.read_bytes()).hexdigest()
assert hashes==package['expected'],'CPU assembly differs'
print('HELDOUT_STAGE_JSON='+json.dumps(hashes,sort_keys=True),flush=True)
""".replace('PACKED', repr(packed)).replace('DEST', repr(plan['afs_root'] + '/launch'))
    command = 'timeout --signal=TERM --kill-after=15 240 python3 -u -c ' + shlex.quote(source)
    intent = {'id': 'stage', 'name': 'b1k-heldout20-1017-stage', 'uid': None, 'gpu': 0,
              'image': plan['image'], 'creation_attempted': False, 'startup_sha256': sha(source.encode())}
    transport.new(ROOT / 'intents/stage.json', intent)
    native_create(intent, command, cpu=True)


def stage_receipt(client, plan):
    if (ROOT / 'stage_accepted.json').exists():
        return True
    intent = load(ROOT / 'intents/stage.json')
    state = observe(intent)
    save(ROOT / 'observations/stage.json', state)
    if state['state'] != 'SUCCEEDED':
        if state['state'] in TERMINAL:
            raise ValueError('CPU staging failed')
        return False
    key = plan['object_prefix'] + '/stage_receipt.json'
    try:
        response = client.get_object(Bucket='260010165', Key=key)
        try:
            receipt = json.loads(response['Body'].read(1024*1024))
        finally:
            response['Body'].close()
    except Exception as exc:
        if getattr(exc, 'response', {}).get('Error', {}).get('Code') not in ('NoSuchKey', '404'):
            raise
        reader_path = ROOT / 'intents/stage-readback.json'
        if not reader_path.exists():
            url = client.generate_presigned_url('put_object', Params={'Bucket':'260010165','Key':key}, ExpiresIn=86400)
            config = {'root': plan['afs_root'] + '/launch', 'ids': list(load(ROOT / 'startup_hashes.json')),
                      'stage_uid': state['uid'], 'put_url': url}
            source = """import hashlib,json,urllib.request
from pathlib import Path
config=CONFIG
root=Path(config['root'])
hashes={identity:hashlib.sha256((root/(identity+'.py')).read_bytes()).hexdigest() for identity in config['ids']}
body=json.dumps({'hashes':hashes,'stage_uid':config['stage_uid'],'log_contents_exported':False}).encode()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(urllib.request.Request(config['put_url'],data=body,method='PUT'),timeout=90):pass
print('metadata receipt saved',flush=True)
""".replace('CONFIG', repr(config))
            reader = {'id':'stage-readback','name':'b1k-heldout20-1017-readback','uid':None,'gpu':0,
                      'image':plan['image'],'creation_attempted':False,'source_sha256':sha(source.encode())}
            transport.new(reader_path, reader)
            command = 'timeout --signal=TERM --kill-after=15 240 python3 -u -c ' + shlex.quote(source)
            native_create(reader, command, cpu=True)
        else:
            reader = observe(load(reader_path))
            if reader['state'] in TERMINAL and reader['state'] != 'SUCCEEDED':
                raise ValueError('Staging metadata readback failed')
        return False
    if receipt['hashes'] != load(ROOT / 'startup_hashes.json') or receipt['stage_uid'] != state['uid']:
        raise ValueError('AFS startup readback differs')
    save(ROOT / 'stage_accepted.json', {'uid': state['uid'], 'hashes': receipt['hashes'],
                                      'accepted_at': time.time(), 'metadata_key': key})
    return True


def submit_group(plan, group):
    path = ROOT / 'intents' / (group['id'] + '.json')
    if path.exists():
        return
    hashes = load(ROOT / 'startup_hashes.json')
    receipt = load(ROOT / 'stage_accepted.json')
    if hashes != receipt['hashes']:
        raise ValueError('Staging receipt changed')
    local = ROOT / 'private' / (group['id'] + '.py')
    if sha(local.read_bytes()) != hashes[group['id']]:
        raise ValueError('Local immutable startup changed')
    remote = plan['afs_root'] + '/launch/' + group['id'] + '.py'
    launcher = ('import hashlib;from pathlib import Path;p=Path(' + repr(remote) + ');b=p.read_bytes();'
                'assert hashlib.sha256(b).hexdigest()==' + repr(hashes[group['id']]) + ';'
                'exec(compile(b,str(p),"exec"))')
    command = ('timeout --signal=TERM --kill-after=60 ' + str(plan['job_walltime_seconds']) + ' ' +
               shlex.quote(plan['runtime'] + '/envs/b1k-eval/bin/python') + ' -c ' + shlex.quote(launcher))
    intent = {'id': group['id'], 'name': group['name'], 'uid': None, 'gpu': 1, 'image': plan['image'],
              'members': group['members'], 'startup_sha256': hashes[group['id']], 'creation_attempted': False}
    transport.new(path, intent)
    native_create(intent, command)


def fetch(client, plan, identity):
    try:
        response = client.get_object(Bucket='260010165', Key=plan['object_prefix'] + '/reports/' + identity + '.json')
    except Exception as exc:
        code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
        if code in ('NoSuchKey', '404'):
            return None
        raise
    try:
        raw = response['Body'].read(4 * 1024**2)
    finally:
        response['Body'].close()
    report = json.loads(raw)
    if (report.get('id') != identity or report.get('protocol_sha256') != plan['protocol_sha256'] or
            report.get('worker_bundle_sha256') != sha((ROOT / 'frozen_sha256.json').read_bytes())):
        raise ValueError('Report identity differs')
    if report.get('state') == 'accepted':
        role, index = identity.rsplit('-', 1)
        expected_hash = (plan['expected_checkpoint_manifest_sha256'] if role == 'candidate'
                         else sha(json.dumps(plan['baseline_files'], sort_keys=True).encode()))
        if report['role'] != role or report['checkpoint_manifest_sha256'] != expected_hash:
            raise ValueError('Reported model identity differs')
        row = report['record']
        if [row['task'], row['instance_id']] != plan['pairs'][int(index)]:
            raise ValueError('Reported pair differs')
        if row['truncated'] and row['steps'] != plan['native_task_budgets'][row['task']] + 1:
            raise ValueError('Incomplete native budget')
        validate({'protocol_sha256': plan['protocol_sha256'], 'records': [row]},
                 [plan['pairs'][int(index)]], plan['protocol_sha256'])
    save(ROOT / 'reports' / (identity + '.json'), report)
    return report


def finish(plan, reports):
    pairs = list(map(tuple, plan['pairs']))
    models = {}
    for role in ('baseline', 'candidate'):
        records = [reports[role + '-' + f'{i:02d}']['record'] for i in range(20)]
        models[role] = validate({'protocol_sha256': plan['protocol_sha256'], 'records': records}, pairs, plan['protocol_sha256'])
    result = {'complete': True, 'scope': 'same-task instances excluded from current fine-tuning indexes',
              'baseline_pretraining_exclusion_proven': False,
              'summaries': {role: summary(rows, pairs) for role, rows in models.items()},
              'candidate_vs_baseline': paired(models['baseline'], models['candidate'], pairs),
              'protocol_sha256': plan['protocol_sha256'],
              'report_sha256': {identity: sha((ROOT / 'reports' / (identity + '.json')).read_bytes()) for identity in reports}}
    save(ROOT / 'comparison.json', result)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['task', 'instance', 'baseline_success', 'baseline_q', 'expert_success', 'expert_q', 'q_delta'])
    for pair in pairs:
        old, new = models['baseline'][pair], models['candidate'][pair]
        writer.writerow([*pair, old['model_success'], old['q_score'], new['model_success'], new['q_score'], new['q_score']-old['q_score']])
    (ROOT / 'comparison.csv').write_text(output.getvalue(), encoding='utf-8-sig')


def main():
    with (ROOT / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = (ROOT / 'plan.json').read_bytes()
        plan = json.loads(raw)
        verify_local(plan)
        if not plan['execution_enabled']:
            raise ValueError('Admission not enabled')
        save(ROOT / 'launch.json', {'pid': os.getpid(), 'started_at': time.time(), 'plan_sha256': sha(raw)})
        client = transport.aoss_client()
        prepare_stage(client, plan)
        ids = [role + '-' + f'{i:02d}' for role in ('baseline', 'candidate') for i in range(20)]
        while time.time() < plan['deadline_epoch']:
            if (ROOT / 'plan.json').read_bytes() != raw:
                raise ValueError('Plan changed while running')
            if not stage_receipt(client, plan):
                time.sleep(30)
                continue
            if not (ROOT / 'STOP').exists():
                for group in plan['groups']:
                    submit_group(plan, group)
            intents = [load(ROOT / 'intents' / (g['id'] + '.json')) for g in plan['groups']
                       if (ROOT / 'intents' / (g['id'] + '.json')).exists()]
            with ThreadPoolExecutor(max_workers=6) as pool:
                states = list(pool.map(observe, intents))
                values = list(pool.map(lambda identity: fetch(client, plan, identity), ids))
            reports = dict(zip(ids, values))
            accepted = {role: sum(bool(reports[role+'-'+f'{i:02d}'] and reports[role+'-'+f'{i:02d}']['state']=='accepted') for i in range(20))
                        for role in ('baseline', 'candidate')}
            complete = accepted == {'baseline': 20, 'candidate': 20} and all(s['state'] in TERMINAL for s in states)
            save(ROOT / 'status.json', {'observed_at': time.time(), 'accepted': accepted,
                  'states': states, 'complete': complete, 'gpu_jobs_created': len(intents)})
            if complete:
                finish(plan, reports)
                return
            if (ROOT / 'STOP').exists():
                return
            if len(states) == 10 and all(s['state'] in TERMINAL for s in states):
                save(ROOT / 'terminal_incomplete.json', {'at': time.time(), 'accepted': accepted})
                return
            time.sleep(60)
        save(ROOT / 'deadline_reached.json', {'at': time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(ROOT / 'controller_error.json', {'at': time.time(), 'type': type(exc).__name__,
             'message': str(exc)[:200] if not isinstance(exc, subprocess.SubprocessError) else 'subprocess failure'})
        raise
