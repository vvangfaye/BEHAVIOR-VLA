"""Finite CPU preparation/export or a single native closed-loop episode."""
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import time
import urllib.request

from native_helpers import build_env, stop_process, wait_for_policy, verify_video


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def source_preflight(plan, full_baseline=False):
    code, runtime = Path(plan['code']), Path(plan['runtime'])
    if sha(code / '_PUBLIC_TEST_SOURCE_READY') != plan['source_manifest_sha256']:
        raise ValueError('Source receipt changed')
    for name, digest in plan['critical_files'].items():
        if sha(code / name) != digest:
            raise ValueError('Frozen evaluation/model source changed: ' + name)
    supplement = plan['supplemental_library']
    if sha(Path(supplement['root']) / '_READY') != supplement['ready_manifest_sha256']:
        raise ValueError('Supplemental graphics library changed')
    for name in ('envs/b1k-eval/bin/python', 'envs/b1k-pi05/bin/python', '_RUNTIME_READY'):
        if not (runtime / name).is_file():
            raise ValueError('Runtime file missing: ' + name)
    if full_baseline:
        base = Path(plan['baseline_checkpoint'])
        for row in plan['baseline_files']:
            p = base / row['relative_path']
            if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
                raise ValueError('Baseline checkpoint differs: ' + str(p))
    stats = runtime / 'omnigibson-data/2026-challenge-task-instances/metadata/task.jsonl'
    tasks = {row['task_name']: row for line in stats.read_text().splitlines() if line.strip()
             for row in [json.loads(line)]}
    budgets = {task: int(1.5*tasks[task]['length']) for task, _ in plan['pairs']}
    return {'critical_source_files': len(plan['critical_files']), 'native_task_budgets': budgets,
            'baseline_full_sha256_verified': full_baseline, 'task_metadata_sha256': sha(stats)}


def download(row, destination):
    relative = Path(row['relative_path'])
    if relative.is_absolute() or '..' in relative.parts or 'train_state' in relative.parts:
        raise ValueError('Unsafe inference export path')
    path = destination / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(4):
        try:
            req = urllib.request.Request(row['get_url'], headers={'If-Match': row['etag']})
            with opener.open(req, timeout=90) as source, tmp.open('wb') as out:
                if source.headers.get('ETag', '').strip('"') != row['etag']:
                    raise ValueError('Source object identity changed')
                shutil.copyfileobj(source, out, length=8*1024**2)
            if tmp.stat().st_size != row['bytes']:
                raise ValueError('Source object truncated')
            digest = sha(tmp)
            if row.get('sha256') and digest != row['sha256']:
                raise ValueError('Source metadata SHA256 differs')
            tmp.replace(path)
            # Read from the final AFS path as well, not just the transfer buffer.
            if sha(path) != digest:
                raise ValueError('AFS inference readback differs')
            return {'relative_path': str(relative), 'bytes': row['bytes'], 'sha256': digest,
                    'source_etag': row['etag']}
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def prepare(payload):
    plan = payload['plan']
    root = Path(plan['afs_root'])
    root.mkdir(parents=True, exist_ok=True)
    preflight = source_preflight(plan, full_baseline=True)
    if shutil.disk_usage(root).free < 30*1024**3:
        raise ValueError('Need30GiB of own AFS free space for inference export')
    return dict(state='accepted', preflight=preflight,
                baseline_manifest_sha256=hashlib.sha256(json.dumps(plan['baseline_files'], sort_keys=True).encode()).hexdigest())


def export(payload):
    plan, snap = payload['plan'], payload['snapshot']
    if snap['state'] != 'committed_checkpoint5000' or snap['formal_contract'] != plan['formal_contract']:
        raise ValueError('Export lacks exact committed formal checkpoint proof')
    if snap['consumed_step'] != 5000 or not snap['metadata'].get('commit_timestamp_nsecs'):
        raise ValueError('Export is not committed5000')
    dest = Path(plan['afs_root']) / 'checkpoint5000'
    dest.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(dest).free < sum(row['bytes'] for row in snap['files']) + 5*1024**3:
        raise ValueError('Insufficient export space')
    entries = [download(row, dest) for row in snap['files']]
    norm = 'assets/' + plan['repo_id'] + '/norm_stats.json'
    by_name = {row['relative_path']: row for row in entries}
    if by_name[norm]['sha256'] != plan['norm_sha256']:
        raise ValueError('Candidate normalization differs from baseline')
    if load(dest / 'assets/subtask_conditioning.json')['action_conditioning'] != 'task_and_subtask':
        raise ValueError('Candidate conditioning contract differs')
    raw = ''.join(row['sha256'] + '  ' + row['relative_path'] + '\n' for row in entries).encode()
    (dest / 'SHA256SUMS.inference').write_bytes(raw)
    ready = dict(state='accepted', completed_updates=5000, parent_checkpoint=1688977,
                 checkpoint=str(dest), files=entries, bytes=sum(r['bytes'] for r in entries),
                 manifest_sha256=hashlib.sha256(raw).hexdigest(), commit=snap['metadata'],
                 source_objects_etag_and_size_verified=True, full_afs_sha256_readback=True,
                 optimizer_exported=False, training_modified=False)
    write(dest / '_READY', ready)
    return ready


def transfer_probe(payload):
    plan = payload['plan']
    if payload['snapshot']['state'] != 'read_only_transfer_probe':
        raise ValueError('Wrong transfer probe')
    dest = Path(plan['afs_root']) / 'transfer-probe'
    dest.mkdir(parents=True,exist_ok=False)
    entries = [download(row,dest) for row in payload['snapshot']['files']]
    norm = next(row for row in entries if row['relative_path'].endswith('/norm_stats.json'))
    if norm['sha256'] != plan['norm_sha256']:
        raise ValueError('Transferred source normalization differs')
    return dict(state='accepted',direct_aliyun_oss_to_sensecore_afs=True,
                source_sha256_and_afs_full_readback=True,files=entries)


def evaluate(payload):
    plan = payload['plan']
    task, instance = payload['pair']
    if [task, instance] not in plan['pairs'] or payload['role'] not in ('baseline', 'candidate'):
        raise ValueError('Episode outside frozen20 scope')
    own = Path(plan['afs_root']) / 'episodes' / payload['id']
    own.mkdir(parents=True, exist_ok=False)
    source_preflight(plan)
    checkpoint = Path(plan['baseline_checkpoint']) if payload['role'] == 'baseline' else Path(plan['afs_root']) / 'checkpoint5000'
    if payload['role'] == 'candidate':
        ready = load(checkpoint / '_READY')
        if ready['completed_updates'] != 5000 or ready['manifest_sha256'] != sha(checkpoint / 'SHA256SUMS.inference'):
            raise ValueError('Candidate export not accepted')
        manifest_sha = ready['manifest_sha256']
    else:
        manifest_sha = hashlib.sha256(json.dumps(plan['baseline_files'], sort_keys=True).encode()).hexdigest()
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip().splitlines()
    if len(gpu_rows) != 1 or not any(s in gpu_rows[0] for s in ('5090', 'N12LP')):
        raise ValueError('Need exactly one5090 GPU')
    driver = gpu_rows[0].split(',')[2].strip()
    libs = [p for folder in ('/usr/local/nvidia/lib64','/usr/lib/x86_64-linux-gnu','/usr/lib64','/lib64')
            for p in Path(folder).glob('libEGL_nvidia.so.' + driver)]
    if not libs:
        raise ValueError('Matching host graphics driver missing')
    vk, egl = own / 'vk.json', own / 'egl.json'
    for path in (vk, egl):
        write(path, {'file_format_version':'1.0.0', 'ICD': {'library_path':str(libs[0]), **({'api_version':'1.3.242'} if path == vk else {})}})
    cache = Path('/var/tmp/behavior-1k-runtime') / payload['job_name']
    cache.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(cache).free < 40*1024**3 or cache.stat().st_dev == Path(plan['runtime']).stat().st_dev:
        raise ValueError('Need40GiB separate worker-local simulator cache')
    for name in ('appdata','cache','config','warp'):
        (cache / name).mkdir()
    params = dict(plan, port=8930, startup_timeout_seconds=1200)
    libpath = load(Path(plan['supplemental_library']['root']) / '_READY')['library_path']
    policy_env = build_env(params, cache, vk, egl, libpath, process='policy')
    eval_env = build_env(params, cache, vk, egl, libpath, process='evaluator')
    bundle = Path(__file__).parent
    for env in (policy_env, eval_env):
        env['PYTHONPATH'] = str(bundle) + ':' + env['PYTHONPATH']
        env['PYTHONUNBUFFERED'] = '1'
    policy_env['PAIRED_NATIVE_SERVER'] = str(Path(plan['code']) / 'openpi/scripts/b1k/serve_b1k.py')
    eval_env['PAIRED_SEED_PROOF'] = str(own / 'seed_ack.json')
    token = secrets.token_hex(32)
    runtime = Path(plan['runtime'])
    output = own / 'results'
    server_cmd = [str(runtime / 'envs/b1k-pi05/bin/python'), str(bundle / 'server_entry.py'), '--robot','b1k/R1Pro','--task','b1k/'+task,'--repo-id',plan['repo_id'],'--policy.config','pi05_b1k','--policy.dir',str(checkpoint),'--control_mode','receding_horizon','--action_horizon','16','--subtask-prediction','--max-token-len','512','--max-subtask-len','64','--subtask-temperature','0.0','--subtask-action-conditioning','task_and_subtask','--server-token',token,'--port','8930']
    eval_cmd = [str(runtime / 'envs/b1k-eval/bin/python'), str(bundle / 'evaluator_entry.py'),'--task-name',task,'--host','127.0.0.1','--port','8930','--mode','train','--prompt-mode','whole_task','--num-rollouts','1','--instance-indices',str(instance),'--expected-server-token',token,'--output-dir',str(output),'--write-video','--headless']
    server = evaluator = None
    try:
        with (own/'policy.log').open('x') as plog, (own/'eval.log').open('x') as elog:
            server = subprocess.Popen(server_cmd, env=policy_env, cwd=Path(plan['code'])/'openpi', stdout=plog, stderr=subprocess.STDOUT, start_new_session=True)
            metadata = wait_for_policy(params, server, token, eval_env, own/'policy_metadata.json')
            if metadata.get('full_episode_seed_protocol') != 'native_full_episode_seed_ack_v1':
                raise ValueError('Seed-capable server metadata missing')
            evaluator = subprocess.Popen(eval_cmd, env=eval_env, cwd=Path(plan['code'])/'BEHAVIOR-1K', stdout=elog, stderr=subprocess.STDOUT, start_new_session=True)
            evaluator.wait(timeout=26000)
            if evaluator.returncode != 0 or server.poll() is not None:
                raise RuntimeError('Native evaluation process failed; no automatic extra rollout')
        native_path = output/'json'/f'{task}_{instance}_0.json'
        if len(list((output/'json').glob('*.json'))) != 1:
            raise ValueError('Expected exactly one native result')
        row = load(native_path)
        ack = load(own/'seed_ack.json')
        if ack != dict(protocol='native_full_episode_seed_ack_v1',policy_seed=42,server_seed_acknowledged=True,environment_seed=42):
            raise ValueError('Actual seed acknowledgement mismatch')
        if row['task'] != task or row['instance_id'] != instance or row['rollout_id'] != 0 or row['steps'] <= 0:
            raise ValueError('Native result identity mismatch')
        if not (row['terminated'] or row['truncated']):
            raise ValueError('Episode did not finish its native budget')
        q = row['q_score']['final']
        if type(row['success']) is not bool or not math.isfinite(q) or not 0 <= q <= 1 or row['success'] != (q == 1):
            raise ValueError('Native success and goal Q inconsistent')
        video = output/'videos'/f'{task}_{instance}_0.mp4'
        video_proof = verify_video(params,video,eval_env,own)
        artifacts = [{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in (native_path,video,own/'policy.log',own/'eval.log',own/'seed_ack.json')]
        result = dict(state='accepted',id=payload['id'],role=payload['role'],checkpoint_manifest_sha256=manifest_sha,
                      protocol_sha256=plan['protocol_sha256'],artifacts=artifacts,video_validation=video_proof,
                      record=dict(task=task,instance_id=instance,artifact_accepted=True,policy_seed=42,
                                  policy_seed_acknowledged=True,steps=row['steps'],model_success=row['success'],q_score=q,
                                  terminated=row['terminated'],truncated=row['truncated'],result_sha256=sha(native_path)))
        write(own/'acceptance.json',result)
        return result
    finally:
        stop_process(evaluator)
        stop_process(server)


def main(payload):
    if payload['mode'] == 'prepare':
        return prepare(payload)
    if payload['mode'] == 'export':
        return export(payload)
    if payload['mode'] == 'transfer_probe':
        return transfer_probe(payload)
    if payload['mode'] == 'evaluate':
        return evaluate(payload)
    raise ValueError('Unknown finite worker mode')
