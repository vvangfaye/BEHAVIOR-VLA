#!/usr/bin/env python3
"""OSS-only single-host bounded subtask collection/export/fine-tuning jobs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]
KIND = 'subtask_finetune_nebula'
GIB = 1024**3
TRAIN_QUEUE = 'demand_specific_na175_h20'
TRAIN_GPU_CARD_TYPE = 'h20'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from nebula.submit_oss import OSSClient, object_key, validate_manifest


class GateError(ValueError):
    """Safe local validation message; SDK exception text is never disclosed."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def safe_path(value):
    p = PurePosixPath(value)
    if not p.parts or p.is_absolute() or '..' in p.parts or any(c in value for c in ('\\', '\n', '\r', '\x00')):
        raise GateError('Artifact path escapes its root')
    return p.as_posix()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2); f.write('\n')


def inventory(root):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise GateError('Submission bundle must not contain symlinks')
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def snapshot(bucket, key, path, checksum=None):
    safe_path(key); safe_path(path)
    head = bucket.head_object(key)
    if head.content_length <= 0 or not head.etag: raise GateError('Missing or empty frozen OSS input')
    item = dict(key=key, path=path, bytes=head.content_length, etag=head.etag)
    if checksum is not None: item['sha256'] = checksum
    return item


def download(bucket, item, root):
    target = Path(root) / safe_path(item['path'])
    if target.exists() or target.is_symlink(): raise GateError('Download target already exists')
    head = bucket.head_object(item['key'])
    if (head.content_length, head.etag) != (item['bytes'], item['etag']):
        raise GateError('OSS input changed after preflight')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    h = hashlib.sha256(); n = 0
    response = bucket.get_object(item['key'], headers={'If-Match': item['etag']})
    try:
        if response.etag != item['etag']: raise GateError('Download ETag differs')
        with partial.open('xb') as f:
            while block := response.read(8 * 1024**2): f.write(block); h.update(block); n += len(block)
    finally: response.resp.response.close()
    if n != item['bytes'] or item.get('sha256', h.hexdigest()) != h.hexdigest():
        raise GateError('Truncated or checksum-invalid OSS download')
    partial.rename(target)
    return {**item, 'sha256': h.hexdigest()}


def validate_plan(plan):
    if plan.get('kind') != KIND or plan.get('fingerprint') != digest({k: v for k, v in plan.items() if k != 'fingerprint'}):
        raise GateError('Nebula fine-tuning plan changed')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', plan['run_id']): raise GateError('Unsafe run ID')
    if type(plan['devices']) is not int or plan['devices'] not in (1, 2, 4, 8) or type(plan['batch_size']) is not int or not plan['devices'] <= plan['batch_size'] <= 64 or plan['batch_size'] % plan['devices'] or type(plan['steps']) is not int or not 128 <= plan['steps'] <= 5000:
        raise GateError('Invalid bounded training allocation')
    resources = plan.get('resources', {})
    if not isinstance(resources, dict) or type(resources.get('worker_count')) is not int or resources['worker_count'] != 1 or not isinstance(resources.get('queue'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', resources['queue']) or not isinstance(resources.get('gpu_card_type'), str) or not re.fullmatch(r'[a-z0-9_-]{1,32}', resources['gpu_card_type']):
        raise GateError('Invalid frozen training queue/card contract')
    known_cards = {TRAIN_QUEUE: 'h20', 'amap_common_aios_nm125_l20': 'l20'}
    if resources['queue'] in known_cards and resources['gpu_card_type'] != known_cards[resources['queue']]:
        raise GateError('Training queue and GPU card type disagree')
    paths = [safe_path(i['path']) for i in plan['objects']]
    if len(paths) != len(set(paths)) or any(type(i['bytes']) is not int or i['bytes'] <= 0 or not i['etag'] for i in plan['objects']):
        raise GateError('Invalid or duplicate frozen objects')
    for i in plan['objects']: safe_path(i['key'])
    if not plan['objects'] or plan['download_bytes'] != sum(i['bytes'] for i in plan['objects']):
        raise GateError('Input byte budget differs')
    collection = plan['collection_plan']
    if collection.get('fingerprint') != digest({k: v for k, v in collection.items() if k != 'fingerprint'}) or collection.get('panel_fingerprint') != plan['panel']['fingerprint']:
        raise GateError('Nested collection plan changed')
    if plan['collection_plan']['checkpoint_identity']['checkpoint_step'] != 1375000:
        raise GateError('Fine-tuning must use the verified baseline 1375000')
    if plan['replay_plan']['held_out_instances'] != plan['collection_plan']['held_out_instances']:
        raise GateError('Replay holdout differs from the frozen collection holdout')
    if plan['conditioning'] != 'task_and_subtask' or plan['video_backend'] != 'pyav' or plan['training_horizon'] != 32:
        raise GateError('Training semantics changed')


def freeze(client, *, panel, collection_plan, collection_source, metadata, replay_plan, labels,
           checkpoint, base_repo_id, policy_archive, cache_archive, output, run_id, devices=4, batch_size=8, steps=1000, queue=TRAIN_QUEUE, gpu_card_type=TRAIN_GPU_CARD_TYPE):
    from stage_subtask_replay import validate as validate_replay
    validate_replay(replay_plan, metadata)
    if collection_plan['panel_fingerprint'] != panel['fingerprint'] or not 1 <= collection_plan['expected_trials'] <= 300:
        raise GateError('Collection must match the panel and pilot budget')
    if collection_plan.get('fingerprint') != digest({k: v for k, v in collection_plan.items() if k != 'fingerprint'}):
        raise GateError('Collection selection changed')
    if replay_plan['held_out_instances'] != collection_plan['held_out_instances']:
        raise GateError('Replay holdout differs from collection holdout')
    source = client.key(collection_source)
    destination = client.key(output)
    for origin in (source, client.key(checkpoint), client.key(replay_plan['source'])):
        if destination == origin or destination.startswith(origin + '/') or origin.startswith(destination + '/'):
            raise GateError('Output must be separate from immutable input roots')
    marker = json.loads(client.read(source + '/_SUCCESS'))
    contract = json.loads(client.read(source + '/collection_contract.json'))
    if marker.get('kind') != 'subtask_success_collection' or marker.get('complete') is not True or marker.get('ready_for_export') is not True or marker.get('plan_fingerprint') != collection_plan['fingerprint']:
        raise GateError('Collection has no completed successful new trajectories')
    if contract.get('kind') != 'subtask_success_collection' or contract.get('plan_fingerprint') != collection_plan['fingerprint'] or contract.get('parent_panel_fingerprint') != panel['fingerprint'] or contract.get('checkpoint_identity') != collection_plan['checkpoint_identity'] or not contract.get('run_id'):
        raise GateError('Collection identity differs from baseline and plan')
    objects = []
    found = client.inventory(source)
    if any(k.endswith('/_FAILED') or k.endswith('/_RUNNING') for k in found):
        raise GateError('Collection is failed or still running')
    for key, size in sorted(found.items()):
        if size > 0:
            relative = safe_path(key[len(source) + 1:])
            objects.append(snapshot(client.bucket, key, 'collection/' + relative))
    for name in ('_SUCCESS', 'collection_contract.json', 'collection_summary.json'):
        if source + '/' + name not in found: raise GateError('Incomplete collection artifacts')
    model = client.key(checkpoint); model_objects = client.inventory(model)
    manifest = client.read(model + '/SHA256SUMS.inference')
    entries = validate_manifest(manifest, model, model_objects)
    identity = collection_plan['checkpoint_identity']; ready = json.loads(client.read(model + '/_READY'))
    if ready.get('step') != 1375000 or ready.get('manifest_sha256') != hashlib.sha256(manifest).hexdigest() or ready['manifest_sha256'] != identity['checkpoint_manifest_sha256']:
        raise GateError('Initialization differs from evaluated checkpoint')
    required = {model + '/params/_METADATA', model + '/_CHECKPOINT_METADATA',
                model + '/assets/' + base_repo_id + '/norm_stats.json', model + '/assets/subtask_conditioning.json'}
    if not required <= entries.keys() or any(k.startswith(model + '/params/') and k not in entries for k in model_objects):
        raise GateError('Model manifest does not cover required parameters and assets')
    if model + '/model.safetensors' in model_objects: raise GateError('Expected one JAX checkpoint format')
    conditioning = json.loads(client.read(model + '/assets/subtask_conditioning.json'))
    if conditioning.get('action_conditioning') != 'task_and_subtask': raise GateError('Wrong model conditioning')
    covered = {**entries, model + '/SHA256SUMS.inference': hashlib.sha256(manifest).hexdigest(),
               model + '/_READY': hashlib.sha256(client.read(model + '/_READY')).hexdigest()}
    for key, checksum in sorted(covered.items()):
        objects.append(snapshot(client.bucket, key, 'checkpoint/' + key[len(model) + 1:], checksum))
    replay_source = client.key(replay_plan['source'])
    for item in replay_plan['objects']:
        remote = snapshot(client.bucket, replay_source + '/' + safe_path(item['path']), 'replay/' + item['path'])
        if (remote['bytes'], remote['etag']) != (item['bytes'], item['etag']): raise GateError('Replay input changed')
        objects.append(remote)
    for name, uri in (('policy.tar.gz', policy_archive), ('cache.tar.gz', cache_archive)):
        objects.append(snapshot(client.bucket, client.key(uri), 'archives/' + name))
    result = dict(schema_version=1, kind=KIND, run_id=run_id, output='oss://' + client.bucket_name + '/' + client.key(output),
        bucket=client.bucket_name, endpoint=os.environ.get('OSS_ENDPOINT', ''), devices=devices, batch_size=batch_size,
        steps=steps, resources=dict(queue=queue, gpu_card_type=gpu_card_type, worker_count=1), conditioning='task_and_subtask', video_backend='pyav', training_horizon=32,
        base_repo_id=base_repo_id, collection_source='oss://' + client.bucket_name + '/' + source,
        collection_plan=collection_plan, panel=panel, replay_plan=replay_plan,
        metadata_files=replay_plan['metadata'], labels_sha256=sha(labels),
        objects=objects, download_bytes=sum(i['bytes'] for i in objects),
        checkpoint_bytes=sum(i['bytes'] for i in objects if i['path'].startswith('checkpoint/')))
    result['fingerprint'] = digest(result); validate_plan(result)
    client.unused(client.key(output)); client.unused(client.key(output) + '.submission_claim.json', directory=False)
    return result


def bundle(plan, metadata, labels, destination):
    validate_plan(plan)
    destination = Path(destination)
    if destination.exists(): raise GateError('Submission preparation requires a fresh root')
    destination.mkdir(); payload = destination / 'bundle'; payload.mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.venv', 'outputs', '.git')
    for relative in ('openpi/src', 'openpi/packages/openpi-client'):
        shutil.copytree(REPO / relative, payload / relative, ignore=ignore)
    copies = ['openpi/pyproject.toml', 'openpi/README.md', 'openpi/LICENSE',
        'openpi/scripts/compute_norm_stats.py', 'openpi/scripts/train.py',
        'openpi/scripts/b1k/train_b1k.py', 'openpi/scripts/b1k/check_subtask_data.py',
        'openpi/scripts/b1k/check_frame_memory.py', 'openpi/scripts/b1k/distributed_preflight.py',
        'nebula/subtask_finetune.py', 'nebula/submit_oss.py', 'nebula/eval_storage.py']
    names = ['finetune_subtask_collection', 'export_subtask_collection', 'verify_subtask_collection_export',
        'stage_subtask_replay', 'strip_depth_from_info', 'subtask_improvement', 'subtask_sample_panel',
        'subtask_trajectory', 'collect_b1k_subtasks', 'eval_b1k_subtask_samples', 'eval_b1k_queue', 'pi05_eval_queue']
    copies += ['scripts/' + name + '.py' for name in names]
    copies += ['scripts/train_pi05_b1k.sh', 'scripts/train_pi05_subtask_b1k.sh']
    for relative in copies:
        target = payload / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(REPO / relative, target)
    for item in plan['metadata_files']:
        target = payload / 'inputs/metadata' / safe_path(item['path']); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(metadata) / item['path'], target)
        if sha(target) != item['sha256']: raise GateError('Frozen replay metadata changed')
    target = payload / 'inputs/subtask_segments.parquet'; shutil.copyfile(labels, target)
    if sha(target) != plan['labels_sha256']: raise GateError('Original label table changed')
    write(payload / 'nebula/finetune_job.json', plan)
    write(payload / 'nebula/cluster.json', dict(worker=dict(cpu=3200, gpu=plan['devices'] * 100, memory=160000)))
    (payload / 'requirements.txt').write_text('oss2>=2.19,<3\n')
    (payload / 'nebula/finetune_entry.py').write_text(
        "import pathlib,subprocess,sys\np=pathlib.Path(__file__).with_name('subtask_finetune.py')\n"
        "raise SystemExit(subprocess.call([sys.executable,str(p),'worker','--plan',str(p.with_name('finetune_job.json'))]))\n")
    write(destination / 'bundle_manifest.json', inventory(payload))
    return payload


def submit(client, prepared):
    prepared = Path(prepared); payload = prepared / 'bundle'
    if inventory(payload) != read(prepared / 'bundle_manifest.json'): raise GateError('Submission code or inputs changed')
    plan = read(payload / 'nebula/finetune_job.json'); validate_plan(plan)
    output = client.key(plan['output']); claim = output + '.submission_claim.json'
    if any((prepared / name).exists() for name in ('submission_claim.json', 'submission_receipt.json', 'submission.stdout.log')):
        raise GateError('Prior submission attempt exists; inspect its exact task instead of resubmitting')
    client.unused(output); client.unused(claim, directory=False)
    secrets = [os.environ.get(k, '') for k in ('AI_HUB_TOKEN', 'OSS_ACCESS_ID', 'OSS_ACCESS_KEY')]
    if not all(secrets) or any(any(c in s for c in (',', '\n', '\r')) for s in secrets[1:]):
        raise GateError('Missing or invalid private launcher credentials')
    command = [os.environ.get('AI_HUB_CLI', 'ai-hub-cli'), 'train', 'mdl',
        '--token=' + secrets[0], '--namespace=' + os.environ['AI_HUB_NAMESPACE'],
        '--name=' + os.environ.get('AIHUB_MODEL_NAME', 'behavior'),
        '--queue=' + plan['resources']['queue'],
        '--entry=nebula/finetune_entry.py', '--worker_count=1', '--file.cluster_file=nebula/cluster.json',
        '--oss_access_id=' + secrets[1], '--oss_access_key=' + secrets[2],
        '--oss_bucket=' + client.bucket_name, '--oss_endpoint=' + os.environ['OSS_ENDPOINT'],
        '--env=OSS_ACCESS_KEY_ID=' + secrets[1] + ',OSS_ACCESS_KEY_SECRET=' + secrets[2] + ',PIP_NO_DEPS=0',
        '--job_name=' + plan['run_id'], '--algo_name=' + os.environ.get('ALGO_NAME', 'pytorch2120'),
        '--gpu_card_type=' + plan['resources']['gpu_card_type'], '--oss_appendable=true',
        '--max_failover_times=0', '--debug_mode', '--ignore=__pycache__,*.pyc,*.pyo']
    write(prepared / 'submission_claim.json', dict(run_id=plan['run_id'], plan_fingerprint=plan['fingerprint'], state='submission_reserved', resubmit_allowed=False))
    client.put_new(claim, prepared / 'submission_claim.json')
    client.put_new(output + '.bundle_manifest.json', prepared / 'bundle_manifest.json')
    # CLI failures can echo arguments; redact before saving or displaying output.
    receipt = dict(kind=KIND, run_id=plan['run_id'], plan_fingerprint=plan['fingerprint'], output=plan['output'],
        devices=plan['devices'], steps=plan['steps'], resources=plan['resources'], bundled_files_manifest_sha256=sha(prepared / 'bundle_manifest.json'), job_identifiers=[], submission_state='unknown', resubmit_allowed=False)
    try:
        process = subprocess.run(command, cwd=payload, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        clean = process.stdout
        for secret in secrets: clean = clean.replace(secret, '[REDACTED]')
        (prepared / 'submission.stdout.log').write_text(clean)
        receipt.update(exit_code=process.returncode, submission_state='submitted' if process.returncode == 0 else 'failed_or_unknown',
            job_identifiers=identifiers(clean))
    finally:
        write(prepared / 'submission_receipt.json', receipt)
        client.put_new(output + '.submission_receipt.json', prepared / 'submission_receipt.json')
    return receipt


def identifiers(text):
    found = set()
    for line in text.splitlines():
        try: value = json.loads(line)
        except json.JSONDecodeError: value = None
        if isinstance(value, dict):
            for key in ('job_id', 'jobId', 'task_id', 'taskId', 'instance_id', 'iid', 'IID'):
                if value.get(key) is not None: found.add(str(value[key]))
        for match in re.finditer(r'(?i)\b(?:iid|(?:job|task|instance)[_ -]?id)\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9._:-]*)', line):
            found.add(match.group(1))
    return sorted(found)


def checked_space(root, required):
    available = shutil.disk_usage(root).free
    if available < required: raise GateError('Insufficient scratch for actual inputs/export/train-state storage')
    return dict(required_disk_bytes=required, available_disk_bytes=available)


def restore_policy(archive, target):
    target = Path(target)
    if target.exists(): raise GateError('Policy environment needs a fresh owned root')
    target.mkdir()
    with tarfile.open(archive, 'r:gz') as source: source.extractall(target, filter='data')
    if not (target / 'bin/python').is_file(): raise GateError('Archived policy environment lacks Python')
    unpack = target / 'bin/conda-unpack'
    if unpack.is_file(): subprocess.run([str(unpack)], check=True)
    return target / 'bin/python'


def prepare_worker_data(plan, runtime, *, task_prompts=None):
    """Verify completed collection/replay, export real files and freeze training."""
    from subtask_improvement import load_plan
    from stage_subtask_replay import validate as validate_replay
    from strip_depth_from_info import main as strip_depth
    from collect_b1k_subtasks import validate_collection
    from export_subtask_collection import accepted_trials, export_collection
    from finetune_subtask_collection import prepare
    validate_plan(plan); runtime = Path(runtime); panel = plan['panel']; collection = plan['collection_plan']
    panel_path = runtime / 'panel.json'; write(panel_path, panel)
    collection_path = runtime / 'collection_plan.json'; write(collection_path, collection)
    load_plan(collection_path, panel)
    original = REPO / 'inputs/metadata'; validate_replay(plan['replay_plan'], original)
    labels = REPO / 'inputs/subtask_segments.parquet'
    if sha(labels) != plan['labels_sha256']: raise GateError('Bundled original labels changed')
    actual = validate_collection(runtime / 'collection', panel, collection)
    if actual != read(runtime / 'collection/collection_summary.json') or not actual['ready_for_export']:
        raise GateError('Collection did not produce verified successful new trajectories')
    accepted, _ = accepted_trials(runtime / 'collection', collection, {s['sample_id']: s for s in panel['samples']})
    frames = sum(r['steps'] for _, r in accepted) + plan['replay_plan']['original_replay_frames']
    first = accepted[0][1]
    per_frame = sum(math.prod(s) for s in first['camera_shapes'].values()) + 4 * (first['state_shape'][0] + 23)
    budget = checked_space(runtime, frames * per_frame * 2 + 8 * plan['checkpoint_bytes'] + 16 * GIB)
    write(runtime / 'export_storage_preflight.json', {**budget, 'maximum_export_bytes': frames * per_frame * 2})
    replay = runtime / 'replay'
    for item in plan['metadata_files']:
        target = replay / safe_path(item['path']); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / item['path'], target)
        if sha(target) != item['sha256']: raise GateError('Replay metadata copy changed')
    strip_depth(str(replay))
    episodes = [r['episode_index'] for r in plan['replay_plan']['episodes']]
    (replay / 'selected_episodes.txt').write_text(''.join(str(i) + '\n' for i in episodes))
    exported = runtime / 'dataset'
    export_collection(runtime / 'collection', exported, 'b1k_collection_' + plan['run_id'], panel, collection,
        base_dataset=replay, base_segments=labels, base_episodes=episodes, task_prompts=task_prompts)
    training = prepare(exported, runtime / 'collection', runtime / 'checkpoint', plan['base_repo_id'], panel, collection,
        runtime / 'training', run_id=plan['run_id'], devices=plan['devices'], batch_size=plan['batch_size'], steps=plan['steps'])
    return training


def verify_training_cards(plan):
    validate_plan(plan)
    names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    expected = plan['resources']['gpu_card_type'].upper()
    if len(names) < plan['devices'] or any(not re.search(r'(?<![A-Z0-9])' + re.escape(expected) + r'(?![A-Z0-9])', name.upper()) for name in names[:plan['devices']]):
        raise GateError('Allocated GPUs differ from the frozen training card contract')
    return dict(queue=plan['resources']['queue'], requested_gpu_card_type=plan['resources']['gpu_card_type'],
        allocated_gpu_names=names, requested_devices=plan['devices'], verified=True)


def pipeline(plan, runtime):
    from finetune_subtask_collection import run, stage_final
    runtime = Path(runtime)
    write(runtime / 'training_gpu_preflight.json', verify_training_cards(plan))
    training = prepare_worker_data(plan, runtime)
    for phase in ('smoke', 'full'): run(training, phase)
    stage_final(training, runtime / 'inference_checkpoint')
    write(runtime / 'pipeline_completion.json', dict(complete=True, kind=KIND, plan_fingerprint=plan['fingerprint'],
        training_plan_fingerprint=training['fingerprint'], training_receipt=read(runtime / 'training/training_receipt.json')))


def publish(client, plan, runtime):
    runtime = Path(runtime); proof = read(runtime / 'pipeline_completion.json')
    if proof.get('complete') is not True or proof.get('plan_fingerprint') != plan['fingerprint']:
        raise GateError('Cannot publish incomplete training')
    receipt = read(runtime / 'training/training_receipt.json')
    if receipt != proof['training_receipt'] or receipt.get('inference_staging_complete') is not True or receipt.get('training_plan_fingerprint') != proof['training_plan_fingerprint']:
        raise GateError('Training output is not inference verified')
    training = read(runtime / 'training/training_plan.json')
    provenance = read(runtime / 'dataset/meta/collection_provenance.json')
    if (receipt.get('complete') is not True or receipt.get('training_run_id') != plan['run_id']
            or receipt.get('conditioning') != 'task_and_subtask'
            or receipt.get('collection_plan_fingerprint') != plan['collection_plan']['fingerprint']
            or receipt.get('init_checkpoint_manifest_sha256') != plan['collection_plan']['checkpoint_identity']['checkpoint_manifest_sha256']
            or receipt.get('export_provenance_sha256') != digest(provenance)
            or training.get('fingerprint') != digest({k:v for k,v in training.items() if k!='fingerprint'})
            or training['fingerprint'] != receipt['training_plan_fingerprint']
            or inventory(runtime / 'dataset') != training['dataset_files']):
        raise GateError('Completed training/data provenance changed before publication')
    reserved = {tuple(g) for g in plan['collection_plan']['held_out_instances']}
    if any((r['task_name'], r['task_instance_id']) in reserved for r in provenance):
        raise GateError('Published dataset leaks reserved instances')
    inference = runtime / 'inference_checkpoint'
    manifest = inference / 'SHA256SUMS.inference'
    if sha(manifest) != receipt['output_checkpoint_manifest_sha256'] or read(inference / '_READY').get('manifest_sha256') != sha(manifest):
        raise GateError('Published inference manifest differs from actual training')
    for line in manifest.read_text().splitlines():
        checksum, name = line.split('  ', 1)
        if sha(inference / safe_path(name)) != checksum:
            raise GateError('Staged model changed before publication')
    output = client.key(plan['output']); files = []
    for area in ('dataset', 'training', 'inference_checkpoint'):
        for p in sorted((runtime / area).rglob('*')):
            if p.is_symlink(): raise GateError('Output must not contain symlinks')
            if p.is_file() and not p.relative_to(runtime).as_posix().startswith('training/smoke/checkpoints/'):
                files.append(p)
    files += [runtime / name for name in ('job_plan.json', 'pipeline_completion.json', 'training_gpu_preflight.json', 'download_receipt.json', 'export_storage_preflight.json', 'worker.log') if (runtime / name).is_file()]
    uploaded = []
    for path in files:
        relative = path.relative_to(runtime).as_posix(); checksum = sha(path); key = output + '/' + relative
        client.bucket.put_object_from_file(key, str(path), headers={'x-oss-forbid-overwrite': 'true', 'x-oss-meta-sha256': checksum})
        head = client.bucket.head_object(key)
        if head.content_length != path.stat().st_size: raise GateError('Published object byte count differs')
        uploaded.append(dict(path=relative, bytes=head.content_length, etag=head.etag, sha256=checksum))
    write(runtime / 'uploaded_outputs.json', uploaded)
    client.put_new(output + '/uploaded_outputs.json', runtime / 'uploaded_outputs.json')
    write(runtime / '_SUCCESS', dict(complete=True, kind=KIND, plan_fingerprint=plan['fingerprint'],
        training_receipt=receipt, output_objects=len(uploaded), official_score_improvement_verified=False))
    client.bucket.delete_object(output + '/_RUNNING')
    client.put_new(output + '/_SUCCESS', runtime / '_SUCCESS')


def worker(plan):
    validate_plan(plan)
    for source, target in (('OSS_ACCESS_KEY_ID', 'OSS_ACCESS_ID'), ('OSS_ACCESS_KEY_SECRET', 'OSS_ACCESS_KEY')):
        os.environ[target] = os.environ[source]
    os.environ['OSS_BUCKET'] = plan['bucket']; os.environ['OSS_ENDPOINT'] = plan['endpoint']
    client = OSSClient(); output = client.key(plan['output']); client.unused(output)
    runtime = Path(tempfile.mkdtemp(prefix='b1k-finetune-' + plan['run_id'] + '-', dir='/var/tmp'))
    try:
        write(runtime / 'job_plan.json', plan)
        write(runtime / '_RUNNING', dict(plan_fingerprint=plan['fingerprint'])); client.put_new(output + '/_RUNNING', runtime / '_RUNNING')
        archives = [i for i in plan['objects'] if i['path'].startswith('archives/')]
        checked_space(runtime, sum(i['bytes'] for i in archives) + 16 * GIB)
        print(json.dumps(dict(phase='restoring_policy_environment', output=plan['output'])), flush=True)
        receipts = [download(client.bucket, i, runtime) for i in archives]
        archive = runtime / 'archives/policy.tar.gz'
        with tarfile.open(archive, 'r:gz') as source: expanded = sum(m.size for m in source if m.isfile())
        others = [i for i in plan['objects'] if not i['path'].startswith('archives/')]
        checked_space(runtime, expanded + sum(i['bytes'] for i in others) + 8 * plan['checkpoint_bytes'] + 16 * GIB)
        python = restore_policy(archive, runtime / 'policy_env')
        from nebula.eval_storage import restore_tokenizer
        restore_tokenizer(runtime / 'archives/cache.tar.gz', runtime / 'cache')
        site = subprocess.check_output([str(python), '-c', 'import site;print(site.getsitepackages()[0])'], text=True).strip()
        Path(site, '00-b1k-finetune-source.pth').write_text(str(REPO / 'openpi/src') + '\n' + str(REPO / 'openpi/packages/openpi-client/src') + '\n')
        (REPO / 'openpi/.venv').symlink_to(runtime / 'policy_env', target_is_directory=True)
        print(json.dumps(dict(phase='staging_verified_collection_checkpoint_and_replay', output=plan['output'])), flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool: receipts += list(pool.map(lambda i: download(client.bucket, i, runtime), others))
        write(runtime / 'download_receipt.json', dict(plan_fingerprint=plan['fingerprint'], objects=receipts))
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'OSS_ACCESS_ID', 'OSS_ACCESS_KEY', 'OSS_ACCESS_KEY_ID', 'OSS_ACCESS_KEY_SECRET', 'AI_HUB_TOKEN')}
        env.update(OPENPI_DATA_HOME=str(runtime / 'cache'), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
        with (runtime / 'worker.log').open('x') as log:
            process = subprocess.Popen([str(python), str(Path(__file__)), 'pipeline', '--plan', str(REPO / 'nebula/finetune_job.json'), '--runtime', str(runtime)], env=env, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=30)
                    if code: raise subprocess.CalledProcessError(code, 'bounded training pipeline')
                    break
                except subprocess.TimeoutExpired:
                    try:
                        with (runtime / 'worker.log').open('rb') as latest:
                            latest.seek(max(0, latest.seek(0, 2) - 1024**2))
                            client.bucket.put_object(output + '/worker.live.log', latest.read())
                    except Exception as error:
                        print('Live-log upload failed (' + type(error).__name__ + '); training continues.', flush=True)
        print(json.dumps(dict(phase='publishing_verified_dataset_and_checkpoint', output=plan['output'])), flush=True)
        publish(client, plan, runtime)
        print(json.dumps(dict(kind=KIND, state='completed', output=plan['output'])), flush=True)
    except BaseException as error:
        write(runtime / '_FAILED', dict(kind=KIND, plan_fingerprint=plan['fingerprint'], error_type=type(error).__name__, error_message=str(error) if isinstance(error, GateError) else None))
        client.bucket.delete_object(output + '/_RUNNING')
        client.put_new(output + '/_FAILED', runtime / '_FAILED')
        if (runtime / 'worker.log').is_file(): client.bucket.put_object_from_file(output + '/worker.failed.log', str(runtime / 'worker.log'), headers={'x-oss-forbid-overwrite': 'true'})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__); commands = p.add_subparsers(dest='command', required=True)
    f = commands.add_parser('prepare')
    for name in ('panel', 'collection-plan', 'metadata', 'replay-plan', 'labels', 'destination'): f.add_argument('--' + name, type=Path, required=True)
    for name in ('collection-source', 'checkpoint', 'base-repo-id', 'policy-archive', 'cache-archive', 'output', 'run-id'): f.add_argument('--' + name, required=True)
    f.add_argument('--queue', default=os.environ.get('TRAIN_QUEUE', TRAIN_QUEUE))
    f.add_argument('--gpu-card-type', default=os.environ.get('TRAIN_GPU_CARD_TYPE', TRAIN_GPU_CARD_TYPE))
    f.add_argument('--devices', type=int, default=4); f.add_argument('--batch-size', type=int, default=8); f.add_argument('--steps', type=int, default=1000)
    s = commands.add_parser('submit'); s.add_argument('--prepared', type=Path, required=True)
    for name in ('worker', 'pipeline'):
        parser = commands.add_parser(name); parser.add_argument('--plan', type=Path, required=True)
        if name == 'pipeline': parser.add_argument('--runtime', type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == 'prepare':
            from subtask_sample_panel import load_panel
            from subtask_improvement import load_plan
            panel = load_panel(a.panel); collection, _ = load_plan(a.collection_plan, panel)
            plan = freeze(OSSClient(), panel=panel, collection_plan=collection, collection_source=a.collection_source,
                metadata=a.metadata, replay_plan=read(a.replay_plan), labels=a.labels, checkpoint=a.checkpoint,
                base_repo_id=a.base_repo_id, policy_archive=a.policy_archive, cache_archive=a.cache_archive,
                output=a.output, run_id=a.run_id, devices=a.devices, batch_size=a.batch_size, steps=a.steps, queue=a.queue, gpu_card_type=a.gpu_card_type)
            payload = bundle(plan, a.metadata, a.labels, a.destination)
            print(json.dumps(dict(prepared=str(payload.parent), plan_fingerprint=plan['fingerprint'], download_bytes=plan['download_bytes'], resources=plan['resources'])))
        elif a.command == 'submit': print(json.dumps(submit(OSSClient(), a.prepared)))
        elif a.command == 'pipeline': pipeline(read(a.plan), a.runtime)
        else: worker(read(a.plan))
    except GateError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        # SDK/subprocess error strings can contain signed URLs or private args.
        print('Fine-tuning operation failed (' + type(error).__name__ + '); inspect owned evidence, no resubmission attempted.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__': raise SystemExit(main())
