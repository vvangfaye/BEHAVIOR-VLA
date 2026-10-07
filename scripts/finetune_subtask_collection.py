#!/usr/bin/env python3
"""Bounded joint fine-tuning of verified successful subtask collection data."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

from subtask_improvement import digest, load_plan
from subtask_sample_panel import load_panel

KIND = 'subtask_collection_finetuning'
REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def tree(root):
    root = Path(root)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Training artifacts cannot contain symlinks')
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def verify_checkpoint(root, identity, repo_id):
    root = Path(root)
    ready = read(root / '_READY')
    manifest = root / 'SHA256SUMS.inference'
    if ready.get('step') != identity['checkpoint_step'] or ready.get('manifest_sha256') != identity['checkpoint_manifest_sha256'] or sha(manifest) != identity['checkpoint_manifest_sha256']:
        raise ValueError('Initialization differs from the verified baseline checkpoint')
    entries = {}
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line)
        if not match:
            raise ValueError('Invalid checkpoint checksum manifest')
        expected, name = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or name in entries:
            raise ValueError('Unsafe or duplicate checkpoint manifest path')
        source = root / name
        if source.resolve() != root.resolve() / name or not source.is_file() or sha(source) != expected:
            raise ValueError('Checkpoint content changed: ' + name)
        entries[name] = expected
    norm_name = f'assets/{repo_id}/norm_stats.json'
    if not {'params/_METADATA', '_CHECKPOINT_METADATA', norm_name} <= entries.keys():
        raise ValueError('Checkpoint manifest omits required initialization artifacts')
    params = {p.relative_to(root).as_posix() for p in (root / 'params').rglob('*') if p.is_file()}
    if not params or not params <= entries.keys() or (root / 'model.safetensors').exists():
        raise ValueError('Initialization is not a completely verified JAX checkpoint')
    conditioning = read(root / 'assets/subtask_conditioning.json')
    if conditioning.get('action_conditioning') != 'task_and_subtask':
        raise ValueError('Initialization conditioning differs from legacy joint training')
    stats = read(root / norm_name)['norm_stats']
    for feature in ('state', 'actions'):
        values = stats[feature]
        for key in ('mean', 'std', 'q01', 'q99'):
            array = values[key]
            if len(array) != 23 or any(isinstance(v, bool) or not math.isfinite(v) for v in array):
                raise ValueError('Base normalization must have 23 finite transformed dimensions')
        if any(v < 0 for v in values['std']) or any(a > b for a, b in zip(values['q01'], values['q99'])):
            raise ValueError('Base normalization has invalid scale or quantiles')
    return entries, norm_name


def audit_export(root, collection_root, panel, collection_plan, verifier=None):
    from subtask_improvement import KIND as COLLECTION_KIND
    from collect_b1k_subtasks import validate_collection
    from export_subtask_collection import accepted_trials
    from verify_subtask_collection_export import verify_export
    root, collection_root = Path(root), Path(collection_root)
    marker = read(collection_root / '_SUCCESS')
    contract = read(collection_root / 'collection_contract.json')
    if marker.get('kind') != COLLECTION_KIND or marker.get('complete') is not True or marker.get('ready_for_export') is not True or marker.get('plan_fingerprint') != collection_plan['fingerprint']:
        raise ValueError('Training requires completed, exportable real collection attempts')
    if contract.get('kind') != COLLECTION_KIND or contract.get('plan_fingerprint') != collection_plan['fingerprint'] or contract.get('checkpoint_identity') != collection_plan['checkpoint_identity'] or contract.get('parent_panel_fingerprint') != panel['fingerprint'] or not contract.get('run_id'):
        raise ValueError('Collection runtime provenance differs from the frozen plan')
    actual = validate_collection(collection_root, panel, collection_plan)
    if actual != read(collection_root / 'collection_summary.json') or not actual['ready_for_export']:
        raise ValueError('Collection summary does not match real successful trajectories')
    accepted_trials(collection_root, collection_plan, {s['sample_id']: s for s in panel['samples']})
    verification = (verifier or verify_export)(root, read(root / 'export_report.json')['repo_id'], collection_root=collection_root, horizon=32)
    report, provenance = read(root / 'export_report.json'), read(root / 'meta/collection_provenance.json')
    if verification.get('complete') is not True or verification.get('action_horizon') != 32 or verification.get('aligned_source_frames_verified') != report['new_frames'] or not report['new_frames']:
        raise ValueError('Actual export read-back lacks aligned successful new controls')
    if report.get('plan_fingerprint') != collection_plan['fingerprint'] or report.get('conditioning') != 'task_and_subtask':
        raise ValueError('Export differs from the selected collection or conditioning')
    reserved = {tuple(g) for g in collection_plan['held_out_instances']}
    if not provenance or any((r['task_name'], r['task_instance_id']) in reserved for r in provenance):
        raise ValueError('Training export leaks reserved evaluation instances')
    episodes = (root / 'retained_episodes.txt').read_text().splitlines()
    if episodes != [str(i) for i in range(report['episodes'])]:
        raise ValueError('Training must use exactly the complete dense export episode set')
    return report, provenance, verification


def prepare(dataset_root, collection_root, checkpoint, base_repo_id, panel, collection_plan, run_root, *, run_id, devices=4, batch_size=8, steps=1000):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', run_id):
        raise ValueError('Training run ID must be a safe fresh name')
    if type(devices) is not int or devices not in (1, 2, 4, 8) or type(batch_size) is not int or not devices <= batch_size <= 64 or batch_size % devices or type(steps) is not int or not 128 <= steps <= 5000:
        raise ValueError('Pilot training budget must be 128..5000 steps with a divisible batch <=64')
    run_root = Path(run_root).resolve()
    if run_root.exists():
        raise ValueError('Fine-tuning requires a fresh experiment root')
    for source in (dataset_root, collection_root, checkpoint):
        path = Path(source).resolve()
        if path == run_root or path in run_root.parents or run_root in path.parents:
            raise ValueError('Experiment output must be separate from immutable input artifacts')
    identity = collection_plan.get('checkpoint_identity') or {}
    if identity.get('checkpoint_step') != 1375000:
        raise ValueError('This experiment must initialize from verified checkpoint 1375000')
    entries, norm = verify_checkpoint(checkpoint, identity, base_repo_id)
    report, provenance, verification = audit_export(dataset_root, collection_root, panel, collection_plan)
    if report['frames'] < batch_size:
        raise ValueError('Export does not contain a complete training batch')
    payload = dict(schema_version=1, kind=KIND, training_run_id=run_id, conditioning='task_and_subtask',
        collection_plan_fingerprint=collection_plan['fingerprint'], init_checkpoint_identity=identity,
        export_provenance_sha256=digest(provenance), dataset_root=str(Path(dataset_root).resolve()),
        collection_root=str(Path(collection_root).resolve()), checkpoint=str(Path(checkpoint).resolve()),
        base_repo_id=base_repo_id, repo_id='b1k_collection_' + run_id, run_root=str(run_root),
        dataset_files=tree(dataset_root), checkpoint_files=entries, base_norm_path=norm,
        base_norm_sha256=entries[norm], export_verification=verification,
        devices=devices, batch_size=batch_size, steps=steps, smoke_steps=2,
        training_action_horizon=32, execution_horizon=16, optimizer_resume=False, video_backend="pyav",
        learning_rate=dict(peak=1e-5, floor=1e-6, warmup=100, decay=steps), seed=42)
    plan = {**payload, 'fingerprint': digest(payload)}
    run_root.mkdir(parents=True)
    (run_root / 'training_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def environment(plan, phase):
    if phase not in ('smoke', 'full'):
        raise ValueError('Unknown training phase')
    root = Path(plan['run_root']) / phase
    steps = plan['smoke_steps'] if phase == 'smoke' else plan['steps']
    # Explicit values prevent inherited whole-data, resume or factorized settings.
    env = {k: v for k, v in os.environ.items() if k == 'OPENPI_DATA_HOME' or not k.startswith(('OPENPI_', 'VAL_', 'EXPECTED_', 'FRAMESAMP_', 'MEMORY_', 'DEBUGPY_', 'TRAIN_', 'LR_'))}
    env.update(CODE_ROOT=str(REPO), OPENPI_DIR=str(REPO / 'openpi'),
        DATASET_ROOT=plan['dataset_root'], EPISODES_FILE=str(Path(plan['dataset_root']) / 'retained_episodes.txt'),
        SUBTASK_PROMPTS=str(Path(plan['dataset_root']) / 'meta/subtask_segments.parquet'),
        SUBTASK_ACTION_CONDITIONING='task_and_subtask', SUBTASK_EMPTY_LABEL_POLICY='error',
        SUBTASK_PREDICTION='1', MAX_TOKEN_LEN='512', MAX_SUBTASK_LEN='64', WHOLE_TASK_PROB='0',
        SUBTASK_LOSS_WEIGHT='1.0', SUBTASK_LOSS_CHUNK_SIZE='8',
        FRAME_STRIDE='1', TRAIN_EPOCHS='', RESUME='0', OVERWRITE='0', FRAMESAMP_MODUL='0',
        MODEL_ACTION_HORIZON='32', TRAIN_SEED=str(plan['seed']), B1K_VIDEO_BACKEND=plan['video_backend'],
        EXP_NAME=plan['training_run_id'] + '_' + phase, REPO_ID=plan['repo_id'],
        NUM_GPUS=str(plan['devices']), CUDA_DEVICES=','.join(map(str, range(plan['devices']))),
        FSDP_DEVICES=str(plan['devices']), BATCH_SIZE=str(plan['batch_size']), NUM_WORKERS='2',
        TRAIN_STEPS=str(steps), SAVE_INTERVAL=str(steps), KEEP_PERIOD=str(steps), LOG_INTERVAL='1',
        INIT_PARAMS=str(Path(plan['checkpoint']) / 'params'),
        ASSETS_BASE_DIR=str(root / 'assets'), CHECKPOINT_BASE_DIR=str(root / 'checkpoints'),
        PREFLIGHT_OUTPUT_DIR=str(root / 'preflight'), PREFLIGHT_ONLY='0',
        LR_PEAK=str(plan['learning_rate']['peak']), LR_FLOOR=str(plan['learning_rate']['floor']),
        LR_WARMUP_STEPS=str(1 if phase == 'smoke' else plan['learning_rate']['warmup']), LR_DECAY_STEPS=str(steps),
        OPENPI_VERIFY_FINAL_CHECKPOINT='1', OPENPI_REQUIRE_FINITE_METRICS='1',
        OPENPI_SUCCESS_PATH=str(root / 'checkpoint_restore_verified.json'))
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def run(plan, phase):
    if plan.get('fingerprint') != digest({k: v for k, v in plan.items() if k != 'fingerprint'}):
        raise ValueError('Training plan changed')
    if tree(plan['dataset_root']) != plan['dataset_files']:
        raise ValueError('Exported training files changed after preflight')
    verify_checkpoint(plan['checkpoint'], plan['init_checkpoint_identity'], plan['base_repo_id'])
    root = Path(plan['run_root']) / phase
    if root.exists():
        raise ValueError('Training phase already exists; inspect its evidence before any retry')
    if phase == 'full':
        smoke = read(Path(plan['run_root']) / 'smoke/phase_receipt.json')
        if smoke.get('complete') is not True or smoke.get('training_plan_fingerprint') != plan['fingerprint']:
            raise ValueError('Full fine-tuning requires successful training on this data and model')
    env = environment(plan, phase)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    if len(rows) < plan['devices']:
        raise ValueError('Allocated GPU count does not match fine-tuning plan')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if processes:
        raise ValueError('Training GPUs have existing compute processes; do not reuse unrelated allocations')
    root.mkdir()
    norm = Path(env['ASSETS_BASE_DIR']) / 'pi05_b1k' / plan['repo_id'] / 'norm_stats.json'
    norm.parent.mkdir(parents=True)
    shutil.copyfile(Path(plan['checkpoint']) / plan['base_norm_path'], norm)
    if sha(norm) != plan['base_norm_sha256']:
        raise ValueError('Fresh training assets changed the base normalization')
    (root / '_RUNNING').touch()
    try:
        with (root / 'train.log').open('x') as log:
            for mode in ('check', phase):
                subprocess.run(['/bin/bash', str(REPO / 'scripts/train_pi05_subtask_b1k.sh'), mode], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        proof = read(env['OPENPI_SUCCESS_PATH'])
        expected_steps = int(env['TRAIN_STEPS'])
        if proof.get('restored_step') != expected_steps or proof.get('action_horizon') != 32 or proof.get('conditioning_mode') != 'joint' or proof.get('global_devices') != plan['devices'] or proof.get('processes') != 1:
            raise ValueError('Actual final checkpoint restore differs from bounded joint training')
        checkpoint = Path(env['CHECKPOINT_BASE_DIR']) / 'pi05_b1k' / env['EXP_NAME'] / str(expected_steps - 1)
        if not (checkpoint / 'params/_METADATA').is_file() or read(checkpoint / 'assets/subtask_conditioning.json').get('action_conditioning') != 'task_and_subtask':
            raise ValueError('Saved model lacks joint inference parameters or matching conditioning')
        if read(checkpoint / 'assets' / plan['repo_id'] / 'norm_stats.json') != read(norm):
            raise ValueError('Saved checkpoint changed the base normalization values')
        receipt = dict(complete=True, kind=KIND, phase=phase, training_plan_fingerprint=plan['fingerprint'],
            training_run_id=plan['training_run_id'], conditioning='task_and_subtask',
            collection_plan_fingerprint=plan['collection_plan_fingerprint'],
            export_provenance_sha256=plan['export_provenance_sha256'],
            init_checkpoint_manifest_sha256=plan['init_checkpoint_identity']['checkpoint_manifest_sha256'],
            saved_checkpoint=str(checkpoint), checkpoint_restore_evidence=proof,
            training_log_sha256=sha(root / 'train.log'), base_norm_sha256=plan['base_norm_sha256'],
            output_checkpoint_manifest_sha256=None, inference_staging_complete=False)
        (root / 'phase_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt
    except BaseException as error:
        (root / '_FAILED').write_text(type(error).__name__ + '\n')
        raise
    finally:
        (root / '_RUNNING').unlink(missing_ok=True)


def stage_final(plan, destination):
    """Stage actual completed inference parameters; only then link both models."""
    if plan.get('fingerprint') != digest({k: v for k, v in plan.items() if k != 'fingerprint'}):
        raise ValueError('Training plan changed')
    if tree(plan['dataset_root']) != plan['dataset_files']:
        raise ValueError('Exported training files changed after training')
    verify_checkpoint(plan['checkpoint'], plan['init_checkpoint_identity'], plan['base_repo_id'])
    if (Path(plan['run_root']) / 'training_receipt.json').exists():
        raise ValueError('Training receipt already exists; inspect instead of repeating staging')
    root = Path(plan['run_root']) / 'full'
    receipt = read(root / 'phase_receipt.json')
    proof = read(root / 'checkpoint_restore_verified.json')
    if (root / '_RUNNING').exists() or (root / '_FAILED').exists() or receipt.get('complete') is not True or receipt.get('training_plan_fingerprint') != plan['fingerprint']:
        raise ValueError('Inference staging requires completed training on this frozen plan')
    if receipt.get('checkpoint_restore_evidence') != proof or proof.get('restored_step') != plan['steps'] or proof.get('action_horizon') != 32 or proof.get('conditioning_mode') != 'joint' or proof.get('global_devices') != plan['devices'] or proof.get('processes') != 1 or receipt.get('training_log_sha256') != sha(root / 'train.log'):
        raise ValueError('Actual training completion evidence changed or is mismatched')
    expected = root / 'checkpoints/pi05_b1k' / (plan['training_run_id'] + '_full') / str(plan['steps'] - 1)
    source = Path(receipt['saved_checkpoint'])
    if source.resolve() != expected.resolve():
        raise ValueError('Saved inference parameters belong to another experiment')
    if not (source / 'params/_METADATA').is_file() or read(source / 'assets/subtask_conditioning.json').get('action_conditioning') != 'task_and_subtask':
        raise ValueError('Saved joint checkpoint is missing required inference artifacts')
    norm_source = Path(plan['checkpoint']) / plan['base_norm_path']
    if sha(norm_source) != plan['base_norm_sha256'] or read(source / 'assets' / plan['repo_id'] / 'norm_stats.json') != read(norm_source):
        raise ValueError('Trained checkpoint normalization differs from the paired base model')
    destination = Path(destination).resolve()
    if destination.exists() or source.resolve() == destination or source.resolve() in destination.parents or destination in source.resolve().parents:
        raise ValueError('Inference staging requires a fresh separate directory')
    for path in (plan['dataset_root'], plan['collection_root'], plan['checkpoint']):
        path = Path(path).resolve()
        if path == destination or path in destination.parents or destination in path.parents:
            raise ValueError('Inference output must be separate from immutable inputs')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Checkpoint staging rejects symlinks')
    required = sum(p.stat().st_size for area in ('params', 'assets') for p in (source / area).rglob('*') if p.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + 1024**3:
        raise ValueError('Insufficient disk to stage verified inference parameters plus reserve')
    destination.mkdir()
    for name in ('params', 'assets'):
        shutil.copytree(source / name, destination / name)
    shutil.copyfile(source / '_CHECKPOINT_METADATA', destination / '_CHECKPOINT_METADATA')
    # Preserve identical numeric statistics and exact runtime hash for retesting.
    shutil.copyfile(norm_source, destination / 'assets' / plan['repo_id'] / 'norm_stats.json')
    entries = tree(destination)
    manifest = ''.join(value + '  ' + name + '\n' for name, value in sorted(entries.items()))
    (destination / 'SHA256SUMS.inference').write_text(manifest)
    manifest_sha = sha(destination / 'SHA256SUMS.inference')
    if manifest_sha == plan['init_checkpoint_identity']['checkpoint_manifest_sha256']:
        raise ValueError('Trained inference artifact is identical to the base manifest')
    (destination / '_READY').write_text(json.dumps(dict(step=plan['steps'] - 1,
        restored_train_state_step=plan['steps'], manifest_sha256=manifest_sha,
        init_checkpoint_step=1375000, training_run_id=plan['training_run_id'])) + '\n')
    result = {**receipt, 'output_checkpoint_manifest_sha256': manifest_sha,
              'inference_staging_complete': True, 'staged_checkpoint': str(destination)}
    with (Path(plan['run_root']) / 'training_receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('prepare')
    for name in ('dataset-root', 'collection-root', 'checkpoint', 'panel', 'collection-plan', 'run-root'):
        freeze.add_argument('--' + name, required=True, type=Path)
    freeze.add_argument('--base-repo-id', required=True); freeze.add_argument('--run-id', required=True)
    freeze.add_argument('--devices', type=int, default=4); freeze.add_argument('--batch-size', type=int, default=8)
    freeze.add_argument('--steps', type=int, default=1000)
    execute = commands.add_parser('run'); execute.add_argument('--plan', required=True, type=Path)
    execute.add_argument('--phase', required=True, choices=('smoke', 'full'))
    staging = commands.add_parser('stage'); staging.add_argument('--plan', required=True, type=Path)
    staging.add_argument('--destination', required=True, type=Path)
    a = p.parse_args()
    if a.command == 'prepare':
        panel = load_panel(a.panel); collection, _ = load_plan(a.collection_plan, panel)
        result = prepare(a.dataset_root, a.collection_root, a.checkpoint, a.base_repo_id, panel, collection,
            a.run_root, run_id=a.run_id, devices=a.devices, batch_size=a.batch_size, steps=a.steps)
        print(json.dumps({k: result[k] for k in ('fingerprint', 'training_run_id', 'devices', 'batch_size', 'steps')}))
    elif a.command == 'run':
        print(json.dumps(run(read(a.plan), a.phase)))
    else:
        print(json.dumps(stage_final(read(a.plan), a.destination)))


if __name__ == '__main__':
    main()
