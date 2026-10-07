"""Fine-tuning initialization identity, held-out data and owned-run safeguards."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_subtask_collection as train
import test_subtask_improvement as fixture
from collect_b1k_subtasks import validate_collection


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / 'init_model'
    stats = {'norm_stats': {name: {'mean': [0.] * 23, 'std': [1.] * 23,
        'q01': [-1.] * 23, 'q99': [1.] * 23} for name in ('state', 'actions')}}
    files = {'params/_METADATA': b'{}', 'params/array': b'fixture model bytes',
        '_CHECKPOINT_METADATA': b'{}', 'assets/subtask_conditioning.json': b'{"action_conditioning":"task_and_subtask"}',
        'assets/base/norm_stats.json': json.dumps(stats).encode()}
    for name, contents in files.items():
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(contents)
    manifest = ''.join(hashlib.sha256(data).hexdigest() + '  ' + name + '\n' for name, data in sorted(files.items()))
    (root / 'SHA256SUMS.inference').write_text(manifest)
    identity = dict(checkpoint_step=1375000, checkpoint_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(), checkpoint_source='oss://fixture/base')
    (root / '_READY').write_text(json.dumps(dict(step=1375000, manifest_sha256=identity['checkpoint_manifest_sha256'])))
    return root, identity


def test_actual_checkpoint_hashes_cover_all_params_and_norm(checkpoint):
    root, identity = checkpoint
    entries, norm = train.verify_checkpoint(root, identity, 'base')
    assert len(entries) == 5 and norm == 'assets/base/norm_stats.json'


@pytest.mark.parametrize('damage', ['weights', 'unknown_params', 'ready', 'identity', 'symlink', 'model_format'])
def test_changed_uncovered_or_wrong_initialization_rejected(checkpoint, damage):
    root, identity = checkpoint
    if damage == 'weights': (root / 'params/array').write_bytes(b'changed')
    elif damage == 'unknown_params': (root / 'params/another-array').write_bytes(b'uncovered')
    elif damage == 'ready': (root / '_READY').write_text('{"step":0}')
    elif damage == 'identity': identity['checkpoint_step'] = 1200000
    elif damage == 'symlink':
        original = root / 'params/array'; original.unlink(); original.symlink_to(root / '_CHECKPOINT_METADATA')
    else: (root / 'model.safetensors').write_bytes(b'other format')
    with pytest.raises(ValueError): train.verify_checkpoint(root, identity, 'base')


@pytest.fixture
def exported(tmp_path, checkpoint):
    base, identity = checkpoint
    panel, summary = fixture.inputs()
    collection_plan = fixture.plan_for(panel, summary, held_out_instances=[('task', 0)], checkpoint_identity=identity)
    collection = tmp_path / 'collection'
    for i, trial in enumerate(collection_plan['trials']):
        fixture.write_trial(collection, collection_plan, panel, trial, value=i + 1)
    result = validate_collection(collection, panel, collection_plan)
    (collection / '_SUCCESS').write_text(json.dumps(dict(kind='subtask_success_collection', complete=True, ready_for_export=True, plan_fingerprint=collection_plan['fingerprint'])))
    (collection / 'collection_summary.json').write_text(json.dumps(result))
    (collection / 'collection_contract.json').write_text(json.dumps(dict(kind='subtask_success_collection', run_id='cpu-contract-fixture', plan_fingerprint=collection_plan['fingerprint'], parent_panel_fingerprint=panel['fingerprint'], checkpoint_identity=identity)))
    output = tmp_path / 'export'; (output / 'meta').mkdir(parents=True)
    report = dict(repo_id='fixture/output', new_frames=8, frames=16, episodes=2,
                  plan_fingerprint=collection_plan['fingerprint'], conditioning='task_and_subtask')
    provenance = [dict(task_name='task', task_instance_id=1, source_kind='collected'),
                  dict(task_name='task', task_instance_id=2, source_kind='original')]
    (output / 'export_report.json').write_text(json.dumps(report))
    (output / 'meta/collection_provenance.json').write_text(json.dumps(provenance))
    (output / 'retained_episodes.txt').write_text('0\n1\n')
    verified = dict(complete=True, action_horizon=32, aligned_source_frames_verified=8)
    return base, panel, collection_plan, collection, output, verified


def test_audit_recomputes_collection_and_rejects_original_replay_holdout(exported):
    _, panel, plan, collection, output, verified = exported
    train.audit_export(output, collection, panel, plan, verifier=lambda *a, **k: verified)
    p = output / 'meta/collection_provenance.json'; rows = train.read(p)
    rows[1]['task_instance_id'] = 0; p.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='reserved'):
        train.audit_export(output, collection, panel, plan, verifier=lambda *a, **k: verified)


def test_cannot_prepare_training_from_intended_but_unfinished_collection(exported):
    _, panel, plan, collection, output, verified = exported
    (collection / 'collection_summary.json').write_text('{}')
    with pytest.raises(ValueError, match='summary'):
        train.audit_export(output, collection, panel, plan, verifier=lambda *a, **k: verified)


@pytest.mark.parametrize('damage', ['wrong_horizon', 'lost_new_actions', 'missing_episode'])
def test_actual_training_data_contract_rejected(exported, damage):
    _, panel, plan, collection, output, verified = exported
    if damage == 'wrong_horizon': verified['action_horizon'] = 16
    elif damage == 'lost_new_actions': verified['aligned_source_frames_verified'] = 0
    else: (output / 'retained_episodes.txt').write_text('0\n')
    with pytest.raises(ValueError):
        train.audit_export(output, collection, panel, plan, verifier=lambda *a, **k: verified)


@pytest.fixture
def prepared(exported, tmp_path, monkeypatch):
    base, panel, collection, source, output, verification = exported
    # Unit orchestration only: real reader verification is separately required.
    monkeypatch.setattr(train, 'audit_export', lambda *a: (train.read(output / 'export_report.json'), train.read(output / 'meta/collection_provenance.json'), verification))
    return train.prepare(output, source, base, 'base', panel, collection, tmp_path / 'run', run_id='test', steps=128)


def test_preserves_base_stats_and_stops_inherited_resume_full_dataset_configuration(prepared, monkeypatch):
    for k, v in dict(RESUME='1', OVERWRITE='1', TRAIN_EPOCHS='2', TRAIN_STEPS='1688978',
        DATASET_ROOT='/original/full', SUBTASK_ACTION_CONDITIONING='subtask_only',
        EPISODES_FILE='/old/retained', VAL_EPISODES_FILE='/old/validation',
        OPENPI_PROCESS_COUNT='2', OPENPI_DATA_HOME='/owned/cache', SUBTASK_LOSS_WEIGHT='0').items():
        monkeypatch.setenv(k, v)
    env = train.environment(prepared, 'smoke')
    assert env['TRAIN_STEPS'] == '2' and env['SUBTASK_LOSS_WEIGHT'] == '1.0'
    assert env['RESUME'] == '0' and env['OVERWRITE'] == '0' and env['TRAIN_EPOCHS'] == ''
    assert env['DATASET_ROOT'] == prepared['dataset_root'] and env['EPISODES_FILE'].endswith('retained_episodes.txt')
    assert env['SUBTASK_ACTION_CONDITIONING'] == 'task_and_subtask'
    assert env['OPENPI_DATA_HOME'] == '/owned/cache' and 'OPENPI_PROCESS_COUNT' not in env and 'VAL_EPISODES_FILE' not in env
    assert env['INIT_PARAMS'] == str(Path(prepared['checkpoint']) / 'params')
    assert env['OPENPI_VERIFY_FINAL_CHECKPOINT'] == '1' and env['OPENPI_REQUIRE_FINITE_METRICS'] == '1'
    assert env['MODEL_ACTION_HORIZON'] == '32' and env['TRAIN_SEED'] == '42' and env['B1K_VIDEO_BACKEND'] == 'pyav'


def test_dataset_mutation_rejected_before_any_gpu_process(prepared, monkeypatch):
    (Path(prepared['dataset_root']) / 'retained_episodes.txt').write_text('0\n')
    monkeypatch.setattr(train.subprocess, 'check_output', lambda *a, **k: pytest.fail('GPU query before data integrity check'))
    with pytest.raises(ValueError, match='files changed'): train.run(prepared, 'smoke')


def test_existing_owned_phase_cannot_be_silently_overwritten(prepared):
    (Path(prepared['run_root']) / 'smoke').mkdir()
    with pytest.raises(ValueError, match='already exists'): train.run(prepared, 'smoke')


def test_full_training_requires_same_data_model_smoke(prepared):
    p = Path(prepared['run_root']) / 'smoke'; p.mkdir()
    (p / 'phase_receipt.json').write_text(json.dumps(dict(complete=True, training_plan_fingerprint='another plan')))
    with pytest.raises(ValueError, match='successful training'): train.run(prepared, 'full')


def test_existing_gpu_process_never_terminated_or_used(prepared, monkeypatch):
    def query(args, **kwargs):
        return '111\n' if '--query-compute-apps=pid' in args else '\n'.join(f'{i}, L20, 45000' for i in range(4))
    monkeypatch.setattr(train.subprocess, 'check_output', query)
    with pytest.raises(ValueError, match='existing compute'): train.run(prepared, 'smoke')
    assert not (Path(prepared['run_root']) / 'smoke').exists()


def test_training_failure_preserves_log_and_never_mints_success_receipt(prepared, monkeypatch):
    monkeypatch.setattr(train.subprocess, 'check_output', lambda args, **kwargs: '' if '--query-compute-apps=pid' in args else '\n'.join(f'{i}, L20, 45000' for i in range(4)))
    def fail(*a, **k): raise subprocess.CalledProcessError(1, 'training preflight')
    monkeypatch.setattr(train.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError): train.run(prepared, 'smoke')
    p = Path(prepared['run_root']) / 'smoke'
    assert (p / '_FAILED').is_file() and (p / 'train.log').is_file()
    assert not (p / '_RUNNING').exists() and not (p / 'phase_receipt.json').exists()
    norm = p / 'assets/pi05_b1k' / prepared['repo_id'] / 'norm_stats.json'
    assert train.sha(norm) == prepared['base_norm_sha256']


@pytest.fixture
def completed(prepared):
    import shutil
    root = Path(prepared['run_root']) / 'full'; root.mkdir()
    source = root / 'checkpoints/pi05_b1k/test_full' / str(prepared['steps'] - 1)
    source.mkdir(parents=True)
    base = Path(prepared['checkpoint'])
    for name in ('params', 'assets'): shutil.copytree(base / name, source / name)
    (source / 'params/array').write_bytes(b'changed model fixture bytes')
    (source / 'assets/base').rename(source / 'assets' / prepared['repo_id'])
    shutil.copyfile(base / '_CHECKPOINT_METADATA', source / '_CHECKPOINT_METADATA')
    proof = dict(restored_step=prepared['steps'], action_horizon=32, conditioning_mode='joint', global_devices=4, processes=1)
    (root / 'checkpoint_restore_verified.json').write_text(json.dumps(proof))
    (root / 'train.log').write_text('CPU staging fixture; no real GPU training\n')
    receipt = dict(complete=True, training_plan_fingerprint=prepared['fingerprint'], saved_checkpoint=str(source),
        training_run_id=prepared['training_run_id'], conditioning='task_and_subtask',
        collection_plan_fingerprint=prepared['collection_plan_fingerprint'],
        export_provenance_sha256=prepared['export_provenance_sha256'],
        init_checkpoint_manifest_sha256=prepared['init_checkpoint_identity']['checkpoint_manifest_sha256'],
        checkpoint_restore_evidence=proof, training_log_sha256=train.sha(root / 'train.log'))
    (root / 'phase_receipt.json').write_text(json.dumps(receipt))
    return prepared, root, source


def test_staging_links_frozen_data_to_new_model_and_preserves_exact_base_norm(completed, tmp_path):
    plan, root, source = completed
    out = tmp_path / 'staged'; result = train.stage_final(plan, out)
    assert result['inference_staging_complete'] and result['complete']
    assert result['output_checkpoint_manifest_sha256'] != plan['init_checkpoint_identity']['checkpoint_manifest_sha256']
    ready = train.read(out / '_READY')
    train.verify_checkpoint(out, dict(checkpoint_step=plan['steps'] - 1,
        checkpoint_manifest_sha256=ready['manifest_sha256']), plan['repo_id'])
    assert train.sha(out / 'assets' / plan['repo_id'] / 'norm_stats.json') == plan['base_norm_sha256']
    assert not (out / 'train_state').exists()
    assert train.read(Path(plan['run_root']) / 'training_receipt.json')['export_provenance_sha256'] == plan['export_provenance_sha256']


@pytest.mark.parametrize('damage', ['failed', 'running', 'restore_proof', 'log', 'norm', 'dataset', 'another_checkpoint'])
def test_failed_or_mismatched_training_cannot_publish_ready_inference(completed, tmp_path, damage):
    plan, root, source = completed
    if damage == 'failed': (root / '_FAILED').touch()
    elif damage == 'running': (root / '_RUNNING').touch()
    elif damage == 'restore_proof': (root / 'checkpoint_restore_verified.json').write_text('{}')
    elif damage == 'log': (root / 'train.log').write_text('replaced')
    elif damage == 'norm': (source / 'assets' / plan['repo_id'] / 'norm_stats.json').write_text('{}')
    elif damage == 'dataset': (Path(plan['dataset_root']) / 'retained_episodes.txt').write_text('0\n')
    else:
        receipt = train.read(root / 'phase_receipt.json'); receipt['saved_checkpoint'] = plan['checkpoint']
        (root / 'phase_receipt.json').write_text(json.dumps(receipt))
    out = tmp_path / 'staged'
    with pytest.raises(ValueError): train.stage_final(plan, out)
    assert not out.exists() and not (Path(plan['run_root']) / 'training_receipt.json').exists()
