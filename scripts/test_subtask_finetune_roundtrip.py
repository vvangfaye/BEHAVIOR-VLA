"""Real worker export/preparation on synthetic codec files, with no GPU training."""
from pathlib import Path
import json
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip('lerobot.datasets', reason='Requires actual BEHAVIOR LeRobot and video codecs')
from test_finetune_subtask_collection import checkpoint
from test_subtask_collection_roundtrip import test_real_mixed_export_and_training_horizon_readback as _build_fixture
from stage_subtask_replay import freeze
from nebula.test_subtask_finetune import Client
from nebula import subtask_finetune as job


def test_real_oss_staging_and_worker_export_training_preparation(tmp_path, monkeypatch, checkpoint):
    source = tmp_path / 'source'
    _build_fixture(source, monkeypatch, checkpoint)
    base, identity = checkpoint
    # Recover the frozen collection from actual trajectory metadata in the fixture.
    import h5py
    with h5py.File(next((source/'collection').glob('pi05/*/trials/*/trajectory.hdf5')), 'r') as f:
        one = json.loads(f.attrs['metadata'])
    import test_subtask_improvement as fixture
    panel, summary = fixture.inputs()
    collection = fixture.plan_for(panel, summary, checkpoint_identity=identity, held_out_instances=[['task',99]])
    assert collection['fingerprint'] == one['plan_fingerprint']
    objects = {'original/'+p.relative_to(source/'base').as_posix():p.read_bytes()
               for area in ('data','videos') for p in (source/'base'/area).rglob('*') if p.is_file()}
    objects.update({'collection/'+p.relative_to(source/'collection').as_posix():p.read_bytes()
                    for p in (source/'collection').rglob('*') if p.is_file()})
    objects.update({'model/'+p.relative_to(base).as_posix():p.read_bytes() for p in base.rglob('*') if p.is_file()})
    objects.update({'runtime/policy.tar.gz':b'fixture-not-restored', 'runtime/cache.tar.gz':b'fixture-not-restored'})
    client=Client(objects);monkeypatch.setenv('OSS_ENDPOINT','https://fixture.invalid')
    replay=freeze(source/'base','oss://fixture/original',[1],[['task',99]],client.bucket)
    plan=job.freeze(client,panel=panel,collection_plan=collection,collection_source='oss://fixture/collection',
        metadata=source/'base',replay_plan=replay,labels=source/'base/labels.parquet',checkpoint='oss://fixture/model',
        base_repo_id='base',policy_archive='oss://fixture/runtime/policy.tar.gz',cache_archive='oss://fixture/runtime/cache.tar.gz',
        output='oss://fixture/fresh/codec-worker',run_id='codec-worker',steps=128)
    payload=job.bundle(plan,source/'base',source/'base/labels.parquet',tmp_path/'prepared')
    runtime=tmp_path/'worker';runtime.mkdir()
    receipts=[job.download(client.bucket,i,runtime) for i in plan['objects'] if not i['path'].startswith('archives/')]
    job.write(runtime/'download_receipt.json',dict(objects=receipts,plan_fingerprint=plan['fingerprint']))
    monkeypatch.setattr(job,'REPO',payload)
    prepared=job.prepare_worker_data(plan,runtime,task_prompts={'task':'Do task'})
    assert prepared['export_verification']['real_lerobot_reader']
    assert prepared['export_verification']['frames']==44
    assert prepared['export_verification']['aligned_source_frames_verified']==4
    assert prepared['video_backend']=='pyav' and prepared['training_action_horizon']==32
    assert prepared['init_checkpoint_identity']['checkpoint_manifest_sha256']==identity['checkpoint_manifest_sha256']
    provenance=json.loads((runtime/'dataset/meta/collection_provenance.json').read_text())
    assert not any(row['task_instance_id']==99 for row in provenance)
    assert [row['source_episode_index'] for row in provenance if row['source_kind']=='original']==[1]
    assert not (runtime/'pipeline_completion.json').exists() and not (runtime/'_SUCCESS').exists()
    job.write(tmp_path/'worker_preparation_evidence.json',dict(fixture_kind='synthetic_format_only',
        real_video_codec=True,real_worker_export_and_training_preparation=True,frames=44,new_frames=4,
        original_replay_frames=40,training_horizon=32,video_backend='pyav',held_out_leak=False,
        real_GPU_training=False,real_simulator_collection=False,plan_fingerprint=plan['fingerprint']))
