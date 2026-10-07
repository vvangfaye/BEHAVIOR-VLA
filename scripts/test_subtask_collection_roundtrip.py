"""Real codec/LeRobot integration on synthetic arrays, not simulator successes."""
from pathlib import Path
import copy
import dataclasses
import json
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
pytest.importorskip('lerobot.datasets',reason='Requires BEHAVIOR LeRobot dataset extra and real codecs')
from lerobot.datasets import LeRobotDataset
import test_subtask_improvement as fixture
from subtask_trajectory import CAMERAS,capture_observation,inspect_trajectory
from export_subtask_collection import export_collection,feature_contract
from verify_subtask_collection_export import verify_export
from stage_subtask_replay import freeze,stage
from test_stage_subtask_replay import Bucket
from test_finetune_subtask_collection import checkpoint
from collect_b1k_subtasks import validate_collection
from finetune_subtask_collection import prepare,environment


def observations(value=0):
    return {'robot_r1::proprio':np.full(57,value,np.float32),
            **{source:np.full((224,224,4),value,np.uint8) for source in CAMERAS.values()}}


def test_real_mixed_export_and_training_horizon_readback(tmp_path,monkeypatch,checkpoint):
    monkeypatch.setattr(fixture,'obs',observations)
    base_checkpoint,identity=checkpoint
    panel,summary=fixture.inputs();plan=fixture.plan_for(panel,summary,checkpoint_identity=identity,held_out_instances=[['task',99]])
    for i,trial in enumerate(plan['trials']):
        fixture.write_trial(tmp_path/'collection',plan,panel,trial,value=i+1,success=i<2)
    path=next((tmp_path/'collection').glob('pi05/*/trials/*/trajectory.hdf5'))
    features=feature_contract(inspect_trajectory(path,require_success=False))
    base=LeRobotDataset.create(repo_id='fixture/base',root=tmp_path/'base',fps=30,
        features=copy.deepcopy(features),robot_type='r1pro',use_videos=True)
    for ep in range(2):
        for step in range(40):
            base.add_frame({**capture_observation(observations(10+ep+step)),
                            'action':np.full(23,step,np.float32),'task':'Do task'})
        base.save_episode()
    base.finalize()
    for path in (tmp_path/'base/meta/episodes').glob('chunk-*/*.parquet'):
        rows=pq.read_table(path).to_pylist()
        for row in rows:row.update(task_index=3,task_name='task',task_instance_id=99+row['episode_index'])
        pq.write_table(pa.Table.from_pylist(rows),path)
    labels=tmp_path/'base/labels.parquet'
    pq.write_table(pa.Table.from_pylist([dict(episode_index=i,start=0,end=40,subtask='move') for i in range(2)]),labels)
    # Read actual shared-file metadata and stage only explicit replay via OSS API.
    bucket=Bucket({'original/'+p.relative_to(tmp_path/'base').as_posix():p.read_bytes()
                   for area in ('data','videos') for p in (tmp_path/'base'/area).rglob('*') if p.is_file()})
    replay_plan=freeze(tmp_path/'base','oss://fixture/original',[1],[['task',99]],bucket)
    staging=stage(replay_plan,tmp_path/'base',tmp_path/'replay_staged',bucket,reserve_bytes=0)
    assert len(staging['objects'])==4
    output=tmp_path/'export'
    report=export_collection(tmp_path/'collection',output,'fixture/exported',panel,plan,
        base_dataset=tmp_path/'replay_staged',base_segments=labels,base_episodes=[1],task_prompts={'task':'Do task'})
    assert report['episodes']==3 and report['frames']==44 and report['new_frame_fraction']==4/44
    checked=verify_export(output,'fixture/exported',collection_root=tmp_path/'collection')
    assert checked['real_lerobot_reader'] and checked['action_horizon']==32
    assert checked['aligned_source_frames_verified']==4 and checked['label_coverage']
    assert (output/'_VERIFIED_EXPORT').is_file()
    assert len(list((output/'videos').glob('**/*.mp4')))==3
    # CPU format proof only; exercise preparation through real encoded data and
    # recorded fixture trials, with dummy model bytes, without claiming GPU work.
    collection=tmp_path/'collection'
    collection_summary=validate_collection(collection,panel,plan)
    (collection/'collection_summary.json').write_text(json.dumps(collection_summary))
    (collection/'_SUCCESS').write_text(json.dumps(dict(kind='subtask_success_collection',
        complete=True,ready_for_export=True,plan_fingerprint=plan['fingerprint'])))
    (collection/'collection_contract.json').write_text(json.dumps(dict(kind='subtask_success_collection',
        run_id='cpu-format-fixture',plan_fingerprint=plan['fingerprint'],
        parent_panel_fingerprint=panel['fingerprint'],checkpoint_identity=identity)))
    training=prepare(output,collection,base_checkpoint,'base',panel,plan,tmp_path/'training',
        run_id='cpu-format-fixture',steps=128)
    assert training['export_verification']['real_lerobot_reader']
    assert training['export_verification']['frames']==44
    assert training['training_action_horizon']==32 and training['optimizer_resume'] is False
    assert environment(training,'smoke')['TRAIN_STEPS']=='2'
    assert environment(training,'full')['TRAIN_STEPS']=='128'
    # Exercise the actual training entry point, not only the export verifier.
    # Tokenization is outside this reader contract and requires model assets.
    from openpi.training import config as config_lib, data_loader
    from openpi import transforms
    monkeypatch.setattr(config_lib, 'ModelTransformFactory',
                        lambda: lambda model: transforms.Group())
    config=config_lib.get_config('pi05_b1k')
    model=dataclasses.replace(config.model,train_subtask_prediction=True,
        max_token_len=512,subtask_action_conditioning='task_and_subtask')
    factory=dataclasses.replace(config.data,repo_id='fixture/exported',dataset_root=str(output),
        episodes_index=[0,1,2],video_backend=training['video_backend'])
    data_config=factory.create(tmp_path/'reader_assets',model)
    assert data_config.dataset_kwargs['video_backend']=='pyav'
    assert data_config.dataset_kwargs['tolerance_s']==5e-3
    training_reader=data_loader.create_b1k_dataset(data_config,model.action_horizon)
    assert len(training_reader)==44
    source_reader=LeRobotDataset(repo_id='fixture/exported',root=output,download_videos=False,
        video_backend='pyav',delta_timestamps={'action':[i/30 for i in range(32)]})
    decoded_training_samples=0
    for index in [0,1,2,3,22,43]:
        actual=training_reader[index];source=source_reader[index]
        assert actual['prompt']=='Do task'
        expected_subtask='pick up' if int(actual['episode_index'])<2 else 'move'
        assert actual['subtask']==expected_subtask
        np.testing.assert_array_equal(actual['action'],source['action'])
        assert actual['action'].shape==(32,23)
        np.testing.assert_array_equal(actual['observation.state'],source['observation.state'])
        for key in CAMERAS:
            np.testing.assert_array_equal(actual[key],source[key])
        decoded_training_samples+=1
    # Source numerical alignment must be checked, even with readable videos.
    data_path=next((output/'data').glob('chunk-*/*.parquet'))
    table=pq.read_table(data_path);rows=table.to_pylist();rows[0]['action'][0]+=1
    pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),data_path)
    with pytest.raises(ValueError,match='pairing'):verify_export(output,'fixture/exported',collection_root=tmp_path/'collection')
    # Leave the persisted positive fixture intact after the deliberate corruption.
    pq.write_table(table,data_path)
    final=verify_export(output,'fixture/exported',collection_root=tmp_path/'collection')
    (tmp_path/'test_evidence.json').write_text(json.dumps(dict(
        fixture='synthetic_format_only',real_simulator_collection=False,
        negative_pairing_check_passed=True,final_positive_check=final,
        bounded_training_preparation_verified=True,actual_training_reader_verified=True,
        actual_training_reader_decoded_samples=decoded_training_samples,training_video_backend=training['video_backend'],
        real_GPU_training=False,training_plan_fingerprint=training['fingerprint']),indent=2)+'\n')
