"""Real HDF5 offset tests with a byte-range fake; not simulator fidelity proof."""
import io
import json
from types import SimpleNamespace
import h5py
import numpy as np
import pytest

import audit_b1k_grasp_sources as audit


class Bucket:
    def __init__(self, data): self.data, self.calls, self.heads = data, [], 0
    def head_object(self, key):
        self.heads += 1
        return SimpleNamespace(content_length=len(self.data), etag='frozen')
    def get_object(self, key, *, byte_range, headers):
        assert headers == {'If-Match': 'frozen'}
        self.calls.append(byte_range)
        start, end = byte_range
        result = io.BytesIO(self.data[start:end + 1])
        result.headers = {'ETag': 'frozen'}
        return result


def inputs(tmp_path):
    path = tmp_path/'source.hdf5'
    with h5py.File(path,'w') as f:
        data=f.create_group('data')
        data.attrs['config']=json.dumps({'task':{'activity_name':'task'},'robots':[{'name':'r1','grasping_mode':'assisted'}]})
        data.attrs['scene_file']=json.dumps({'state':{'registry':{'object_registry':{'r1':{'controllers':{},'ag_obj_constraint_params':{'left':{},'right':{}}}}}},'objects_info':{'init_info':{'r1':{'args':{}},'toy':{'args':{'fixed_base':False}}}}})
        demo=data.create_group('demo_0');demo.attrs['transitions']='{}';demo.attrs['num_samples']=121
        demo.create_group('init_metadata')
        state=np.zeros((121,12),np.float32)
        state[70,1]=audit.MAGIC;state[80,11]=audit.MAGIC
        demo.create_dataset('state',data=state)
        demo.create_dataset('state_size',data=np.full(121,10,np.int64))
        actions=np.zeros((120,23),np.float32);actions[80,14]=1
        demo.create_dataset('action',data=actions)
    samples=[{'sample_id':str(n),'task':'task','episode_length':120,'start':n,'episode_index':0,'raw_episode_id':0,'task_index':0,'task_instance_id':0} for n in [0,70,80]]
    item={'key':'key','relative_path':'task-0000/episode_00000000.hdf5','bytes':path.stat().st_size,'etag':'frozen'}
    return Bucket(path.read_bytes()),item,samples


def test_actual_contiguous_hdf5_ranges_inspect_only_valid_selected_rows(tmp_path):
    bucket,item,samples=inputs(tmp_path)
    result=audit.audit_episode(bucket,item,samples)
    assert [r['ag_magic_present'] for r in result['samples']]==[False,True,False]
    assert [r['legacy_assisted_without_selected_row_grasp_signal'] for r in result['samples']]==[True,False,True]
    assert result['samples'][0]['reconstruction_65_frame_history_available'] is False
    assert result['samples'][-1]['reconstruction_32_closing_commands_both_arms'] is False
    assert result['receipt']['source_reheaded_unchanged'] is True and bucket.heads==2
    assert result['receipt']['whole_file_sha256_verified'] is False
    assert all(r['metric_eligible'] is False and r['runtime_grasp_restoration_verified'] is False for r in result['samples'])
    assert all(audit.digest(bucket.data[r['offset']:r['offset']+r['bytes']])==r['sha256'] for r in result['receipt']['range_receipts'])


@pytest.mark.parametrize('bad',['inventory','range_etag','short','after_change','budget'])
def test_range_pin_length_and_budget_fail_closed(tmp_path,bad):
    bucket,item,_=inputs(tmp_path)
    if bad=='inventory':item['etag']='changed'
    if bad=='budget':
        reader=audit.FrozenRanges(bucket,item,limit=1)
        with pytest.raises(ValueError):reader.exact(0,2)
        assert not bucket.calls
        return
    if bad in ['range_etag','short']:
        original=bucket.get_object
        def mutated(*args,**kwargs):
            value=original(*args,**kwargs)
            if bad=='range_etag':value.headers={'ETag':'changed'}
            else:value=io.BytesIO(value.read()[:-1]);value.headers={'ETag':'frozen'}
            return value
        bucket.get_object=mutated
    if bad=='after_change':
        reader=audit.FrozenRanges(bucket,item)
        bucket.data+=b'x'
        with pytest.raises(ValueError):reader.finish()
    else:
        with pytest.raises(ValueError):audit.FrozenRanges(bucket,item).exact(0,4)


def test_full_panel_counts_cannot_hide_failed_or_missing_episodes(tmp_path):
    bucket,item,samples=inputs(tmp_path)
    panel={'fingerprint':'fp','samples':samples,'exclusions':[{'reason':'unsupported_predicate'}]}
    inventory={'panel_fingerprint':'fp','episodes':[item]}
    result=audit.audit_panel(bucket,panel,inventory,tmp_path/'audit',workers=1)
    assert result['complete_selected_row_audit'] and result['audited_starts']==3
    assert result['scheduled_tasks']==1 and result['scheduled_episodes']==1
    assert result['selected_rows_with_ag_magic']==1
    assert result['current_two_arm_reconstruction_preconditions_met']==0
    assert result['task_coverage'][0]['audited_starts']==3
    with pytest.raises(ValueError):audit.audit_panel(bucket,panel,inventory,tmp_path/'audit')
    inventory['episodes']=[]
    with pytest.raises(ValueError):audit.audit_panel(bucket,panel,inventory,tmp_path/'missing')


def test_failures_are_incomplete_and_do_not_expose_sdk_error_text(tmp_path):
    bucket,item,samples=inputs(tmp_path)
    def failed(*args): raise RuntimeError('private signed URL or credential')
    bucket.head_object=failed
    panel={'fingerprint':'fp','samples':samples,'exclusions':[]}
    result=audit.audit_panel(bucket,panel,{'panel_fingerprint':'fp','episodes':[item]},tmp_path/'audit')
    assert not result['complete_selected_row_audit'] and result['failed_episodes']==1
    assert result['audited_starts']==0 and result['task_coverage'][0]['scheduled_starts']==3
    assert 'private' not in json.dumps(result)


def test_uses_same_last_demo_and_aligned_layout_as_real_evaluation_reader(tmp_path):
    bucket,item,samples=inputs(tmp_path)
    path=tmp_path/'source.hdf5'
    with h5py.File(path,'a') as f:
        demo=f['data'].create_group('demo_2')
        demo.attrs['transitions']='{}';demo.attrs['num_samples']=120
        demo.create_group('init_metadata')
        state=np.zeros((120,12),np.float32);state[80,1]=audit.MAGIC
        demo.create_dataset('state',data=state)
        demo.create_dataset('state_size',data=np.full(120,10,np.int64))
        demo.create_dataset('action',data=np.zeros((120,23),np.float32))
    bucket=Bucket(path.read_bytes());item['bytes']=len(bucket.data)
    result=audit.audit_episode(bucket,item,samples)
    assert result['demo_group']=='demo_2' and result['state_layout']=='aligned'
    assert [s['ag_magic_present'] for s in result['samples']]==[False,False,True]
