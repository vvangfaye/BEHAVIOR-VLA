"""Check isolation of physics diagnostics from model success-rate validation."""
import json
import pytest

import diagnose_b1k_grasp_restore as diag
import eval_b1k_subtask_probe as probe
from test_subtask_probe import inputs


def test_source_motion_measures_coupling_without_using_target_labels():
    history=[]
    for frame in range(10):
        history.append({'arms':{'right':{'eef_position':[frame*.01,0,0],
            'gripper_joint_positions':[.001], 'objects':{
                'carried':{'relative_position':[0,0,.04],'relative_orientation':[0,0,0,1 if frame%2 else -1]},
                'stationary':{'relative_position':[-frame*.01,0,.04],'relative_orientation':[0,0,0,1]}}}}})
    result=diag.source_motion_summary(history)['right']
    objects={x['object']:x for x in result['objects']}
    assert result['eef_world_excursion_m']==pytest.approx(.09)
    assert objects['carried']['relative_position_max_deviation_m']<1e-10
    assert objects['carried']['relative_orientation_max_deviation_rad']==pytest.approx(0)
    assert objects['stationary']['relative_position_max_deviation_m']>.04


def test_diagnostic_plan_is_explicit_and_has_a_stricter_step_budget(tmp_path):
    panel,plan,_=inputs();plan.update(kind=diag.KIND,max_steps=32)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    assert probe.load_plan(path,panel)[0]['kind']==diag.KIND
    plan['max_steps']=33;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='limited to 32'):probe.load_plan(path,panel)
    plan['kind']='unknown';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='Unsupported'):probe.load_plan(path,panel)


def output_fixture(root,sample,panel):
    # These are validation fixtures, not actual videos or runtime grasp evidence.
    out=root/'pi05'/sample['task']/'grasp_diagnostic';(out/'videos').mkdir(parents=True,exist_ok=True)
    video=out/'videos'/(sample['sample_id']+'.mp4');video.write_bytes(b'CPU validation fixture video payload')
    def observation(frame,steps):
        return {'source_frame':frame,'physics_steps':steps,'arms':{'right':{
            'eef_position':[0,0,0],'eef_orientation':[0,0,0,1],
            'gripper_joint_positions':[.01],'gripper_joint_lower_limits':[0],'gripper_joint_upper_limits':[.04],
            'contacts_observed_after_physics':bool(steps),'objects':{'object':{
                'world_position':[0,0,.04],'world_orientation':[0,0,0,1],
                'relative_position':[0,0,.04],'relative_orientation':[0,0,0,1]}}}}}
    history=[observation(i,0) for i in range(max(0,sample['start']-diag.HISTORY_FRAMES),sample['start']+1)]
    value={**sample,'instance_id':sample['task_instance_id'],'source_start':sample['start'],
        'kind':diag.KIND,'official_benchmark':False,'metric_eligible':False,
        'panel_fingerprint':panel['fingerprint'],'raw_sha256':'a'*64,
        'diagnostic_source_sha256':'b'*64,'native_robot_source_sha256':'c'*64,
        'video_sha256':diag.runner.file_sha256(video),'model_action_requests':0,
        'physics_steps':1,'recorded_actions':[[0]*23],
        'advancement_source':'recorded_demonstration_actions',
        'candidate_selection':'recorded_scene_non_fixed_objects','movable_candidates':['object'],
        'source_history':history, 'source_motion_summary':diag.source_motion_summary(history),
        'observations_after_restore':[observation(sample['start'],0),observation(sample['start']+1,1)],
        'constraint_reconstruction_attempted':False,'exact_legacy_grasp_restoration_verified':False}
    path=out/(sample['sample_id']+'.json');path.write_text(json.dumps(value));return path,value


@pytest.mark.parametrize('bad',['scored','model_action','forced_grasp','exact_claim','zero_steps','policy','nan','video','identity','no_arm','history_gap','action_dim','summary'])
def test_diagnostic_validation_rejects_scoring_claims_and_missing_evidence(tmp_path,bad):
    panel,plan,samples=inputs();plan.update(kind=diag.KIND,max_steps=32)
    for s in samples:s['start']=5
    path,r=output_fixture(tmp_path,samples[0],panel);output_fixture(tmp_path,samples[1],panel)
    if bad=='scored':r['metric_eligible']=True
    elif bad=='model_action':r['model_action_requests']=1
    elif bad=='forced_grasp':r['constraint_reconstruction_attempted']=True
    elif bad=='exact_claim':r['exact_legacy_grasp_restoration_verified']=True
    elif bad=='zero_steps':r['physics_steps']=0
    elif bad=='policy':r['advancement_source']='model_actions'
    elif bad=='nan':r['source_motion_summary']={'position':float('nan')}
    elif bad=='video':(path.parent/'videos'/(path.stem+'.mp4')).write_bytes(b'changed')
    elif bad=='no_arm':r['observations_after_restore'][1]['arms']={}
    elif bad=='history_gap':r['source_history'].pop()
    elif bad=='action_dim':r['recorded_actions'][0]=[0]*22
    elif bad=='summary':r['source_motion_summary']['right']['eef_world_excursion_m']=1
    else:r['source_start']=99
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):probe.validate_results(tmp_path,panel,plan,samples)


def test_complete_diagnostic_still_does_not_pass_model_or_grasp_fidelity_gate(tmp_path):
    panel,plan,samples=inputs();plan.update(kind=diag.KIND,max_steps=32)
    for s in samples:s['start']=5;output_fixture(tmp_path,s,panel)
    result=probe.validate_results(tmp_path,panel,plan,samples)
    assert result['complete'] and result['recorded_action_physics_steps']==2
    assert result['metric_eligible'] is False and result['exact_legacy_grasp_restoration_verified'] is False
    assert result['model_action_requests']==0
    plan['kind']=probe.KIND
    with pytest.raises(ValueError,match='incomplete'):probe.validate_results(tmp_path,panel,plan,samples)


def test_recomputed_summary_allows_only_derived_float_rounding():
    # Actual Linux GPU / macOS CPU recomputation of identical recorded inputs.
    expected={'angle': {'relative_orientation_max_deviation_rad': 4.2146848510894035e-8},
              'eef_world_excursion_m': .09, 'observed_frames': 1,
              'source_gripper_joint_positions_first': [.01], 'object': 'recorded_object'}
    actual={**expected,'angle': {'relative_orientation_max_deviation_rad': 2.9802322387695312e-8}}
    assert diag.source_motion_summaries_match(actual,expected)
    for bad in [
        {**actual,'angle': {'relative_orientation_max_deviation_rad': 1e-4}},
        {**actual,'eef_world_excursion_m': .09000001},
        {**actual,'observed_frames': True},
        {**actual,'source_gripper_joint_positions_first': [.0100000001]},
        {**actual,'object': 'different_object'},
        {**actual,'extra_field': 0},
    ]:
        assert not diag.source_motion_summaries_match(bad,expected)
