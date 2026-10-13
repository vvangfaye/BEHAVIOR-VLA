"""Saved-field and source-role regressions; no fixture is simulator proof."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import recorded_native_state as native


def inputs():
    names=['robot','box_1','box_2','table_1']
    info={n:{'args':{'name':n,'category':n.rsplit('_',1)[0]}} for n in names}
    scene={'objects_info':{'init_info':info},'state':{'registry':{'object_registry':{'robot':{'controller_groups':{},'joint_pos':[0]*28}}}},
           'metadata':{'task':{'inst_to_name':{'agent.n.01_1':'robot','box.n.01_1':'box_1','box.n.01_2':'box_2','table.n.01_1':'table_1'}}}}
    row=[0,0,0,0,0,0,1,native.uuid('object_registry'),1,native.uuid('robot'),0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    row+=list(range(28))+[0]*28+[0]*10
    packet=[float(native.MAGIC),0,native.uuid('box_1'),0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0]
    row+=packet
    cache={'recorded_config':{'robots':[{'model':'r1pro','name':'robot','grasping_mode':'assisted'}]},'scene_file':scene,
           'states':np.asarray([row,row],np.float32),'state_sizes':np.asarray([len(row)]*2),'sha256':'a'*64}
    sample={'sample_id':'sample','start':1}
    return cache,sample


def runtime(proof):
    root=SimpleNamespace(dump_state=lambda serialized=False:{'pos':proof['root_position'],'ori':proof['root_orientation']})
    objects={p['target_object']:SimpleNamespace(name=p['target_object'],uuid=p['target_uuid'],links={'root':object()}) for p in proof['recorded_grasps']}
    held={'left':None,'right':None};params={'left':None,'right':None}
    for p in proof['recorded_grasps']:
        arm=['left','right'][p['arm_index']];held[arm]=objects[p['target_object']]
        params[arm]={'target_obj':held[arm],'target_link_name':'root','joint_type':p['joint_type'],
                     **{k:copy.deepcopy(p[k]) for k in ['parent_frame_pos','parent_frame_orn','child_frame_pos','child_frame_orn']}}
    robot=SimpleNamespace(name='robot',arm_names=['left','right'],root_link=root,get_joint_positions=lambda:proof['full_robot_joint_positions'],
                          _ag_obj_in_hand=held,_ag_obj_constraint_params=params,_ag_release_counter={'left':None,'right':None})
    return SimpleNamespace(robot=robot)


def test_source_decode_retains_literal_saved_constraint_fields_and_sparse_robot_history():
    cache,sample=inputs();proof=native.decode_start(cache,sample)
    assert proof['recorded_grasps'][0]['target_object']=='box_1'
    assert proof['full_robot_joint_positions']==list(range(28))
    assert proof['source_robot_last_written_frame']==1
    cache['states'][1,:]=0;cache['states'][1,7:9]=[native.uuid('object_registry'),0];cache['state_sizes'][1]=9
    later=native.decode_start(cache,sample)
    assert later['source_robot_last_written_frame']==0 and later['recorded_grasps']==proof['recorded_grasps']
    assert later['selected_state_sha256']!=proof['selected_state_sha256']


def sliced_inputs():
    cache,sample=inputs()
    info=lambda name: {'args':{'name':name,'category':'half_box'}}
    cache['transition_by_step']={0:{'objects':{'remove':['box_1'],
        'add':[info('half_box_1_0'),info('half_box_1_1')]}}}
    pos=int(np.flatnonzero(cache['states'][1]==native.MAGIC)[0])
    cache['states'][1,pos+2]=native.uuid('half_box_1_0')
    return cache,sample,pos


def test_transition_grasp_and_annotation_roles_use_literal_recorded_slicing_lineage():
    cache,sample,_=sliced_inputs()
    before=native.decode_start(cache,{**sample,'start':0})
    assert before['recorded_grasps'][0]['target_object']=='box_1'
    proof=native.decode_start(cache,sample)
    assert proof['recorded_grasps'][0]['target_object']=='half_box_1_0'
    proof=native.validate_runtime(proof,runtime(proof))
    binding=native.source_role_bindings(cache,proof,
        {'targets':['half_box'],'references':['table'],'operation':'placed'})
    assert binding['resolved_roles']=={'targets':['half_box_1_0'],'references':['table_1']}
    entry=binding['bindings'][0]
    assert entry['method']=='unique_initial_recorded_native_grasp'
    assert entry['source_task_instances']==['box.n.01_1']
    assert entry['source_transition']['step']==0
    assert entry['source_transition']['parent']=='box_1'
    assert len(entry['source_transition']['init_info_sha256'])==64
    assert cache['scene_file']['objects_info']['init_info'].keys()=={'robot','box_1','box_2','table_1'}


@pytest.mark.parametrize('damage',['future','removed_target','unrelated_name','wrong_category'])
def test_transition_identity_does_not_admit_future_removed_or_unrelated_objects(damage):
    cache,sample,pos=sliced_inputs()
    if damage=='future':
        cache['transition_by_step'][1]=cache['transition_by_step'].pop(0)
    elif damage=='removed_target':
        cache['states'][1,pos+2]=native.uuid('box_1')
    elif damage=='unrelated_name':
        cache['transition_by_step'][0]['objects']['add'][0]['args']['name']='unrelated_half'
        cache['states'][1,pos+2]=native.uuid('unrelated_half')
    else:
        cache['transition_by_step'][0]['objects']['add'][0]['args']['category']='other'
    if damage in ('future','removed_target'):
        with pytest.raises(ValueError,match='identity fields'):
            native.decode_start(cache,sample)
    else:
        proof=native.decode_start(cache,sample)
        with pytest.raises(ValueError,match='absent or ambiguous'):
            native.preview_source_role_bindings(cache,proof,
                {'targets':['half_box'],'references':['table'],'operation':'placed'})


@pytest.mark.parametrize('bad',['legacy','wrong_model','wrong_qcount','duplicate','unknown_uuid','arm','link','quat','nonfinite'])
def test_rejects_unproven_or_corrupt_native_payloads(bad):
    cache,sample=inputs()
    if bad=='legacy':cache['scene_file']['state']['registry']['object_registry']['robot']={'controllers':{}}
    elif bad=='wrong_model':cache['recorded_config']['robots'][0]['model']='other'
    elif bad=='wrong_qcount':cache['scene_file']['state']['registry']['object_registry']['robot']['joint_pos']=[0]*27
    else:
        pos=int(np.flatnonzero(cache['states'][1]==native.MAGIC)[0])
        if bad=='duplicate':
            packet=cache['states'][1,pos:pos+19].copy()
            cache['states']=np.pad(cache['states'],((0,0),(0,19)))
            cache['states'][1,-19:]=packet;cache['state_sizes'][1]+=19
        elif bad=='unknown_uuid':cache['states'][1,pos+2]=123
        elif bad=='arm':cache['states'][1,pos+1]=2
        elif bad=='link':cache['states'][1,pos+3]=-1
        elif bad=='quat':cache['states'][1,pos+7:pos+11]=0
        elif bad=='nonfinite':cache['states'][1,30]=np.nan
    with pytest.raises(ValueError):native.decode_start(cache,sample)


@pytest.mark.parametrize('bad',['joints','pose','target','anchor','link','joint_type','release','invented','robot_name'])
def test_pre_policy_check_rejects_changed_saved_state(bad):
    cache,sample=inputs();proof=native.decode_start(cache,sample);env=runtime(proof);robot=env.robot
    if bad=='joints':robot.get_joint_positions=lambda:[-1]*28
    elif bad=='pose':robot.root_link.dump_state=lambda serialized=False:{'pos':[.01,0,0],'ori':[0,0,0,1]}
    elif bad=='target':robot._ag_obj_in_hand['left'].name='box_2'
    elif bad=='anchor':robot._ag_obj_constraint_params['left']['child_frame_pos'][0]=.01
    elif bad=='link':robot._ag_obj_constraint_params['left']['target_link_name']='missing'
    elif bad=='joint_type':robot._ag_obj_constraint_params['left']['joint_type']='SphericalJoint'
    elif bad=='release':robot._ag_release_counter['left']=1
    elif bad=='invented':robot._ag_obj_in_hand['right']=robot._ag_obj_in_hand['left']
    elif bad=='robot_name':robot.name='other'
    with pytest.raises(ValueError):native.validate_runtime(proof,env)


def test_only_source_scope_and_saved_hold_disambiguate_generic_placement_target():
    cache,sample=inputs();proof=native.decode_start(cache,sample)
    proof=native.validate_runtime(proof,runtime(proof))
    pred={'targets':['box'],'references':['table'],'operation':'placed'}
    bound=native.source_role_bindings(cache,proof,pred)
    assert bound['resolved_roles']=={'targets':['box_1'],'references':['table_1']}
    assert bound['bindings'][0]['method']=='unique_initial_recorded_native_grasp'
    assert bound['desired_predicate_truth_used'] is False
    # Reference binding cannot use the held object to resolve an ambiguous container.
    with pytest.raises(ValueError):native.source_role_bindings(cache,proof,{'targets':['table'],'references':['box'],'operation':'placed'})
    with pytest.raises(ValueError):native.source_role_bindings(cache,proof,{**pred,'operation':'grasped'})
    with pytest.raises(ValueError):native.source_role_bindings(cache,{**proof,'recorded_native_fields_verified_before_policy':False},pred)


def test_same_category_dual_hold_remains_ambiguous_and_free_hands_do_not_invent_target():
    cache,sample=inputs();proof=native.decode_start(cache,sample);proof=native.validate_runtime(proof,runtime(proof))
    pred={'targets':['box'],'references':['table'],'operation':'placed'}
    for grasps in [[],[{'target_object':'box_1'},{'target_object':'box_2'}]]:
        with pytest.raises(ValueError):native.source_role_bindings(cache,{**proof,'recorded_grasps':grasps},pred)


def demonstration_inputs(operation='grasped'):
    cache, sample = inputs()
    cache['states'] = np.repeat(cache['states'][:1], 4, axis=0)
    cache['state_sizes'] = np.repeat(cache['state_sizes'][:1], 4)
    pos = int(np.flatnonzero(cache['states'][0] == native.MAGIC)[0])
    cache['states'][2:, pos + 2] = native.uuid('box_2')
    predicate = {'targets': ['box'], 'operation': operation}
    sample = {'sample_id': 'sample', 'start': 0, 'end': 3, 'predicate': predicate}
    cache['source_samples'] = {'sample': sample}
    if operation == 'near':
        sample['end'] = 1
        pickup = {'sample_id': 'pickup', 'start': 1, 'end': 3,
                  'predicate': {'targets': ['box'], 'operation': 'grasped'}}
        cache['source_samples']['pickup'] = pickup
    return cache, sample, predicate, pos


@pytest.mark.parametrize('operation', ['grasped', 'near'])
def test_literal_demonstration_pickup_identifies_new_object_without_changing_original_start(operation):
    cache, sample, predicate, _ = demonstration_inputs(operation)
    before = native.decode_start(cache, sample)
    binding = native.preview_source_role_bindings(cache, before, predicate)
    assert binding['contract'] == native.DEMONSTRATION_BINDING
    assert binding['resolved_roles']['targets'] == ['box_2']
    entry = binding['bindings'][0]
    assert entry['method'] == 'unique_recorded_demonstration_grasp_event'
    event = entry['source_demonstration_event']
    assert event['event_frame'] == 2 and event['newly_held_matching_objects'] == ['box_2']
    assert event['initial_held_objects'] == ['box_1']
    assert event['event_recorded_grasps'][0]['target_object'] == 'box_2'
    assert event['event_sample_id'] == ('pickup' if operation == 'near' else 'sample')
    assert binding['desired_predicate_truth_used'] is False
    assert native.decode_start(cache, sample) == before
    assert binding['runtime_native_fields_verified'] is False


@pytest.mark.parametrize('damage', ['gap', 'different_category', 'not_pickup', 'same_hold', 'outside', 'bad_packet'])
def test_demonstration_identity_requires_adjacent_same_category_and_valid_new_hold(damage):
    cache, sample, predicate, pos = demonstration_inputs('near')
    pickup = cache['source_samples']['pickup']
    if damage == 'gap': pickup['start'] = 2
    elif damage == 'different_category': pickup['predicate']['targets'] = ['table']
    elif damage == 'not_pickup': pickup['predicate']['operation'] = 'released'
    elif damage == 'same_hold': cache['states'][2, pos + 2] = native.uuid('box_1')
    elif damage == 'outside': pickup['end'] = 8
    else: cache['states'][2, pos + 2] = 123
    with pytest.raises(ValueError, match='absent or ambiguous'):
        native.preview_source_role_bindings(cache, native.decode_start(cache, sample), predicate)


def test_demonstration_with_two_new_objects_does_not_choose_an_arbitrary_target():
    cache, sample, predicate, pos = demonstration_inputs()
    cache['state_sizes'][0] = pos
    packet = cache['states'][2, pos:pos + 19].copy()
    packet[1] = 1; packet[2] = native.uuid('box_1')
    cache['states'] = np.pad(cache['states'], ((0, 0), (0, 19)))
    cache['states'][2, -19:] = packet
    cache['state_sizes'][2] += 19
    with pytest.raises(ValueError, match='absent or ambiguous'):
        native.preview_source_role_bindings(cache, native.decode_start(cache, sample), predicate)


def test_saved_scope_supports_object_category_alias_without_room_wide_guessing():
    cache,sample=inputs();proof=native.decode_start(cache,sample);proof=native.validate_runtime(proof,runtime(proof))
    meta=cache['scene_file']['metadata']['task']['inst_to_name'];meta['carryall.n.01_1']='table_1'
    cache['scene_file']['objects_info']['init_info']['table_1']['args']['category']='tote'
    result=native.source_role_bindings(cache,proof,{'targets':['box_1'],'references':['tote'],'operation':'placed'})
    assert result['resolved_roles']['references']==['table_1']
    assert result['bindings'][0]['method']=='unique_recorded_task_instance'


def test_unique_bddl_reference_does_not_resolve_ambiguous_scene_category():
    cache,sample=inputs()
    cache['scene_file']['objects_info']['init_info']['table_2']={'args':{'name':'table_2','category':'table'}}
    proof=native.decode_start(cache,sample)
    predicate={'targets':['box'],'references':['table'],'operation':'placed'}
    with pytest.raises(ValueError,match='ambiguous outside task scope'):
        native.preview_source_role_bindings(cache,proof,predicate)
    bound=native.preview_source_role_bindings(cache,proof,{**predicate,'references':['table_1']})
    assert bound['resolved_roles']['references']==['table_1']


def test_recorded_particle_scope_does_not_exclude_unrelated_physical_object_roles():
    cache,sample=inputs()
    cache['scene_file']['state']['registry']['system_registry']={'dirt':{},'stain':{}}
    cache['scene_file']['metadata']['task']['inst_to_name'].update(
        {'dirt.n.02_1':'dirt','stain.n.01_1':'stain'})
    proof=native.decode_start(cache,sample)
    pred={'targets':['box'],'references':['table'],'operation':'placed'}
    binding=native.preview_source_role_bindings(cache,proof,pred)
    assert binding['resolved_roles']=={'targets':['box_1'],'references':['table_1']}
    assert binding['desired_predicate_truth_used'] is False
    with pytest.raises(ValueError,match='absent or ambiguous'):
        native.preview_source_role_bindings(cache,proof,{**pred,'targets':['dirt']})


def test_missing_mapping_member_is_not_assumed_to_be_a_recorded_particle_system():
    cache,sample=inputs()
    cache['scene_file']['metadata']['task']['inst_to_name']['dirt.n.02_1']='dirt'
    proof=native.decode_start(cache,sample)
    with pytest.raises(ValueError,match='task instance mapping invalid'):
        native.preview_source_role_bindings(cache,proof,
            {'targets':['box'],'references':['table'],'operation':'placed'})


def test_root_quaternion_sign_equivalence_is_not_a_pose_change():
    cache,sample=inputs();proof=native.decode_start(cache,sample);env=runtime(proof)
    env.robot.root_link.dump_state=lambda serialized=False:{'pos':proof['root_position'],'ori':[-v for v in proof['root_orientation']]}
    assert native.validate_runtime(proof,env)['recorded_native_fields_verified_before_policy']



def test_source_preview_does_not_certify_simulator_restoration():
    cache,sample=inputs();proof=native.decode_start(cache,sample)
    preview=native.preview_source_role_bindings(cache,proof,{'targets':['box'],'operation':'placed'})
    assert preview['runtime_native_fields_verified'] is False
    assert 'recorded_native_fields_verified_before_policy' not in proof
    with pytest.raises(ValueError):native.source_role_bindings(cache,proof,{'targets':['box'],'operation':'placed'})


def test_monitor_preserves_annotation_and_uses_source_identity_without_scene_wide_guessing(monkeypatch):
    import subtask_completion as completion
    cache,sample=inputs();proof=native.decode_start(cache,sample);env=runtime(proof)
    proof=native.validate_runtime(proof,env)
    objects=[SimpleNamespace(name=n) for n in ('box_1','box_2','table_1','box_3')]
    env.env=SimpleNamespace(task=SimpleNamespace(object_scope={}),scene=SimpleNamespace(objects=objects))
    monkeypatch.setattr(completion.RoleStateObserver,'_state',lambda *args:False)
    monkeypatch.setattr(completion.RoleStateObserver,'_grasped',lambda *args:True)
    pred={'kind':'role_bound','targets':['box'],'references':['table'],'operation':'placed','state':'OnTop'}
    monitor=completion.RoleCompletionMonitor(env,pred,sample,source_context=(cache,proof))
    assert monitor.metric_eligible and monitor.predicate==pred
    assert monitor.observer.targets[0].name=='box_1'
    assert monitor.setup_evidence['source_role_binding']['runtime_native_fields_verified']
    ambiguous={**pred,'targets':['table'],'references':['box']}
    excluded=completion.RoleCompletionMonitor(env,ambiguous,sample,source_context=(cache,proof))
    assert not excluded.metric_eligible and excluded.setup_exclusion_reason=='predicate_unobservable'


def test_changed_recorded_joint_aborts_before_completion_or_policy(tmp_path,monkeypatch):
    import eval_b1k_subtask_samples as runner
    import subtask_completion as completion
    cache,sample=inputs();proof=native.decode_start(cache,sample);env=runtime(proof)
    env.robot.get_joint_positions=lambda:[-1]*28
    sample['subtask']='place box';sample['predicate']={'kind':'role_bound'}
    monkeypatch.setattr(runner,'restore_materialized_sample_state',lambda *args:{})
    monkeypatch.setattr(completion,'RoleCompletionMonitor',lambda *args,**kw:pytest.fail('Monitor started before native check'))
    env.step=lambda:pytest.fail('Policy used an invalid native start')
    with pytest.raises(ValueError,match='full joints'):
        runner.run_sample(SimpleNamespace(),env,SimpleNamespace(),sample,cache,{},tmp_path,'panel')
    assert not (tmp_path/'json').exists()


def native_result_inputs():
    cache,sample=inputs();proof=native.decode_start(cache,sample)
    proof=native.validate_runtime(proof,runtime(proof))
    pred={'kind':'role_bound','targets':['box'],'references':['table'],'operation':'placed','state':'OnTop'}
    sample.update(task_index=0,raw_episode_id=4,predicate=pred,task='task',episode_index=3,subtask='place box')
    plan={'schema_version':1,'kind':'subtask_state_validation','panel_fingerprint':'frozen',
          'sample_ids':[sample['sample_id']],'max_steps':256,'require_recorded_native_state':True,
          'checkpoint_step':1375000,'checkpoint_manifest_sha256':'b'*64,
          'raw_source_sha256':{'task-0000/episode_00000004.hdf5':cache['sha256']},
          'selected_state_sha256':{sample['sample_id']:proof['selected_state_sha256']}}
    result={'predicate':pred,'metric_eligible':True,'restoration':{'recorded_native_source':proof},
            'setup_evidence':{'source_role_binding':native.source_role_bindings(cache,proof,pred)},
            'steps':1,'actual_model_control_steps':1,'production_difficulty_baseline':False,
            'observed_subtasks':['place box'],'observed_subtask_sources':['provided'],
            'model_action_audit':[{'step':0,'preceding_observation_state':[0.]*61,'executed_action':[0.]*23}]}
    return sample,plan,result


@pytest.mark.parametrize('bad',['raw_hash','row_hash','sample','start','future_frame','joints','missing_actual',
                                'anchor','role','binding','unrecorded_claim','root_error'])
def test_gpu_receipt_cannot_pass_with_tampered_source_or_missing_role_evidence(bad):
    sample,plan,result=native_result_inputs();native.validate_result_proof(result,sample,plan)
    proof=result['restoration']['recorded_native_source']
    if bad=='raw_hash':proof['raw_sha256']='c'*64
    elif bad=='row_hash':proof['selected_state_sha256']='c'*64
    elif bad=='sample':proof['sample_id']='other'
    elif bad=='start':proof['source_start']=0
    elif bad=='future_frame':proof['source_robot_last_written_frame']=2
    elif bad=='joints':proof['full_robot_joint_positions']=[0]*27
    elif bad=='missing_actual':proof['actual_restored_grasps']=[]
    elif bad=='anchor':proof['actual_restored_grasps'][0]['child_frame_pos'][0]=.01
    elif bad=='role':result['predicate']={**sample['predicate'],'targets':['box_2']}
    elif bad=='binding':result['setup_evidence']={}
    elif bad=='unrecorded_claim':proof['unrecorded_physics_caches_or_timers_verified']=True
    elif bad=='root_error':proof['root_pose_errors']['root_position_max_abs_error']=.01
    with pytest.raises(ValueError):native.validate_result_proof(result,sample,plan)


@pytest.mark.parametrize('bad',['flag','checkpoint','manifest','raw_scope','row_scope','kind'])
def test_native_plan_requires_full_source_checkpoint_and_selected_row_bindings(tmp_path,bad):
    import json
    import eval_b1k_subtask_probe as probe
    sample,plan,result=native_result_inputs()
    panel={'fingerprint':'frozen','samples':[sample]}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    assert probe.load_plan(path,panel)==(plan,[sample])
    if bad=='flag':plan['require_recorded_native_state']='yes'
    elif bad=='checkpoint':plan['checkpoint_step']=True
    elif bad=='manifest':plan['checkpoint_manifest_sha256']='unknown'
    elif bad=='raw_scope':plan['raw_source_sha256']['foreign.hdf5']='a'*64
    elif bad=='row_scope':plan['selected_state_sha256']={}
    elif bad=='kind':plan['kind']=probe.GRASP_DIAGNOSTIC_KIND;plan['max_steps']=32
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):probe.load_plan(path,panel)



@pytest.mark.parametrize('bad',['nonfinite','dimension','missing','wrong_prompt','model_generated_source'])
def test_native_result_requires_actual_finite_controls_and_fixed_subtask_responses(bad):
    sample,plan,result=native_result_inputs()
    if bad=='nonfinite':result['model_action_audit'][0]['executed_action'][0]=float('nan')
    elif bad=='dimension':result['model_action_audit'][0]['preceding_observation_state']=[0.]*60
    elif bad=='missing':result['model_action_audit']=[]
    elif bad=='wrong_prompt':result['observed_subtasks']=['another subtask']
    else:result['observed_subtask_sources']=['model']
    with pytest.raises(ValueError):native.validate_result_proof(result,sample,plan)
