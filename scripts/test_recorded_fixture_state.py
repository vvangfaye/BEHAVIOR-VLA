"""Fixture identity and before-policy checks; synthetic rows are not GPU proof."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import recorded_native_state as native
import recorded_fixture_state as fixture
from test_recorded_native_state import inputs, runtime


def world(operation='near', field='ToggledOn', task_scope=False):
    cache, _ = inputs()
    category = 'switch' if field == 'ToggledOn' else 'sliding_door'
    names = [category + '_1', category + '_2']
    registry = cache['scene_file']['state']['registry']['object_registry']
    base = cache['states'][0].tolist()
    rows = []
    for frame in range(4):
        row = list(base)
        for index, name in enumerate(names):
            cache['scene_file']['objects_info']['init_info'][name] = {'args': {'name': name, 'category': category}}
            initial = {'is_asleep': False, 'root_link': {'pos': [0, 0, 0], 'ori': [0, 0, 0, 1],
                       'lin_vel': [0, 0, 0], 'ang_vel': [0, 0, 0]}, 'non_kin': {}}
            packet = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            if field == 'ToggledOn':
                initial['non_kin'] = {'ToggledOn': {'value': True, 'hand_in_marker_steps': 0}}
                packet += [0 if index == 1 and frame >= 2 else 1, 0]
            else:
                initial.update(joint_pos=[0], joint_vel=[0])
                packet += [0.5 if index == 1 and frame >= 2 else 0, 0]
            registry[name] = initial
            row += [native.uuid(name), *packet]
            if task_scope:
                cache['scene_file']['metadata']['task']['inst_to_name'][category + '.n.01_' + str(index + 1)] = name
        rows.append(row)
    cache['states'] = np.asarray(rows, np.float32)
    cache['state_sizes'] = np.asarray([len(row) for row in rows])
    predicate = {'kind': 'role_bound', 'contract': 'annotation_roles_v1', 'operation': operation, 'targets': [category]}
    if operation == 'state': predicate.update(state=field, value=False)
    sample = {'sample_id': 'current', 'start': 0, 'end': 2 if operation == 'near' else 4,
              'predicate': predicate, 'episode_length': 4, 'task_index': 1, 'raw_episode_id': 10, 'subtask': 'fixture action'}
    event = {**sample, 'sample_id': 'following', 'start': 2, 'end': 4,
             'predicate': {**predicate, 'operation': 'state', 'state': field, 'value': False}}
    cache['source_samples'] = {s['sample_id']: s for s in ([sample, event] if operation == 'near' else [sample])}
    proof = native.decode_start(cache, sample)
    return cache, sample, proof, names


def binding_world(operation='state', field='ToggledOn', task_scope=False):
    cache, sample, proof, names = world(operation, field, task_scope)
    proof = native.validate_runtime(proof, runtime(proof))
    binding = native.source_role_bindings(cache, proof, sample['predicate'])
    source = binding['bindings'][0]['source_fixture_event']['initial_native_fields']
    actual = {'root_link': {'pos': source['root_position'], 'ori': source['root_orientation']},
              'joint_pos': source['joint_positions'], 'joint_vel': source['joint_velocities'],
              'non_kin': copy.deepcopy(source['non_kin'])}
    obj = SimpleNamespace(name=names[1], uuid=native.uuid(names[1]), dump_state=lambda serialized=False: actual)
    evaluator = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda key, value: obj if value == obj.name else None)))
    return cache, sample, proof, binding, evaluator, actual, obj


@pytest.mark.parametrize('operation', ['near', 'state'])
@pytest.mark.parametrize('field', ['ToggledOn', 'Open'])
@pytest.mark.parametrize('task_scope', [False, True])
def test_original_fixture_identity_from_unique_literal_field_change(operation, field, task_scope):
    cache, sample, proof, names = world(operation, field, task_scope)
    binding = native.preview_source_role_bindings(cache, proof, sample['predicate'])
    assert binding['contract'] == fixture.FIXTURE_BINDING
    assert binding['resolved_roles']['targets'] == [names[1]]
    entry = binding['bindings'][0]
    assert entry['source_task_instances'] == ([names[1].rsplit('_', 1)[0] + '.n.01_2'] if task_scope else [])
    assert entry['source_scene_object']['object_uuid'] == native.uuid(names[1])
    event = entry['source_fixture_event']
    assert event['event_frame'] == 3 and event['desired_predicate_value_used'] is False
    assert event['initial_native_fields']['source_frame'] == sample['start']
    assert [x['object'] for x in event['candidates'] if x['changed']] == [names[1]]
    if operation == 'state':
        sample['predicate']['value'] = True
        changed_goal = native.preview_source_role_bindings(cache, proof, sample['predicate'])
        assert changed_goal['resolved_roles'] == binding['resolved_roles']


@pytest.mark.parametrize('damage', ['gap', 'different_target', 'nonstate', 'two_changes', 'no_change', 'unknown_schema', 'transition'])
def test_unsupported_or_nonunique_demonstration_never_selects_a_fixture(damage):
    cache, sample, proof, names = world()
    event = cache['source_samples']['following']
    if damage == 'gap': event['start'] = 3
    elif damage == 'different_target': event['predicate']['targets'] = ['other']
    elif damage == 'nonstate': event['predicate']['operation'] = 'grasped'
    elif damage in ('two_changes', 'no_change'):
        target = names[0] if damage == 'two_changes' else names[1]
        offset = int(np.flatnonzero(cache['states'][3] == native.uuid(target))[0])
        cache['states'][3, offset + 15] = 0 if damage == 'two_changes' else 1
    elif damage == 'unknown_schema': cache['scene_file']['state']['registry']['object_registry'][names[0]]['non_kin']['Temperature'] = {'value': 20}
    else: cache['transition_by_step'] = {1: {'objects': {'remove': [names[1]]}}}
    with pytest.raises(ValueError, match='absent or ambiguous'):
        native.preview_source_role_bindings(cache, proof, sample['predicate'])


def test_sparse_fixture_packet_uses_latest_written_fields_and_ignores_grasp_uuid_references():
    cache, _, _, names = world()
    offset = int(np.flatnonzero(cache['states'][0] == native.MAGIC)[0])
    cache['states'][0, offset + 2] = native.uuid(names[1])
    assert fixture.decode_fixture_fields(cache, names[1], 0)['non_kin']['ToggledOn']['value'] is True
    cache['states'][3] = 0;cache['states'][3, 7:9] = [native.uuid('object_registry'), 0];cache['state_sizes'][3] = 9
    decoded = fixture.decode_fixture_fields(cache, names[1], 3)
    assert decoded['last_written_frame'] == 2 and decoded['non_kin']['ToggledOn']['value'] is False


@pytest.mark.parametrize('damage', ['uuid', 'position', 'joint', 'velocity', 'toggle', 'timer'])
def test_before_policy_runtime_mismatch_cannot_become_eligible(damage):
    _, _, _, binding, evaluator, actual, obj = binding_world(field='Open' if damage in ('joint', 'velocity') else 'ToggledOn')
    if damage == 'uuid': obj.uuid += 1
    elif damage == 'position': actual['root_link']['pos'] = [1, 0, 0]
    elif damage == 'joint': actual['joint_pos'] = [0.2]
    elif damage == 'velocity': actual['joint_vel'] = [0.2]
    elif damage == 'toggle': actual['non_kin']['ToggledOn']['value'] = False
    else: actual['non_kin']['ToggledOn']['hand_in_marker_steps'] = 1
    with pytest.raises(ValueError, match='differs|differ') as error:
        fixture.validate_runtime_fixture_roles(binding, evaluator)
    if damage in ('toggle', 'timer'):
        import json
        difference = json.loads(str(error.value).split(': ', 1)[1])
        assert difference['object'] == obj.name
        assert difference['expected'] == binding['bindings'][0]['source_fixture_event']['initial_native_fields']['non_kin']
        assert difference['actual'] == actual['non_kin']


def test_complete_receipt_requires_original_scene_identity_and_before_policy_fields():
    cache, sample, proof, binding, evaluator, _, _ = binding_world()
    verified = fixture.validate_runtime_fixture_roles(binding, evaluator)
    fixture.validate_fixture_result_binding(verified, sample, proof)
    relative = 'task-0001/episode_00000010.hdf5'
    result = {'predicate': sample['predicate'], 'restoration': {'recorded_native_source': proof},
              'setup_evidence': {'source_role_binding': verified}, 'metric_eligible': True,
              'steps': 1, 'actual_model_control_steps': 1, 'production_difficulty_baseline': False,
              'model_action_audit': [{'step': 0, 'preceding_observation_state': [0] * 61, 'executed_action': [0] * 23}],
              'observed_subtasks': [sample['subtask']], 'observed_subtask_sources': ['provided']}
    plan = {'max_steps': 8, 'raw_source_sha256': {relative: cache['sha256']},
            'selected_state_sha256': {sample['sample_id']: proof['selected_state_sha256']}}
    native.validate_result_proof(result, sample, plan)
    for damage in ['runtime_flag', 'source_hash', 'scene_identity', 'chosen_object', 'attestation']:
        broken = copy.deepcopy(result);b = broken['setup_evidence']['source_role_binding'];entry = b['bindings'][0]
        if damage == 'runtime_flag': b['runtime_fixture_fields_verified_before_policy'] = False
        elif damage == 'source_hash': entry['source_fixture_event']['initial_native_fields']['raw_sha256'] = 'f' * 64
        elif damage == 'scene_identity': entry['source_scene_object']['object_uuid'] += 1
        elif damage == 'chosen_object': entry['source_fixture_event']['unique_changed_object'] = 'other'
        else: b['runtime_fixture_field_attestations'][0]['actual_non_kin']['ToggledOn']['value'] = False
        with pytest.raises(ValueError): native.validate_result_proof(broken, sample, plan)
