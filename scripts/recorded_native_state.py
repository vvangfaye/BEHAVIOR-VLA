"""Independent saved-field checks and source-grounded annotation role binding.

These checks verify recorded native fields, not unrecorded PhysX caches/timers.
Legacy controller layouts are deliberately outside this protocol.
"""
from __future__ import annotations
import hashlib
import json
import re
import numpy as np
from recorded_fixture_state import FIXTURE_BINDING, METHOD as FIXTURE_METHOD

CONTRACT = 'recorded_native_fields_v1'
BINDING = 'recorded_task_scope_and_initial_native_grasp_v1'
DEMONSTRATION_BINDING = 'recorded_task_scope_and_demonstration_identity_v2'
MAGIC = np.float32(100123456)


def uuid(name):
    return int(np.float32(int(hashlib.md5(name.encode()).hexdigest(), 16) % 10**8))


def is_native_recording(cache):
    robots = cache.get('recorded_config', {}).get('robots', [])
    if len(robots) != 1:
        return False
    state = cache.get('scene_file', {}).get('state', {}).get('registry', {}).get('object_registry', {}).get(robots[0].get('name'), {})
    return 'controller_groups' in state and 'controllers' not in state


def _recorded_object_topology(cache, frame):
    """Read identities from the original scene and recorded transitions before a row.

    Transitions execute after their recorded frame. Slicing descendants are tied
    to a removed task participant using SlicingRule's recorded name/category
    convention, without consulting any predicate truth or rollout.
    """
    infos = dict(cache['scene_file']['objects_info']['init_info'])
    mapping = cache['scene_file'].get('metadata', {}).get('task', {}).get('inst_to_name', {})
    lineage = {}
    transitions = cache.get('transition_by_step')
    if transitions is None:
        transitions = {int(k): v for k, v in cache.get('transitions', {}).items()}
    for step, transition in sorted(transitions.items()):
        if step >= frame:
            break
        objects = transition.get('objects', {})
        removed = {name: infos.get(name) for name in objects.get('remove', [])}
        previous_lineage = dict(lineage)
        for name in removed:
            infos.pop(name, None)
            lineage.pop(name, None)
        for info in objects.get('add', []):
            args = info.get('args', {})
            name = args.get('name')
            if not isinstance(name, str) or not name:
                continue  # System templates do not represent graspable objects.
            if name in infos:
                raise ValueError('Recorded transition adds an existing object identity')
            infos[name] = info
            for parent, parent_info in removed.items():
                if parent_info is None:
                    continue
                parent_category = parent_info.get('args', {}).get('category')
                if (not re.fullmatch('half_' + re.escape(parent) + r'_\d+', name)
                        or args.get('category') != 'half_' + str(parent_category)):
                    continue
                instances = sorted(k for k, v in mapping.items() if v == parent)
                if not instances and parent in previous_lineage:
                    instances = previous_lineage[parent]['source_task_instances']
                if instances:
                    lineage[name] = {'step': step, 'object': name, 'parent': parent,
                        'category': args['category'], 'source_task_instances': instances,
                        'init_info_sha256': hashlib.sha256(json.dumps(info, sort_keys=True,
                            separators=(',', ':')).encode()).hexdigest()}
    return infos, lineage


def decode_start(cache, sample):
    if not is_native_recording(cache):
        raise ValueError('Recorded native verification requires controller_groups schema')
    robot = cache['recorded_config']['robots'][0]
    if str(robot.get('model') or robot.get('type')).lower() != 'r1pro' or robot.get('grasping_mode') not in ('assisted', 'sticky'):
        raise ValueError('Recorded native protocol requires one assisted/sticky R1Pro')
    name = robot['name']; scene = cache['scene_file']
    count = len(scene['state']['registry']['object_registry'][name]['joint_pos'])
    if count != 28:
        raise ValueError('Recorded R1Pro joint layout differs')
    start = sample['start']; robot_row = None
    # Sparse frames may omit an unchanged robot. Its latest explicit row is
    # the saved robot state; absence in one row must not erase a prior grasp.
    for frame in range(start, -1, -1):
        row = np.asarray(cache['states'][frame, :int(cache['state_sizes'][frame])])
        registry = np.flatnonzero(row == np.float32(uuid('object_registry')))
        if len(registry) != 1:
            raise ValueError('Recorded object registry boundary is ambiguous')
        begin = int(registry[0]) + 2
        hits = np.flatnonzero(row[begin:] == np.float32(uuid(name)))
        if len(hits) > 1:
            raise ValueError('Recorded robot boundary is ambiguous')
        if len(hits):
            robot_row = (frame, row, begin + int(hits[0])); break
    if robot_row is None:
        raise ValueError('No explicit recorded robot state before selected start')
    frame, row, offset = robot_row
    infos, _ = _recorded_object_topology(cache, frame)
    names = {uuid(n): n for n in infos}
    if len(names) != len(infos):
        raise ValueError('Recorded object UUIDs collide')
    if row.dtype != np.dtype('float32') or offset + 15 + 28 > len(row) or row[offset + 1] not in (0, 1):
        raise ValueError('Recorded robot numeric schema differs')
    q = row[offset + 15:offset + 43]
    if not np.isfinite(row).all():
        raise ValueError('Nonfinite recorded native state')
    packets = []
    for pos in np.flatnonzero(row == MAGIC):
        pos = int(pos)
        if pos <= offset + 43 or pos + 19 > len(row):
            raise ValueError('Native grasp packet is outside its supported robot payload')
        values = row[pos:pos + 19]
        arm, target, link = int(values[1]), int(values[2]), int(values[3])
        if values[1] != arm or arm not in (0, 1) or values[2] != target or target not in names or values[3] != link or link < 0 or values[18] not in (0, 1):
            raise ValueError('Native grasp identity fields invalid')
        if abs(np.linalg.norm(values[7:11]) - 1) > 1e-4 or abs(np.linalg.norm(values[14:18]) - 1) > 1e-4:
            raise ValueError('Native grasp quaternion invalid')
        packets.append({'arm_index': arm, 'target_uuid': target, 'target_object': names[target],
                        'target_link_index': link, 'parent_frame_pos': values[4:7].tolist(),
                        'parent_frame_orn': values[7:11].tolist(), 'child_frame_pos': values[11:14].tolist(),
                        'child_frame_orn': values[14:18].tolist(),
                        'joint_type': 'SphericalJoint' if values[18] else 'FixedJoint',
                        'source_float_offset': pos})
    if len(packets) > 2 or len({p['arm_index'] for p in packets}) != len(packets):
        raise ValueError('Duplicate/unsupported native grasp arms')
    selected = np.asarray(cache['states'][start, :int(cache['state_sizes'][start])])
    return {'contract': CONTRACT, 'sample_id': sample['sample_id'], 'raw_sha256': cache['sha256'],
            'source_start': start, 'source_robot_last_written_frame': frame,
            'selected_state_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
            'robot_name': name, 'robot_uuid': uuid(name), 'source_robot_offset': offset,
            'full_robot_joint_positions': q.tolist(), 'root_position': row[offset+2:offset+5].tolist(),
            'root_orientation': row[offset+5:offset+9].tolist(), 'recorded_grasps': packets,
            'unrecorded_physics_caches_or_timers_verified': False}


def _list(value):
    return value.tolist() if hasattr(value, 'tolist') else list(value)


def validate_runtime(proof, evaluator):
    robot = evaluator.robot
    if robot.name != proof['robot_name'] or list(robot.arm_names) != ['left', 'right']:
        raise ValueError('Runtime robot identity/arm order differs from source')
    if _list(robot.get_joint_positions()) != proof['full_robot_joint_positions']:
        raise ValueError('Runtime full joints differ from saved source joints')
    root = robot.root_link.dump_state(serialized=False)
    position = np.asarray(_list(root['pos']))
    orientation = np.asarray(_list(root['ori']))
    expected_orientation = np.asarray(proof['root_orientation'])
    errors = {'root_position_max_abs_error': float(np.max(np.abs(position - proof['root_position']))),
              'root_orientation_max_abs_error': float(min(np.max(np.abs(orientation - expected_orientation)),
                                                         np.max(np.abs(orientation + expected_orientation))))}
    if not all(np.isfinite(v) and v <= 1e-5 for v in errors.values()):
        raise ValueError('Runtime root pose differs from saved source pose')
    expected = {p['arm_index']: p for p in proof['recorded_grasps']}
    actual = []
    for idx, arm in enumerate(robot.arm_names):
        held, params = robot._ag_obj_in_hand[arm], robot._ag_obj_constraint_params[arm]
        if robot._ag_release_counter[arm] is not None:
            raise ValueError('Native restart has a release window instead of its saved constraint')
        if idx not in expected:
            if held is not None or params:
                raise ValueError('Native restart invented an unrecorded held object')
            continue
        p = expected[idx]
        if held is None or held.name != p['target_object'] or int(held.uuid) != p['target_uuid'] or not params or params['target_obj'] is not held:
            raise ValueError('Native restart held-object identity differs from saved source')
        links = sorted(held.links)
        if p['target_link_index'] >= len(links) or params['target_link_name'] != links[p['target_link_index']] or params['joint_type'] != p['joint_type']:
            raise ValueError('Native restart link/joint type differs from saved source')
        for key in ['parent_frame_pos', 'parent_frame_orn', 'child_frame_pos', 'child_frame_orn']:
            if _list(params[key]) != p[key]:
                raise ValueError('Native restart changed a saved local grasp anchor')
        actual.append({'arm': arm, 'target_object': held.name, 'target_uuid': int(held.uuid),
                       'target_link_name': params['target_link_name'], 'joint_type': params['joint_type'],
                       **{key: _list(params[key]) for key in ['parent_frame_pos', 'parent_frame_orn', 'child_frame_pos', 'child_frame_orn']}})
    return {**proof, 'recorded_native_fields_verified_before_policy': True,
            'full_robot_joints_match': True, 'root_pose_errors': errors, 'actual_restored_grasps': actual}


def source_role_bindings(cache, proof, predicate):
    """Bind actual verified starts without consulting desired predicate truth."""
    from legacy_history_recovery import CONTRACT as HISTORY_CONTRACT
    if proof.get('contract') == HISTORY_CONTRACT:
        from legacy_history_recovery import source_role_bindings as history_bindings
        return history_bindings(cache, proof, predicate)
    if proof.get('recorded_native_fields_verified_before_policy') is not True:
        raise ValueError('Source role binding requires verified native initial fields')
    return {**preview_source_role_bindings(cache, proof, predicate),
            'runtime_native_fields_verified': True}


def _demonstration_grasp_identity(cache, proof, predicate, candidates):
    """Resolve a category from a literal recorded grasp, never a policy rollout.

    A pickup uses its own final recorded frame. Navigation may use only the
    immediately adjacent pickup annotation for the same category. Subtracting
    the initial hold distinguishes a new pickup from an already carried object.
    Missing, two-object, or non-adjacent evidence remains ambiguous.
    """
    samples = cache.get('source_samples', {})
    sample = samples.get(proof['sample_id'])
    if (not sample or sample.get('start') != proof['source_start']
            or sample.get('predicate') != predicate
            or predicate['operation'] not in ('grasped', 'near')):
        return None
    event_sample = sample
    if predicate['operation'] == 'near':
        following = [s for s in samples.values() if s['start'] >= sample['end']]
        if not following:
            return None
        event_sample = min(following, key=lambda s: (s['start'], s['end'], s['sample_id']))
        if (event_sample['start'] != sample['end']
                or event_sample['predicate']['operation'] != 'grasped'
                or event_sample['predicate'].get('targets') != predicate.get('targets')):
            return None
    frame = event_sample['end'] - 1
    if not proof['source_start'] <= frame < len(cache['states']):
        return None
    try:
        event = decode_start(cache, {**event_sample, 'start': frame})
    except ValueError:
        return None  # An unverified later packet cannot establish an identity.
    initial = {p['target_object'] for p in proof['recorded_grasps']}
    final = {p['target_object'] for p in event['recorded_grasps']}
    newly_held = (final - initial) & candidates
    if len(newly_held) != 1:
        return None
    name = next(iter(newly_held))
    return name, {'source_sample_id': sample['sample_id'],
        'event_sample_id': event_sample['sample_id'],
        'event_annotation_start': event_sample['start'],
        'event_annotation_end': event_sample['end'], 'event_frame': frame,
        'raw_sha256': event['raw_sha256'],
        'event_selected_state_sha256': event['selected_state_sha256'],
        'event_robot_last_written_frame': event['source_robot_last_written_frame'],
        'initial_held_objects': sorted(initial),
        'event_recorded_grasps': event['recorded_grasps'],
        'newly_held_matching_objects': [name]}


def preview_source_role_bindings(cache, proof, predicate, *, demonstration_identity_resolver=None):
    """Audit source identities only; this does not certify simulator restoration.

    No desired predicate truth or model actions are consulted. A recorded
    demonstration grasp can establish an otherwise ambiguous annotation identity;
    it never supplies a rollout observation or a completion result.
    """
    if proof['raw_sha256'] != cache['sha256']:
        raise ValueError('Source role binding differs from source file hash')
    scene = cache['scene_file']
    mapping = scene.get('metadata', {}).get('task', {}).get('inst_to_name')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Recorded task instance mapping absent')
    initial_infos = scene['objects_info']['init_info']
    registry = scene.get('state', {}).get('registry', scene.get('state', {}))
    recorded_systems = registry.get('system_registry', {})
    system_names = set(recorded_systems) if isinstance(recorded_systems, dict) else set()
    # BDDL scope includes particle systems as well as physical objects. A saved
    # system must not invalidate unrelated object roles; it is not an object
    # candidate for the physical role bindings below.
    if any(not isinstance(k, str) or not isinstance(v, str)
           or (v not in initial_infos and v not in system_names)
           for k, v in mapping.items()):
        raise ValueError('Recorded task instance mapping invalid')
    infos, lineage = _recorded_object_topology(cache, proof['source_start'])
    held = {p['target_object'] for p in proof['recorded_grasps']}
    resolved, evidence = {}, []
    for role in ('targets', 'references'):
        resolved[role] = []
        for identifier in predicate.get(role, []):
            exact = {v for k, v in mapping.items() if v in infos and identifier in (k, v)}
            candidates = exact or {v for k, v in mapping.items() if v in infos
                if identifier in (k.split('.n.', 1)[0], re.sub(r'_\d+$', '', v), infos[v]['args'].get('category'))}
            if not exact:
                candidates |= {name for name, birth in lineage.items()
                    if identifier in (name, infos[name]['args'].get('category'))}
                if role == 'references':
                    scene_candidates = {name for name, info in infos.items()
                        if identifier in (re.sub(r'_\d+$', '', name), info['args'].get('category'))}
                    if scene_candidates - candidates:
                        raise ValueError('Source annotation reference absent or ambiguous outside task scope')
            method = 'unique_recorded_task_instance'
            source_event = None
            fixture_event = None
            if (role == 'targets' and predicate['operation'] in ('placed', 'released')
                    and (len(candidates) > 1 or any(name in lineage for name in candidates))):
                candidates &= held; method = 'unique_initial_recorded_native_grasp'
            if role == 'targets' and len(candidates) > 1:
                event = (demonstration_identity_resolver or _demonstration_grasp_identity)(cache, proof, predicate, candidates)
                if event is not None:
                    name, source_event = event
                    candidates = {name}; method = 'unique_recorded_demonstration_grasp_event'
            if role == 'targets' and len(candidates) != 1:
                from recorded_fixture_state import demonstration_fixture_identity
                event = demonstration_fixture_identity(cache, proof, predicate, identifier)
                if event is not None:
                    name, fixture_event = event
                    candidates = {name}; method = FIXTURE_METHOD
            if len(candidates) != 1:
                raise ValueError('Source annotation role absent or ambiguous')
            name = next(iter(candidates)); resolved[role].append(name)
            entry = {'role': role, 'annotation_id': identifier, 'object': name, 'method': method,
                     'source_task_instances': sorted(k for k, v in mapping.items() if v == name)}
            if source_event is not None:
                entry['source_demonstration_event'] = source_event
            if fixture_event is not None:
                entry['source_fixture_event'] = fixture_event
                fields = fixture_event['initial_native_fields']
                entry['source_scene_object'] = {k: fields[k] for k in ('object', 'object_uuid', 'init_info_sha256')}
            if name in lineage:
                # A descendant is not a newly invented BDDL instance: preserve
                # the recorded parent's instances and the exact creation event.
                entry.update(source_task_instances=lineage[name]['source_task_instances'],
                             source_transition=lineage[name])
            evidence.append(entry)
    demonstration_used = any('source_demonstration_event' in e for e in evidence)
    fixture_used = any('source_fixture_event' in e for e in evidence)
    contract = FIXTURE_BINDING if fixture_used else DEMONSTRATION_BINDING if demonstration_used else BINDING
    return {'contract': contract, 'raw_sha256': proof['raw_sha256'],
            'source_state_sha256': proof['selected_state_sha256'], 'sample_id': proof['sample_id'],
            'resolved_roles': resolved, 'bindings': evidence, 'desired_predicate_truth_used': False,
            'runtime_native_fields_verified': False}



def validate_result_proof(result, sample, plan):
    """Require the before-policy source proof and original frozen annotation roles.

    This validates the receipt contract; independent raw-file comparison remains
    necessary before interpreting physical restoration as verified evidence.
    """
    proof = result.get('restoration', {}).get('recorded_native_source', {})
    relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
    if (proof.get('contract') != CONTRACT or
            proof.get('recorded_native_fields_verified_before_policy') is not True or
            proof.get('full_robot_joints_match') is not True or
            proof.get('unrecorded_physics_caches_or_timers_verified') is not False or
            proof.get('sample_id') != sample['sample_id'] or
            proof.get('source_start') != sample['start'] or
            proof.get('raw_sha256') != plan['raw_source_sha256'][relative] or
            proof.get('selected_state_sha256') != plan['selected_state_sha256'][sample['sample_id']] or
            result.get('predicate') != sample['predicate']):
        raise ValueError('Recorded native result lacks its frozen before-policy source proof')
    frame = proof.get('source_robot_last_written_frame')
    joints = np.asarray(proof.get('full_robot_joint_positions', []))
    errors = proof.get('root_pose_errors', {})
    if (type(frame) is not int or not 0 <= frame <= sample['start'] or
            joints.shape != (28,) or not np.isfinite(joints).all() or
            set(errors) != {'root_position_max_abs_error', 'root_orientation_max_abs_error'} or
            any(not np.isfinite(v) or not 0 <= v <= 1e-5 for v in errors.values())):
        raise ValueError('Recorded native result has invalid source robot evidence')
    saved = proof.get('recorded_grasps', [])
    actual = proof.get('actual_restored_grasps', [])
    if (not isinstance(saved, list) or not isinstance(actual, list) or
            len(saved) != len(actual) or len(saved) > 2 or
            len({p.get('arm_index') for p in saved}) != len(saved) or
            len({p.get('arm') for p in actual}) != len(actual)):
        raise ValueError('Recorded native result has invalid grasp counts')
    by_arm = {p['arm']: p for p in actual}
    for packet in saved:
        arm = {0: 'left', 1: 'right'}.get(packet.get('arm_index'))
        restored = by_arm.get(arm, {})
        keys = ['target_object', 'target_uuid', 'joint_type', 'parent_frame_pos',
                'parent_frame_orn', 'child_frame_pos', 'child_frame_orn']
        if arm is None or any(restored.get(k) != packet.get(k) for k in keys) or not restored.get('target_link_name'):
            raise ValueError('Recorded native result changed saved grasp fields')
    binding = result.get('setup_evidence', {}).get('source_role_binding')
    if result.get('metric_eligible') is True or binding is not None:
        if (not isinstance(binding, dict) or binding.get('contract') not in (BINDING, DEMONSTRATION_BINDING, FIXTURE_BINDING) or
                binding.get('runtime_native_fields_verified') is not True or
                binding.get('desired_predicate_truth_used') is not False or
                binding.get('raw_sha256') != proof['raw_sha256'] or
                binding.get('source_state_sha256') != proof['selected_state_sha256'] or
                binding.get('sample_id') != sample['sample_id']):
            raise ValueError('Recorded native result lacks source role binding evidence')
        expected = [(role, identifier) for role in ('targets', 'references')
                    for identifier in sample['predicate'].get(role, [])]
        entries = binding.get('bindings', [])
        if [(e.get('role'), e.get('annotation_id')) for e in entries] != expected:
            raise ValueError('Recorded native result changed original annotation roles')
        for role in ('targets', 'references'):
            names = [e.get('object') for e in entries if e['role'] == role]
            if binding.get('resolved_roles', {}).get(role) != names or any(not n for n in names):
                raise ValueError('Recorded native result has inconsistent role identities')
        if any(e.get('method') not in ('unique_recorded_task_instance', 'unique_initial_recorded_native_grasp',
                                     'unique_recorded_demonstration_grasp_event', FIXTURE_METHOD)
               or (e.get('method') != FIXTURE_METHOD and not e.get('source_task_instances')) for e in entries):
            raise ValueError('Recorded native result lacks source task identity evidence')
        demonstration_entries = [e for e in entries if e.get('method') == 'unique_recorded_demonstration_grasp_event']
        fixture_entries = [e for e in entries if e.get('method') == FIXTURE_METHOD]
        expected_contract = FIXTURE_BINDING if fixture_entries else DEMONSTRATION_BINDING if demonstration_entries else BINDING
        if binding['contract'] != expected_contract:
            raise ValueError('Recorded native result changed its annotation identity contract')
        if fixture_entries:
            from recorded_fixture_state import validate_fixture_result_binding
            validate_fixture_result_binding(binding, sample, proof)
        for entry in demonstration_entries:
            event = entry.get('source_demonstration_event', {})
            frame, end = event.get('event_frame'), event.get('event_annotation_end')
            if (type(frame) is not int or type(end) is not int or frame != end - 1
                    or not sample['start'] <= frame < sample['episode_length']
                    or event.get('source_sample_id') != sample['sample_id']
                    or event.get('raw_sha256') != proof['raw_sha256']
                    or event.get('newly_held_matching_objects') != [entry['object']]
                    or entry['object'] in event.get('initial_held_objects', [])):
                raise ValueError('Recorded native result lacks its literal demonstration identity event')
        for entry in entries:
            transition = entry.get('source_transition')
            if transition is None:
                continue
            if (not isinstance(transition, dict) or type(transition.get('step')) is not int
                    or not 0 <= transition['step'] < sample['start']
                    or transition.get('object') != entry['object']
                    or transition.get('source_task_instances') != entry['source_task_instances']
                    or not isinstance(transition.get('parent'), str)
                    or not re.fullmatch('half_' + re.escape(transition['parent']) + r'_\d+', entry['object'])
                    or not re.fullmatch(r'[0-9a-f]{64}', transition.get('init_info_sha256', ''))):
                raise ValueError('Recorded native role has invalid source transition lineage')

    steps = result.get('steps')
    rows = result.get('model_action_audit')
    if (type(steps) is not int or not 0 <= steps <= plan['max_steps'] or
            not isinstance(rows, list) or len(rows) != steps or
            result.get('actual_model_control_steps') != steps or
            [r.get('step') for r in rows] != list(range(steps)) or
            result.get('production_difficulty_baseline') is not plan.get('production_difficulty_baseline', False)):
        raise ValueError('Recorded native model action audit is incomplete')
    for row in rows:
        state = np.asarray(row.get('preceding_observation_state', []))
        action = np.asarray(row.get('executed_action', []))
        if state.shape != (61,) or action.shape != (23,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('Recorded native model actions require finite 61-state and 23-control values')
    if result.get('metric_eligible') is True:
        if (steps == 0 or result.get('observed_subtasks') != [sample['subtask']] * steps or
                result.get('observed_subtask_sources') != ['provided'] * steps):
            raise ValueError('Recorded native model lacks provided-subtask policy response evidence')
    elif steps != 0:
        raise ValueError('Excluded recorded native setup executed model controls')


def validate_baseline_result(result, sample, raw_provenance, max_steps):
    """Require production controls and native fields bound to raw preflight."""
    if not isinstance(raw_provenance, dict):
        raise ValueError('Production native evidence requires independent raw preflight')
    for key in ('sha256', 'selected_state_sha256'):
        if not isinstance(raw_provenance.get(key), str) or not re.fullmatch('[0-9a-f]{64}', raw_provenance[key]):
            raise ValueError('Production native evidence requires whole-file and selected-row hashes')
    relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
    validate_result_proof(result, sample, {
        'max_steps': max_steps, 'production_difficulty_baseline': True,
        'raw_source_sha256': {relative: raw_provenance['sha256']},
        'selected_state_sha256': {sample['sample_id']: raw_provenance['selected_state_sha256']},
    })
