"""Evaluate original old-format starts with disclosed physical-history recovery."""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

CONTRACT = 'recorded_actions_before_start_native_grasp_recovery_v1'
BINDING = 'recorded_task_scope_with_physical_history_grasp_identity_v1'
METHOD = 'recorded_full_states_with_native_physical_action_prefix_v1'


def enabled():
    return os.environ.get('B1K_LEGACY_HISTORY_RECOVERY') == CONTRACT


def _counter_state(robot):
    return {name: copy.deepcopy(getattr(robot, name)) for name in
            ('_ag_grasp_counter', '_ag_release_counter')}


def _grasps(robot_state, state):
    from recorded_native_state import uuid
    objects = state[0]['registry']['object_registry']
    result = []
    for index, arm in enumerate(('left', 'right')):
        params = robot_state.get('ag_obj_constraint_params', {}).get(arm)
        if not params:
            continue
        name = params['target_obj']
        if name not in objects:
            raise ValueError('History recovery references a missing object')
        result.append({'arm_index': index, 'target_object': name, 'target_uuid': uuid(name),
                       'target_link_name': params['target_link_name'], 'joint_type': params['joint_type'],
                       **{key: params[key].tolist() for key in
                          ('parent_frame_pos', 'parent_frame_orn', 'child_frame_pos', 'child_frame_orn')}})
    return result


def materialize(evaluator, cache, samples):
    """Replay each recorded action once, saving starts before their own action."""
    import h5py
    import torch
    import omnigibson as og
    from omnigibson.macros import gm
    import eval_b1k_subtask_samples as runner
    from diagnose_b1k_grasp_history import carry_native_grasp, cpu_tree

    if not enabled() or not cache['grasp_serialization_audit']['legacy_missing_assisted_grasp_state']:
        raise ValueError('Physical-history evaluation requires its explicit old-format protocol')
    runner._validate_recorded_episode(evaluator, cache, samples[0])
    starts = {int(s['start']) for s in samples}
    # Later annotation endpoints determine role identity only. Their actions
    # cannot mutate a previously captured starting state or its prefix proof.
    identity_frames = {int(s['end']) - 1 for s in cache['source_samples'].values()}
    targets = starts | identity_frames
    last = max(targets)
    reference_frames = set(cache.get('robot_reference_capture_frames', []))
    if any(type(f) is not int or not 0 <= f <= last for f in reference_frames):
        raise ValueError('Robot reference capture must stay within the existing replay prefix')
    targets |= reference_frames
    if min(targets) < 0 or last >= len(cache['states']):
        raise ValueError('History evaluation target is outside the frozen episode')
    root = Path(os.environ['SCO_EPHEMERAL_RUNTIME_ROOT']).resolve()
    if not root.is_relative_to(Path('/var/tmp/behavior-1k-runtime')):
        raise ValueError('History source fixtures require the allocated worker temporary disk')
    fixtures = Path(tempfile.mkdtemp(prefix='history-prefix-', dir=root))
    budget = min(64 * 1024**3, shutil.disk_usage(root).free // 2)
    chunk_size = 64
    robot_name = evaluator.robot.name
    enabled_rules = gm.ENABLE_TRANSITION_RULES
    snapshots = {}
    counters = {}
    identity = {}
    prefix_hashes = {}
    started = time.monotonic()
    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = False
    try:
        runner._restore_episode_baseline(evaluator, cache)
        chunk = {}
        used = 0
        for frame in range(last + 1):
            row = torch.as_tensor(cache['states'][frame, :int(cache['state_sizes'][frame])])
            og.sim.load_state(row, serialized=True)
            chunk[frame] = cpu_tree(runner._dump_full_simulator_state(evaluator.env.scene, og.sim))
            if len(chunk) == chunk_size or frame == last:
                path = fixtures / (str(frame // chunk_size) + '.pt.gz')
                with gzip.open(path, 'wb', compresslevel=1) as output:
                    torch.save(chunk, output)
                used += path.stat().st_size
                if used > budget:
                    raise ValueError('History fixtures exceed the available worker disk budget')
                chunk = {}
            if frame in cache['transition_by_step'] and frame < last:
                runner._apply_transition(evaluator.env.scene, cache['transition_by_step'][frame], cache['scene_file'])
            if frame % 1000 == 0:
                print(f'HISTORY_SOURCE_FIXTURES frame={frame}/{last} bytes={used}', flush=True)
        runner._restore_episode_baseline(evaluator, cache)
        for arm in evaluator.robot.arm_names:
            evaluator.robot._ag_grasp_counter[arm] = None
            evaluator.robot._ag_release_counter[arm] = None
        with h5py.File(cache['path'], 'r', locking=False) as file:
            prefix = np.asarray(file['data'][cache['details']['demo_group']]['action'][:last], dtype=np.float32)
        if prefix.shape != (last, 23) or not np.isfinite(prefix).all():
            raise ValueError('Original history action prefix is incomplete')
        digest = hashlib.sha256()
        loaded_chunk = -1
        for frame in range(last + 1):
            number = frame // chunk_size
            if number != loaded_chunk:
                path = fixtures / (str(number) + '.pt.gz')
                with gzip.open(path, 'rb') as source:
                    chunk = torch.load(io.BytesIO(source.read()), map_location='cpu', weights_only=True)
                loaded_chunk = number
            recorded = chunk.pop(frame)
            state = carry_native_grasp(recorded, robot_name, evaluator.robot.dump_state(serialized=False))
            og.sim.load_state(state, serialized=False)
            og.sim.update_handles()
            if frame in targets:
                og.sim.render()
                if frame in reference_frames:
                    from codex_robot_guidance import capture_robot_anchor
                    anchor = capture_robot_anchor(evaluator.robot, frame)
                    anchor['recorded_state_sha256'] = hashlib.sha256(
                        np.asarray(cache['states'][frame, :int(cache['state_sizes'][frame])]).tobytes()).hexdigest()
                    cache.setdefault('robot_reference_anchors', {})[frame] = anchor
                captured = cpu_tree(runner._dump_full_simulator_state(evaluator.env.scene, og.sim))
                native_robot = captured[0]['registry']['object_registry'][robot_name]
                identity[frame] = _grasps(native_robot, captured)
                if frame in starts:
                    snapshots[frame] = captured
                    counters[frame] = _counter_state(evaluator.robot)
                    prefix_hashes[frame] = digest.copy().hexdigest()
            if frame == last:
                break
            evaluator.robot.apply_action(torch.from_numpy(prefix[frame].copy()))
            og.sim.step()
            digest.update(prefix[frame].tobytes())
            if frame in cache['transition_by_step']:
                runner._apply_transition(evaluator.env.scene, cache['transition_by_step'][frame], cache['scene_file'])
            if frame % 250 == 0:
                print(f'HISTORY_GRASP_RECOVERY frame={frame+1}/{last} captured={len(snapshots)}', flush=True)
        cache['history_identity_grasps'] = identity
        cache['history_recovery_protocol'] = CONTRACT
        materialized = {}
        for sample in samples:
            frame = sample['start']
            transitions = sorted(step for step in cache['transition_by_step'] if step < frame)
            evidence = copy.deepcopy(cache['sample_evidence'][sample['sample_id']])
            evidence.update(full_scene_path=getattr(evaluator, '_exact_state_full_scene_path', None),
                            scene_backfilled_states=getattr(evaluator, '_exact_state_backfilled_states', 0),
                            state_prefix_samples=frame+1, topology_transition_steps=transitions,
                            full_snapshot=True, full_snapshot_format=runner.FULL_SNAPSHOT_FORMAT,
                            grasp_recovery_protocol=CONTRACT, grasp_recovery_method=METHOD,
                            recorded_prefix_physics_steps=frame, prefix_actions_sha256=prefix_hashes[frame],
                            future_actions_used_to_prepare_start=False, manual_grasp_establishment_used=False,
                            exact_original_grasp_anchor_bytes_verified=False)
            materialized[sample['sample_id']] = {'state': snapshots[frame],
                'state_format': runner.FULL_SNAPSHOT_FORMAT, 'transition_steps': transitions,
                'restoration': evidence, 'history_counter_state': counters[frame]}
        print(f'HISTORY_GRASP_RECOVERY_COMPLETE starts={len(samples)} prefix_steps={last} elapsed_seconds={time.monotonic()-started:.1f}', flush=True)
        return materialized
    finally:
        with gm.unlocked():
            gm.ENABLE_TRANSITION_RULES = enabled_rules
        shutil.rmtree(fixtures)


def restored_context(evaluator, cache, sample, materialized):
    """Attach the disclosed recovered initial fields to a model rollout."""
    from recorded_native_state import uuid
    for attribute, values in materialized['history_counter_state'].items():
        setattr(evaluator.robot, attribute, copy.deepcopy(values))
    name = evaluator.robot.name
    robot_state = materialized['state'][0]['registry']['object_registry'][name]
    root = robot_state['root_link']
    proof = {'contract': CONTRACT, 'method': METHOD, 'sample_id': sample['sample_id'],
             'raw_sha256': cache['sha256'], 'source_start': sample['start'],
             'selected_state_sha256': cache['sample_evidence'][sample['sample_id']]['selected_state_sha256'],
             'robot_name': name, 'robot_uuid': uuid(name),
             'full_robot_joint_positions': robot_state['joint_pos'].tolist(),
             'root_position': root['pos'].tolist(), 'root_orientation': root['ori'].tolist(),
             'recovered_grasps': copy.deepcopy(cache['history_identity_grasps'][sample['start']]),
             'runtime_history_snapshot_loaded_before_policy': True,
             'recorded_prefix_physics_steps': sample['start'],
             'prefix_actions_sha256': materialized['restoration']['prefix_actions_sha256'],
             'future_actions_used_to_prepare_start': False, 'manual_grasp_establishment_used': False,
             'exact_original_grasp_anchor_bytes_verified': False}
    return {'history_recovery_source': proof}


def _demonstration_identity(cache, proof, predicate, candidates):
    sample = cache['source_samples'].get(proof['sample_id'])
    if sample is None or predicate['operation'] not in ('grasped', 'near'):
        return None
    event = sample
    if predicate['operation'] == 'near':
        following = [s for s in cache['source_samples'].values() if s['start'] >= sample['end']]
        if not following:
            return None
        event = min(following, key=lambda s: (s['start'], s['end'], s['sample_id']))
        if event['start'] != sample['end'] or event['predicate']['operation'] != 'grasped' or event['predicate'].get('targets') != predicate.get('targets'):
            return None
    frame = event['end'] - 1
    final = cache['history_identity_grasps'].get(frame)
    if final is None:
        return None
    initial = {g['target_object'] for g in proof['recorded_grasps']}
    matching = ({g['target_object'] for g in final} - initial) & candidates
    if len(matching) != 1:
        return None
    name = next(iter(matching))
    row = np.asarray(cache['states'][frame, :int(cache['state_sizes'][frame])], dtype=np.float32)
    return name, {'source_sample_id': sample['sample_id'], 'event_sample_id': event['sample_id'],
                  'event_annotation_start': event['start'], 'event_annotation_end': event['end'],
                  'event_frame': frame, 'raw_sha256': cache['sha256'],
                  'event_selected_state_sha256': hashlib.sha256(row.tobytes()).hexdigest(),
                  'initial_held_objects': sorted(initial), 'event_history_recovered_grasps': copy.deepcopy(final),
                  'newly_held_matching_objects': [name], 'recovery_protocol': CONTRACT}


def source_role_bindings(cache, proof, predicate):
    from recorded_native_state import preview_source_role_bindings
    if proof.get('contract') != CONTRACT or proof.get('runtime_history_snapshot_loaded_before_policy') is not True:
        raise ValueError('History annotation binding lacks its before-policy recovered start')
    internal = {**proof, 'recorded_grasps': proof['recovered_grasps']}
    binding = preview_source_role_bindings(cache, internal, predicate,
                                          demonstration_identity_resolver=_demonstration_identity)
    for entry in binding['bindings']:
        if entry['method'] == 'unique_initial_recorded_native_grasp':
            entry['method'] = 'unique_initial_history_recovered_grasp'
        elif entry['method'] == 'unique_recorded_demonstration_grasp_event':
            entry['method'] = 'unique_recorded_demonstration_history_grasp_event'
    return {**binding, 'contract': BINDING, 'runtime_native_fields_verified': True,
            'grasp_recovery_protocol': CONTRACT, 'exact_original_grasp_anchor_bytes_verified': False}
