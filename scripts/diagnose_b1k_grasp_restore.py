#!/usr/bin/env python3
"""Observe legacy start-state grasp fidelity without scoring or reconstruction."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import eval_b1k_subtask_samples as runner

KIND = 'subtask_grasp_restoration_diagnostic'
HISTORY_FRAMES = 64


def source_motion_summary(history):
    import numpy as np
    output = {}
    for arm in history[-1]['arms']:
        ee = np.asarray([r['arms'][arm]['eef_position'] for r in history], dtype=float)
        candidates = []
        for name in history[-1]['arms'][arm]['objects']:
            rows = [r['arms'][arm]['objects'][name] for r in history]
            pos = np.asarray([r['relative_position'] for r in rows], dtype=float)
            quats = np.asarray([r['relative_orientation'] for r in rows], dtype=float)
            quats /= np.linalg.norm(quats, axis=1, keepdims=True)
            candidates.append({'object': name,
                'median_distance_m': float(np.median(np.linalg.norm(pos, axis=1))),
                'relative_position_max_deviation_m': float(np.max(np.linalg.norm(pos-pos.mean(axis=0), axis=1))),
                'relative_orientation_max_deviation_rad': float(np.max(2*np.arccos(np.clip(np.abs(quats@quats[0]),0,1))))})
        output[arm] = {'observed_frames': len(history),
            'eef_world_excursion_m': float(np.max(np.linalg.norm(ee-ee[0], axis=1))),
            'source_gripper_joint_positions_first': history[0]['arms'][arm]['gripper_joint_positions'],
            'source_gripper_joint_positions_last': history[-1]['arms'][arm]['gripper_joint_positions'],
            'objects': candidates}
    return output


def source_motion_summaries_match(actual, expected, field=None):
    """Compare recomputed summaries across NumPy / BLAS implementations.

    acos near a unit quaternion dot product amplifies machine rounding to
    about 1e-8 radians. Only that derived angular field gets this tolerance;
    copied source joints and all categorical fields must still match exactly.
    """
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            source_motion_summaries_match(actual[key], value, key)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            source_motion_summaries_match(a, e, field) for a, e in zip(actual, expected)
        )
    if isinstance(expected, float):
        if not isinstance(actual, float) or not math.isfinite(actual):
            return False
        if field in {'relative_orientation_max_deviation_rad', 'max_orientation_difference_rad', 'max_relative_orientation_change_rad'}:
            return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-7)
        if field in {'median_distance_m', 'relative_position_max_deviation_m', 'eef_world_excursion_m',
                     'max_position_difference_m', 'max_relative_position_change_m', 'final_relative_position_change_m'}:
            return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return type(actual) is type(expected) and actual == expected


def observe(evaluator, objects, *, frame=None, physics_steps=0):
    from omnigibson.utils import transform_utils as T
    robot = evaluator.robot
    q = robot.get_joint_positions()
    arms = {}
    for arm in robot.arm_names:
        ee_pos, ee_orn = robot.get_eef_pose(arm)
        positions = {}
        for obj in objects:
            pos, orn = obj.get_position_orientation()
            relative_pos, relative_orn = T.relative_pose_transform(pos, orn, ee_pos, ee_orn)
            positions[obj.name] = {'world_position': pos.tolist(), 'world_orientation': orn.tolist(),
                                  'relative_position': relative_pos.tolist(), 'relative_orientation': relative_orn.tolist()}
        held = robot._ag_obj_in_hand[arm]
        params = robot._ag_obj_constraint_params[arm]
        candidate = None
        contacts = []
        # Contact caches before a physical step are not treated as fresh evidence.
        if physics_steps:
            touched, _ = robot._find_gripper_contacts(arm)
            contacts = sorted(touched)
            value = robot._calculate_in_hand_object(arm)
            if value is not None:
                candidate = {'object': value[0].name, 'link': value[1]}
        idx = robot.gripper_control_idx[arm]
        arms[arm] = {'eef_position': ee_pos.tolist(), 'eef_orientation': ee_orn.tolist(),
                     'gripper_joint_positions': q[idx].tolist(),
                     'gripper_joint_lower_limits': robot.joint_lower_limits[idx].tolist(),
                     'gripper_joint_upper_limits': robot.joint_upper_limits[idx].tolist(),
                     'held_object': None if held is None else held.name,
                     'constraint_joint_type': None if params is None else params['joint_type'],
                     'release_counter': robot._ag_release_counter[arm],
                     'contacts_observed_after_physics': bool(physics_steps),
                     'contact_links': contacts, 'native_grasp_candidate': candidate, 'objects': positions}
    return {'source_frame': frame, 'physics_steps': physics_steps, 'arms': arms}


def materialize_source_observations(evaluator, samples, cache):
    import omnigibson as og
    from omnigibson.macros import gm
    import torch as th
    # Movable candidates are selected from recorded scene metadata, never from
    # the requested predicate or its target IDs.
    infos = cache['scene_file']['objects_info']['init_info']
    names = sorted(n for n, item in infos.items()
                   if n != evaluator.robot.name and not item['args'].get('fixed_base', False))
    objects = [evaluator.env.scene.object_registry('name', name) for name in names]
    if not objects or any(obj is None for obj in objects):
        raise ValueError('Diagnostic movable scene candidates are missing')
    last = max(s['start'] for s in samples)
    windows = {s['sample_id']: range(max(0, s['start']-HISTORY_FRAMES), s['start']+1) for s in samples}
    watched = {frame for frames in windows.values() for frame in frames}
    histories, snapshots = {}, {}
    runner._validate_recorded_episode(evaluator, cache, samples[0])
    runner._restore_episode_baseline(evaluator, cache)
    enabled = gm.ENABLE_TRANSITION_RULES
    started = last_log = time.monotonic()
    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = False
    try:
        for action, frame in runner.sparse_materialization_plan([s['start'] for s in samples], cache['transitions']):
            if action == 'load':
                og.sim.load_state(th.as_tensor(cache['states'][frame, :int(cache['state_sizes'][frame])]), serialized=True)
                if frame in watched:
                    og.sim.update_handles()
                    og.sim.render()  # Propagate restored articulation poses before querying EEF links.
                    histories[frame] = observe(evaluator, objects, frame=frame)
                now = time.monotonic()
                if frame % 1000 == 0 or now-last_log >= 30:
                    print(f'GRASP_DIAGNOSTIC_SOURCE frame={frame}/{last} elapsed_seconds={now-started:.1f}', flush=True)
                    last_log = now
            elif action == 'capture':
                snapshots[frame] = runner._dump_full_simulator_state(evaluator.env.scene, og.sim)
            else:
                runner._apply_transition(evaluator.env.scene, cache['transition_by_step'][frame])
        print(f'GRASP_DIAGNOSTIC_SOURCE_COMPLETE frames={last+1} snapshots={len(snapshots)}', flush=True)
    finally:
        with gm.unlocked():
            gm.ENABLE_TRANSITION_RULES = enabled
    return objects, histories, snapshots, windows


def diagnose(base, evaluator, args, panel, plan, samples, cache):
    import h5py
    import numpy as np
    import omnigibson as og
    from omnigibson.macros import gm
    import torch as th
    objects, histories, snapshots, windows = materialize_source_observations(evaluator, samples, cache)
    names = [obj.name for obj in objects]
    enabled = gm.ENABLE_TRANSITION_RULES
    out = Path(args.output_dir)/'grasp_diagnostic'
    out.mkdir(parents=True, exist_ok=True)
    video_dir = out/'videos'; video_dir.mkdir(exist_ok=True)
    with h5py.File(cache['path'], 'r', locking=False) as file:
        actions = file['data'][cache['details']['demo_group']]['action']
        for sample in samples:
            runner._restore_episode_baseline(evaluator, cache)
            with gm.unlocked():
                gm.ENABLE_TRANSITION_RULES = False
            try:
                for step in sorted(cache['transition_by_step']):
                    if step < sample['start']:
                        runner._apply_transition(evaluator.env.scene, cache['transition_by_step'][step])
                snapshot = copy.deepcopy(snapshots[sample['start']])
                runner._validate_full_snapshot_objects(og.sim, snapshot)
                og.sim.load_state(snapshot, serialized=False)
                og.sim.update_handles()
                for _ in range(3):og.sim.render()
                evaluator.external_subtask = sample['subtask']
                runner._reset_evaluator_rollout(evaluator)
                observations = [observe(evaluator, objects, frame=sample['start'])]
                stem = runner._artifact_stem(sample)
                video = video_dir/(stem+'.mp4')
                evaluator.start_recording(str(video), rate=args.video_fps)
                executed = []
                try:
                    for index in range(plan['max_steps']):
                        frame = sample['start']+index
                        if frame >= len(actions):break
                        act = np.asarray(actions[frame], dtype=np.float32)
                        if act.shape != (23,) or not np.isfinite(act).all():
                            raise ValueError('Recorded diagnostic action must be finite and exactly 23-wide')
                        obs, _, terminated, truncated, _ = evaluator.env.step(th.from_numpy(act.copy()), n_render_iterations=1)
                        evaluator.obs = evaluator._preprocess_obs(evaluator._sync_lights_and_get_obs(obs))
                        evaluator._write_video('GRASP RESTORATION DIAGNOSTIC: recorded demonstration actions')
                        executed.append(act.tolist())
                        observations.append(observe(evaluator, objects, frame=frame+1, physics_steps=index+1))
                        if terminated or truncated:break
                finally:evaluator.stop_recording()
            finally:
                with gm.unlocked():gm.ENABLE_TRANSITION_RULES = enabled
            history = [histories[frame] for frame in windows[sample['sample_id']]]
            result = {'kind': KIND, 'official_benchmark': False, 'metric_eligible': False,
                'sample_id': sample['sample_id'], 'task': sample['task'],
                'instance_id': sample['task_instance_id'], 'episode_index': sample['episode_index'],
                'raw_episode_id': sample['raw_episode_id'], 'source_start': sample['start'],
                'panel_fingerprint': panel['fingerprint'], 'raw_sha256': cache['sha256'],
                'grasp_serialization_audit': cache['grasp_serialization_audit'],
                'movable_candidates': names, 'candidate_selection': 'recorded_scene_non_fixed_objects',
                'model_action_requests': 0, 'physics_steps': len(executed),
                'advancement_source': 'recorded_demonstration_actions', 'recorded_actions': executed,
                'source_history': history, 'source_motion_summary': source_motion_summary(history),
                'observations_after_restore': observations,
                'constraint_reconstruction_attempted': False,
                'exact_legacy_grasp_restoration_verified': False,
                'diagnostic_source_sha256': runner.file_sha256(Path(__file__)),
                'native_robot_source_sha256': runner.file_sha256(Path(og.__file__).parent/'robots/robot.py'),
                'video_sha256': runner.file_sha256(video)}
            runner.atomic_write_json(out/(stem+'.json'), result)
            print(f'GRASP_DIAGNOSTIC_SAMPLE sample={sample["sample_id"]} recorded_action_steps={len(executed)}', flush=True)


def validate_results(root, panel, plan, samples):
    expected = {s['sample_id']:s for s in samples}
    seen = set(); steps = 0
    for path in Path(root).glob('pi05/*/grasp_diagnostic/*.json'):
        r = json.loads(path.read_text());sid=r.get('sample_id')
        if sid not in expected or sid in seen:raise ValueError('Unexpected/duplicate grasp diagnostic sample')
        s=expected[sid]
        for key in ('task','episode_index','raw_episode_id'):
            if r.get(key)!=s[key]:raise ValueError('Grasp diagnostic source identity mismatch')
        if r.get('instance_id')!=s['task_instance_id'] or r.get('source_start')!=s['start'] or r.get('panel_fingerprint')!=panel['fingerprint']:
            raise ValueError('Grasp diagnostic differs from the frozen source start')
        if r.get('kind')!=KIND or r.get('official_benchmark') is not False or r.get('metric_eligible') is not False:
            raise ValueError('Grasp diagnostic must never be labeled as scored evaluation')
        if r.get('model_action_requests')!=0 or r.get('advancement_source')!='recorded_demonstration_actions' or r.get('constraint_reconstruction_attempted') is not False or r.get('exact_legacy_grasp_restoration_verified') is not False:
            raise ValueError('Grasp diagnostic has unsupported scoring or reconstruction claims')
        n=r.get('physics_steps')
        if type(n) is not int or not 1<=n<=plan['max_steps'] or len(r.get('recorded_actions',[]))!=n or len(r.get('observations_after_restore',[]))!=n+1:
            raise ValueError('Grasp diagnostic lacks actual bounded physics evidence')
        if not r.get('source_history') or not r.get('source_motion_summary') or r.get('candidate_selection')!='recorded_scene_non_fixed_objects':
            raise ValueError('Grasp diagnostic lacks source motion evidence')
        for key in ('raw_sha256','diagnostic_source_sha256','native_robot_source_sha256','video_sha256'):
            value=r.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Grasp diagnostic lacks source/video hash')
        def finite(v):
            if isinstance(v,dict):return all(finite(x) for x in v.values())
            if isinstance(v,list):return all(finite(x) for x in v)
            return not isinstance(v,float) or math.isfinite(v)
        if not finite(r):raise ValueError('Non-finite grasp diagnostic evidence')
        candidates = r.get('movable_candidates')
        if not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates):
            raise ValueError('Grasp diagnostic has invalid movable candidates')
        def observation(value, expected_frame, expected_steps):
            if value.get('source_frame') != expected_frame or value.get('physics_steps') != expected_steps:
                raise ValueError('Grasp diagnostic observation timing is inconsistent')
            arms = value.get('arms')
            if not isinstance(arms, dict) or not arms:
                raise ValueError('Grasp diagnostic lacks observed arms')
            for arm in arms.values():
                if len(arm.get('eef_position', [])) != 3 or len(arm.get('eef_orientation', [])) != 4:
                    raise ValueError('Grasp diagnostic lacks actual EEF poses')
                joints = arm.get('gripper_joint_positions')
                if not isinstance(joints, list) or not joints or any(
                    len(arm.get(key, [])) != len(joints) for key in ('gripper_joint_lower_limits', 'gripper_joint_upper_limits')
                ):
                    raise ValueError('Grasp diagnostic lacks gripper joint evidence')
                if arm.get('contacts_observed_after_physics') is not bool(expected_steps):
                    raise ValueError('Grasp diagnostic contact-cache freshness is inconsistent')
                if set(arm.get('objects', {})) != set(candidates):
                    raise ValueError('Grasp diagnostic does not observe every movable candidate')
                for obj in arm['objects'].values():
                    if any(len(obj.get(key, [])) != size for key, size in
                        (('world_position', 3), ('world_orientation', 4), ('relative_position', 3), ('relative_orientation', 4))):
                        raise ValueError('Grasp diagnostic lacks observed candidate poses')
        expected_frames = list(range(max(0, s['start']-HISTORY_FRAMES), s['start']+1))
        if len(r['source_history']) != len(expected_frames):
            raise ValueError('Grasp diagnostic source motion window is incomplete')
        for value, frame in zip(r['source_history'], expected_frames):
            observation(value, frame, 0)
        for i, value in enumerate(r['observations_after_restore']):
            observation(value, s['start']+i, i)
        if any(not isinstance(action, list) or len(action) != 23 for action in r['recorded_actions']):
            raise ValueError('Grasp diagnostic action dimensions are invalid')
        if not source_motion_summaries_match(r['source_motion_summary'], source_motion_summary(r['source_history'])):
            raise ValueError('Grasp diagnostic summary differs from actual source observations')
        video=path.parent/'videos'/(path.stem+'.mp4')
        if not video.is_file() or video.stat().st_size==0 or runner.file_sha256(video)!=r['video_sha256']:
            raise ValueError('Grasp diagnostic video is missing or changed')
        seen.add(sid);steps+=n
    if seen!=set(expected):raise ValueError('Grasp diagnostic is incomplete')
    return {'kind':KIND,'complete':True,'official_benchmark':False,'metric_eligible':False,
            'samples':len(seen),'recorded_action_physics_steps':steps,
            'model_action_requests':0,'exact_legacy_grasp_restoration_verified':False,
            'panel_fingerprint':panel['fingerprint']}
