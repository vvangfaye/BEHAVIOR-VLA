#!/usr/bin/env python3
"""Test native grasp recovery from actions strictly before original starts."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import diagnose_b1k_grasp_restore as diag
import eval_b1k_subtask_samples as runner

KIND = 'subtask_grasp_history_diagnostic'
METHOD = 'recorded_full_states_with_native_physical_action_prefix_v1'
PREFIX_LIMIT = 2500


def carry_native_grasp(source_state, robot_name, native_robot_state):
    """Keep only the native grasp while restoring every recorded state field."""
    if set(source_state) != {0}:
        raise ValueError('History diagnosis requires one recorded scene')
    state = copy.deepcopy(source_state)
    robots = state[0]['registry']['object_registry']
    if robot_name not in robots or 'controller_groups' not in robots[robot_name]:
        raise ValueError('History diagnosis lacks the decoded recorded robot')
    params = native_robot_state.get('ag_obj_constraint_params')
    if not isinstance(params, dict):
        raise ValueError('Native assisted-grasp state is unavailable')
    if set(params) - {'left', 'right'}:
        raise ValueError('Native assisted-grasp state has unsupported arms')
    robots[robot_name]['ag_obj_constraint_params'] = copy.deepcopy(params)
    return state


def cpu_tree(value):
    import torch as th
    if isinstance(value, th.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def restore(evaluator, cache, sample, state):
    import omnigibson as og
    runner._restore_episode_baseline(evaluator, cache)
    for frame, transition in sorted(cache['transition_by_step'].items()):
        if frame < sample['start']:
            runner._apply_transition(evaluator.env.scene, transition, cache['scene_file'])
    runner._validate_full_snapshot_objects(og.sim, state)
    og.sim.load_state(copy.deepcopy(state), serialized=False)
    og.sim.update_handles()
    for _ in range(3):
        og.sim.render()
    runner._reset_evaluator_rollout(evaluator)


def run_followup(evaluator, sample, actions, steps, objects, video):
    import numpy as np
    import torch as th
    observations = [diag.observe(evaluator, objects, frame=sample['start'])]
    executed = []
    evaluator.start_recording(str(video), rate=30)
    try:
        for i in range(steps):
            action = np.asarray(actions[sample['start']+i], dtype=np.float32)
            if action.shape != (23,) or not np.isfinite(action).all():
                raise ValueError('Invalid original followup action')
            obs, _, terminated, truncated, _ = evaluator.env.step(th.from_numpy(action.copy()), n_render_iterations=1)
            evaluator.obs = evaluator._preprocess_obs(evaluator._sync_lights_and_get_obs(obs))
            evaluator._write_video('SOURCE HISTORY GRASP DIAGNOSTIC: recorded actions')
            executed.append(action.tolist())
            observations.append(diag.observe(evaluator, objects, frame=sample['start']+i+1, physics_steps=i+1))
            if terminated or truncated:
                break
    finally:
        evaluator.stop_recording()
    return {'recorded_actions': executed, 'observations': observations,
            'physics_steps': len(executed), 'video_sha256': runner.file_sha256(video)}


def diagnose(base, evaluator, args, panel, plan, samples, cache):
    import h5py
    import numpy as np
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    last = max(s['start'] for s in samples)
    if not 64 <= last <= PREFIX_LIMIT or len(samples) > 2:
        raise ValueError('History diagnosis exceeds its bounded original prefix')
    if not cache['grasp_serialization_audit']['legacy_missing_assisted_grasp_state']:
        raise ValueError('History diagnosis requires explicit old-format source')
    root = Path(args.output_dir)/'grasp_history'
    videos = root/'videos';videos.mkdir(parents=True, exist_ok=True)
    native_dir = root/'native_snapshots';native_dir.mkdir(exist_ok=True)
    ephemeral = Path(os.environ['SCO_EPHEMERAL_RUNTIME_ROOT']).resolve()
    if not ephemeral.is_relative_to(Path('/var/tmp/behavior-1k-runtime')):
        raise ValueError('Prefix fixtures must use the verified worker temporary disk')
    fixtures = ephemeral/('source-prefix-'+str(samples[0]['raw_episode_id']))
    fixtures.mkdir(exist_ok=False)
    infos = cache['scene_file']['objects_info']['init_info']
    names = sorted(n for n, info in infos.items() if n != evaluator.robot.name and not info['args'].get('fixed_base', False))
    objects = [evaluator.env.scene.object_registry('name', name) for name in names]
    if not objects or any(obj is None for obj in objects):
        raise ValueError('Original movable scene candidates are unavailable')
    windows = {s['sample_id']: list(range(s['start']-64, s['start']+1)) for s in samples}
    watched = {frame for frames in windows.values() for frame in frames}
    histories = {};source_states = {};recovered_states = {};prefix_events = []
    target_frames = {s['start'] for s in samples}
    enabled = gm.ENABLE_TRANSITION_RULES
    with gm.unlocked():gm.ENABLE_TRANSITION_RULES = False
    try:
        runner._validate_recorded_episode(evaluator, cache, samples[0])
        runner._restore_episode_baseline(evaluator, cache)
        fixture_bytes = 0
        for frame in range(last+1):
            row = th.as_tensor(cache['states'][frame, :int(cache['state_sizes'][frame])])
            og.sim.load_state(row, serialized=True)
            if frame in watched:
                og.sim.update_handles();og.sim.render()
                histories[frame] = diag.observe(evaluator, objects, frame=frame)
            state = cpu_tree(runner._dump_full_simulator_state(evaluator.env.scene, og.sim))
            path = fixtures/(str(frame)+'.pt');th.save(state, path)
            fixture_bytes += path.stat().st_size
            if fixture_bytes > 10*1024**3:
                raise ValueError('Bounded history fixtures exceed 10 GiB temporary disk budget')
            if frame in target_frames:source_states[frame] = state
            if frame in cache['transition_by_step'] and frame < last:
                runner._apply_transition(evaluator.env.scene, cache['transition_by_step'][frame], cache['scene_file'])
            if frame % 250 == 0:
                print(f'GRASP_HISTORY_FIXTURE frame={frame}/{last} bytes={fixture_bytes}', flush=True)
        runner._restore_episode_baseline(evaluator, cache)
        for arm in evaluator.robot.arm_names:
            if evaluator.robot._ag_obj_in_hand[arm] is not None:
                raise ValueError('Original frame-zero baseline has an unexpected held object')
            evaluator.robot._ag_grasp_counter[arm] = None
            evaluator.robot._ag_release_counter[arm] = None
        with h5py.File(cache['path'], 'r', locking=False) as file:
            actions = file['data'][cache['details']['demo_group']]['action']
            prefix = np.asarray(actions[:last], dtype=np.float32)
            if prefix.shape != (last, 23) or not np.isfinite(prefix).all():
                raise ValueError('Original prefix actions are incomplete')
            prefix_hash = hashlib.sha256(prefix.tobytes()).hexdigest()
            previous_relation = None
            for frame in range(last+1):
                recorded = th.load(fixtures/(str(frame)+'.pt'), map_location='cpu', weights_only=True)
                merged = carry_native_grasp(recorded, evaluator.robot.name, evaluator.robot.dump_state(serialized=False))
                runner._validate_full_snapshot_objects(og.sim, merged)
                og.sim.load_state(merged, serialized=False);og.sim.update_handles()
                if frame in target_frames:
                    og.sim.render()
                    recovered_states[frame] = cpu_tree(runner._dump_full_simulator_state(evaluator.env.scene, og.sim))
                relation = {arm: {'held_object':None if evaluator.robot._ag_obj_in_hand[arm] is None else evaluator.robot._ag_obj_in_hand[arm].name,
                                 'release_counter':evaluator.robot._ag_release_counter[arm],
                                 'constraint_joint_type':None if evaluator.robot._ag_obj_constraint_params[arm] is None else evaluator.robot._ag_obj_constraint_params[arm]['joint_type']}
                            for arm in evaluator.robot.arm_names}
                if relation != previous_relation:
                    prefix_events.append({'source_frame':frame,'arms':relation});previous_relation=relation
                if frame == last:break
                # This is the action/physics portion of Environment.step. Reward,
                # termination and policy counters are not advanced by preparation.
                evaluator.robot.apply_action(th.from_numpy(prefix[frame].copy()))
                og.sim.step()
                if frame in cache['transition_by_step']:
                    runner._apply_transition(evaluator.env.scene, cache['transition_by_step'][frame], cache['scene_file'])
                if frame % 100 == 0:
                    print(f'GRASP_HISTORY_PHYSICS frame={frame+1}/{last}', flush=True)
            for sample in samples:
                frame = sample['start'];stem=runner._artifact_stem(sample)
                history = [histories[f] for f in windows[sample['sample_id']]]
                base.seed_everything(1)
                restore(evaluator, cache, sample, source_states[frame])
                original = run_followup(evaluator, sample, actions, plan['max_steps'], objects, videos/(stem+'-unmodified.mp4'))
                base.seed_everything(1)
                restore(evaluator, cache, sample, recovered_states[frame])
                native_path=native_dir/(stem+'.pt');th.save(recovered_states[frame], native_path)
                recovered = run_followup(evaluator, sample, actions, plan['max_steps'], objects, videos/(stem+'-history.mp4'))
                out={'kind':KIND,'method':METHOD,'official_benchmark':False,'metric_eligible':False,
                     'model_action_requests':0,'exact_legacy_grasp_restoration_verified':False,
                     'sample_id':sample['sample_id'],'task':sample['task'],'instance_id':sample['task_instance_id'],
                     'episode_index':sample['episode_index'],'raw_episode_id':sample['raw_episode_id'],'source_start':frame,
                     'panel_fingerprint':panel['fingerprint'],'raw_sha256':cache['sha256'],
                     'diagnostic_source_sha256':runner.file_sha256(Path(__file__)),
                     'source_history':history,'source_motion_summary':diag.source_motion_summary(history),
                     'prefix_action_range':[0,frame],'prefix_action_count':frame,
                     'prefix_actions_sha256':hashlib.sha256(prefix[:frame].tobytes()).hexdigest(),
                     'episode_prefix_actions_sha256':prefix_hash,'episode_prefix_physics_steps':last,
                     'prefix_relation_events':[event for event in prefix_events if event['source_frame']<=frame],
                     'future_actions_used_to_prepare_start':False,'manual_grasp_establishment_used':False,
                     'full_recorded_state_clamped_before_each_prefix_action':True,
                     'native_snapshot_sha256':runner.file_sha256(native_path),
                     'branches':{'unmodified':original,'history_recovered':recovered}}
                (root/(stem+'.json')).write_text(json.dumps(out,indent=2)+'\n')
                print(f'GRASP_HISTORY_SAMPLE sample={sample["sample_id"]} prefix_steps={frame}',flush=True)
    finally:
        with gm.unlocked():gm.ENABLE_TRANSITION_RULES=enabled


def validate_results(root, panel, plan, samples):
    expected={s['sample_id']:s for s in samples};seen=set();followup=0
    for path in Path(root).glob('pi05/*/grasp_history/*.json'):
        result=json.loads(path.read_text());sid=result['sample_id']
        if sid not in expected or sid in seen:raise ValueError('Unexpected or duplicate history sample')
        sample=expected[sid]
        if any(result[key]!=sample[key] for key in ['task','episode_index','raw_episode_id']):raise ValueError('History source identity differs')
        if result['source_start']!=sample['start'] or result['instance_id']!=sample['task_instance_id'] or result['panel_fingerprint']!=panel['fingerprint']:raise ValueError('History original start differs')
        if result['kind']!=KIND or result['method']!=METHOD or result['metric_eligible'] is not False or result['official_benchmark'] is not False or result['exact_legacy_grasp_restoration_verified'] is not False or result['model_action_requests']!=0:raise ValueError('History diagnostic has unsupported scoring claims')
        if result['future_actions_used_to_prepare_start'] is not False or result['manual_grasp_establishment_used'] is not False or result['full_recorded_state_clamped_before_each_prefix_action'] is not True:raise ValueError('History preparation used an unsupported shortcut')
        if result['prefix_action_count']!=sample['start'] or result['prefix_action_range']!=[0,sample['start']]:raise ValueError('History preparation has a wrong action range')
        if result['episode_prefix_physics_steps']!=max(s['start'] for s in samples):raise ValueError('History prefix physics count differs')
        if [h['source_frame'] for h in result['source_history']]!=list(range(sample['start']-64,sample['start']+1)):raise ValueError('History source motion window differs')
        if not diag.source_motion_summaries_match(result['source_motion_summary'],diag.source_motion_summary(result['source_history'])):raise ValueError('History source coupling summary differs')
        for branch_name,branch in result['branches'].items():
            if branch_name not in ('unmodified','history_recovered') or branch['physics_steps']!=plan['max_steps'] or len(branch['observations'])!=plan['max_steps']+1 or len(branch['recorded_actions'])!=plan['max_steps']:raise ValueError('History paired physical branch is incomplete')
            if [(x['source_frame'],x['physics_steps']) for x in branch['observations']]!=[(sample['start']+i,i) for i in range(plan['max_steps']+1)]:raise ValueError('History branch observation timing differs')
            suffix='unmodified' if branch_name=='unmodified' else 'history'
            video=path.parent/'videos'/(path.stem+'-'+suffix+'.mp4')
            if not video.is_file() or runner.file_sha256(video)!=branch['video_sha256']:raise ValueError('History branch video differs')
            followup+=branch['physics_steps']
        if set(result['branches'])!={'unmodified','history_recovered'}:raise ValueError('Missing history paired branch')
        native=path.parent/'native_snapshots'/(path.stem+'.pt')
        if not native.is_file() or runner.file_sha256(native)!=result['native_snapshot_sha256']:raise ValueError('History native snapshot differs')
        seen.add(sid)
    if seen!=set(expected):raise ValueError('History original sample coverage is incomplete')
    return {'kind':KIND,'complete':True,'samples':len(seen),'recorded_action_physics_steps':followup,
            'recorded_prefix_physics_steps':max(s['start'] for s in samples),
            'model_action_requests':0,'metric_eligible':False,'official_benchmark':False,
            'exact_legacy_grasp_restoration_verified':False,'panel_fingerprint':panel['fingerprint']}
