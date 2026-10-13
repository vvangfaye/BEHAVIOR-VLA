"""Diagnostic-only replay and closed-gripper dwell probes of a frozen attempt."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


class RecordedActions:
    current_subtask_source = 'provided'

    def __init__(self, actions, subtask):
        self.actions, self.current_subtask, self.index = actions, subtask, 0

    def reset(self):
        self.index = 0

    def forward(self, obs):
        import torch
        if self.index >= len(self.actions):
            raise RuntimeError('Recorded diagnostic actions exhausted')
        action = torch.as_tensor(self.actions[self.index].copy(), dtype=torch.float32)
        self.index += 1
        return action


def observe(evaluator, step, phase):
    from omnigibson.utils.usd_utils import RigidContactAPI
    robot = evaluator.robot
    target = evaluator.env.scene.object_registry('name', 'digital_camera_87')
    assert target is not None
    target_paths = set(target.link_prim_paths)
    row = dict(step=step, phase=phase, target=target.name,
               target_position=jsonable(target.get_position_orientation()[0]), arms={})
    for arm in robot.arm_names:
        contacts, by_object = robot._find_gripper_contacts(arm=arm)
        raycasts = robot._find_gripper_raycast_collisions(arm=arm)
        fingers = [link.prim_path for link in robot.finger_links[arm]]
        touching = sorted({p for obj, links in by_object.items() if obj in target_paths for p in links})
        row['arms'][arm] = dict(
            held_object=getattr(robot._ag_obj_in_hand[arm], 'name', None),
            grasp_counter=jsonable(robot._ag_grasp_counter[arm]),
            release_counter=jsonable(robot._ag_release_counter[arm]),
            finger_joint_positions=jsonable(robot.get_joint_positions()[robot.gripper_control_idx[arm]]),
            target_contact_fingers=touching, target_two_finger_contact=len(set(fingers) & set(touching)) >= 2,
            target_raycast_hit=bool(target_paths & raycasts),
            finger_contacts={str(k): sorted(v) for k, v in by_object.items()},
            raycast_hits=sorted(raycasts),
            eef_world_position=jsonable(robot.eef_links[arm].get_position_orientation()[0]))
    row['target_contacts'] = jsonable(sorted(RigidContactAPI.get_contact_pairs(
        target.scene.idx, {target}, None, current_only=True)))
    return row


def run(config):
    import h5py
    import numpy as np
    from PIL import Image
    import omnigibson as og
    import eval_b1k_subtask_samples as runner
    from collect_b1k_codex import collection_evaluator, time_limit
    from collect_b1k_codex_subtask import _restore, make_monitor
    from codex_subtask_collection import load_plan, inspect_attempt, HISTORY, file_sha256
    from legacy_history_recovery import materialize, _counter_state
    from diagnose_b1k_grasp_history import cpu_tree
    from subtask_trajectory import capture_observation, CAMERAS

    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    stage = 'input_validation'

    def publish(state, **fields):
        row = dict(state=state, stage=stage, updated_at=time.time(), diagnostic_only=True,
                   training_eligible=False, model_calls=0, **fields)
        (root/'status.json').write_text(json.dumps(row, allow_nan=False, indent=2)+'\n')
        print('CAMERA_REPLAY_STATUS='+json.dumps(row), flush=True)

    def save_images(evaluator, prefix):
        frame = capture_observation(evaluator.obs)
        for i, key in enumerate(CAMERAS):
            Image.fromarray(frame[key]).save(root/f'{prefix}-camera-{i}.png')

    try:
        plan = load_plan(config['plan_path']); sample = plan['sample']
        assert plan['fingerprint'] == config['plan_fingerprint']
        assert sample['task'] == 'attach_a_camera_to_a_tripod' and sample['task_instance_id'] == 4
        original = Path(config['attempt_root']); info = inspect_attempt(original, require_success=False)
        assert info['metadata']['assisted_plan'] == plan
        assert info['result']['steps'] == config['expected_steps'] == 628
        assert info['result']['success'] is False and info['result']['actual_teacher_control_steps'] == 180
        with h5py.File(original/'trajectory.hdf5', 'r', locking=False) as f:
            actions = f['action'][:]
        assert actions.shape == (628, 23) and actions.dtype == np.float32 and np.isfinite(actions).all()
        close_ends = set()
        for path in sorted((original/'teacher').glob('step-*/decision/*/response.json')):
            decision = json.loads(path.read_text()); correction = decision.get('correction') or {}
            if correction.get('gripper') == 'close':
                request = json.loads((path.parent/'request.json').read_text())['request']
                close_ends.add(request['step']+correction['steps'])
        assert close_ends == {519, 545, 560, 571, 584, 608}
        base = runner._load_base_module()
        args = base.parse_args(['--task-name',sample['task'],'--mode','train','--prompt-mode','whole_task',
            '--num-rollouts','1','--max-steps','2048','--robot-config',config['robot_config'],
            '--output-dir',str(root),'--write-video','--headless'])
        args.external_subtask=sample['subtask'];base.gm.HEADLESS=True;base.seed_everything(plan['policy_seed'])
        policy=RecordedActions(actions,sample['subtask'])
        class ReplayEvaluator(base.Evaluator):
            def load_policy(self): return policy
        proxy=SimpleNamespace(**vars(base));proxy.Evaluator=ReplayEvaluator
        cache=runner.load_episode_cache(runner.raw_hdf5_path(plan['raw_data_root'],sample),[sample],allow_unverified_grasp_state=True)
        relative=f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
        assert cache['sha256']==plan['source_plan']['raw_source_sha256'][relative]
        cfg=runner._build_evaluator_cfg(base,args,plan['policy_seed'])
        stage='simulator_startup';publish('starting')
        def failure(error): publish('failed',error_type=type(error).__name__)
        with time_limit(5400), collection_evaluator(
                lambda:runner.create_sample_evaluator(proxy,cfg,cache,sample,2048),failure) as evaluator:
            stage='original_history_recovery';publish('running')
            states=materialize(evaluator,cache,[sample])
            assert states[sample['sample_id']]['restoration']['prefix_actions_sha256']==plan['source_plan']['prefix_actions_sha256'][sample['sample_id']]
            restoration=_restore(evaluator,cache,sample,states[sample['sample_id']],HISTORY)
            assert restoration['history_recovery_source']['selected_state_sha256']==plan['source_plan']['selected_state_sha256'][sample['sample_id']]
            monitor=make_monitor(evaluator,sample,cache,restoration)
            robot_module=importlib.import_module(type(evaluator.robot).__module__)
            runtime=dict(robot_module=robot_module.__name__,robot_source_sha256=file_sha256(robot_module.__file__),
                env_config={k:evaluator.env.env_config[k] for k in ('action_frequency','physics_frequency','rendering_frequency')},physics_dt=og.sim.get_physics_dt(),
                grasping_mode=evaluator.robot.grasping_mode)
            # The active robot class can inherit its grasp implementation from another module.
            grasp_module=importlib.import_module(evaluator.robot._handle_assisted_grasping.__func__.__module__)
            runtime['grasp_window_seconds']=float(grasp_module.m.GRASP_WINDOW)
            (root/'runtime.json').write_text(json.dumps(runtime,indent=2,allow_nan=False)+'\n')
            keyframes={0,192,256,418,628}|close_ends|{s+3 for s in close_ends}
            snapshots={}; rows=[]; differences=[]
            evaluator.start_recording(str(root/'replay.mp4'),rate=30)
            stage='recorded_action_replay';publish('running',executed_steps=0)
            with h5py.File(original/'trajectory.hdf5','r',locking=False) as f,(root/'contacts.jsonl').open('x',buffering=1) as stream:
                for step in range(629):
                    row=observe(evaluator,step,'original_action_replay');row['monitor']=monitor.result()
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                    if step in close_ends:
                        snapshots[step]=dict(state=cpu_tree(runner._dump_full_simulator_state(evaluator.env.scene,og.sim)),
                            counters=_counter_state(evaluator.robot))
                    if step in keyframes:save_images(evaluator,f'replay-{step:04d}')
                    if step==628:break
                    frame=capture_observation(evaluator.obs);delta=frame['observation.state']-f['observation.state'][step]
                    diff=dict(step=step,state_max_abs=float(np.abs(delta).max()))
                    if step in keyframes:
                        diff['rgb_mae']={k:float(np.abs(frame[k].astype(float)-f[k][step].astype(float)).mean()) for k in CAMERAS}
                    differences.append(diff);evaluator.external_subtask=sample['subtask']
                    terminated,truncated=evaluator.step()
                    assert np.array_equal(np.asarray(evaluator.robot_action).reshape(-1),actions[step])
                    assert not terminated and not truncated
                    monitor.refresh(step+1)
                    if (step+1)%64==0:publish('running',executed_steps=step+1)
            evaluator.stop_recording()
            (root/'replay_fidelity.json').write_text(json.dumps(differences,indent=2)+'\n')
            stage='diagnostic_close_dwell_probes';publish('running',executed_steps=628)
            probes=[]
            for end in sorted(close_ends):
                saved=snapshots[end]
                branch=dict(state=saved['state'],state_format=runner.FULL_SNAPSHOT_FORMAT,transition_steps=states[sample['sample_id']]['transition_steps'],
                    restoration=dict(diagnostic_branch=True,original_action_replay_step=end,training_eligible=False))
                runner.restore_materialized_sample_state(evaluator,cache,branch)
                for name,value in saved['counters'].items():setattr(evaluator.robot,name,copy.deepcopy(value))
                hold=np.tile(actions[end-1],(15,1))
                assert np.all(hold[:,evaluator.robot.controller_action_idx['base']]==0)
                assert np.all(hold[:,evaluator.robot.controller_action_idx['gripper_left']]==-1)
                evaluator.policy=RecordedActions(hold,sample['subtask'])
                records=[];save_images(evaluator,f'probe-{end:04d}-initial')
                with (root/f'probe-{end:04d}.jsonl').open('x',buffering=1) as stream:
                    for i in range(16):
                        row=observe(evaluator,i,'diagnostic_close_dwell');row['original_action_replay_step']=end
                        records.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                        if i<15:
                            terminated,truncated=evaluator.step();assert not terminated and not truncated
                save_images(evaluator,f'probe-{end:04d}-final')
                probes.append(dict(original_step=end,hold_steps=15,first_grasp_step=next((r['step'] for r in records if any(v['held_object']=='digital_camera_87' for v in r['arms'].values())),None),
                    two_finger_contact_steps=[r['step'] for r in records if r['arms']['left']['target_two_finger_contact']],
                    raycast_hit_steps=[r['step'] for r in records if r['arms']['left']['target_raycast_hit']]))
            summary=dict(kind='camera_grasp_contact_replay_v1',state='complete',diagnostic_only=True,
                training_eligible=False,model_calls=0,original_results_modified=False,executed_steps=628,
                byte_identical_actions=True,plan_fingerprint=plan['fingerprint'],
                original_trajectory_sha256=file_sha256(original/'trajectory.hdf5'),
                actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                max_proprio_abs=max(d['state_max_abs'] for d in differences),
                max_keyframe_rgb_mae=max(v for d in differences for v in d.get('rgb_mae',{}).values()),
                unrecorded_physics_caches_verified=False,runtime=runtime,close_ends=sorted(close_ends),dwell_probes=probes,
                target_held_replay_steps=[r['step'] for r in rows if any(v['held_object']=='digital_camera_87' for v in r['arms'].values())],
                two_finger_replay_steps=[r['step'] for r in rows if r['arms']['left']['target_two_finger_contact']])
            (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
            stage='complete';publish('complete',executed_steps=628)
            return summary
    except BaseException as error:
        publish('failed',error_type=type(error).__name__)
        (root/'failure.json').write_text(json.dumps(dict(stage=stage,error_type=type(error).__name__))+'\n')
        raise


if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
