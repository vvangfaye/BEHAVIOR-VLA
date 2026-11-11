"""Bounded simulator restore check, with no model or training calls.

Results only qualify the one newly initialized state and action sequence tested.
They never establish fidelity to a historical expert or policy rollout.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def physical_state(evaluator):
    result = {}
    for obj in evaluator.env.scene.objects:
        p, q = obj.get_position_orientation()
        result[obj.name] = {'position': p.detach().cpu().tolist(),
                            'orientation': q.detach().cpu().tolist()}
    robot = evaluator.robot
    result['__robot__'] = {'joint_positions': robot.get_joint_positions().detach().cpu().tolist(),
                           'joint_velocities': robot.get_joint_velocities().detach().cpu().tolist(),
                           'held_objects': {arm: getattr(obj, 'name', None)
                                            for arm,obj in robot._ag_obj_in_hand.items()},
                           'grasp_counter': copy.deepcopy(robot._ag_grasp_counter),
                           'release_counter': copy.deepcopy(robot._ag_release_counter)}
    return result


def state_error(left, right):
    import numpy as np
    if left.keys() != right.keys():
        raise ValueError('Restored object set changed')
    positions, angles = [], []
    for name in left:
        if name == '__robot__':
            continue
        a,b = left[name],right[name]
        positions.append(float(np.linalg.norm(np.asarray(a['position'])-b['position'])))
        qa,qb = np.asarray(a['orientation']),np.asarray(b['orientation'])
        dot = abs(float(qa@qb))/(np.linalg.norm(qa)*np.linalg.norm(qb))
        angles.append(float(2*np.arccos(np.clip(dot,0,1))))
    a,b = left['__robot__'],right['__robot__']
    return {'max_object_position_error_m': max(positions,default=0),
            'max_object_orientation_error_rad': max(angles,default=0),
            'max_joint_position_error': float(np.max(np.abs(np.asarray(a['joint_positions'])-b['joint_positions']))),
            'max_joint_velocity_error': float(np.max(np.abs(np.asarray(a['joint_velocities'])-b['joint_velocities']))),
            'held_objects_equal': a['held_objects']==b['held_objects'],
            'grasp_counters_equal': a['grasp_counter']==b['grasp_counter'],
            'release_counters_equal': a['release_counter']==b['release_counter']}


def passed(error, tolerance):
    return (all(error[k] for k in ('held_objects_equal','grasp_counters_equal','release_counters_equal'))
            and error['max_object_position_error_m']<=tolerance
            and error['max_object_orientation_error_rad']<=tolerance
            and error['max_joint_position_error']<=tolerance
            and error['max_joint_velocity_error']<=tolerance)


def run(output, plan):
    import numpy as np
    import torch
    import omnigibson as og
    from omnigibson.macros import gm
    from helpers import _build_evaluator_cfg, _dump_full_simulator_state, _validate_full_snapshot_objects
    import importlib

    base = importlib.import_module('omnigibson.eval.eval')
    for name in ('omnigibson.eval.eval','omnigibson.eval.evaluator','omnigibson.eval.single_env'):
        module=importlib.import_module(name)
        relative='BEHAVIOR-1K/OmniGibson/'+name.replace('.','/')+'.py'
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=plan['source_sha256'][relative]:
            raise ValueError('Imported runtime source differs from the frozen plan: '+name)
    args = base.parse_args(['--task-name',plan['task'],'--mode','train','--prompt-mode','whole_task',
                            '--num-rollouts','1','--max-steps','512','--headless',
                            '--robot-config',plan['robot_config'],'--output-dir',str(output)])
    gm.HEADLESS = True
    base.seed_everything(plan['seed'])
    class NoModel:
        def reset(self): pass
        def forward(self,obs): raise RuntimeError('This probe must not query a policy')
    class ProbeEvaluator(base.Evaluator):
        def load_policy(self): return NoModel()
    evaluator = ProbeEvaluator(_build_evaluator_cfg(base,args,plan['seed']))
    evaluator.reset(prepare_for_task_instance=True)
    evaluator.load_task_instance(plan['instance'])
    evaluator.reset()
    robot=evaluator.robot
    if robot.action_dim!=23 or robot.grasping_mode!='assisted':
        raise ValueError('Unexpected robot/action/grasping configuration')
    # Use the robot's controller-index mapping, never assume joint-vector order.
    hold=torch.zeros(23,device=robot.get_joint_positions().device)
    for name in ('trunk','arm_left','arm_right'):
        hold[robot.controller_action_idx[name]]=robot.get_joint_positions()[robot.controller_joint_idx[name]]
    for name in ('gripper_left','gripper_right'):
        hold[robot.controller_action_idx[name]]=1.0
    for _ in range(16):
        evaluator.env.step(hold.clone(),n_render_iterations=1)

    state=_dump_full_simulator_state(evaluator.env.scene,og.sim)
    snapshot_side={'grasp_counter':copy.deepcopy(robot._ag_grasp_counter),
                   'release_counter':copy.deepcopy(robot._ag_release_counter),
                   'env_step':int(evaluator.env._current_step),
                   'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                   'torch_rng':torch.get_rng_state(),
                   'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    initial=physical_state(evaluator)
    torch.save({'simulator':state,'side':snapshot_side},output/'snapshot.pt')
    actions=[]
    for index in range(plan['action_steps']):
        action=hold.clone()
        action[robot.controller_action_idx['base'][0]]=0.025 if index<16 else 0.0
        actions.append(action)
    np.save(output/'actions.npy',torch.stack(actions).cpu().numpy(),allow_pickle=False)
    trials=[]
    for repeat in range(plan['repeats']+1):
        if repeat:
            _validate_full_snapshot_objects(og.sim,state)
            og.sim.load_state(copy.deepcopy(state),serialized=False)
            robot._ag_grasp_counter=copy.deepcopy(snapshot_side['grasp_counter'])
            robot._ag_release_counter=copy.deepcopy(snapshot_side['release_counter'])
            evaluator.env._current_step=snapshot_side['env_step']
            random.setstate(snapshot_side['python_rng']);np.random.set_state(snapshot_side['numpy_rng'])
            torch.set_rng_state(snapshot_side['torch_rng'])
            if snapshot_side['cuda_rng']:torch.cuda.set_rng_state_all(snapshot_side['cuda_rng'])
            og.sim.update_handles()
            for _ in range(3):og.sim.render()
        start=physical_state(evaluator)
        trajectory=[]
        for index,action in enumerate(actions):
            _,_,terminated,truncated,_=evaluator.env.step(action.clone(),n_render_iterations=1)
            trajectory.append(physical_state(evaluator))
            if bool(terminated) or bool(truncated):
                raise ValueError('Probe terminated before its frozen action sequence ended')
        trials.append({'repeat':repeat,'restored':bool(repeat),'initial':start,'trajectory':trajectory,
                       'simulator_time_at_end':float(og.sim.current_time)})
        write(output/'progress.json',{'state':'running','completed_sequences':repeat+1,
                                     'physics_action_steps':16+(repeat+1)*len(actions)})
    checks=[]
    for trial in trials[1:]:
        before=state_error(initial,trial['initial'])
        errors=[state_error(a,b) for a,b in zip(trials[0]['trajectory'],trial['trajectory'],strict=True)]
        checks.append({'repeat':trial['repeat'],'restore_error':before,'step_errors':errors,
                       'passed':passed(before,plan['restore_tolerance']) and
                                all(passed(e,plan['trajectory_tolerance']) for e in errors)})
    result={'state':'complete','kind':'new_native_state_restore_physics_probe','plan':plan,
            'model_calls':0,'training':False,'historical_rollout_fidelity_claimed':False,
            'held_object_present':any(initial['__robot__']['held_objects'].values()),
            'checks':checks,'all_checks_passed':all(c['passed'] for c in checks),
            'trials':trials,'absolute_simulator_clock_restored':False,
            'limitation':'Only the tested fresh state, action sequence and duration are qualified. '
              'This does not establish contact-state restoration or general simulator determinism.'}
    write(output/'result.json',result)
    print(json.dumps({k:result[k] for k in ('state','all_checks_passed','held_object_present','model_calls')}),flush=True)
    og.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads(args.plan.read_text())
    write(args.output/'progress.json',{'state':'initializing','started_at':time.time()})
    try:run(args.output,plan)
    except BaseException as error:
        write(args.output/'failure.json',{'state':'failed','error_type':type(error).__name__,
                                         'error':str(error)[:1000]})
        raise
