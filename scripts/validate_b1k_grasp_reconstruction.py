#!/usr/bin/env python3
"""Paired physical validation of source-inferred fixed grasps, never scoring."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import math

import diagnose_b1k_grasp_restore as diag
import eval_b1k_subtask_samples as runner

KIND = 'subtask_grasp_reconstruction_diagnostic'
BRANCHES = ('unmodified', 'source_contact_reconstruction')
METHOD = 'unique_source_comotion_and_matching_native_contact_fixed_joint'


def propose_grasps(history):
    """Select objects from observed motion, with no task/predicate inputs."""
    if len(history) != diag.HISTORY_FRAMES + 1:
        raise ValueError('Reconstruction requires a complete 65-frame source window')
    frames = [v['source_frame'] for v in history]
    if frames != list(range(frames[0], frames[0] + len(history))):
        raise ValueError('Source history is not contiguous')
    summary = diag.source_motion_summary(history)
    proposals = {}
    for arm, motion in summary.items():
        source = history[-1]['arms'][arm]
        if motion['eef_world_excursion_m'] < .03:
            raise ValueError('Source motion is too small to disambiguate grasp coupling')
        candidates = [v for v in motion['objects'] if v['median_distance_m'] < .25
                      and v['relative_position_max_deviation_m'] < .001
                      and v['relative_orientation_max_deviation_rad'] < .02]
        if len(candidates) != 1:
            raise ValueError('Source coupling is absent or ambiguous')
        if all(math.isclose(q, upper, abs_tol=1e-5, rel_tol=0)
               for q, upper in zip(source['gripper_joint_positions'], source['gripper_joint_upper_limits'])):
            raise ValueError('Source gripper is fully open')
        proposals[arm] = {**candidates[0], 'eef_world_excursion_m': motion['eef_world_excursion_m'],
                          'method': METHOD, 'original_grasp_annotation_available': False}
    if len({p['object'] for p in proposals.values()}) != len(proposals):
        raise ValueError('Two arms infer the same object; dual constraints are unsupported')
    return proposals


def pose_errors(before, after):
    import numpy as np
    distances, angles = [], []
    for arm, source in before['arms'].items():
        current = after['arms'][arm]
        pairs = [(source['eef_position'], source['eef_orientation'],
                  current['eef_position'], current['eef_orientation'])]
        pairs += [(obj['world_position'], obj['world_orientation'],
                   current['objects'][name]['world_position'], current['objects'][name]['world_orientation'])
                  for name, obj in source['objects'].items()]
        for pos, orn, target_pos, target_orn in pairs:
            distances.append(float(np.linalg.norm(np.asarray(pos)-target_pos)))
            a=np.asarray(orn,dtype=float); b=np.asarray(target_orn,dtype=float)
            angles.append(float(2*np.arccos(np.clip(abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),0,1))))
    return {'max_position_difference_m': max(distances), 'max_orientation_difference_rad': max(angles)}


def retention(observations, proposals):
    import numpy as np
    output={}
    for arm, p in proposals.items():
        name=p['object']; initial=observations[0]['arms'][arm]['objects'][name]
        rows=[v['arms'][arm] for v in observations[1:]]
        errors=[float(np.linalg.norm(np.asarray(v['objects'][name]['relative_position'])-initial['relative_position']))
                for v in rows]
        angles=[]
        q0=np.asarray(initial['relative_orientation'],dtype=float);q0/=np.linalg.norm(q0)
        for v in rows:
            q=np.asarray(v['objects'][name]['relative_orientation'],dtype=float);q/=np.linalg.norm(q)
            angles.append(float(2*np.arccos(np.clip(abs(np.dot(q,q0)),0,1))))
        held=[v['held_object']==name and v['release_counter'] is None and v['constraint_joint_type']=='FixedJoint'
              for v in rows]
        output[arm]={'object':name,'physical_steps':len(rows),'held_physical_steps':sum(held),
                     'max_relative_position_change_m':max(errors),'final_relative_position_change_m':errors[-1],
                     'max_relative_orientation_change_rad':max(angles),
                     'passed':all(held) and max(errors)<.02 and max(angles)<.1}
    return output


def _restore(evaluator, cache, sample, snapshot):
    import omnigibson as og
    runner._restore_episode_baseline(evaluator, cache)
    for step in sorted(cache['transition_by_step']):
        if step < sample['start']:
            runner._apply_transition(evaluator.env.scene, cache['transition_by_step'][step])
    runner._validate_full_snapshot_objects(og.sim, snapshot)
    og.sim.load_state(copy.deepcopy(snapshot), serialized=False)
    og.sim.update_handles()
    for _ in range(3): og.sim.render()
    evaluator.external_subtask=sample['subtask']
    runner._reset_evaluator_rollout(evaluator)


def _run_actions(evaluator, sample, actions, max_steps, objects, video, seed, base):
    import numpy as np
    import torch as th
    base.seed_everything(seed)
    observations=[diag.observe(evaluator,objects,frame=sample['start'])];executed=[]
    evaluator.start_recording(str(video),rate=30)
    try:
        for i in range(max_steps):
            act=np.asarray(actions[sample['start']+i],dtype=np.float32)
            if act.shape!=(23,) or not np.isfinite(act).all(): raise ValueError('Invalid recorded action')
            obs,_,terminated,truncated,_=evaluator.env.step(th.from_numpy(act.copy()),n_render_iterations=1)
            evaluator.obs=evaluator._preprocess_obs(evaluator._sync_lights_and_get_obs(obs))
            evaluator._write_video('SOURCE GRASP RECONSTRUCTION DIAGNOSTIC: recorded actions')
            executed.append(act.tolist())
            observations.append(diag.observe(evaluator,objects,frame=sample['start']+i+1,physics_steps=i+1))
            if terminated or truncated:break
    finally:evaluator.stop_recording()
    return {'recorded_actions':executed,'observations':observations,'physics_steps':len(executed),
            'video_sha256':runner.file_sha256(video),'paired_seed':seed}


def diagnose(base,evaluator,args,panel,plan,samples,cache):
    import h5py
    import numpy as np
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    objects,histories,snapshots,windows=diag.materialize_source_observations(evaluator,samples,cache)
    output=Path(args.output_dir)/'grasp_reconstruction';output.mkdir(parents=True,exist_ok=True)
    videos=output/'videos';videos.mkdir(exist_ok=True)
    native_snapshots=output/'native_snapshots';native_snapshots.mkdir(exist_ok=True)
    enabled=gm.ENABLE_TRANSITION_RULES
    with h5py.File(cache['path'],'r',locking=False) as f:
        actions=f['data'][cache['details']['demo_group']]['action']
        with gm.unlocked():gm.ENABLE_TRANSITION_RULES=False
        try:
            for sample in samples:
                stem=runner._artifact_stem(sample);start=sample['start'];snapshot=snapshots[start]
                history=[histories[i] for i in windows[sample['sample_id']]]
                proposals=propose_grasps(history)
                # Both arms must close in this bounded experiment; opening a coupled
                # object would test a release instead of retention.
                selected=np.asarray(actions[start:start+plan['max_steps']],dtype=np.float32)
                if selected.shape!=(plan['max_steps'],23) or np.any(selected[:,[14,22]]>=1):
                    raise ValueError('Paired retention requires recorded closing gripper commands')
                base.seed_everything(plan['paired_seed'])
                _restore(evaluator,cache,sample,snapshot)
                branches={'unmodified':_run_actions(evaluator,sample,actions,plan['max_steps'],objects,
                                                    videos/(stem+'-unmodified.mp4'),plan['paired_seed'],base)}
                base.seed_everything(plan['paired_seed'])
                _restore(evaluator,cache,sample,snapshot)
                source=diag.observe(evaluator,objects,frame=start)
                # One real physical source action refreshes contacts. No stale
                # before-physics cache is used as contact evidence.
                act=np.asarray(actions[start],dtype=np.float32)
                evaluator.env.step(th.from_numpy(act.copy()),n_render_iterations=1)
                contact_observation=diag.observe(evaluator,objects,frame=start+1,physics_steps=1)
                contacts={}
                for arm,p in proposals.items():
                    candidate=contact_observation['arms'][arm]['native_grasp_candidate']
                    if not candidate or candidate['object']!=p['object']:
                        raise ValueError('Physical native candidate does not match source coupling')
                    target=evaluator.env.scene.object_registry('name',candidate['object'])
                    link=candidate['link'];joint_type=evaluator.robot._get_assisted_grasp_joint_type(target,link)
                    point=evaluator.robot._find_finger_contact_position(arm,target.links[link].prim_path)
                    if point is None or joint_type!='FixedJoint':
                        raise ValueError('No physical contact anchor or supported fixed grasp')
                    contacts[arm]={'object':target.name,'link':link,'world_contact_position':point.tolist(),
                                   'joint_type':joint_type,'link_mass_kg':float(target.links[link].mass)}
                # Return to the exact saved poses and joints before adding the
                # supported constraint. Contact-probe advancement is not hidden.
                _restore(evaluator,cache,sample,snapshot)
                before=diag.observe(evaluator,objects,frame=start)
                q_before=evaluator.robot.get_joint_positions().clone()
                for arm,c in contacts.items():
                    target=evaluator.env.scene.object_registry('name',c['object'])
                    evaluator.robot._establish_grasp(target,c['link'],arm,th.as_tensor(c['world_contact_position'],device=q_before.device,dtype=q_before.dtype),'FixedJoint')
                after=diag.observe(evaluator,objects,frame=start)
                unchanged=pose_errors(before,after)
                if unchanged['max_position_difference_m']>1e-5 or unchanged['max_orientation_difference_rad']>1e-5 or not th.equal(q_before,evaluator.robot.get_joint_positions()):
                    raise ValueError('Creating source-inferred grasp changed the saved initial pose/joints')
                native=runner._dump_full_simulator_state(evaluator.env.scene,og.sim)
                native_path=native_snapshots/(stem+'.pt')
                # This is an internally produced native state, with no credentials.
                th.save(native,native_path)
                known_before=diag.observe(evaluator,objects,frame=start)
                # Release constraints and physically relocate their objects so an
                # unchanged restore cannot pass a no-op roundtrip.
                for arm,c in contacts.items():
                    evaluator.robot.release_grasp_immediately(arm=arm)
                    obj=evaluator.env.scene.object_registry('name',c['object'])
                    pos,orn=obj.get_position_orientation();obj.set_position_orientation(pos+pos.new_tensor([.2,0,0]),orn)
                og.sim.update_handles();og.sim.render()
                released=diag.observe(evaluator,objects,frame=start)
                if any(released['arms'][arm]['held_object'] is not None for arm in proposals):
                    raise ValueError('Known snapshot test did not actually release the grasps')
                og.sim.load_state(copy.deepcopy(native),serialized=False);og.sim.update_handles()
                for _ in range(3):og.sim.render()
                reloaded=diag.observe(evaluator,objects,frame=start)
                roundtrip=pose_errors(known_before,reloaded)
                if roundtrip['max_position_difference_m']>1e-5 or roundtrip['max_orientation_difference_rad']>1e-5:
                    raise ValueError('Known native snapshot did not restore source poses')
                if any(reloaded['arms'][arm]['held_object']!=p['object'] or reloaded['arms'][arm]['constraint_joint_type']!='FixedJoint'
                       or reloaded['arms'][arm]['release_counter'] is not None for arm,p in proposals.items()):
                    raise ValueError('Known native snapshot did not restore its recorded grasp')
                q_reloaded=evaluator.robot.get_joint_positions().tolist()
                runner._reset_evaluator_rollout(evaluator)
                branches['source_contact_reconstruction']=_run_actions(evaluator,sample,actions,plan['max_steps'],objects,
                                        videos/(stem+'-source_contact_reconstruction.mp4'),plan['paired_seed'],base)
                for branch in branches.values():branch['retention']=retention(branch['observations'],proposals)
                result={'kind':KIND,'official_benchmark':False,'metric_eligible':False,
                        'sample_id':sample['sample_id'],'task':sample['task'],'instance_id':sample['task_instance_id'],
                        'episode_index':sample['episode_index'],'raw_episode_id':sample['raw_episode_id'],'source_start':start,
                        'panel_fingerprint':panel['fingerprint'],'raw_sha256':cache['sha256'],
                        'model_action_requests':0,'advancement_source':'recorded_demonstration_actions',
                        'method':METHOD,'proposals':proposals,'source_history':history,'source_observation':source,
                        'physical_contact_probe':{'recorded_action':act.tolist(),'physics_steps':1,
                                                  'observation':contact_observation,'contacts':contacts},
                        'initial_pose_unchanged':unchanged,'initial_robot_joint_positions':q_before.tolist(),
                        'known_native_snapshot':{
                            'sha256':runner.file_sha256(native_path),'before':known_before,'after_release':released,
                            'after_reload':reloaded,'pose_errors':roundtrip,
                            'after_reload_joint_positions':q_reloaded,
                            'recreated_known_constraints':True,
                            'original_legacy_grasp_verified':False},
                        'branches':branches,'exact_legacy_grasp_restoration_verified':False,
                        'reconstruction_source_sha256':runner.file_sha256(Path(__file__)),
                        'native_robot_source_sha256':runner.file_sha256(Path(og.__file__).parent/'robots/robot.py')}
                runner.atomic_write_json(output/(stem+'.json'),result)
                print(f'GRASP_RECONSTRUCTION_SAMPLE sample={sample["sample_id"]} action_steps={sum(v["physics_steps"] for v in branches.values())+1}',flush=True)
        finally:
            with gm.unlocked():gm.ENABLE_TRANSITION_RULES=enabled


def validate_results(root,panel,plan,samples):
    expected={s['sample_id']:s for s in samples};seen=set();steps=0;passed=0
    for path in Path(root).glob('pi05/*/grasp_reconstruction/*.json'):
        d=json.loads(path.read_text());sid=d.get('sample_id')
        if sid not in expected or sid in seen:raise ValueError('Unexpected/duplicate reconstruction sample')
        sample=expected[sid]
        for key,reference in [('task','task'),('episode_index','episode_index'),('raw_episode_id','raw_episode_id'),
                              ('source_start','start'),('instance_id','task_instance_id')]:
            if d.get(key)!=sample[reference]:raise ValueError('Reconstruction source identity differs')
        if d.get('kind')!=KIND or d.get('metric_eligible') is not False or d.get('official_benchmark') is not False or d.get('exact_legacy_grasp_restoration_verified') is not False or d.get('model_action_requests')!=0:
            raise ValueError('Reconstruction must not claim scoring or original grasp verification')
        if d.get('panel_fingerprint')!=panel['fingerprint'] or d.get('method')!=METHOD or d.get('advancement_source')!='recorded_demonstration_actions':
            raise ValueError('Reconstruction contract differs')
        def finite(v):
            if isinstance(v,dict):return all(finite(x) for x in v.values())
            if isinstance(v,list):return all(finite(x) for x in v)
            return not isinstance(v,float) or math.isfinite(v)
        if not finite(d):raise ValueError('Nonfinite reconstruction evidence')
        history=d['source_history']
        if [v['source_frame'] for v in history]!=list(range(sample['start']-64,sample['start']+1)):
            raise ValueError('Source motion window differs')
        if not diag.source_motion_summaries_match(d['proposals'],propose_grasps(history)):
            raise ValueError('Proposal differs from actual source motion')
        known=d['known_native_snapshot']
        native=path.parent/'native_snapshots'/(path.stem+'.pt')
        if not native.is_file() or runner.file_sha256(native)!=known['sha256'] or known['recreated_known_constraints'] is not True or known['original_legacy_grasp_verified'] is not False:
            raise ValueError('Missing known native state roundtrip evidence')
        for key in ('raw_sha256','reconstruction_source_sha256','native_robot_source_sha256'):
            value=d.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Missing source hash')
        probe=d['physical_contact_probe']
        if probe['physics_steps']!=1 or len(probe['recorded_action'])!=23:raise ValueError('Missing actual physical contact probe')
        for arm,p in d['proposals'].items():
            contact=probe['contacts'][arm];candidate=probe['observation']['arms'][arm]['native_grasp_candidate']
            if contact['object']!=p['object'] or contact['joint_type']!='FixedJoint' or not candidate or candidate['object']!=p['object'] or candidate['link']!=contact['link']:
                raise ValueError('Physical contact does not support inferred source object')
            if known['before']['arms'][arm]['held_object']!=p['object'] or known['after_release']['arms'][arm]['held_object'] is not None or known['after_reload']['arms'][arm]['held_object']!=p['object']:
                raise ValueError('Known native constraint release/reload is unverified')
        if pose_errors(known['before'],known['after_release'])['max_position_difference_m'] < .1:
            raise ValueError('Known native roundtrip did not actually relocate objects')
        initial_errors=pose_errors(d['source_observation'],known['before'])
        if not diag.source_motion_summaries_match(initial_errors,d['initial_pose_unchanged']) or max(initial_errors.values())>1e-5:
            raise ValueError('Reconstruction changed the recorded start pose')
        if d['initial_robot_joint_positions']!=known['after_reload_joint_positions']:
            raise ValueError('Known native restored robot joints differ')
        actual_pose_errors=pose_errors(known['before'],known['after_reload'])
        if not diag.source_motion_summaries_match(known['pose_errors'],actual_pose_errors) or max(actual_pose_errors.values())>1e-5:
            raise ValueError('Known native restored poses differ')
        if set(d['branches'])!=set(BRANCHES):raise ValueError('Missing paired branch')
        good=True
        for name,branch in d['branches'].items():
            n=branch['physics_steps'];obs=branch['observations']
            if type(n) is not int or n!=plan['max_steps'] or len(obs)!=n+1 or len(branch['recorded_actions'])!=n or branch['paired_seed']!=plan['paired_seed']:
                raise ValueError('Incomplete paired physics evidence')
            if any(len(a)!=23 for a in branch['recorded_actions']):raise ValueError('Invalid paired actions')
            if [(v['source_frame'],v['physics_steps']) for v in obs]!=[(sample['start']+i,i) for i in range(n+1)]:raise ValueError('Observation timing differs')
            if not diag.source_motion_summaries_match(branch['retention'],retention(obs,d['proposals'])):raise ValueError('Retention report differs from actual observations')
            video=path.parent/'videos'/(path.stem+'-'+name+'.mp4')
            if not video.is_file() or runner.file_sha256(video)!=branch['video_sha256']:raise ValueError('Missing paired video')
            steps+=n
            if name=='source_contact_reconstruction':good=all(v['passed'] for v in branch['retention'].values())
        if d['branches']['unmodified']['recorded_actions']!=d['branches']['source_contact_reconstruction']['recorded_actions']:
            raise ValueError('Paired branches use different actions')
        passed+=int(good);steps+=1;seen.add(sid)
    if seen!=set(expected):raise ValueError('Incomplete reconstruction diagnosis')
    return {'kind':KIND,'complete':True,'official_benchmark':False,'metric_eligible':False,
            'samples':len(seen),'recorded_action_physics_steps':steps,'model_action_requests':0,
            'source_inferred_retention_passed':passed,'exact_legacy_grasp_restoration_verified':False}
