#!/usr/bin/env python3
"""Collect real executed subtask trajectories through the existing evaluator."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import eval_b1k_subtask_samples as runner
from subtask_improvement import KIND, load_plan
from subtask_trajectory import TrajectoryWriter, inspect_trajectory


class SeededClient:
    def __init__(self, policy):
        self.wrapped=policy
        self.next_seed=None
    def __getattr__(self,key):return getattr(self.wrapped,key)
    def reset(self):self.wrapped.reset()
    def forward(self,obs):
        if self.next_seed is not None:
            metadata=self.wrapped.policy.get_server_metadata()
            if metadata.get('collection_seeding') is not True:
                raise RuntimeError('Server cannot honor per-trial collection seeds')
            obs={**obs,'_collection_policy_seed':self.next_seed}
            self.next_seed=None
        return self.wrapped.forward(obs=obs)



class AuditedTrajectoryWriter:
    """Record training arrays and per-control proof in the same callback."""
    def __init__(self, writer):
        from validate_b1k_native_snapshot_model import ActionAudit
        self.writer, self.audit = writer, ActionAudit()
    def record(self, frame, action, terminated, truncated):
        self.audit.record(frame, action, terminated, truncated)
        self.writer.record(frame, action, terminated, truncated)


def validate_collection(root, panel, plan, *, require_complete=True):
    """All attempts must finish; accepted files require proven content and video."""
    import h5py
    import math
    samples = {s['sample_id']: s for s in panel['samples']}
    allowed = {t['trial_id']: t for t in plan['trials']}
    found, accepted, eligible, excluded = set(), 0, 0, 0
    for done_path in sorted(Path(root).glob('pi05/*/trials/*/completed.json')):
        directory = done_path.parent
        trial_id = directory.name
        if trial_id not in allowed or trial_id in found:
            raise ValueError('Unknown or duplicate trial in collection')
        found.add(trial_id)
        done = json.loads(done_path.read_text())
        trial = allowed[trial_id]
        sample = samples[trial['sample_id']]
        with h5py.File(directory/'trajectory.hdf5', 'r') as stream:
            if not stream.attrs.get('complete'):
                raise ValueError('Unfinished collection trajectory')
            metadata = json.loads(stream.attrs['metadata'])
            result = json.loads(stream.attrs['result'])
            n = int(stream.attrs['control_steps'])
        if (metadata.get('plan_fingerprint') != plan['fingerprint'] or metadata.get('trial') != trial
                or metadata.get('sample') != sample or done.get('plan_fingerprint') != plan['fingerprint']
                or done.get('trial_id') != trial_id):
            raise ValueError('Collection provenance differs from frozen plan')
        if (type(result.get('steps')) is not int or result['steps'] != n or not 0 <= n <= plan['max_steps']
                or any(done.get(k) != result.get(k) for k in ('success', 'metric_eligible', 'steps'))
                or result.get('sample_id') != sample['sample_id'] or result.get('panel_fingerprint') != panel['fingerprint']):
            raise ValueError('Collection result identity or action budget differs')
        if plan.get('require_recorded_native_state') or plan.get('require_recorded_source_state'):
            if plan.get('require_recorded_source_state'):
                from full_subtask_collection import validate_result
                validate_result(result, sample, panel['fingerprint'], plan)
            else:
                from recorded_native_state import validate_result_proof
                validate_result_proof(result, sample, plan)
            video = directory/'videos'/(runner._artifact_stem(sample)+'.mp4')
            if result['metric_eligible'] and runner.file_sha256(video) != result.get('video_sha256'):
                raise ValueError('Native collection video changed')
        if result.get('metric_eligible') is True:
            eligible += 1
            score = result.get('q_score', float('nan'))
            if isinstance(score, dict):
                score = score.get('final', float('nan'))
            try:
                finite_score = math.isfinite(float(score))
            except (TypeError, ValueError):
                finite_score = False
            if n <= 0 or not finite_score:
                raise ValueError('Eligible attempt lacks actual actions or finite simulator score')
            inspect_trajectory(directory/'trajectory.hdf5', require_success=False)
            video = directory/'videos'/(runner._artifact_stem(sample)+'.mp4')
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError('Eligible attempt lacks nonempty video')
        elif result.get('metric_eligible') is False and n == 0 and result.get('success') is False:
            excluded += 1
        else:
            raise ValueError('Invalid collection eligibility')
        receipt = directory/'accepted.json'
        if result.get('success') is True:
            inspected = inspect_trajectory(directory/'trajectory.hdf5')
            if not receipt.is_file() or json.loads(receipt.read_text()).get('content_sha256') != inspected['content_sha256']:
                raise ValueError('Successful trajectory lacks a verified acceptance receipt')
            accepted += 1
        elif receipt.exists():
            raise ValueError('Failed attempt must not be labeled accepted')
    if require_complete and found != set(allowed):
        raise ValueError('Collection is incomplete')
    complete = found == set(allowed)
    return dict(kind=KIND, complete=complete, official_benchmark=False, plan_fingerprint=plan['fingerprint'],
                expected_trials=len(allowed), completed_trials=len(found), eligible_trials=eligible,
                setup_excluded=excluded, accepted_trials=accepted,
                acceptance_rate=accepted/eligible if eligible else None, ready_for_export=complete and accepted > 0,
                completed_trial_ids=sorted(found))


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--subtask-panel',type=Path,required=True)
    parser.add_argument('--collection-plan',type=Path,required=True)
    parser.add_argument('--raw-data-root',type=Path,required=True)
    parser.add_argument('--sample-max-steps',type=int,required=True)
    parser.add_argument('--episode-index',type=int)
    parser.add_argument('--resume-completed',action='store_true')
    parser.add_argument('--task-name',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    collection_args,base_argv=parser.parse_known_args(argv)
    panel=runner.load_panel(collection_args.subtask_panel)
    plan,sample_by_id=load_plan(collection_args.collection_plan,panel)
    trials=[t for t in plan['trials'] if sample_by_id[t['sample_id']]['task']==collection_args.task_name]
    if not trials:raise ValueError('Task is outside collection plan')
    completed=set()
    if collection_args.resume_completed:
        completed=set(validate_collection(collection_args.output_dir.parent.parent,panel,plan,
                                         require_complete=False)['completed_trial_ids'])
    if collection_args.episode_index is None:
        import subprocess, sys
        # One simulator process per source episode, as in the newer Nebula collector.
        # Isaac Sim native memory and scene objects do not survive between episodes.
        original=list(argv if argv is not None else sys.argv[1:])
        episodes=sorted({sample_by_id[t['sample_id']]['episode_index'] for t in trials
                         if t['trial_id'] not in completed})
        for episode in episodes:
            subprocess.run([sys.executable,__file__,*original,'--episode-index',str(episode)],check=True)
        return
    base=runner._load_base_module()
    args=base.parse_args(['--task-name',collection_args.task_name,'--output-dir',str(collection_args.output_dir),*base_argv])
    if args.mode!='train' or args.num_rollouts!=1 or not args.write_video or args.external_subtask or args.external_subtask_selector:
        raise ValueError('Collection requires train states, fixed annotated subtasks, videos, and one logical rollout')
    if args.env_wrapper != 'omnigibson.eval.wrappers.DefaultWrapper':
        raise ValueError('Collection storage contract requires the default 224x224 RGB wrapper')
    panel=runner.load_panel(collection_args.subtask_panel)
    plan,sample_by_id=load_plan(collection_args.collection_plan,panel)
    if plan['max_steps']!=collection_args.sample_max_steps:
        raise ValueError('Collection step limit differs from frozen plan')
    args.sample_max_steps=plan['max_steps']
    trials=[t for t in plan['trials'] if sample_by_id[t['sample_id']]['task']==args.task_name]
    if not trials:raise ValueError('Task is outside collection plan')
    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    episodes=[collection_args.episode_index]
    if episodes[0] not in {sample_by_id[t['sample_id']]['episode_index'] for t in trials}:
        raise ValueError('Episode is outside collection plan')
    for episode_index in episodes:
        episode_trials=[t for t in trials if sample_by_id[t['sample_id']]['episode_index']==episode_index
                        and t['trial_id'] not in completed]
        if not episode_trials:continue
        samples=list({t['sample_id']:sample_by_id[t['sample_id']] for t in episode_trials}.values())
        full_protocol = plan.get('require_recorded_source_state') is True
        cache=runner.load_episode_cache(runner.raw_hdf5_path(collection_args.raw_data_root,samples[0]),samples,
                                       allow_unverified_grasp_state=full_protocol)
        history_recovery = cache['grasp_serialization_audit']['legacy_missing_assisted_grasp_state']
        if history_recovery:
            from legacy_history_recovery import enabled, CONTRACT
            if not full_protocol or not enabled() or any(plan['source_protocol_by_sample'][s['sample_id']] != CONTRACT for s in samples):
                raise ValueError('History collection lacks its explicit frozen source protocol')
        if plan.get('require_recorded_native_state') or full_protocol:
            from recorded_native_state import decode_start
            relative=f"task-{samples[0]['task_index']:04d}/episode_{samples[0]['raw_episode_id']:08d}.hdf5"
            if cache['sha256'] != plan['raw_source_sha256'][relative]:
                raise ValueError('Collection source differs from its frozen native file')
            for sample in samples:
                if history_recovery:
                    import hashlib, h5py, numpy as np
                    source_hash = cache['sample_evidence'][sample['sample_id']]['selected_state_sha256']
                    with h5py.File(cache['path'], 'r', locking=False) as source:
                        actions = np.asarray(source['data'][cache['details']['demo_group']]['action'][:sample['start']], dtype=np.float32)
                    if hashlib.sha256(actions.tobytes()).hexdigest() != plan['prefix_actions_sha256'][sample['sample_id']]:
                        raise ValueError('Collection history prefix changed')
                else:
                    source_hash = decode_start(cache,sample)['selected_state_sha256']
                if source_hash != plan['selected_state_sha256'][sample['sample_id']]:
                    raise ValueError('Collection source row differs from its frozen native start')
        base.gm.HEADLESS=args.headless
        evaluator_args=copy.copy(args);evaluator_args.external_subtask=samples[0]['subtask'];evaluator_args.max_steps=args.sample_max_steps
        cfg=runner._build_evaluator_cfg(base,evaluator_args,base.seed_everything(base.DEFAULT_EVAL_SEED))
        with runner.create_sample_evaluator(base,cfg,cache,samples[0],args.sample_max_steps) as evaluator:
            if history_recovery:
                from legacy_history_recovery import materialize
                materialized=materialize(evaluator,cache,samples)
            else:
                materialized=runner.materialize_episode_start_states(evaluator,cache,samples)
            evaluator.policy=SeededClient(evaluator.policy)
            for trial in episode_trials:
                out=root/'trials'/trial['trial_id']
                if out.exists():raise ValueError('Each collection trial requires a fresh directory')
                out.mkdir(parents=True)
                sample=sample_by_id[trial['sample_id']]
                base.seed_everything(trial['policy_seed'])
                evaluator.policy.next_seed=trial['policy_seed']
                metadata=dict(schema_version=1,plan_fingerprint=plan['fingerprint'],checkpoint=plan['checkpoint'],
                              conditioning=plan['conditioning'],trial=trial,sample=sample,
                              fps=int(evaluator.env.env_config['rendering_frequency']),
                              action_semantics='executed_R1Pro_control_23d',pairing='observation_before_executed_action')
                writer=TrajectoryWriter(out/'trajectory.hdf5',metadata)
                try:
                    audited=AuditedTrajectoryWriter(writer) if plan.get('require_recorded_native_state') or full_protocol else writer
                    restoration_validator = None
                    if history_recovery:
                        from legacy_history_recovery import restored_context
                        restoration_validator = lambda actual: restored_context(actual, cache, sample, materialized[sample['sample_id']])
                    result=runner.run_sample(base,evaluator,args,sample,cache,materialized[sample['sample_id']],out,
                                             panel['fingerprint'],trajectory_writer=audited,
                                             restoration_validator=restoration_validator)
                    if plan.get('require_recorded_native_state') or full_protocol:
                        video=out/'videos'/(runner._artifact_stem(sample)+'.mp4')
                        result.update(model_action_audit=audited.audit.rows,
                            actual_model_control_steps=len(audited.audit.rows),
                            production_difficulty_baseline=False,
                            video_sha256=runner.file_sha256(video) if result['metric_eligible'] else None)
                        if history_recovery:
                            from legacy_history_recovery import CONTRACT, METHOD
                            result.update(state_recovery_protocol=CONTRACT, grasp_recovery_method=METHOD,
                                exact_original_grasp_anchor_bytes_verified=False, original_sample_scope_preserved=True)
                        if full_protocol:
                            from full_subtask_collection import validate_result
                            validate_result(result,sample,panel['fingerprint'],plan)
                        else:
                            from recorded_native_state import validate_result_proof
                            validate_result_proof(result,sample,plan)
                        runner.atomic_write_json(out/'json'/(runner._artifact_stem(sample)+'.json'),result)
                    writer.finish(result)
                    if result['metric_eligible'] and result['success']:
                        inspected=inspect_trajectory(out/'trajectory.hdf5')
                        video=out/'videos'/(runner._artifact_stem(sample)+'.mp4')
                        if not video.is_file() or video.stat().st_size==0:raise RuntimeError('Accepted trajectory lacks video')
                        runner.atomic_write_json(out/'accepted.json',dict(trial_id=trial['trial_id'],
                            plan_fingerprint=plan['fingerprint'],content_sha256=inspected['content_sha256'],steps=result['steps']))
                    runner.atomic_write_json(out/'completed.json',dict(trial_id=trial['trial_id'],success=result['success'],
                        metric_eligible=result['metric_eligible'],steps=result['steps'],plan_fingerprint=plan['fingerprint']))
                    if plan.get('require_recorded_source_state'):
                        from collection_checkpoint import save_trial
                        save_trial(out,plan)
                    print(f"Collected trial {trial['trial_id']}: steps={result['steps']} success={result['success']}",flush=True)
                finally:writer.close()


if __name__=='__main__':main()
