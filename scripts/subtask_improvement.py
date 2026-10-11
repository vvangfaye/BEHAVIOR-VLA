#!/usr/bin/env python3
"""Freeze an auditable collection plan from complete exact-state diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from subtask_sample_panel import load_panel

KIND = 'subtask_success_collection'
# Conservative per-step allowance for three RGB streams, proprio/action,
# HDF5 chunk overhead and diagnostic video; compression is not assumed.
OUTPUT_BYTES_PER_STEP = 3*224*224*3 + 512*4 + 23*4 + 2 + 2*65536


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze_plan(summary, panel, *, max_groups=3, max_starts=10, trials=10, min_eligible=10,
                threshold=0.5, max_steps=2048, seed=1, held_out_instances=(), checkpoint_identity=None):
    if summary.get('complete') is not True or summary.get('evaluation_kind') != 'training_subtask_closed_loop':
        raise ValueError('Selection requires a complete production exact-state diagnostic')
    fidelity = summary.get('state_fidelity', {})
    if (fidelity.get('contract') != 'recorded_native_fields_v1' or fidelity.get('required') is not True
            or fidelity.get('verified') is not True or summary.get('sample_max_steps') != 2048
            or fidelity.get('eligible_results_verified') != summary.get('overall', {}).get('metric_eligible')):
        raise ValueError('Selection requires verified recorded-native fields and the formal 2048-step baseline')
    if summary.get('panel_fingerprint') != panel['fingerprint']:
        raise ValueError('Diagnostic and panel identities differ')
    for field in ['missing_results', 'invalid_results', 'unexpected_results', 'duplicate_results']:
        if summary.get(field):
            raise ValueError(f'Diagnostic has {field}')
    if (any(type(v) is not int or v <= 0 for v in (max_groups, max_starts, trials, min_eligible, max_steps))
            or max_steps > 4096 or type(seed) is not int or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Collection budgets and evidence requirements must be positive')
    excluded_instances = {(str(t), int(i)) for t, i in held_out_instances}
    sample_by_id = {s['sample_id']: s for s in panel['samples']}
    setup_exclusions = summary.get('setup_exclusions', [])
    source_exclusions = summary.get('source_exclusions', [])
    if not isinstance(setup_exclusions, list) or not isinstance(source_exclusions, list):raise ValueError('Invalid baseline setup/source exclusions')
    source_excluded_ids = {item.get('sample_id') for item in source_exclusions if isinstance(item, dict)}
    setup_exclusions = setup_exclusions + source_exclusions
    excluded_starts = set()
    for item in setup_exclusions:
        sid = item.get('sample_id') if isinstance(item, dict) else None
        if sid not in sample_by_id or sid in excluded_starts:raise ValueError('Unknown or duplicate baseline setup exclusion')
        sample = sample_by_id[sid]
        if item.get('task') != sample['task'] or item.get('subtask') != sample['subtask']:
            raise ValueError('Baseline setup exclusion differs from frozen sample')
        excluded_starts.add(sid)
    groups, underpowered = [], []
    for task, prompts in sorted(summary['per_prompt_label'].items()):
        for prompt, counts in sorted(prompts.items()):
            n, successes = counts['metric_eligible'], counts['successes']
            if type(n) is not int or type(successes) is not int or not 0 <= successes <= n:
                raise ValueError('Invalid eligible/success counts')
            if n == 0:
                continue
            rate = successes / n
            reported = counts.get('completion_rate')
            if reported is None or not math.isclose(rate, reported, abs_tol=1e-12):
                raise ValueError('Success rate denominator differs from eligible counts')
            if rate >= threshold:
                continue
            record = dict(task=task, subtask=prompt, eligible=n, successes=successes, rate=rate)
            if n < min_eligible:
                underpowered.append(record)
                continue
            groups.append(record)
    groups.sort(key=lambda g: (g['rate'], -g['eligible'], g['task'], g['subtask']))
    origins, selected_groups = [], []
    for group in groups:
        starts = [s for s in panel['samples'] if (s['task'], s['subtask']) == (group['task'], group['subtask'])
                  and (s['task'], s['task_instance_id']) not in excluded_instances
                  and s['sample_id'] not in excluded_starts]
        starts.sort(key=lambda s: digest([seed, s['sample_id']]))
        # Prefer distinct instances before selecting another segment from one instance.
        chosen, deferred, instances = [], [], set()
        for sample in starts:
            key = (sample['task'], sample['task_instance_id'])
            if key in instances:
                deferred.append(sample)
            else:
                chosen.append(sample); instances.add(key)
        chosen = (chosen + deferred)[:max_starts]
        if not chosen:
            continue
        selected_groups.append({**group, 'selected_starts':len(chosen), 'distinct_instances':len({s['task_instance_id'] for s in chosen})})
        origins.extend(s['sample_id'] for s in chosen)
        if len(selected_groups) == max_groups:
            break
    trial_records = []
    for sample_id in origins:
        for repeat in range(trials):
            policy_seed = int(digest([seed, sample_id, repeat])[:8], 16)
            trial_records.append({'trial_id':digest([sample_id, repeat, policy_seed])[:24],
                                  'sample_id':sample_id, 'repeat':repeat, 'policy_seed':policy_seed})
    payload = dict(schema_version=1, kind=KIND, panel_fingerprint=panel['fingerprint'],
                   evaluation_summary_sha256=digest(summary), checkpoint=summary['checkpoint'],
                   checkpoint_identity=checkpoint_identity,
                   conditioning='task_and_subtask', mode='train', official_benchmark=False,
                   max_steps=max_steps, seed=seed, selected_groups=selected_groups,
                   underpowered_groups=underpowered, held_out_instances=sorted(map(list, excluded_instances)),
                   baseline_setup_excluded_sample_ids=sorted(excluded_starts - source_excluded_ids),
                   baseline_source_excluded_sample_ids=sorted(source_excluded_ids),
                   trials=trial_records, expected_trials=len(trial_records),
                   expected_starts=len(origins), maximum_control_steps=len(trial_records)*max_steps,
                   maximum_output_bytes=len(trial_records)*max_steps*OUTPUT_BYTES_PER_STEP,
                   selection=dict(max_groups=max_groups,max_starts=max_starts,trials_per_start=trials,
                                  min_eligible=min_eligible,threshold=threshold),
                   augmentation='distinct_recorded_training_starts',
                   quality=dict(require_success=True,require_video=True,require_finite_arrays=True,
                                require_action_dimension=23,require_fixed_subtask=True))
    return {**payload, 'fingerprint':digest(payload)}



PILOT_PURPOSE = 'small_data_augmentation_feasibility_v1'


def freeze_pilot_plan(panel, sample_ids, source_proofs, runtime_evidence, *,
                      checkpoint_identity, held_out_instances, trials=3, max_steps=768, seed=1):
    """Select explicit feasibility targets without claiming statistical difficulty.

    Unlike difficulty selection, this bounded experiment need not wait for the
    full benchmark. Native source checks and independent instance holds remain.
    """
    samples = {s['sample_id']: s for s in panel['samples']}
    if (not isinstance(sample_ids, list) or not 1 <= len(sample_ids) <= 6
            or len(set(sample_ids)) != len(sample_ids) or set(sample_ids) - samples.keys()
            or type(trials) is not int or not 1 <= trials <= 6
            or type(max_steps) is not int or not 1 <= max_steps <= 1024
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('Feasibility pilot requires at most 6 starts, 6 seeds and 1024 steps')
    held_out = sorted({(str(t), int(i)) for t, i in held_out_instances})
    if not held_out or any((samples[s]['task'], samples[s]['task_instance_id']) in held_out for s in sample_ids):
        raise ValueError('Pilot requires independent reserved instances and cannot collect them')
    selected = [samples[s] for s in sample_ids]
    groups = sorted({(s['task'], s['subtask']) for s in selected})
    if len(groups) > 3:
        raise ValueError('Feasibility pilot supports at most 3 subtask groups')
    if (runtime_evidence.get('verified') is not True or runtime_evidence.get('complete') is not True
            or runtime_evidence.get('checkpoint_step') != checkpoint_identity.get('checkpoint_step')
            or runtime_evidence.get('checkpoint_manifest_sha256') != checkpoint_identity.get('checkpoint_manifest_sha256')
            or runtime_evidence.get('panel_fingerprint') != panel['fingerprint']):
        raise ValueError('Pilot requires a completed native GPU proof for the same checkpoint and panel')
    verified = {s['sample_id'] for s in runtime_evidence.get('samples', [])
                if s.get('metric_eligible') is True and s.get('raw_native_fields_independently_compared') is True
                and type(s.get('steps')) is int and s['steps'] > 0}
    if any(not any(s['sample_id'] in verified for s in selected if (s['task'], s['subtask']) == g) for g in groups):
        raise ValueError('Each pilot group needs a previously verified actual native model rollout')
    raw_hashes, state_hashes = {}, {}
    import re
    for s in selected:
        proof = source_proofs.get(s['sample_id'], {})
        relative = f"task-{s['task_index']:04d}/episode_{s['raw_episode_id']:08d}.hdf5"
        if (proof.get('contract') != 'recorded_native_fields_v1' or proof.get('sample_id') != s['sample_id']
                or proof.get('source_start') != s['start']
                or any(not isinstance(proof.get(k), str) or not re.fullmatch(r'[0-9a-f]{64}', proof[k])
                       for k in ('raw_sha256', 'selected_state_sha256'))):
            raise ValueError('Pilot start lacks frozen original native source evidence')
        if relative in raw_hashes and raw_hashes[relative] != proof['raw_sha256']:
            raise ValueError('Conflicting original source hashes')
        raw_hashes[relative] = proof['raw_sha256']; state_hashes[s['sample_id']] = proof['selected_state_sha256']
    records = []
    for sid in sample_ids:
        for repeat in range(trials):
            policy_seed = int(digest([seed, sid, repeat])[:8], 16)
            records.append(dict(trial_id=digest([sid, repeat, policy_seed])[:24],
                                sample_id=sid, repeat=repeat, policy_seed=policy_seed))
    payload = dict(schema_version=1, kind=KIND, panel_fingerprint=panel['fingerprint'],
        evaluation_summary_sha256=None, checkpoint=checkpoint_identity['checkpoint_source'],
        checkpoint_identity=checkpoint_identity, conditioning='task_and_subtask', mode='train',
        official_benchmark=False, experiment_purpose=PILOT_PURPOSE, difficulty_selection_verified=False,
        production_difficulty_baseline=False, require_recorded_native_state=True,
        raw_source_sha256=raw_hashes, selected_state_sha256=state_hashes,
        runtime_validation_evidence=runtime_evidence, max_steps=max_steps, seed=seed,
        held_out_instances=[list(v) for v in held_out],
        selected_groups=[dict(task=t, subtask=p, eligible=None, successes=None, rate=None,
            selected_starts=sum((s['task'], s['subtask']) == (t,p) for s in selected),
            distinct_instances=len({s['task_instance_id'] for s in selected if (s['task'],s['subtask']) == (t,p)})) for t,p in groups],
        underpowered_groups=[], baseline_setup_excluded_sample_ids=[], baseline_source_excluded_sample_ids=[],
        trials=records, expected_trials=len(records), expected_starts=len(sample_ids),
        maximum_control_steps=len(records)*max_steps, maximum_output_bytes=len(records)*max_steps*OUTPUT_BYTES_PER_STEP,
        selection=dict(method='explicit_feasibility_targets', max_groups=3, trials_per_start=trials,
                       min_eligible=None, threshold=None), augmentation='distinct_recorded_training_starts',
        quality=dict(require_success=True, require_video=True, require_finite_arrays=True,
                     require_action_dimension=23, require_fixed_subtask=True))
    return {**payload, 'fingerprint':digest(payload)}


def load_plan(path, panel):
    plan = json.loads(Path(path).read_text())
    payload = {k:v for k,v in plan.items() if k != 'fingerprint'}
    if plan.get('fingerprint') != digest(payload) or plan.get('schema_version') != 1 or plan.get('kind') != KIND:
        raise ValueError('Invalid or modified collection plan')
    if plan.get('panel_fingerprint') != panel['fingerprint'] or plan.get('conditioning') != 'task_and_subtask' or plan.get('mode') != 'train':
        raise ValueError('Collection differs from the panel/training conditioning contract')
    if type(plan.get('max_steps')) is not int or not 1 <= plan['max_steps'] <= 4096:
        raise ValueError('Invalid collection step limit')
    trials = plan.get('trials')
    if not isinstance(trials, list) or not trials or len(trials) != plan['expected_trials']:
        raise ValueError('Collection has no trials or inconsistent trial count')
    if len({t['trial_id'] for t in trials}) != len(trials):
        raise ValueError('Duplicate collection trials')
    if (plan.get('expected_starts') != len({t['sample_id'] for t in trials})
            or plan.get('maximum_control_steps') != len(trials) * plan['max_steps']
            or plan.get('maximum_output_bytes') != len(trials) * plan['max_steps'] * OUTPUT_BYTES_PER_STEP):
        raise ValueError('Collection start count or maximum control budget differs')
    samples = {s['sample_id']:s for s in panel['samples']}
    held_out = {tuple(k) for k in plan['held_out_instances']}
    setup_excluded = plan.get('baseline_setup_excluded_sample_ids', [])
    source_excluded = plan.get('baseline_source_excluded_sample_ids', [])
    if not isinstance(setup_excluded, list) or not isinstance(source_excluded, list):
        raise ValueError('Invalid excluded baseline start identities')
    excluded = setup_excluded + source_excluded
    if any(not isinstance(sid,str) or sid not in samples for sid in excluded) or len(set(excluded)) != len(excluded):
        raise ValueError('Invalid excluded baseline start identities')
    excluded = set(excluded)
    for trial in trials:
        if trial['sample_id'] not in samples or type(trial['policy_seed']) is not int or not 0 <= trial['policy_seed'] < 2**32:
            raise ValueError('Invalid sample identity or random seed')
        if type(trial.get('repeat')) is not int or trial['repeat'] < 0:
            raise ValueError('Invalid repeat identity')
        expected_seed = int(digest([plan['seed'], trial['sample_id'], trial['repeat']])[:8], 16)
        expected_id = digest([trial['sample_id'], trial['repeat'], expected_seed])[:24]
        if trial['policy_seed'] != expected_seed or trial['trial_id'] != expected_id:
            raise ValueError('Trial identity does not match deterministic sampling seed')
        if trial['sample_id'] in excluded:raise ValueError('Collection would repeat a setup-excluded baseline start')
        s = samples[trial['sample_id']]
        if (s['task'],s['task_instance_id']) in held_out:
            raise ValueError('Collection would leak an independent evaluation instance')
    if plan.get('experiment_purpose') is not None:
        if plan['experiment_purpose'] != PILOT_PURPOSE:
            raise ValueError('Unsupported collection experiment purpose')
        ids = sorted({t['sample_id'] for t in trials})
        candidate = freeze_pilot_plan(panel, ids, {sid:dict(contract='recorded_native_fields_v1',
            sample_id=sid, source_start=samples[sid]['start'],
            raw_sha256=plan['raw_source_sha256'][f"task-{samples[sid]['task_index']:04d}/episode_{samples[sid]['raw_episode_id']:08d}.hdf5"],
            selected_state_sha256=plan['selected_state_sha256'][sid]) for sid in ids},
            plan['runtime_validation_evidence'], checkpoint_identity=plan['checkpoint_identity'],
            held_out_instances=plan['held_out_instances'], trials=plan['selection']['trials_per_start'],
            max_steps=plan['max_steps'], seed=plan['seed'])
        # Canonical trial ordering is irrelevant, but every trial and provenance
        # field must match the bounded pilot constructor exactly.
        for key in candidate:
            if key in ('fingerprint', 'trials'): continue
            if plan.get(key) != candidate[key]: raise ValueError('Modified pilot contract: ' + key)
        if sorted(trials, key=lambda t:t['trial_id']) != sorted(candidate['trials'], key=lambda t:t['trial_id']):
            raise ValueError('Pilot trials differ from its bounded seed schedule')
    if plan.get('selection_basis') is not None:
        if plan['selection_basis'] == 'all_accepted_low_success_subtasks_v2':
            from full_subtask_collection import validate
        else:
            from accepted_subtask_collection import validate
        validate(plan, samples)
    return plan, samples


def verified_baseline_identity(summary,panel,receipt,marker):
    """Bind the completed summary to the exact submission and completion marker."""
    run_id=summary.get('run_id')
    if not run_id or receipt.get('run_id')!=run_id or marker.get('run_id')!=run_id:
        raise ValueError('Baseline run identity is missing or mismatched')
    if receipt.get('evaluation_kind')!='training_subtask_closed_loop' or receipt.get('exit_code')!=0:
        raise ValueError('Baseline requires the successful diagnostic submission receipt')
    if receipt.get('panel_fingerprint')!=panel['fingerprint'] or marker.get('panel_fingerprint')!=panel['fingerprint'] or marker.get('summary_sha256')!=digest(summary):
        raise ValueError('Baseline completion marker does not verify its panel and summary')
    expected={'expected_episode_jobs':len(panel['episodes']),'expected_segment_rollouts':len(panel['samples']),
              'expected_annotated_segments':len(panel['samples'])+len(panel.get('exclusions',[])),
              'expected_exclusions':len(panel.get('exclusions',[]))}
    if any(marker.get(k)!=v for k,v in expected.items()):raise ValueError('Baseline completion counts differ from frozen panel')
    identity={k:receipt.get(k) for k in ('checkpoint_step','checkpoint_manifest_sha256','checkpoint_source')}
    sha=identity['checkpoint_manifest_sha256']
    if type(identity['checkpoint_step']) is not int or identity['checkpoint_step']<0 or not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha) or not identity['checkpoint_source']:
        raise ValueError('Verified checkpoint identity is missing from baseline receipt')
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',required=True,type=Path)
    parser.add_argument('--summary',required=True,type=Path)
    parser.add_argument('--success-marker',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--submission-receipt',required=True,type=Path)
    parser.add_argument('--max-groups',type=int,default=3)
    parser.add_argument('--max-starts',type=int,default=2)
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--min-eligible',type=int,default=10)
    parser.add_argument('--threshold',type=float,default=0.5)
    parser.add_argument('--max-steps',type=int,default=2048)
    parser.add_argument('--held-out-instances',type=Path)
    args = parser.parse_args()
    if not args.success_marker.is_file() or args.success_marker.name != '_SUCCESS':
        parser.error('Complete production _SUCCESS marker is required')
    panel = load_panel(args.panel)
    summary = json.loads(args.summary.read_text())
    receipt = json.loads(args.submission_receipt.read_text())
    try:
        identity=verified_baseline_identity(summary,panel,receipt,json.loads(args.success_marker.read_text()))
    except ValueError as error:
        parser.error(str(error))
    held_out = json.loads(args.held_out_instances.read_text()) if args.held_out_instances else []
    plan = freeze_plan(summary,panel,max_groups=args.max_groups,max_starts=args.max_starts,
                       trials=args.trials,min_eligible=args.min_eligible,threshold=args.threshold,
                       max_steps=args.max_steps,held_out_instances=held_out,checkpoint_identity=identity)
    if not plan['trials']:
        parser.error('No adequately evaluated, low-success groups; expand baseline evaluation first')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(plan,stream,indent=2);stream.write('\n')
    print(json.dumps({k:plan[k] for k in ['fingerprint','expected_trials','expected_starts','maximum_control_steps','selected_groups']}))


if __name__ == '__main__':
    main()
