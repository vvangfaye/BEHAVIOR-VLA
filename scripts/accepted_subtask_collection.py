"""Freeze native-state collection from independently accepted task-level audits.

Individual accepted tasks can be used while other full100 tasks are pending.
This adapter never marks the full100 baseline complete or changes its metrics.
"""
from __future__ import annotations

import math
import re

from subtask_improvement import KIND, OUTPUT_BYTES_PER_STEP, digest


def candidates(metrics, acceptance, panel, *, threshold=0.8, min_eligible=4):
    if not 0 < threshold <= 1 or type(min_eligible) is not int or min_eligible < 1:
        raise ValueError('Invalid collection selection thresholds')
    if (metrics['panel_fingerprint'] != panel['fingerprint']
            or acceptance['panel_fingerprint'] != panel['fingerprint']
            or acceptance['kind'] != 'independently_accepted_task_and_subtask_success_rates'):
        raise ValueError('Accepted rates belong to another source panel')
    accepted = set(acceptance['accepted_task_indices'])
    assignments = {a['index']: a for a in metrics['assignments']}
    groups, deferred = [], []
    for row in metrics['subtasks']:
        index, n, successes = row['assignment_index'], row['metric_eligible'], row['successes']
        if index not in accepted:
            continue
        if type(n) is not int or type(successes) is not int or not 0 <= successes <= n:
            raise ValueError('Invalid accepted success denominator')
        if n == 0:
            continue
        rate = successes / n
        if not math.isclose(rate, row['completion_rate'], abs_tol=1e-12):
            raise ValueError('Accepted rate differs from eligible count')
        if rate >= threshold:
            continue
        a = assignments[index]
        if (a['audited_samples'] != a['expected_samples']
                or a.get('all_required_video_sha_and_first_frames_passed') is not True):
            raise ValueError('Incomplete independent task acceptance')
        group = dict(task=row['task'], subtask=row['subtask'], assignment_index=index,
                     eligible=n, successes=successes, rate=rate,
                     underpowered=n < 10, independent_audit_uid=a.get('corrective_audit_uid'))
        if (a.get('original_scope_and_sample_ids_matched') is not True
                or not a.get('corrective_audit_uid') or not a.get('producing_run_root')):
            deferred.append({**group, 'reason': 'producing_receipt_not_embedded_in_accepted_metrics'})
            continue
        if a.get('independent_all_raw_native_fields_matched') is not True:
            deferred.append({**group, 'reason': 'physical_history_collection_requires_separate_gpu_gate'})
        elif n < min_eligible:
            deferred.append({**group, 'reason': 'below_minimum_eligible_evidence'})
        else:
            groups.append(group)
    # Prefer hard groups with demonstrated nonzero yield for the initial runtime gate.
    groups.sort(key=lambda g: (g['successes'] == 0, g['rate'], -g['eligible'], g['task'], g['subtask']))
    return groups, deferred


def select_starts(panel, groups, held_out_instances, *, max_groups=2, starts_per_group=1, seed=1010):
    held = {tuple(x) for x in held_out_instances}
    selected = []
    for group in groups[:max_groups]:
        starts = [s for s in panel['samples'] if (s['task'], s['subtask']) ==
                  (group['task'], group['subtask']) and (s['task'], s['task_instance_id']) not in held]
        starts.sort(key=lambda s: digest([seed, s['sample_id']]))
        chosen, deferred, instances = [], [], set()
        for sample in starts:
            if sample['task_instance_id'] in instances:
                deferred.append(sample)
            else:
                chosen.append(sample)
                instances.add(sample['task_instance_id'])
        selected.extend((chosen + deferred)[:starts_per_group])
    if not selected:
        raise ValueError('No collection starts outside reserved evaluation instances')
    return selected


def freeze(panel, groups, samples, proofs, checkpoint_identity, held_out_instances, *,
           accepted_input_sha256, repeats=3, max_steps=2048, seed=1010):
    if (type(repeats) is not int or not 1 <= repeats <= 20 or type(max_steps) is not int
            or not 1 <= max_steps <= 4096 or not samples or len(samples) * repeats > 300):
        raise ValueError('Collection exceeds its finite attempt/control budget')
    identity = checkpoint_identity
    if (identity.get('checkpoint_step') != 1375000
            or not re.fullmatch('[0-9a-f]{64}', identity.get('checkpoint_manifest_sha256', ''))
            or not identity.get('checkpoint_source')):
        raise ValueError('Accepted rates require their measured checkpoint 1375000')
    held = {tuple(x) for x in held_out_instances}
    group_keys = {(g['task'], g['subtask']) for g in groups}
    raw, states, baseline, trials = {}, {}, {}, []
    eligible_samples = []
    excluded = []
    for sample in samples:
        sid = sample['sample_id']
        if (sample['task'], sample['task_instance_id']) in held:
            raise ValueError('Collection would leak a reserved evaluation instance')
        if (sample['task'], sample['subtask']) not in group_keys:
            raise ValueError('Sample is outside selected low-success groups')
        proof = proofs[sid]
        if proof['baseline_metric_eligible'] is not True:
            excluded.append(sid)
            continue
        relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
        if relative in raw and raw[relative] != proof['raw_sha256']:
            raise ValueError('Conflicting original source identities')
        for key in ('raw_sha256', 'selected_state_sha256', 'baseline_result_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', proof[key]):
                raise ValueError('Unverified original source or baseline result')
        raw[relative] = proof['raw_sha256']
        states[sid] = proof['selected_state_sha256']
        baseline[sid] = proof['baseline_result_sha256']
        eligible_samples.append(sample)
        for repeat in range(repeats):
            policy_seed = int(digest([seed, sid, repeat])[:8], 16)
            trials.append(dict(trial_id=digest([sid, repeat, policy_seed])[:24], sample_id=sid,
                               repeat=repeat, policy_seed=policy_seed))
    if not trials:
        raise ValueError('Every selected start was excluded by the accepted baseline')
    selected_keys = {(s['task'], s['subtask']) for s in eligible_samples}
    payload = dict(schema_version=1, kind=KIND, panel_fingerprint=panel['fingerprint'],
        checkpoint=identity['checkpoint_source'], checkpoint_identity=identity,
        conditioning='task_and_subtask', mode='train', official_benchmark=False,
        selection_basis='independently_accepted_native_task_subtask_rates_v1',
        full100_baseline_complete=False, accepted_input_sha256=accepted_input_sha256,
        baseline_result_sha256=baseline, require_recorded_native_state=True,
        raw_source_sha256=raw, selected_state_sha256=states,
        selected_groups=[g for g in groups if (g['task'], g['subtask']) in selected_keys],
        held_out_instances=sorted(map(list, held)), baseline_setup_excluded_sample_ids=excluded,
        baseline_source_excluded_sample_ids=[], max_steps=max_steps, seed=seed,
        trials=trials, expected_trials=len(trials), expected_starts=len(eligible_samples),
        maximum_control_steps=len(trials)*max_steps,
        maximum_output_bytes=len(trials)*max_steps*OUTPUT_BYTES_PER_STEP,
        augmentation='distinct_recorded_training_starts_and_independent_policy_seeds',
        selection=dict(trials_per_start=repeats),
        quality=dict(require_success=True, require_video=True, require_finite_arrays=True,
                     require_action_dimension=23, require_fixed_subtask=True))
    return {**payload, 'fingerprint': digest(payload)}


def validate(plan, samples):
    if plan.get('selection_basis') != 'independently_accepted_native_task_subtask_rates_v1':
        raise ValueError('Unknown accepted-rate collection protocol')
    if plan.get('require_recorded_native_state') is not True or plan.get('full100_baseline_complete') is not False:
        raise ValueError('Accepted task subset cannot claim a full100 baseline')
    ids = {t['sample_id'] for t in plan['trials']}
    for name in ('selected_state_sha256', 'baseline_result_sha256'):
        if set(plan[name]) != ids or any(not re.fullmatch('[0-9a-f]{64}', h) for h in plan[name].values()):
            raise ValueError('Collection lacks exact source/baseline sample proofs')
    proofs = {sid: dict(baseline_metric_eligible=True, selected_state_sha256=plan['selected_state_sha256'][sid],
        baseline_result_sha256=plan['baseline_result_sha256'][sid],
        raw_sha256=plan['raw_source_sha256'][f"task-{samples[sid]['task_index']:04d}/episode_{samples[sid]['raw_episode_id']:08d}.hdf5"])
        for sid in ids}
    rebuilt = freeze({'fingerprint': plan['panel_fingerprint']}, plan['selected_groups'],
        [samples[sid] for sid in sorted(ids)], proofs, plan['checkpoint_identity'], plan['held_out_instances'],
        accepted_input_sha256=plan['accepted_input_sha256'], repeats=plan['selection']['trials_per_start'],
        max_steps=plan['max_steps'], seed=plan['seed'])
    for key in rebuilt:
        if key in ('fingerprint', 'trials', 'baseline_setup_excluded_sample_ids'):
            continue
        if plan.get(key) != rebuilt[key]:
            raise ValueError('Modified accepted collection contract: ' + key)
    if sorted(plan['trials'], key=lambda t:t['trial_id']) != sorted(rebuilt['trials'], key=lambda t:t['trial_id']):
        raise ValueError('Modified accepted collection seed schedule')


def task_shards(plan, samples):
    """Partition one frozen plan without changing any seeds or source identities."""
    result = []
    for task in sorted({samples[t['sample_id']]['task'] for t in plan['trials']}):
        trials = [t for t in plan['trials'] if samples[t['sample_id']]['task'] == task]
        result.append(dict(index=len(result), task=task, trial_ids=[t['trial_id'] for t in trials],
            expected_trials=len(trials), maximum_control_steps=len(trials)*plan['max_steps']))
    ids = [tid for shard in result for tid in shard['trial_ids']]
    if len(ids) != len(set(ids)) or set(ids) != {t['trial_id'] for t in plan['trials']}:
        raise ValueError('Collection shards overlap or omit frozen attempts')
    return result
