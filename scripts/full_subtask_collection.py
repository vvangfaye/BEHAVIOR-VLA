"""Include every accepted subtask below the requested rate, with explicit source protocols."""
from __future__ import annotations

import math
import re

from accepted_subtask_collection import freeze as freeze_native
from subtask_improvement import digest

BASIS = 'all_accepted_low_success_subtasks_v2'
NATIVE = 'recorded_native_fields_v1'
HISTORY = 'recorded_actions_before_start_native_grasp_recovery_v1'


def targets(metrics, acceptance, *, threshold=.8):
    if not 0 < threshold <= 1 or metrics['panel_fingerprint'] != acceptance['panel_fingerprint']:
        raise ValueError('Invalid accepted rate selection')
    accepted = set(acceptance['accepted_task_indices'])
    groups = []
    for row in metrics['subtasks']:
        n, success = row['metric_eligible'], row['successes']
        if row['assignment_index'] not in accepted or n == 0:
            continue
        if type(n) is not int or type(success) is not int or not 0 <= success <= n:
            raise ValueError('Invalid accepted counts')
        rate = success / n
        if not math.isclose(rate, row['completion_rate'], abs_tol=1e-12):
            raise ValueError('Accepted denominator mismatch')
        if rate < threshold:
            groups.append(dict(task=row['task'], subtask=row['subtask'],
                assignment_index=row['assignment_index'], eligible=n, successes=success,
                rate=rate, underpowered=n < 10))
    keys = {(g['task'], g['subtask']) for g in groups}
    if len(keys) != len(groups):
        raise ValueError('Duplicate accepted group')
    return sorted(groups, key=lambda g: (g['rate'], -g['eligible'], g['task'], g['subtask']))


def choose_starts(panel, group, proofs, held_out_instances, *, maximum=3, seed=1011):
    """Prefer distinct training instances; ineligible baseline starts are never substituted."""
    held = {tuple(x) for x in held_out_instances}
    rows = [s for s in panel['samples'] if (s['task'], s['subtask']) ==
            (group['task'], group['subtask']) and (s['task'], s['task_instance_id']) not in held
            and s['sample_id'] in proofs and proofs[s['sample_id']]['baseline_metric_eligible']]
    rows.sort(key=lambda s: digest([seed, s['sample_id']]))
    distinct, remainder, instances = [], [], set()
    for sample in rows:
        if sample['task_instance_id'] in instances:
            remainder.append(sample)
        else:
            instances.add(sample['task_instance_id']); distinct.append(sample)
    return (distinct + remainder)[:maximum]


def freeze(panel, group, samples, proofs, checkpoint_identity, held_out_instances, *,
           accepted_input_sha256, evidence_manifest_sha256, repeats=10, max_steps=2048, seed=1011):
    if not re.fullmatch('[0-9a-f]{64}', evidence_manifest_sha256):
        raise ValueError('Source audit manifest is required')
    if not 0 <= group['rate'] < .8 or group['eligible'] < 1:
        raise ValueError('Group is outside the user-requested rate scope')
    constructor_proofs = {sid: dict(p) for sid, p in proofs.items()}
    for sid, proof in constructor_proofs.items():
        if proof.get('source_evidence_kind') == 'additional_training_start_preflight':
            proof.update(baseline_metric_eligible=True, baseline_result_sha256=proof['source_preflight_sha256'])
    plan = freeze_native(panel, [group], samples, constructor_proofs, checkpoint_identity, held_out_instances,
        accepted_input_sha256=accepted_input_sha256, repeats=repeats, max_steps=max_steps, seed=seed)
    ids = {t['sample_id'] for t in plan['trials']}
    protocols, prefixes, history_raw = {}, {}, {}
    for sid in ids:
        proof = proofs[sid]
        protocol = proof['state_protocol']
        if protocol not in (NATIVE, HISTORY):
            raise ValueError('Unknown collection restoration protocol')
        protocols[sid] = protocol
        if protocol == HISTORY:
            prefix = proof['prefix_actions_sha256']
            if not re.fullmatch('[0-9a-f]{64}', prefix):
                raise ValueError('History source lacks an independently hashed action prefix')
            prefixes[sid] = prefix
            raw = proof['raw_sample_evidence']
            if any(type(raw.get(k)) is not int or raw[k] < 0 for k in
                   ('selected_state_size', 'transition_count_before_frame')):
                raise ValueError('History source lacks independently checked row dimensions')
            history_raw[sid] = {k: raw[k] for k in ('selected_state_size', 'transition_count_before_frame')}
    plan.update(selection_basis=BASIS, require_recorded_source_state=True,
        require_recorded_native_state=all(p == NATIVE for p in protocols.values()),
        source_protocol_by_sample=protocols, prefix_actions_sha256=prefixes,
        history_raw_sample_evidence=history_raw,
        source_evidence_manifest_sha256=evidence_manifest_sha256,
        low_success_threshold=.8, minimum_eligible_evidence=1)
    additional = {sid: proofs[sid]['source_preflight_sha256'] for sid in ids
                  if proofs[sid].get('source_evidence_kind') == 'additional_training_start_preflight'}
    if additional:
        for sid in additional:plan['baseline_result_sha256'].pop(sid)
        plan.update(additional_start_source_evidence_sha256=additional,
                    baseline_rates_panel_fingerprint=panel['baseline_panel_fingerprint'])
    plan.pop('fingerprint')
    return dict(plan, fingerprint=digest(plan))


def validate(plan, samples):
    if plan.get('selection_basis') != BASIS or plan.get('require_recorded_source_state') is not True:
        raise ValueError('Unknown full-coverage collection contract')
    if len(plan['selected_groups']) != 1:
        raise ValueError('Each campaign shard collects one task-subtask group')
    ids = {t['sample_id'] for t in plan['trials']}
    for key in ('source_protocol_by_sample', 'selected_state_sha256'):
        if set(plan[key]) != ids:
            raise ValueError('Incomplete collection source identities')
    additional=plan.get('additional_start_source_evidence_sha256',{})
    if (set(plan['baseline_result_sha256']) & set(additional) or
            set(plan['baseline_result_sha256']) | set(additional) != ids):
        raise ValueError('Missing or ambiguous baseline/additional source evidence')
    proofs = {}
    for sid in ids:
        sample = samples[sid]
        relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
        proofs[sid] = dict(baseline_metric_eligible=True, raw_sha256=plan['raw_source_sha256'][relative],
            selected_state_sha256=plan['selected_state_sha256'][sid],
            baseline_result_sha256=plan['baseline_result_sha256'].get(sid),
            state_protocol=plan['source_protocol_by_sample'][sid],
            prefix_actions_sha256=plan['prefix_actions_sha256'].get(sid))
        if proofs[sid]['state_protocol'] == HISTORY:
            proofs[sid]['raw_sample_evidence'] = plan['history_raw_sample_evidence'][sid]
        if sid in additional:
            proofs[sid].update(source_evidence_kind='additional_training_start_preflight',
                               source_preflight_sha256=additional[sid])
    rebuilt = freeze({'fingerprint': plan['panel_fingerprint'],
        'baseline_panel_fingerprint': plan.get('baseline_rates_panel_fingerprint')}, plan['selected_groups'][0],
        [samples[sid] for sid in sorted(ids)], proofs, plan['checkpoint_identity'], plan['held_out_instances'],
        accepted_input_sha256=plan['accepted_input_sha256'],
        evidence_manifest_sha256=plan['source_evidence_manifest_sha256'],
        repeats=plan['selection']['trials_per_start'], max_steps=plan['max_steps'], seed=plan['seed'])
    for key in rebuilt:
        if key in ('fingerprint', 'trials'):
            continue
        if plan.get(key) != rebuilt[key]:
            raise ValueError('Changed full collection contract: ' + key)
    if sorted(plan['trials'], key=lambda t:t['trial_id']) != sorted(rebuilt['trials'], key=lambda t:t['trial_id']):
        raise ValueError('Changed collection seeds')


def validate_result(result, sample, panel_fingerprint, plan):
    sid = sample['sample_id']
    if plan['source_protocol_by_sample'][sid] == NATIVE:
        from recorded_native_state import validate_result_proof
        return validate_result_proof(result, sample, plan)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sco'))
    from audit_history_assignment_artifacts import validate_history_result
    relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
    restoration = result.get('restoration', {})
    raw = dict(plan['history_raw_sample_evidence'][sid], sha256=plan['raw_source_sha256'][relative],
               selected_state_sha256=plan['selected_state_sha256'][sid])
    return validate_history_result(result, sample, panel_fingerprint, raw,
        plan['prefix_actions_sha256'][sid], production_difficulty_baseline=False)
