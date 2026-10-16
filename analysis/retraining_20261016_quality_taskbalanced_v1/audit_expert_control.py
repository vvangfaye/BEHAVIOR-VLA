"""Verify a paired expert-only ablation without changing or submitting a training job."""
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'formal_stage/bundle'
sys.path.insert(0, str(BUNDLE / 'openpi/src'))
from openpi.training.subtask_mixture import MixturePlan


def main():
    plan = MixturePlan.load(BUNDLE / 'inputs/compiled/mixture.json')
    sampler = plan._quality_sampler
    tasks = collections.Counter()
    replaced = collections.Counter()
    treatment_hash = hashlib.sha256()
    control_hash = hashlib.sha256()
    for index in range(plan.num_samples):
        treatment = sampler.reference(index)
        control = sampler.reference(index, expert_only=True)
        assert control['source_kind'] == 'replay'
        assert treatment['task_name'] == control['task_name']
        assert control['frame'] >= control['segment_start']
        assert control['frame'] + plan.action_horizon <= control['segment_end']
        if treatment['source_kind'] == 'replay':
            assert treatment == control
        else:
            replaced[control['task_name']] += 1
        tasks[control['task_name']] += 1
        for digest, row in ((treatment_hash, treatment), (control_hash, control)):
            digest.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode() + b'\n')
    assert len(tasks) == 100 and set(tasks.values()) == {9600}
    assert sum(replaced.values()) == sampler.full_draws == 48829
    result = {
        'kind': 'exhaustive_paired_expert_control_audit',
        'source_mixture_fingerprint': plan.fingerprint,
        'samples_checked': plan.num_samples,
        'tasks': len(tasks), 'draws_per_task': 9600,
        'unchanged_expert_windows': plan.num_samples - sum(replaced.values()),
        'replaced_full_episode_windows': sum(replaced.values()),
        'replacements_by_task': dict(sorted(replaced.items())),
        'treatment_reference_sha256': treatment_hash.hexdigest(),
        'expert_control_reference_sha256': control_hash.hexdigest(),
        'control_definition': 'sampler.reference(global_index, expert_only=True)',
        'invariants': ['same task at every index', 'identical existing expert windows',
                       'all replacement windows are legal expert windows',
                       '100 tasks each receive 9600 windows'],
        'remaining_before_execution': ['wire explicit control mode into an isolated training bundle',
            'record control mode in semantic fingerprint and runtime metrics',
            'real-batch smoke test and exact restore validation',
            'use a unique run/output and compare complete matched closed-loop results'],
        'cloud_jobs_created': 0, 'training_configuration_modified': False,
        'model_improvement_proven': False,
    }
    destination = ROOT / 'expert_control_audit.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'replacements_by_task'}))


if __name__ == '__main__':
    main()
