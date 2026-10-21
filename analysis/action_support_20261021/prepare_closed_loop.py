"""Freeze the native evaluation panel and its sources; never submit jobs."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = HERE / 'closed_loop_v1'
    destination.mkdir(exist_ok=True)
    target = destination / 'plan.json'
    if target.exists():
        raise SystemExit('Evaluation plan already frozen')
    previous = PROJECT / 'analysis/repeatability_1688977_seed0_30_20261019'
    old = json.loads((previous / 'plan.json').read_text())
    training_path = HERE / 'vla_pilot/pilot_training_plan.json'
    training = json.loads(training_path.read_text())
    followup = training['closed_loop_followup']
    assets = PROJECT / 'analysis/organizer_rehearsal_1688977_20261016/runtime/datasets/2026-challenge-task-instances'
    rows = {row['task_name']: row for line in (assets / 'metadata/task.jsonl').read_text().splitlines()
            if line.strip() for row in [json.loads(line)]}
    states = list((assets / 'scene_test/public').rglob('*-tro_state.json'))
    pairs = [[task, instance] for task in followup['tasks'] for instance in followup['instances']]
    inventory = []
    for task, instance in pairs:
        matches = [p for p in states if f'_task_{task}_' in p.name and f'_{instance}_' in p.name]
        assert len(matches) == 1, (task, instance)
        path = matches[0]
        inventory.append({'pair': [task, instance], 'relative_path': str(path.relative_to(assets)),
                          'bytes': path.stat().st_size, 'sha256': digest(path)})
    frozen = destination / 'frozen'
    frozen.mkdir(exist_ok=False)
    for name in ['native_helpers.py', 'evaluator_entry.py', 'seed_protocol.py']:
        shutil.copyfile(previous / 'frozen' / name, frozen / name)
    shutil.copyfile(previous / 'frozen/worker.py', frozen / 'source_preflight.py')
    for name in ['closed_loop_worker.py', 'serve_vla_native.py', 'vla_native_policy.py',
                 'vla_runtime.py', 'vla_support_adapter.py', 'support_flow.py']:
        shutil.copyfile(HERE / name, frozen / name)
    protocol = {
        'methods': followup['methods'], 'pairs': pairs, 'policy_seed': 0, 'environment_seed': 0,
        'training_seed': 17, 'training_step': 1000, 'sampler': '10-step Euler',
        'model_horizon': 32, 'executed_prefix': 16, 'precision': 'bf16 autocast, fp32 action expert',
        'generated_subtasks': True, 'subtask_temperature': 0., 'external_agent': False,
        'mode': 'public_test', 'whole_task_prompt': True, 'write_video': True,
        'fresh_processes_per_episode': True, 'fresh_simulator_cache_per_episode': True,
        'action_rng_separate_from_structure_rng': True,
        'primary_outcomes': ['official_q_score', 'native_success'],
        'no_score_based_filter_or_retry': True,
        'inference_backend': 'same converted Torch1688977 for original and trained policies',
        'interpretation': 'Three-task development pilot; not a held-out task generalization claim',
    }
    plan = {k: old[k] for k in ['code', 'runtime', 'baseline_checkpoint', 'baseline_files',
                               'source_manifest_sha256', 'critical_files', 'supplemental_library', 'image']}
    plan.update(schema_version=1, pairs=pairs, protocol=protocol,
                protocol_sha256=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
                training_plan_sha256=digest(training_path),
                task_metadata_sha256=digest(assets / 'metadata/task.jsonl'), instance_assets=inventory,
                native_task_budgets={task: int(1.5 * rows[task]['length']) for task in followup['tasks']},
                afs_root='/mnt/afs/260010165/behavior_1k/action_support_20261021/closed-loop-v1',
                checkpoint=training['checkpoint'], tokenizer=training['tokenizer'],
                upstream='/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01/upstream',
                adapter='/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01/adapter',
                site='/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01/site',
                training_roots={method: '/mnt/afs/260010165/behavior_1k/action_support_20261021/pilot-seed17-' + method
                                for method in training['methods']},
                source_sha256={p.name: digest(p) for p in frozen.glob('*.py')},
                resource_bound={'quota_type': 'spot', 'gpu_per_job': 1, 'max_episode_seconds': 28800,
                                'episodes': len(pairs) * len(followup['methods']),
                                'max_gpu_hours': len(pairs) * len(followup['methods']) * 8,
                                'first_job': ['original1688977', *pairs[0]],
                                'remaining_gate': 'first Torch-policy native episode accepted, and all three final training receipts accepted'},
                status='frozen_plan_only_no_evaluation_submitted')
    with target.open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'plan': str(target), 'pairs': len(pairs), 'episodes': 48,
                      'native_frame_budgets': plan['native_task_budgets'], 'jobs_submitted': 0}))


if __name__ == '__main__':
    main()
