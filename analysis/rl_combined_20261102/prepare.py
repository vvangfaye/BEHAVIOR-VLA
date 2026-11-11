"""Snapshot the proven learner; add only the combined objective dispatch."""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'rl_multigpu_20261101'
SOURCE = HERE.parent / 'rl_scheme_comparison_20261101'


def main():
    runtime = HERE / 'runtime'
    runtime.mkdir(exist_ok=True)
    for path in (OLD / 'runtime').rglob('*.py'):
        target = runtime / path.relative_to(OLD / 'runtime')
        target.parent.mkdir(parents=True, exist_ok=True)
        source = path.read_text()
        if path.name == 'variant_update.py':
            assert source.count("variant=='gamma1_ppo'") == 1
            assert source.count("('balanced_ppo','group_outcome')") == 3
            source = source.replace("variant=='gamma1_ppo'", "variant in ('gamma1_ppo','balanced_gamma1_ppo')")
            source = source.replace("('balanced_ppo','group_outcome')", "('balanced_ppo','balanced_gamma1_ppo','group_outcome')")
        if path.name == 'parallel_training.py':
            assert source.count("('balanced_ppo', 'group_outcome')") == 1
            source = source.replace("('balanced_ppo', 'group_outcome')", "('balanced_ppo', 'balanced_gamma1_ppo', 'group_outcome')")
        if target.exists():
            assert target.read_text() == source
        else:
            target.write_text(source)
    for name in ['simulator.py', 'comparison_driver.py']:
        (runtime / name).write_bytes((SOURCE / 'runtime_source' / name).read_bytes())
    (runtime / 'single_bootstrap.py').write_bytes((SOURCE / 'runtime_source/bootstrap.py').read_bytes())
    tests = (OLD / 'test_parallel.py').read_text()
    tests = tests.replace("variant == 'gamma1_ppo'", "variant in ('gamma1_ppo','balanced_gamma1_ppo')")
    tests = tests.replace("for variant in ['control', 'gamma1_ppo', 'balanced_ppo', 'group_outcome']:",
                          "for variant in ['balanced_gamma1_ppo']:")
    tests = tests.replace("self.compare('balanced_ppo', interrupt=True)", "self.compare('balanced_gamma1_ppo', interrupt=True)")
    (HERE / 'test_combination.py').write_text(tests)
    comparison = json.loads((SOURCE / 'plan.json').read_text())
    plan = {
        'experiment': 'rl_combined_20261102',
        'authorization': 'User: combine trajectory equal weighting and gamma=1; reuse exact existing 32 rollouts, 17 optimizer steps, fixed evaluation set.',
        'variant': 'balanced_gamma1_ppo', 'task': 'bringing_water', 'start_checkpoint': 'original1688977',
        'source_campaign': str(SOURCE), 'source_trajectories': 32, 'source_chunks': 23908,
        'gamma': 1., 'shaping_gamma': 1.,
        'weighting': 'N/(32*T_i), equal total combined actor/critic loss weight per trajectory',
        'optimizer_steps': 17, 'actor_lr': 1e-7, 'value_lr': 1e-4, 'gae_lambda': .95, 'shuffle_seed': 602410,
        'train_gpus': 4, 'eval_parallel_gpus': 16, 'max_total_live_gpus': 32, 'quota_type': 'spot',
        'train_timeout_seconds': 43200, 'eval_timeout_seconds': 5400, 'attempt_cap': 2,
        'max_total_gpu_hours': 192, 'max_wall_hours': 48,
        'eval_cases': comparison['eval_cases'], 'reuse_existing_baselines': True,
        'remote_root': '/mnt/afs/260010165/behavior_1k/rl_combined_20261102',
        'aoss_prefix': 'behavior1k/rl-combined-20261102/',
        'training_storage_limit_bytes': 16 * 1024**3, 'evaluation_storage_limit_bytes': 512 * 1024**2,
        'campaign_storage_reserve_bytes': 36 * 1024**3, 'storage_margin_bytes': 20 * 1024**3,
        'source_rollout_receipts_sha256': hashlib.sha256((SOURCE / 'source_training_receipts.json').read_bytes()).hexdigest(),
        'runtime_sha256': {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(runtime.rglob('*.py'))},
        'baseline_result_sha256': {
            f'i{i}_s{s}': hashlib.sha256((SOURCE / 'evidence/original' / f'i{i}_s{s}' / 'result.json').read_bytes()).hexdigest()
            for i, s in comparison['eval_cases']},
        'comparators': ['original', 'control', 'balanced_ppo', 'gamma1_ppo'],
        'preserve_existing_checkpoints_and_expert_data': True, 'no_new_training_rollouts': True,
    }
    target = HERE / 'plan.json'
    if target.exists():
        previous = json.loads(target.read_text())
        assert {k: v for k, v in previous.items() if k != 'created_at'} == plan
    else:
        plan['created_at'] = time.time()
        target.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print('Prepared immutable combined branch:', len(plan['runtime_sha256']), 'runtime files, 40 fixed cases')


if __name__ == '__main__':
    main()
