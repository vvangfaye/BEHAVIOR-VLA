"""Check task-specific reward contracts and reuse the accepted distributed core tests."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import torch


def main():
    root = Path(sys.argv[1]).absolute()
    sys.path[:0] = [str(root / 'runtime'), str(root)]
    deployment = importlib.import_module('deployment')
    from goals import potential
    from parallel_training import buffers
    from variant_update import prepare_rewards, episode_weights
    config = deployment.campaign()
    source = json.loads((root / 'source_plan.json').read_text())
    receipts = json.loads((root / 'source_training_receipts.json').read_text())
    parent = root.parents[1] / 'rl_combined_20261102'
    core = json.loads((parent / 'verification.json').read_text())
    assert core['passed'] and core['distributed_regression_tests_passed'] == 6
    for name in ['variant_update.py', 'parallel_training.py', 'training.py', 'kl_batch.py', 'b1k_rlinf/model.py']:
        assert (root / 'runtime' / name).read_bytes() == (parent / 'runtime' / name).read_bytes()
    for name in ['goals.py', 'simulator.py']:
        assert (root / 'runtime' / name).read_bytes() == (Path(config['source_campaign']) / name).read_bytes()
    plan = deployment.worker_plan('train')
    assert plan['optimizer']['updates_per_round'] == [17]
    assert plan['optimizer']['gamma'] == plan['shaping']['gamma'] == 1.
    assert plan['critic_warmup'] == source['critic_warmup']
    assert {k: v for k, v in plan['optimizer'].items() if k not in ['gamma', 'updates_per_round']} == {
        k: v for k, v in source['optimizer'].items() if k not in ['gamma', 'updates_per_round']}
    assert set(source['train_instances']).isdisjoint(i for i, s in config['eval_cases'])
    assert sum(r['chunks'] for r in receipts) == config['source_chunks'] and len(receipts) == 32
    assert all(r['policy_hash'] == plan['source_identity']['cold_trainable_hash'] for r in receipts)
    assert config['control_checkpoint']['cumulative_optimizer_steps'] == 17
    for case in [config['eval_cases'][0], config['eval_cases'][-1]]:
        for variant in config['evaluation_variants']:
            evaluated = deployment.worker_plan('eval', case, variant)
            assert evaluated['evaluation']['cases'] == [case]
            assert evaluated['seed'] == evaluated['model_seed'] == 0
            assert evaluated['goal_contract'] == source['goal_contract']
    names = source['goal_contract']['goals']
    def signals(progress):
        values = {name: progress and i == 0 for i, name in enumerate(names)}
        return values if config['task_label'] == 'modem' else {'goals': values}
    episodes = []
    for length, success in [(9, True), (14, False)]:
        rows = []
        for i in range(length):
            rows.append({'task_reward': float(success and i == length - 1) * 10., 'reward': 0.,
                         'terminal': i == length - 1, 'prev_values': torch.tensor([i * .1]),
                         'goals_before': signals(i >= 3), 'goals_after': signals(i >= 2)})
        episodes.append({'instance': 1, 'transitions': rows, 'bootstrap': torch.zeros(1), 'status': {'success': success}})
    gamma = {'comparison_variant': 'gamma1_ppo', 'optimizer': {'gamma': 1., 'gae_lambda': .95},
             'shaping': {'gamma': 1., 'beta': 1.}}
    combined = {**gamma, 'comparison_variant': config['variant']}
    left = prepare_rewards(copy.deepcopy(episodes), gamma)
    right = prepare_rewards(copy.deepcopy(episodes), combined)
    _, a = buffers(left, gamma)
    _, b = buffers(right, combined)
    for key in ['advantages', 'returns', 'prev_values']:
        assert torch.equal(a[key], b[key])
    assert torch.equal(b['weights'], episode_weights(right))
    torch.testing.assert_close(b['weights'][:9].sum(), b['weights'][9:].sum())
    result = {'passed': True, 'at': time.time(), 'task': config['task'],
              'same_distributed_training_core_as_6_passed_regressions': True,
              'task_goal_observer_and_simulator_unchanged': True,
              'all_32_source_receipts_original_policy_and_contract_verified': True,
              'same_optimizer_except_gamma_and_one_round_selection': True,
              'gamma1_buffers_preserved_and_trajectory_weights_match': True,
              'control_checkpoint_exactly_17_updates': True,
              'plan_sha256': hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest()}
    deployment.save(root / 'verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
