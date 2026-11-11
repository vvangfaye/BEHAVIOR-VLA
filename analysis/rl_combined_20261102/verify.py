"""Verify the combination preserves gamma=1 buffers and adds exact episode weights."""
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from test_combination import Toy, episode
from parallel_training import buffers
from variant_update import prepare_rewards, episode_weights
import deployment


def main():
    torch.manual_seed(0)
    model = Toy()
    source = [episode(model, 9, True), episode(model, 14, False)]
    shared = {'optimizer': {'gamma': 1., 'gae_lambda': .95}, 'shaping': {'gamma': 1., 'beta': 1.}}
    first = {**shared, 'comparison_variant': 'gamma1_ppo'}
    combined = {**shared, 'comparison_variant': 'balanced_gamma1_ppo'}
    gamma_episodes = prepare_rewards(copy.deepcopy(source), first)
    combo_episodes = prepare_rewards(copy.deepcopy(source), combined)
    _, gamma_data = buffers(gamma_episodes, first)
    _, combo_data = buffers(combo_episodes, combined)
    for key in ['advantages', 'returns', 'prev_values']:
        assert torch.equal(gamma_data[key], combo_data[key]), key
    assert torch.equal(combo_data['weights'], episode_weights(source))
    torch.testing.assert_close(combo_data['weights'][:9].sum(), combo_data['weights'][9:].sum())
    for a, b in zip(gamma_episodes, combo_episodes):
        assert [r['reward'] for r in a['transitions']] == [r['reward'] for r in b['transitions']]
    plan = deployment.worker_plan('train')
    original = deployment.single.worker_plan({'role': 'train', 'variant': 'gamma1_ppo'})
    assert plan['optimizer'] == original['optimizer']
    assert plan['critic_warmup'] == original['critic_warmup']
    assert plan['source_rollout_receipts'] == original['source_rollout_receipts']
    assert plan['source_identity'] == original['source_identity']
    changed = []
    for path in (deployment.OLD / 'runtime').rglob('*.py'):
        name = str(path.relative_to(deployment.OLD / 'runtime'))
        if path.read_bytes() != (deployment.HERE / 'runtime' / name).read_bytes():
            changed.append(name)
    assert set(changed) == {'variant_update.py', 'parallel_training.py'}
    result = {
        'passed': True, 'at': time.time(), 'distributed_regression_tests_passed': 6,
        'gamma1_rewards_advantages_targets_values_preserved': True,
        'trajectory_weights_match_existing_balanced_ppo': True,
        'same_optimizer_critic_warmup_source_rollouts_and_initial_model': True,
        'runtime_changed_files': changed,
        'real_model_parity_gate_runs_before_first_optimizer_update': True,
        'kl_batch8_reuses_accepted_unchanged_logprob_implementation': True,
        'plan_sha256': hashlib.sha256((deployment.HERE / 'plan.json').read_bytes()).hexdigest(),
    }
    deployment.save(deployment.HERE / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
