"""CPU tests for the rl_eval_power_lr_20261102 runtime changes.

Run (from this directory):
  PYTHONPATH=../runtime:<analysis>/rlinf_adaptation_20261020/upstream <venv-python> -m unittest test_lr_scan -v
"""
import math
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import training as base
from parallel_training import Parallel, Checkpoints, buffers, train
from test_parallel_original import Toy, episode


def make_plan(learning_rate, kl_target):
    return {'comparison_variant': 'control',
            'optimizer': {'actor_lr': learning_rate, 'value_lr': .0001, 'weight_decay': 0., 'shuffle_seed': 5,
                          'updates_per_round': [3], 'gamma': .99, 'gae_lambda': .95, 'clip_ratio': .2,
                          'kl_target': kl_target, 'allow_kl_early_stop': True},
            # Toy.default_forward is single-row only; batched KL was validated separately (kl_batch_gate.json).
            'shaping': {'gamma': 1., 'beta': 1.}, 'distributed': {'kl_batch_size': 1}}


def worker(rank, world, root, learning_rate, kl_target):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method='file://'+str(root/'rendezvous'), rank=rank, world_size=world)
    torch.manual_seed(0)
    model = Toy()
    episodes = [episode(model, 9, True), episode(model, 14, False)]
    plan = make_plan(learning_rate, kl_target)
    parallel = Parallel('cpu')
    optimizer = base.build_optimizer(model, plan)
    rows, data = buffers(episodes, plan)
    manager = Checkpoints(root/'checkpoint', {'variant': 'control'}, parallel)
    result = train(model, optimizer, rows, data, plan, parallel, manager, lambda **_: None)
    if rank == 0:
        torch.save({'model': model.state_dict(), 'result': result}, root/'actual.pt')
    dist.destroy_process_group()


def run(learning_rate, kl_target):
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(4, root, learning_rate, kl_target), nprocs=4, join=True)
        return torch.load(Path(root)/'actual.pt', weights_only=False)


def direct_outside_fraction(state, clip_ratio=.2):
    """Recompute the post-training ratio statistics without the runtime code."""
    torch.manual_seed(0)
    reference = Toy()
    episodes = [episode(reference, 9, True), episode(reference, 14, False)]
    model = Toy()
    model.load_state_dict(state)
    outside, total, abs_sum = 0, 0, 0.0
    for e in episodes:
        for row in e['transitions']:
            now = model.default_forward(row['forward_inputs'], compute_values=False)['logprobs']
            logratio = float((now - row['prev_logprobs']).sum())
            outside += logratio > math.log1p(clip_ratio) or logratio < math.log1p(-clip_ratio)
            abs_sum += abs(logratio)
            total += 1
    return outside / total, abs_sum / total


class LearningRateScanRuntime(unittest.TestCase):
    def test_small_lr_records_diagnostics_and_matches_serial(self):
        # Diagnostics must not change training: compare against the serial reference update.
        torch.manual_seed(0)
        model = Toy()
        episodes = [episode(model, 9, True), episode(model, 14, False)]
        plan = make_plan(.001, .01)
        optimizer = base.build_optimizer(model, plan)
        expected = base.update_policy(model, optimizer, episodes, plan)[0]
        actual = run(.001, .01)
        result = actual['result']
        self.assertEqual(result['optimizer_steps'], expected['optimizer_steps'])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(actual['model'][name], value, atol=1e-6, rtol=1e-5)
        steps = result['update_steps']
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]['ppo_clip_fraction'], 0.0)  # ratio is exactly 1 before any update
        for step in steps:
            for key in ['ppo_clip_fraction', 'ppo_mean_abs_ratio_minus_one', 'post_update_outside_clip_fraction',
                        'post_update_abs_log_ratio_gt_0_05_fraction', 'post_update_mean_abs_log_ratio']:
                self.assertTrue(0.0 <= step[key] <= 1.0 or key == 'post_update_mean_abs_log_ratio', (key, step))
        self.assertEqual(steps[-1]['post_update_outside_clip_fraction'], 0.0)
        outside, mean_abs = direct_outside_fraction(actual['model'])
        self.assertAlmostEqual(result['final_batch_kl']['outside_clip_fraction'], outside, places=12)
        self.assertAlmostEqual(result['final_batch_kl']['mean_abs_log_ratio'], mean_abs, places=6)

    def test_large_step_leaves_clip_interval_when_kl_allows(self):
        actual = run(.3, 1.0)
        result = actual['result']
        steps = result['update_steps']
        self.assertGreater(steps[0]['post_update_outside_clip_fraction'], 0.0)
        self.assertGreater(steps[1]['ppo_clip_fraction'], 0.0)  # PPO clipping binds after the first move
        outside, _ = direct_outside_fraction(actual['model'])
        self.assertAlmostEqual(result['final_batch_kl']['outside_clip_fraction'], outside, places=12)

    def test_kl_guard_rolls_back_and_records_early_stop(self):
        actual = run(1.0, .01)
        result = actual['result']
        self.assertTrue(result['early_stopped'])
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertFalse(result['update_steps'][-1]['accepted'])
        self.assertGreater(result['update_steps'][-1]['post_update_outside_clip_fraction'], 0.0)
        # Rolled back to the original actor parameter.
        torch.manual_seed(0)
        self.assertAlmostEqual(float(actual['model']['actor']), float(Toy().actor), places=7)


if __name__ == '__main__':
    unittest.main()
