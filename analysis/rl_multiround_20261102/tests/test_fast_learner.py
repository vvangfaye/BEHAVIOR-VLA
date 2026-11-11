"""CPU tests for the fast learner (rl_multiround_20261102).

Run from this directory:
  PYTHONPATH=../runtime:<analysis>/rlinf_adaptation_20261020/upstream <venv-python> -m unittest test_fast_learner -v
"""
import math
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import training as base
from parallel_training import (Parallel, Checkpoints, backward, backward_batched, buffers, replay_check_tolerant,
                               train, train_fast)


class BatchToy(torch.nn.Module):
    """Same function as the inherited Toy for batch size 1, but accepts batches."""

    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.tensor([.1]))
        self.value_head = torch.nn.Linear(1, 1)
        self.frozen = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)

    def default_forward(self, inputs, compute_values=True):
        x = inputs['x']
        size = x.shape[0]
        result = {'logprobs': (self.actor * x).reshape(size, 1, 1).expand(size, 1, 23) / 23}
        if compute_values:
            result['values'] = self.value_head(x).reshape(size)
        return result


def make_episode(model, xs, success, instance):
    rows = []
    for i, value in enumerate(xs):
        x = torch.tensor([[float(value)]])
        previous = model.default_forward({'x': x})
        last = i == len(xs) - 1
        rows.append({'forward_inputs': {'x': x}, 'prev_logprobs': previous['logprobs'].detach().clone(),
                     'prev_values': previous['values'].detach().clone(), 'terminal': last,
                     'reward': float(success and last), 'task_reward': float(success and last),
                     'goals_before': {'goals': {'a': False}}, 'goals_after': {'goals': {'a': bool(success)}}})
    return {'instance': instance, 'status': {'success': success}, 'transitions': rows, 'bootstrap': torch.zeros(1)}


def episodes_for(model, layout):
    if layout == 'uniform':
        return [make_episode(model, [1.] * 9, True, 1), make_episode(model, [-1.] * 14, False, 2)]
    # 'masked': only the seeded KL subset rows have tiny |x|; every other row has |x| = 10, so a
    # sampled check sees almost no drift while the full batch drifts far.
    subset = set(torch.randperm(23, generator=torch.Generator().manual_seed(5 + 7)).tolist()[:3])
    values = [.1 if i in subset else 10. for i in range(23)]
    return [make_episode(model, values[:9], True, 1), make_episode(model, [-v for v in values[9:]], False, 2)]


def make_plan(lr, kl_target, updates, fast=None):
    plan = {'comparison_variant': 'control',
            'optimizer': {'actor_lr': lr, 'value_lr': 1e-4, 'weight_decay': 0., 'shuffle_seed': 5,
                          'updates_per_round': [updates], 'gamma': .99, 'gae_lambda': .95, 'clip_ratio': .2,
                          'kl_target': kl_target, 'allow_kl_early_stop': True},
            'shaping': {'gamma': 1., 'beta': 1.}, 'distributed': {'kl_batch_size': 2}}
    if fast:
        plan['distributed']['fast'] = fast
    return plan


class Interrupt(Exception):
    pass


def worker(rank, world, root, mode, lr, kl_target, updates, fast, layout, interrupt_at):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method='file://' + str(root / 'rendezvous'), rank=rank, world_size=world)
    torch.manual_seed(0)
    model = BatchToy()
    episodes = episodes_for(model, layout)
    plan = make_plan(lr, kl_target, updates, fast)
    parallel = Parallel('cpu')
    optimizer = base.build_optimizer(model, plan)
    rows, data = buffers(episodes, plan)
    manager = Checkpoints(root / 'checkpoint', {'variant': 'control'}, parallel)
    trainer = train_fast if mode == 'fast' else train

    def publish(**state):
        if interrupt_at and state.get('stage') == 'update_checkpoint_saved' and state['accepted_optimizer_steps'] == interrupt_at:
            raise Interrupt()
    try:
        result = trainer(model, optimizer, rows, data, plan, parallel, manager, publish)
    except Interrupt:
        directory, metadata = manager.latest()
        model = BatchToy()
        base.load_overlay(model, directory, metadata['trainable_sha256'])
        optimizer = base.build_optimizer(model, plan)
        optimizer.load_state_dict(torch.load(directory / 'optimizer.pt', weights_only=True))
        data = torch.load(directory / 'buffers.pt', weights_only=True)
        result = trainer(model, optimizer, rows, data, plan, parallel, manager, lambda **_: None, resume=metadata)
    if rank == 0:
        torch.save({'model': model.state_dict(), 'result': result}, root / 'actual.pt')
    dist.destroy_process_group()


def run(mode, lr=.001, kl_target=.01, updates=3, fast=None, layout='uniform', interrupt_at=None):
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(4, root, mode, lr, kl_target, updates, fast, layout, interrupt_at), nprocs=4, join=True)
        return torch.load(Path(root) / 'actual.pt', weights_only=False)


def assert_same_model(case, a, b, atol=1e-6):
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=atol, rtol=1e-5, msg=name)


EVERY_STEP_FULL = {'microbatch_size': 4, 'sampled_kl_chunks': 10**6, 'full_kl_every': 1, 'sampled_kl_threshold_fraction': 1.0}


class FastLearner(unittest.TestCase):
    def test_fast_with_full_checks_matches_legacy(self):
        legacy, fast = run('legacy', updates=3), run('fast', updates=3, fast=EVERY_STEP_FULL)
        self.assertEqual(fast['result']['optimizer_steps'], legacy['result']['optimizer_steps'])
        for a, b in zip(legacy['result']['update_steps'], fast['result']['update_steps']):
            self.assertEqual(a['accepted'], b['accepted'])
            self.assertAlmostEqual(a['candidate_kl'], b['candidate_kl'], places=9)
        assert_same_model(self, fast['model'], legacy['model'])
        self.assertAlmostEqual(fast['result']['final_batch_kl']['mean'], legacy['result']['final_batch_kl']['mean'], places=9)

    def test_sampled_check_on_all_rows_matches_full(self):
        sampled = {'microbatch_size': 1, 'sampled_kl_chunks': 10**6, 'full_kl_every': 100, 'sampled_kl_threshold_fraction': 1.0}
        legacy, fast = run('legacy', updates=3), run('fast', updates=3, fast=sampled)
        self.assertEqual(fast['result']['optimizer_steps'], legacy['result']['optimizer_steps'])
        self.assertEqual([s['kl_check'] for s in fast['result']['update_steps']], ['sampled', 'sampled', 'full'])
        assert_same_model(self, fast['model'], legacy['model'], atol=1e-7)

    def test_kl_guard_rejection_matches_legacy(self):
        legacy, fast = run('legacy', lr=1.0, updates=3), run('fast', lr=1.0, updates=3, fast=EVERY_STEP_FULL)
        self.assertTrue(fast['result']['early_stopped'])
        self.assertEqual(fast['result']['optimizer_steps'], legacy['result']['optimizer_steps'])
        assert_same_model(self, fast['model'], legacy['model'])

    def test_final_overshoot_rolls_back_to_verified_state(self):
        fast = {'microbatch_size': 2, 'sampled_kl_chunks': 3, 'full_kl_every': 100, 'sampled_kl_threshold_fraction': .8}
        actual = run('fast', lr=.01, kl_target=.001, updates=2, fast=fast, layout='masked')
        result = actual['result']
        self.assertEqual(result['update_steps'][0]['kl_check'], 'sampled')
        self.assertTrue(result['update_steps'][0]['accepted'])  # the subset hid the drift
        self.assertTrue(result['final_rollback_to_verified'])
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertLessEqual(result['final_batch_kl']['mean'], .001)
        torch.manual_seed(0)
        assert_same_model(self, actual['model'], BatchToy().state_dict(), atol=0)

    def test_resume_from_verified_checkpoint_matches_uninterrupted(self):
        fast = {'microbatch_size': 4, 'sampled_kl_chunks': 10**6, 'full_kl_every': 2, 'sampled_kl_threshold_fraction': 1.0}
        whole = run('fast', updates=4, fast=fast)
        resumed = run('fast', updates=4, fast=fast, interrupt_at=2)
        self.assertEqual(resumed['result']['optimizer_steps'], whole['result']['optimizer_steps'])
        assert_same_model(self, resumed['model'], whole['model'], atol=1e-7)

    def test_batched_gradients_match_single_row(self):
        torch.manual_seed(0)
        model = BatchToy()
        episodes = episodes_for(model, 'uniform')
        plan = make_plan(.001, .01, 3)
        rows, data = buffers(episodes, plan)
        parallel = Parallel('cpu')
        indices = list(range(len(rows)))
        model.zero_grad(set_to_none=True)
        expected_loss = backward(model, rows, data, indices, plan, parallel, distributed=False)
        expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        actual_loss = backward_batched(model, rows, data, indices, plan, parallel, 4, distributed=False)
        self.assertAlmostEqual(actual_loss, expected_loss, places=6)
        for n, p in model.named_parameters():
            if n in expected:
                torch.testing.assert_close(p.grad, expected[n], atol=1e-7, rtol=1e-5)

    def test_tolerant_replay_accepts_small_and_rejects_large_differences(self):
        torch.manual_seed(0)
        model = BatchToy()
        parallel = Parallel('cpu')
        tolerance = {'max_ratio_error': .01, 'max_mean_kl': 1e-5}
        for delta, should_pass in [(0., True), (.004, True), (.05, False)]:
            rows = make_episode(model, [1.] * 6, True, 1)['transitions']
            for row in rows:
                row['prev_logprobs'] = row['prev_logprobs'] - delta / 23
            if should_pass:
                stats = replay_check_tolerant(model, rows, parallel, None, 4, tolerance)
                self.assertTrue(stats['passed'])
                self.assertAlmostEqual(stats['max_probability_ratio_error'], math.expm1(delta), places=6)
            else:
                with self.assertRaises(AssertionError):
                    replay_check_tolerant(model, rows, parallel, None, 4, tolerance)


if __name__ == '__main__':
    unittest.main()


class NoisyBatchToy(BatchToy):
    """Batched forwards differ slightly from single-row forwards (like FP32 kernel order on a real model)."""

    def default_forward(self, inputs, compute_values=True):
        result = super().default_forward(inputs, compute_values)
        if inputs['x'].shape[0] > 1:
            result = {k: v * (1 + 1e-6) for k, v in result.items()}
        return result


class SwappedBatchToy(BatchToy):
    """A batching bug: outputs come back in the wrong row order."""

    def default_forward(self, inputs, compute_values=True):
        result = super().default_forward(inputs, compute_values)
        if inputs['x'].shape[0] > 1:
            result = {k: v.flip(0) for k, v in result.items()}
        return result


def gate_worker(rank, world, root, kind):
    from parallel_training import parity_gate
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method='file://' + str(root / 'rendezvous'), rank=rank, world_size=world)
    torch.manual_seed(0)
    model = {'noisy': NoisyBatchToy, 'swapped': SwappedBatchToy, 'exact': BatchToy}[kind]()
    # Distinct inputs per row so that a row mix-up changes the objective.
    episodes = [make_episode(model, [.5 + .1 * i for i in range(9)], True, 1),
                make_episode(model, [-.5 - .1 * i for i in range(14)], False, 2)]
    plan = make_plan(.001, .01, 3, {'microbatch_size': 4, 'sampled_kl_chunks': 10**6, 'full_kl_every': 1,
                                    'sampled_kl_threshold_fraction': 1.0})
    parallel = Parallel('cpu')
    optimizer = base.build_optimizer(model, plan)
    rows, data = buffers(episodes, plan)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    report = parity_gate(model, optimizer, rows, data, plan, parallel, sample_count=17)
    restored = all(torch.equal(before[n], p) for n, p in model.named_parameters())
    if rank == 0:
        torch.save({'report': report, 'restored': restored}, root / 'gate.pt')
    dist.destroy_process_group()


def run_gate(kind):
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(gate_worker, args=(4, root, kind), nprocs=4, join=True)
        return torch.load(Path(root) / 'gate.pt', weights_only=False)


class BatchedParityGate(unittest.TestCase):
    def test_exact_batching_passes_both_parts(self):
        out = run_gate('exact')
        self.assertTrue(out['report']['passed'] and out['report']['batched']['passed'] and out['restored'])

    def test_small_batched_noise_passes(self):
        out = run_gate('noisy')
        self.assertTrue(out['report']['passed'], out['report'])
        self.assertGreater(out['report']['batched']['gradient_relative_l2'], 0.0)
        self.assertTrue(out['restored'])

    def test_row_mixup_in_batches_fails(self):
        out = run_gate('swapped')
        self.assertFalse(out['report']['batched']['passed'], out['report']['batched'])
        self.assertFalse(out['report']['passed'])
        self.assertTrue(out['restored'])
