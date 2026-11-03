"""CPU tests for the per-round prefix cache (runtime/prefix_cache.py) and its learner integration.

PrefixToy computes BatchToy's function through a frozen "prefix" exactly like Pi0RL.default_forward does
(one _build_prefix_cache_for_actor call per forward, a batch-first per-layer (k, v) tuple used functionally,
values pooled from prefix_out), so serving cached prefixes must reproduce the uncached learner.
"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import kl_batch
import prefix_cache
import training as base
from parallel_training import Parallel, Checkpoints, backward, backward_batched, buffers, train_fast


class PrefixToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.tensor([.1]))
        self.value_head = torch.nn.Linear(1, 1)
        self.frozen = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
        self.prefix_rows = 0

    def _observation_from_forward_inputs(self, inputs):
        return SimpleNamespace(state=inputs['x'], tokenized_prompt=inputs['tokens'])

    def _build_prefix_cache_for_actor(self, observation):
        self.prefix_rows += observation.state.shape[0]
        x = observation.state
        with torch.no_grad():
            kv = tuple((x[:, None, None, :] * 1.0, x[:, None, None, :] * 0.0) for _ in range(2))
            return x[:, None, :] * 1.0, torch.ones(x.shape[0], 1, dtype=torch.bool), kv

    def default_forward(self, inputs, compute_values=True):
        prefix_out, mask, kv = self._build_prefix_cache_for_actor(self._observation_from_forward_inputs(inputs))
        x = torch.cat([kv[0][0], kv[0][1]], dim=1)[:, 0, 0, :]
        size = x.shape[0]
        result = {'logprobs': (self.actor * x).reshape(size, 1, 1).expand(size, 1, 23) / 23}
        if compute_values:
            pooled = (prefix_out * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            result['values'] = self.value_head(pooled).reshape(size)
        return result


def make_episode(model, xs, success, instance):
    rows = []
    for i, value in enumerate(xs):
        inputs = {'x': torch.tensor([[float(value)]]), 'tokens': torch.tensor([[instance, i, 7]])}
        previous = model.default_forward(inputs)
        last = i == len(xs) - 1
        rows.append({'forward_inputs': inputs, 'prev_logprobs': previous['logprobs'].detach().clone(),
                     'prev_values': previous['values'].detach().clone(), 'terminal': last,
                     'reward': float(success and last), 'task_reward': float(success and last),
                     'goals_before': {'goals': {'a': False}}, 'goals_after': {'goals': {'a': bool(success)}}})
    return {'instance': instance, 'status': {'success': success}, 'transitions': rows, 'bootstrap': torch.zeros(1)}


def episodes_for(model):
    return [make_episode(model, [1. + .1 * i for i in range(9)], True, 1),
            make_episode(model, [-1. - .05 * i for i in range(14)], False, 2)]


FAST = {'microbatch_size': 4, 'sampled_kl_chunks': 10**6, 'full_kl_every': 1, 'sampled_kl_threshold_fraction': 1.0}


def make_plan(fast, updates=4, lr=.001):
    return {'comparison_variant': 'control',
            'optimizer': {'actor_lr': lr, 'value_lr': 1e-4, 'weight_decay': 0., 'shuffle_seed': 5,
                          'updates_per_round': [updates], 'gamma': .99, 'gae_lambda': .95, 'clip_ratio': .2,
                          'kl_target': .01, 'allow_kl_early_stop': True},
            'shaping': {'gamma': 1., 'beta': 1.}, 'distributed': {'kl_batch_size': 3, 'fast': fast}}


class Interrupt(Exception):
    pass


def worker(rank, world, root, fast, cached, interrupt_at):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method='file://' + str(root / 'rendezvous'), rank=rank, world_size=world)
    torch.manual_seed(0)
    model = PrefixToy()
    episodes = episodes_for(model)
    plan = make_plan(fast)
    parallel = Parallel('cpu')
    rows, data = buffers(episodes, plan)
    cache = None
    if cached:
        cache, admission = prefix_cache.build_for_rank(model, rows, parallel, 3)
        assert admission['enabled'] and cache.pinned in (True, False)
    computed_after_build = model.prefix_rows
    optimizer = base.build_optimizer(model, plan)
    manager = Checkpoints(root / 'checkpoint', {'variant': 'control'}, parallel)

    def publish(**state):
        if interrupt_at and state.get('stage') == 'update_checkpoint_saved' and state['accepted_optimizer_steps'] == interrupt_at:
            raise Interrupt()
    try:
        result = train_fast(model, optimizer, rows, data, plan, parallel, manager, publish)
    except Interrupt:
        directory, metadata = manager.latest()
        base.load_overlay(model, directory, metadata['trainable_sha256'])
        optimizer = base.build_optimizer(model, plan)
        optimizer.load_state_dict(torch.load(directory / 'optimizer.pt', weights_only=True))
        data = torch.load(directory / 'buffers.pt', weights_only=True)
        result = train_fast(model, optimizer, rows, data, plan, parallel, manager, lambda **_: None, resume=metadata)
    report = cache.report() if cache else None
    reports = [None] * world
    dist.all_gather_object(reports, {'report': report, 'recomputed': model.prefix_rows - computed_after_build})
    if cache:
        cache.uninstall()
    if rank == 0:
        torch.save({'model': model.state_dict(), 'result': result, 'reports': reports}, root / 'actual.pt')
    dist.destroy_process_group()


def run(fast, cached, interrupt_at=None):
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(4, root, fast, cached, interrupt_at), nprocs=4, join=True)
        return torch.load(Path(root) / 'actual.pt', weights_only=False)


class PrefixCacheUnit(unittest.TestCase):
    def setUp(self):
        self.collate = kl_batch.collate

    def tearDown(self):
        if prefix_cache.ACTIVE is not None:
            prefix_cache.ACTIVE.uninstall()
        kl_batch.collate = self.collate

    def fixture(self):
        torch.manual_seed(0)
        model = PrefixToy()
        rows, data = buffers(episodes_for(model), make_plan({**FAST, 'prefix_cache': True}))
        cache = prefix_cache.PrefixCache(model, torch.device('cpu'))
        return model, rows, data, cache

    def test_served_outputs_equal_and_prefix_not_recomputed(self):
        model, rows, _, cache = self.fixture()
        owned = list(range(len(rows)))
        expected = {i: model.default_forward(rows[i]['forward_inputs'])['logprobs'] for i in owned}
        cache.build(rows, owned, 3)
        cache.install()
        before = model.prefix_rows
        for size in (1, 3, 5):
            for group in kl_batch.batches(rows, owned, size):
                inputs, _ = kl_batch.collate(rows, group, torch.device('cpu'))
                out = model.default_forward(inputs)['logprobs']
                for j, i in enumerate(group):
                    self.assertTrue(torch.equal(out[j:j + 1], expected[i]))
        self.assertEqual(model.prefix_rows, before)
        self.assertEqual(cache.stats['computed'], 0)
        self.assertEqual(cache.stats['hits'], 3 * len(rows))

    def test_non_contiguous_rows_and_explicit_select(self):
        model, rows, _, cache = self.fixture()
        cache.build(rows, list(range(len(rows))), 4)
        cache.install()
        group = [7, 2, 19, 11]
        inputs, _ = kl_batch.collate(rows, group, torch.device('cpu'))
        served = model.default_forward(inputs, compute_values=True)
        cache.uninstall()
        plain = model.default_forward(inputs, compute_values=True)
        self.assertTrue(torch.equal(served['logprobs'], plain['logprobs']))
        self.assertTrue(torch.equal(served['values'], plain['values']))
        cache.install()
        prefix_cache.select([5])
        feature = base.prefix_features(model, rows[5]['forward_inputs'])
        self.assertEqual(cache.stats['hits'], 5)
        self.assertTrue(torch.equal(feature, rows[5]['forward_inputs']['x']))

    def test_unselected_mismatched_and_unowned_rows_are_computed(self):
        model, rows, _, cache = self.fixture()
        cache.build(rows, list(range(0, len(rows), 2)), 3)   # owns even rows only
        cache.install()
        model.default_forward(rows[4]['forward_inputs'])          # no collate/select: computed
        self.assertEqual(cache.stats['computed'], 1)
        inputs, _ = kl_batch.collate(rows, [0, 1], torch.device('cpu'))   # row 1 not owned: computed
        model.default_forward(inputs)
        self.assertEqual(cache.stats['computed'], 3)
        prefix_cache.select([2])                                  # wrong row selected: detected, computed
        out = model.default_forward(rows[4]['forward_inputs'])['logprobs']
        self.assertEqual(cache.stats['mismatches'], 1)
        cache.uninstall()
        self.assertTrue(torch.equal(out, model.default_forward(rows[4]['forward_inputs'])['logprobs']))
        self.assertNotIn('_build_prefix_cache_for_actor', model.__dict__)

    def test_pageable_fallback_when_pinning_fails(self):
        model, rows, _, cache = self.fixture()
        real = torch.empty

        def refuse_pinned(*args, **kwargs):
            if kwargs.get('pin_memory'):
                raise RuntimeError('pinning refused (test)')
            return real(*args, **kwargs)
        prefix_cache.torch.empty = refuse_pinned
        try:
            cache.build(rows, list(range(len(rows))), 3)
        finally:
            prefix_cache.torch.empty = real
        self.assertIs(cache.pinned, False)
        cache.install()
        inputs, _ = kl_batch.collate(rows, [3, 4, 5], torch.device('cpu'))
        served = model.default_forward(inputs)['logprobs']
        cache.uninstall()
        self.assertTrue(torch.equal(served, model.default_forward(inputs)['logprobs']))

    def test_owned_partition_gradient_equals_serial(self):
        model, rows, data, cache = self.fixture()
        plan = make_plan({**FAST, 'prefix_cache': True})
        indices = list(range(len(rows)))
        serial = Parallel('cpu')
        model.zero_grad(set_to_none=True)
        backward(model, rows, data, indices, plan, serial, distributed=False)
        expected = [p.grad.clone() for p in model.parameters() if p.requires_grad]

        class Rank(Parallel):
            def __init__(self, rank):
                super().__init__('cpu')
                self.rank, self.world = rank, 4
        total = [torch.zeros_like(g) for g in expected]
        covered = []
        for rank in range(4):
            model.zero_grad(set_to_none=True)
            part = Rank(rank)
            # distributed=False skips the all-reduce; emulate one rank's owned share explicitly
            owned = [i for i in indices if i % 4 == rank]
            covered += owned
            backward_batched(model, rows, data, owned, plan, part, 4, distributed=False)
            for t, p in zip(total, [p for p in model.parameters() if p.requires_grad]):
                t += p.grad * len(owned) / len(indices)
        self.assertEqual(sorted(covered), indices)
        for t, e in zip(total, expected):
            torch.testing.assert_close(t, e, atol=1e-7, rtol=1e-6)


class CachedLearner(unittest.TestCase):
    def test_cached_training_matches_uncached_and_never_recomputes(self):
        plain = run(FAST, cached=False)
        cached = run({**FAST, 'prefix_cache': True, 'checkpoint_every': 2}, cached=True)
        self.assertEqual(plain['result']['optimizer_steps'], cached['result']['optimizer_steps'])
        for name in plain['model']:
            torch.testing.assert_close(cached['model'][name], plain['model'][name], atol=1e-6, rtol=1e-5, msg=name)
        for item in cached['reports']:
            self.assertEqual(item['recomputed'], 0)           # every KL and gradient forward was served
            self.assertEqual(item['report']['mismatches'], 0)
            self.assertGreater(item['report']['hits'], 0)

    def test_resume_with_sparse_checkpoints(self):
        fast = {**FAST, 'prefix_cache': True, 'checkpoint_every': 2}
        straight = run(fast, cached=True)
        resumed = run(fast, cached=True, interrupt_at=2)
        self.assertEqual(straight['result']['optimizer_steps'], resumed['result']['optimizer_steps'])
        for name in straight['model']:
            torch.testing.assert_close(resumed['model'][name], straight['model'][name], atol=1e-7, rtol=1e-6, msg=name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
