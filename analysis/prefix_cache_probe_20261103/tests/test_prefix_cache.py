"""CPU checks of the probe's PrefixCache on a toy model with the same call structure as Pi0RL.default_forward
(one prefix build per call, a batch-first per-layer (k, v) cache used functionally, values from the prefix)."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prefix_probe_driver as probe  # noqa: E402  (runtime modules come from PYTHONPATH)
import kl_batch  # noqa: E402
from parallel_training import Parallel, backward_batched  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Elementwise (batch-invariant) frozen prefix: this test checks the cache mechanism. Whether a
        # real GEMM prefix computed at another batch size differs in the last bits is what the GPU probe
        # measures (on CPU it does: a Linear at batch 3 vs 2 is not bitwise equal).
        self.scale = torch.nn.Parameter(torch.randn(6), requires_grad=False)
        self.actor = torch.nn.Linear(6, 23)
        self.value_head = torch.nn.Linear(6, 1)
        self.prefix_calls = 0

    def _observation_from_forward_inputs(self, inputs):
        return SimpleNamespace(x=inputs['x'], tokenized_prompt=inputs['tokenized_prompt'])

    def _build_prefix_cache_for_actor(self, observation):
        self.prefix_calls += 1
        with torch.no_grad():
            hidden = torch.tanh(torch.cat([observation.x, observation.x[:, :2]], dim=1) * self.scale)  # (B, 6)
            prefix_out = torch.stack([hidden, 2 * hidden], dim=1)          # (B, 2, 6)
            mask = torch.ones(hidden.shape[0], 2, dtype=torch.bool)
            kv = tuple((hidden[:, None, None, :] * (l + 1), hidden[:, None, None, :] - l) for l in range(3))
        return prefix_out, mask, kv

    def default_forward(self, inputs, compute_values=True):
        observation = self._observation_from_forward_inputs(inputs)
        prefix_out, mask, kv = self._build_prefix_cache_for_actor(observation)
        context = sum(torch.cat([k, v], dim=1).sum(dim=(1, 2)) for k, v in kv)   # functional use, like run_suffix
        logprobs = (self.actor(context) / 23).reshape(-1, 1, 23)
        result = {'logprobs': logprobs}
        if compute_values:
            pooled = (prefix_out * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            result['values'] = self.value_head(pooled).reshape(-1)
        return result


def rows_and_data(count=10):
    torch.manual_seed(1)
    rows = []
    for i in range(count):
        rows.append({'forward_inputs': {'x': torch.randn(1, 4), 'tokenized_prompt': torch.tensor([[i, i + 1, 7]])},
                     'prev_logprobs': torch.randn(1, 1, 23) * .01 - 1.})
    data = {'advantages': torch.randn(count), 'returns': torch.randn(count), 'weights': torch.ones(count),
            'prev_values': torch.randn(count, 1)}
    return rows, data


PLAN = {'comparison_variant': 'control', 'optimizer': {'clip_ratio': .2}}


class PrefixCacheTest(unittest.TestCase):
    def setUp(self):
        self.original_collate = kl_batch.collate

    def tearDown(self):
        kl_batch.collate = self.original_collate

    def test_cached_outputs_and_gradients_equal_and_no_recompute(self):
        model, (rows, data) = Toy(), rows_and_data()
        indices, device = list(range(len(rows))), torch.device('cpu')
        cache = probe.PrefixCache(model)
        base3, old = probe.kl_pass(model, rows, indices, 3, device)
        cache.build(rows, indices, 3, device)
        cache.enabled = True
        calls = model.prefix_calls
        cached3, _ = probe.kl_pass(model, rows, indices, 3, device)
        cached2, _ = probe.kl_pass(model, rows, indices, 2, device)
        self.assertEqual(model.prefix_calls, calls)          # the frozen prefix is never recomputed
        cache.enabled = False
        base2, _ = probe.kl_pass(model, rows, indices, 2, device)
        for i in indices:
            self.assertTrue(torch.equal(cached3[i], base3[i]))
            self.assertTrue(torch.equal(cached2[i], base2[i]))
        self.assertEqual(probe.compare(cached3, base3)['max_abs_logprob_difference'], 0.0)
        parallel = Parallel(device)
        grads = {}
        for label, enabled in [('baseline', False), ('cached', True)]:
            cache.enabled = enabled
            model.zero_grad(set_to_none=True)
            backward_batched(model, rows, data, indices[:7], PLAN, parallel, 4, None, distributed=False)
            grads[label] = probe.gradients(model)
        result = probe.agreement(grads['cached'], grads['baseline'])
        self.assertEqual(result['relative_l2'], 0.0)
        self.assertAlmostEqual(result['cosine'], 1.0, places=12)
        self.assertEqual(cache.hits, len(indices) * 2 + 7)

    def test_misaligned_rows_are_rejected(self):
        model, (rows, _) = Toy(), rows_and_data()
        indices, device = list(range(len(rows))), torch.device('cpu')
        cache = probe.PrefixCache(model)
        cache.build(rows, indices, 4, device)
        cache.tokens[3] = cache.tokens[4]
        cache.enabled = True
        with self.assertRaises(AssertionError):
            probe.kl_pass(model, rows, indices, 4, device)

    def test_split_join_round_trip(self):
        tree = (torch.arange(12.).reshape(3, 4), torch.ones(3, 2, dtype=torch.bool),
                tuple((torch.randn(3, 5, 1, 2), torch.randn(3, 5, 1, 2)) for _ in range(2)))
        parts = probe.split_rows(tree, 3)
        joined = probe.join_rows(parts, torch.device('cpu'))
        self.assertTrue(torch.equal(joined[0], tree[0]) and torch.equal(joined[1], tree[1]))
        for (k, v), (k0, v0) in zip(joined[2], tree[2]):
            self.assertTrue(torch.equal(k, k0) and torch.equal(v, v0))
        self.assertEqual(probe.nbytes(parts[0]), probe.nbytes(tree) // 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
