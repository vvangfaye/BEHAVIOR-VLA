"""Frozen-budget spectral controls for the existing generative prototype.

These are continuous DCT-flow controls, not reproductions of FAST/OAT. The
original three-arm experiment and its checkpoints are never modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_activity import data
from train_generative import Flow
from analyze_generative import energy

HERE = Path(__file__).resolve().parent
METHODS = ('dct_full_conditioned', 'dct_fixedrank_conditioned')


def prepare():
    rows = data()
    tasks = sorted({r['task'] for r in rows})
    fold = {}
    for task in tasks:
        eps = sorted({r['episode'] for r in rows if r['task'] == task},
                     key=lambda e: hashlib.sha256(str(e).encode()).hexdigest())
        fold.update({e: i % 5 for i, e in enumerate(eps)})
    train = np.array([fold[r['episode']] in (2, 3, 4) for r in rows])
    test = np.array([fold[r['episode']] == 0 for r in rows])
    raw = np.stack([r['x'] for r in rows]).astype(np.float32)
    task_ids = np.array([tasks.index(r['task']) for r in rows])
    raw = np.concatenate([raw, np.eye(len(tasks), dtype=np.float32)[task_ids]], -1)
    x = (raw - raw[train].mean(0)) / np.maximum(raw[train].std(0), .01)
    x = torch.from_numpy(np.clip(x, -10, 10))
    raw_y = np.stack([r['future'] - r['x'][61:84] for r in rows]).astype(np.float32)
    mean = raw_y[train].mean((0, 1))
    std = np.maximum(raw_y[train].std((0, 1)), .01)
    y = torch.from_numpy((raw_y - mean) / std)
    active = torch.from_numpy(np.stack([r['y'] for r in rows]).astype(np.float32))
    return rows, x, y, active, torch.tensor(np.flatnonzero(train)), torch.tensor(np.flatnonzero(test))


def basis():
    time_index = torch.arange(32, dtype=torch.float64)[None]
    frequency = torch.arange(32, dtype=torch.float64)[:, None]
    b = torch.cos(math.pi * (time_index + .5) * frequency / 32) * math.sqrt(2 / 32)
    b[0] /= math.sqrt(2)
    torch.testing.assert_close(b @ b.T, torch.eye(32, dtype=b.dtype), atol=1e-13, rtol=0)
    return b.float()


def ranks(active, train_ids):
    # Match the expected continuous dimension of the support model on TRAIN.
    # Per-channel allocation retains known marginal activity information.
    expected = 1 + 31 * active[train_ids].mean(0).double().numpy()
    rank = np.floor(expected).astype(int)
    budget = int(np.rint(expected.sum()))
    for j in np.argsort(-(expected - rank))[:budget - int(rank.sum())]:
        rank[j] += 1
    assert rank.min() >= 1 and rank.max() <= 32 and rank.sum() == budget
    return rank, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=[17, 29, 43])
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows, x, y, active, train_ids, test_ids = prepare()
    b = basis()
    encode = lambda v: torch.einsum('kt,ntd->nkd', b, v)
    decode = lambda v: torch.einsum('kt,nkd->ntd', b, v)
    torch.testing.assert_close(decode(encode(y[:8])), y[:8], atol=3e-6, rtol=3e-6)
    rank, expected = ranks(active, train_ids)
    lowpass = (torch.arange(32)[:, None] < torch.from_numpy(rank)[None]).float()
    truth_coef = encode(y)
    output = HERE / f'dct_controls_seed{args.seed}'
    output.mkdir(exist_ok=True)
    protocol = {
        'seed': args.seed, 'methods': METHODS, 'steps': 3000, 'batch': 256,
        'lr': .0003, 'weight_decay': .0001, 'aux_mask_bce_weight': .1,
        'inference_euler_steps': 10, 'test_samples_per_context': 8,
        'train_chunks': len(train_ids), 'test_chunks': len(test_ids),
        'split': 'same episode hash folds: train 2,3,4; test 0; fold 1 unused',
        'parameters': sum(p.numel() for p in Flow(x.shape[-1]).parameters()),
        'lowpass_rank_per_channel': rank.tolist(),
        'train_support_expected_dimensions': float(expected.sum()),
        'lowpass_dimensions': int(rank.sum()),
        'all_methods_receive': 'same true activity labels in training and sampled predicted labels at test',
        'normalization': 'same train-only per-channel action and context normalization as original',
        'rng': 'same initialization and batches; noise sampled in time coordinates then rotated to DCT',
        'scope': 'continuous spectral-flow control; not FAST/OAT reproduction; no visual input or closed loop',
    }
    path = output / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else:
        path.write_text(json.dumps(protocol, indent=2) + '\n')
    start = time.monotonic()
    for method in METHODS:
        path = output / f'{method}.npz'
        if path.exists():
            continue
        torch.manual_seed(args.seed)
        model = Flow(x.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        keep = torch.ones(32, 23) if method == 'dct_full_conditioned' else lowpass
        for step in range(3000):
            ids = train_ids[torch.randint(len(train_ids), (256,))]
            h = model.context(x[ids])
            logits = model.mask(h)
            mask = active[ids]
            noise = encode(torch.randn_like(y[ids])) * keep
            target_action = truth_coef[ids] * keep
            t = torch.rand(len(ids))
            xt = (1 - t[:, None, None]) * noise + t[:, None, None] * target_action
            pred = model.velocity(xt, t, h, mask, 'mask_conditioned_flow') * keep
            loss = F.mse_loss(pred, target_action - noise) + .1 * F.binary_cross_entropy_with_logits(logits, mask)
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if step % 500 == 0:
                print(json.dumps({'seed': args.seed, 'method': method, 'step': step,
                                  'loss': float(loss.detach()), 'seconds': round(time.monotonic() - start, 1)}), flush=True)
        model.eval()
        samples = []
        torch.manual_seed(761)
        with torch.inference_mode():
            for ids in test_ids.split(64):
                h = model.context(x[ids])
                probs = model.mask(h).sigmoid().repeat_interleave(8, 0)
                h = h.repeat_interleave(8, 0)
                mask = torch.bernoulli(probs)
                z = encode(torch.randn(len(h), 32, 23)) * keep
                for k in range(10):
                    v = model.velocity(z, torch.full((len(h),), k / 10), h, mask, 'mask_conditioned_flow')
                    z = z + .1 * v * keep
                samples.append(decode(z).reshape(len(ids), 8, 736))
        samples = torch.cat(samples).numpy()
        target = y[test_ids].flatten(1).numpy()
        scores = energy(samples, target, list(range(23)))
        mse = np.square(samples.mean(1) - target).mean(-1)
        np.savez_compressed(path, energy_score=scores, mean_mse=mse,
                            episode=np.array([rows[i]['episode'] for i in test_ids]),
                            frame=np.array([rows[i]['frame'] for i in test_ids]),
                            samples=samples, target=target)
        torch.save(model.state_dict(), output / f'{method}.pt')
        print(json.dumps({'seed': args.seed, 'method': method, 'energy_score': float(scores.mean()),
                          'mean_mse': float(mse.mean()), 'seconds': time.monotonic() - start}), flush=True)
    result = {'scope': protocol['scope'], 'elapsed_seconds': time.monotonic() - start,
              'methods': {}}
    for method in METHODS:
        with np.load(output / f'{method}.npz') as saved:
            result['methods'][method] = {'energy_score': float(saved['energy_score'].mean()),
                                         'mean_mse': float(saved['mean_mse'].mean())}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
