"""Compare a conditioned VLA sample and its same-mask output projection.

This is inference only. Each pair shares the exact action sample, conditioning
mask, observations, frozen checkpoint and original held-out sample IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from vla_data import VlaPilotData, digest
from vla_runtime import load_base
from vla_support_adapter import SupportActionAdapter


def scores(prediction, target):
    flat = prediction.flatten(2)
    truth = target.flatten(1)
    draws = prediction.shape[1]
    first = (flat - truth[:, None]).norm(dim=-1).mean(1)
    second = torch.cdist(flat, flat).sum((1, 2)) / (draws * (draws - 1))
    return ((first - .5 * second) / truth.shape[-1] ** .5,
            (prediction.mean(1) - target).square().mean((1, 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--trained-directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    run = json.loads((args.trained_directory / 'run.json').read_text())
    result = json.loads((args.trained_directory / 'result.json').read_text())
    completion = json.loads((args.trained_directory / 'completion.json').read_text())
    if not (result['complete'] and result['completed_steps'] == 1000 and completion['accepted']
            and run['method'] == 'conditioned' and run['plan_sha256'] == digest(args.plan)):
        raise ValueError('Expected the completed, fixed-step conditioned training run')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    base = load_base(plan['checkpoint'], plan['tokenizer'], device='cuda', precision='bf16')
    adapter = SupportActionAdapter(base).cuda().eval()
    saved = torch.load(args.trained_directory / 'checkpoint.pt', map_location='cpu', weights_only=False)
    expected = {name for name, param in adapter.named_parameters() if param.requires_grad}
    if not (saved['complete'] and saved['step'] == 1000 and saved['method'] == 'conditioned'
            and saved['plan_sha256'] == digest(args.plan) and set(saved['weights']) == expected):
        raise ValueError('Checkpoint does not match the completed training run')
    loaded = adapter.load_state_dict(saved['weights'], strict=False)
    assert not loaded.unexpected_keys and not (expected & set(loaded.missing_keys))
    del saved
    data = VlaPilotData(plan['data_directory'], base)
    with np.load(args.trained_directory / 'test.npz', allow_pickle=False) as original:
        ids = original['index'].copy()
        prior_samples = original['samples'].copy()
    if not set(ids) <= set(data.test_ids):
        raise ValueError('Evaluation IDs differ from the held-out split')
    rng = torch.Generator(device='cuda').manual_seed(761)
    before, after, targets, masks, records = [], [], [], [], []
    started = time.monotonic()
    with torch.inference_mode():
        for ordinal, index in enumerate(ids):
            if time.monotonic() - started > 3300:
                raise TimeoutError('Posthoc comparison exceeded its one-hour budget')
            observation, target, _ = data.batch([int(index)])
            raw_samples, projected_samples, drawn_masks = [], [], []
            for draw in range(plan['eval_draws']):
                noise = torch.randn(target.shape, device='cuda', generator=rng)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    action, mask, _ = adapter.sample(observation, noise=noise, steps=plan['euler_steps'],
                                                    rng=rng, mode='conditioned')
                    projected = adapter.project_physical(action, mask)
                raw_samples.append(action[..., :23].float().cpu())
                projected_samples.append(projected[..., :23].float().cpu())
                drawn_masks.append(mask.cpu())
            a, b = torch.stack(raw_samples, 1), torch.stack(projected_samples, 1)
            truth = target[..., :23].float().cpu()
            es_a, mse_a = scores(a, truth)
            es_b, mse_b = scores(b, truth)
            records.append({'index': int(index), 'episode': int(data.episode[index]),
                            'energy_conditioned': float(es_a[0]), 'energy_posthoc': float(es_b[0]),
                            'mse_conditioned': float(mse_a[0]), 'mse_posthoc': float(mse_b[0])})
            before.append(a.numpy()); after.append(b.numpy()); targets.append(truth.numpy())
            masks.append(torch.stack(drawn_masks, 1).numpy())
            if ordinal % 16 == 0:
                print(json.dumps({'evaluated': ordinal + 1, 'total': len(ids)}), flush=True)
    original = np.concatenate(before)
    np.savez_compressed(args.output / 'samples.npz', index=ids, episode=data.episode[ids],
                        target=np.concatenate(targets), conditioned=original,
                        posthoc=np.concatenate(after), sampled_masks=np.concatenate(masks))
    report = {'complete': True, 'training_plan_sha256': digest(args.plan), 'records': records,
              'means': {key: float(np.mean([r[key] for r in records])) for key in records[0]
                        if key.startswith(('energy_', 'mse_'))},
              'max_difference_from_original_test_draws': float(np.max(np.abs(original - prior_samples))),
              'paired_same_masks_and_noise': True, 'elapsed_seconds': time.monotonic() - started,
              'scope': 'Held-out expert-observation action scores; no closed-loop success claim'}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))


if __name__ == '__main__':
    main()
