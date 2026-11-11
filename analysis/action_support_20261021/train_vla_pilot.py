"""Bounded matched-budget SFT pilot; frozen VLM, trainable action expert.

This runner consumes an already completed immutable RGB cache. It never
downloads demonstrations, generates simulation data or selects on test scores.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from vla_runtime import load_base
from vla_data import VlaPilotData, digest
from vla_support_adapter import SupportActionAdapter


def atomic_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def choose_eval(data, ids, per_task):
    groups = {}
    for i in ids:
        task = data.plan['samples'][i]['task_index']
        groups.setdefault(task, []).append(i)
    chosen = []
    for task, group in sorted(groups.items()):
        group.sort(key=lambda i: hashlib.sha256(data.plan['samples'][i]['sample_id'].encode()).hexdigest())
        chosen.extend(group[:per_task])
    return np.asarray(chosen, dtype=int)


def evaluate(adapter, data, ids, method, output, *, draws, steps, batch, autocast):
    adapter.eval()
    records, predictions, truths, probabilities = [], [], [], []
    rng = torch.Generator(device=next(adapter.parameters()).device).manual_seed(761)
    with torch.inference_mode():
        for offset in range(0, len(ids), batch):
            selected = ids[offset:offset + batch]
            observation, target, active = data.batch(selected)
            samples = []
            for draw in range(draws):
                noise = torch.randn(target.shape, device=target.device, generator=rng)
                with autocast():
                    generated, _, probs = adapter.sample(observation, noise=noise, steps=steps,
                                                          rng=rng, mode=method)
                samples.append(generated[..., :23].float().cpu())
            prediction = torch.stack(samples, 1)
            truth = target[..., :23].float().cpu()
            flat = prediction.flatten(2)
            dimension = flat.shape[-1]
            distance = (flat - truth.flatten(1)[:, None]).norm(dim=-1).mean(1)
            pair = torch.cdist(flat, flat).sum((1, 2)) / (draws * (draws - 1))
            energy = (distance - .5 * pair) / dimension ** .5
            mse = (prediction.mean(1) - truth).square().mean((1, 2))
            brier = (probs.float().cpu() - active.float().cpu()).square().mean(1)
            for i, es, ms, br in zip(selected, energy.tolist(), mse.tolist(), brier.tolist()):
                records.append({'index': int(i), 'episode': int(data.episode[i]), 'frame': int(data.frame[i]),
                                'energy_score': es, 'mean_action_mse': ms, 'activity_brier': br})
            predictions.append(prediction.numpy())
            truths.append(truth.numpy())
            probabilities.append(probs.float().cpu().numpy())
    np.savez_compressed(output.with_suffix('.npz'), samples=np.concatenate(predictions),
                        target=np.concatenate(truths), probabilities=np.concatenate(probabilities),
                        index=ids, episode=data.episode[ids], frame=data.frame[ids])
    result = {'method': method, 'samples': len(records), 'episodes': len({r['episode'] for r in records}),
              'draws': draws, 'euler_steps': steps,
              'means': {name: float(np.mean([r[name] for r in records]))
                        for name in ['energy_score', 'mean_action_mse', 'activity_brier']},
              'records': records, 'scope': 'Offline physical23 action scores conditioned on existing expert subtask; no closed-loop task score'}
    atomic_json(output, result)
    return result['means']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--method', choices=['standard', 'conditioned', 'projected'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.method not in plan['methods'] or args.seed not in plan['seeds']:
        raise ValueError('Requested run is outside the frozen pilot plan')
    torch.set_num_threads(plan['cpu_threads'])
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output / 'checkpoint.pt'
    plan_sha = digest(args.plan)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    started = time.monotonic()
    device = plan['device']
    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if device.startswith('cuda') else nullcontext
    base = load_base(plan['checkpoint'], plan['tokenizer'], device=device, precision=plan['precision'])
    adapter = SupportActionAdapter(base).to(device)
    data = VlaPilotData(plan['data_directory'], base)
    if data.receipt['data_plan_sha256'] != plan['data_plan_sha256']:
        raise ValueError('Pilot data differs from training protocol')
    expert = [p for p in base.parameters() if p.requires_grad]
    structure = [*adapter.mask_head.parameters(), *adapter.mask_condition.parameters()]
    assert all(p.dtype == torch.float32 for p in expert + structure)
    optimizer = torch.optim.AdamW([{'params': expert, 'lr': plan['expert_lr']},
                                    {'params': structure, 'lr': plan['structure_lr']}],
                                   weight_decay=plan['weight_decay'])
    trainable_names = {n for n, p in adapter.named_parameters() if p.requires_grad}
    step = 0
    if existing.exists():
        saved = torch.load(existing, map_location='cpu', weights_only=False)
        assert saved['plan_sha256'] == plan_sha and saved['method'] == args.method and saved['seed'] == args.seed
        assert set(saved['weights']) == trainable_names
        loaded = adapter.load_state_dict(saved['weights'], strict=False)
        assert not loaded.unexpected_keys and not (set(loaded.missing_keys) & trainable_names)
        optimizer.load_state_dict(saved['optimizer'])
        step = saved['step']
        np_rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(saved['cuda_rng'])
    def save(completed):
        state = {'plan_sha256': plan_sha, 'method': args.method, 'seed': args.seed, 'step': step,
                 'weights': {n: p.detach().cpu() for n, p in adapter.named_parameters() if n in trainable_names},
                 'optimizer': optimizer.state_dict(), 'numpy_rng': np_rng.bit_generator.state,
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state() if device.startswith('cuda') else None,
                 'complete': completed, 'base_checkpoint': plan['checkpoint']}
        temporary = existing.with_suffix('.pt.tmp')
        torch.save(state, temporary)
        temporary.replace(existing)
    atomic_json(args.output / 'run.json', {'plan_sha256': plan_sha, 'method': args.method, 'seed': args.seed,
                                            'trainable_parameters': sum(p.numel() for p in expert + structure),
                                            'data_receipt_sha256': digest(data.directory / 'image_receipt.json'),
                                            'status': 'training', 'resumed_step': step})
    print(json.dumps({'event': 'training_start', 'method': args.method, 'seed': args.seed,
                      'steps': plan['steps'], 'resumed_step': step}), flush=True)
    parameters = expert + structure
    while step < plan['steps'] and not stop[0]:
        if time.monotonic() - started > plan['max_seconds']:
            break
        tick = time.monotonic()
        # Independent deterministic sampler and torch noise streams, shared
        # protocols across methods. No choice depends on target outcomes.
        indices = np_rng.choice(data.train_ids, size=plan['global_batch'], replace=True)
        optimizer.zero_grad(set_to_none=True)
        fraction = min(1., (step + 1) / plan['warmup_steps'])
        optimizer.param_groups[0]['lr'] = plan['expert_lr'] * fraction
        optimizer.param_groups[1]['lr'] = plan['structure_lr'] * fraction
        total = 0.
        for first in range(0, len(indices), plan['micro_batch']):
            selected = indices[first:first + plan['micro_batch']]
            observation, target, active = data.batch(selected)
            noise = torch.randn_like(target)
            time_value = torch.distributions.Beta(1.5, 1.).sample((len(selected),)).to(device) * .999 + .001
            with autocast():
                result = adapter.loss(observation, target, noise=noise, time=time_value,
                                      mode=args.method, raw_active=active)
                loss = result['loss'] * len(selected) / len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, plan['clip_grad'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradients')
        if step == 0:
            assert all(p.grad is None for p in base.parameters() if not p.requires_grad)
            q = base.llm.layers[0].attn.q_proj[1].weight
            assert q.grad is not None and q.grad.norm() > 0
        optimizer.step()
        step += 1
        if step == 1 or step % plan['log_every'] == 0:
            print(json.dumps({'event': 'update', 'method': args.method, 'step': step, 'loss': total,
                              'gradient_norm': float(norm), 'step_seconds': time.monotonic() - tick,
                              'elapsed_seconds': time.monotonic() - started,
                              'peak_gpu_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0}), flush=True)
        if step % plan['save_every'] == 0:
            save(False)
    complete = step == plan['steps']
    save(complete)
    if not complete:
        atomic_json(args.output / 'result.json', {'complete': False, 'completed_steps': step,
                                                  'reason': 'time_cap_or_termination', 'elapsed_seconds': time.monotonic() - started})
        raise SystemExit(2)
    validation = choose_eval(data, data.validation_ids, plan['eval_per_task'])
    test = choose_eval(data, data.test_ids, plan['eval_per_task'])
    results = {}
    for name, ids in [('validation', validation), ('test', test)]:
        results[name] = evaluate(adapter, data, ids, args.method, args.output / f'{name}.json',
                                  draws=plan['eval_draws'], steps=plan['euler_steps'],
                                  batch=plan['eval_batch'], autocast=autocast)
    atomic_json(args.output / 'result.json', {'complete': True, 'completed_steps': step,
                                              'elapsed_seconds': time.monotonic() - started, 'offline': results,
                                              'scope': 'Full VLA action expert SFT; offline scores only; no simulation'})
    print(json.dumps({'event': 'complete', 'method': args.method, 'offline': results}), flush=True)


if __name__ == '__main__':
    main()
