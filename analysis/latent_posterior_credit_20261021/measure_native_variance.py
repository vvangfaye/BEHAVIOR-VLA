"""Measure readout-score second moments under stationary native posterior paths.

No model update or environment calls. The readout score is exact for the final
linear layer, including all hidden coordinates and collapsed endpoint suffix.
It is not the full actor gradient and not a task-return gradient.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'analysis/relative_flow_20261021'))
from adapter import NativeAdapter
from vla_runtime import load_base
from vla_data import VlaPilotData
from rlinf.models.embodiment.openpi.sampling import rl_sampler
from native_transition import OpenPiGaussianTransition
from posterior import GaussianChainPosterior, normal_logprob


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = HERE / 'native_variance_v1'
    pp = root / 'plan.json'
    plan = json.loads(pp.read_text())
    if (root / 'collection_complete.json').exists() or (root / 'state.json').exists():
        raise RuntimeError('Existing run state; no silent restart')
    for p, digest in plan['source_sha256'].items():
        if sha(p) != digest:
            raise ValueError('Frozen source mismatch: ' + p)
    start = time.monotonic()
    calls = 0

    def progress(stage, **kw):
        result = dict(stage=stage, pid=os.getpid(), elapsed_seconds=time.monotonic() - start,
                      observed_at=datetime.now(timezone.utc).isoformat(), transition_calls=calls, **kw)
        (root / 'state.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)

    def deadline():
        if time.monotonic() - start > plan['max_seconds']:
            raise TimeoutError('Frozen native variance time cap reached')

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    progress('loading')
    hook = None
    try:
        base = load_base(plan['checkpoint'], plan['tokenizer'], precision='fp32')
        adapter = NativeAdapter(base).cuda().eval()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        data = VlaPilotData(plan['data_directory'], base)
        hidden = {}

        def capture(module, args):
            hidden['value'] = args[0].detach()

        hook = base.action_out_proj.register_forward_pre_hook(capture)
        W, bias = base.action_out_proj.weight, base.action_out_proj.bias
        W_before, bias_before = W.detach().clone(), bias.detach().clone()
        coeffs = []
        for k in range(plan['steps']):
            v = torch.zeros((1, 1, 1), device='cuda', requires_grad=True)
            mu, _ = rl_sampler.sample_mean_var(torch.zeros_like(v), v,
                     torch.tensor([k], device='cuda'), noise_method='flow_sde',
                     noise_level=plan['noise_level'], num_steps=plan['steps'])
            coeffs.append(float(torch.autograd.grad(mu.sum(), v)[0]))
        metadata = []
        for context in plan['contexts']:
            all_scores, all_observed, all_rewards, all_acceptance = [], [], [], []
            score_seconds = 0.0
            sweep_seconds = 0.0
            parity_error = None
            for offset in range(0, plan['paths_per_context'], plan['batch_size']):
                batch = min(plan['batch_size'], plan['paths_per_context'] - offset)
                observation, target, _ = data.batch([context] * batch)
                prepared = adapter.prepare(observation)
                native = OpenPiGaussianTransition(adapter, prepared, rl_sampler.sample_mean_var,
                           rl_sampler.get_timesteps, steps=plan['steps'], noise_level=plan['noise_level'])

                def transition(x, k):
                    nonlocal calls
                    deadline()
                    calls += 1
                    return native(x, k)

                mask = torch.zeros(native.horizon, native.dim, dtype=torch.bool, device='cuda')
                mask[:plan['executed_steps'], :plan['physical_dimensions']] = True
                mask = mask.flatten()
                engine = GaussianChainPosterior(transition, plan['steps'], mask)
                generator = torch.Generator(device='cuda').manual_seed(plan['seed'] + context * 19 + offset * 101)
                nodes, stds = [], []
                with torch.no_grad():
                    x = torch.randn((batch, native.horizon * native.dim), device='cuda', generator=generator)
                    for k in range(plan['steps']):
                        nodes.append(x.clone())
                        mu, std = transition(x, k)
                        stds.append(float(std.min()))
                        x = mu + std * torch.randn(x.shape, device='cuda', generator=generator)
                    path = torch.stack(nodes, dim=1)
                    observed = x[:, mask].clone()
                    fixed_observed = observed.clone()
                    # Explicitly artificial action-space weight; no task reward claim.
                    target_prefix = target.flatten(1)[:, mask]
                    reward = torch.exp(-(observed - target_prefix).square().mean(1)) - 0.5

                def factor_score(current_path, k):
                    mu, std = transition(current_path[:, k], k)
                    h = hidden['value']
                    if k == plan['steps'] - 1:
                        residual = torch.zeros_like(mu)
                        residual[:, mask] = (observed - mu[:, mask]) / std[:, mask].square()
                    else:
                        residual = (current_path[:, k + 1] - mu) / std.square()
                    sensitivity = (coeffs[k] * residual).reshape(batch, native.horizon, native.dim)
                    gw = torch.einsum('bhd,bhf->bdf', sensitivity, h)
                    gb = sensitivity.sum(1)
                    return torch.cat((gw.flatten(1), gb), dim=1)

                @torch.no_grad()
                def readout_score(current_path):
                    nonlocal score_seconds
                    tick = time.monotonic()
                    result = sum(factor_score(current_path, k) for k in range(plan['steps']))
                    torch.cuda.synchronize()
                    score_seconds += time.monotonic() - tick
                    if not torch.isfinite(result).all():
                        raise ValueError('Nonfinite readout score')
                    return result.cpu().numpy()

                if offset == 0:
                    checks = []
                    for k in (0, plan['steps'] - 1):
                        W.requires_grad_(True); bias.requires_grad_(True)
                        mu, std = transition(path[:, k].detach(), k)
                        if k == plan['steps'] - 1:
                            density = normal_logprob(observed, mu[:, mask], std[:, mask]).sum()
                        else:
                            density = normal_logprob(path[:, k + 1].detach(), mu, std).sum()
                        auto_w, auto_b = torch.autograd.grad(density, (W, bias))
                        with torch.no_grad():
                            computed = factor_score(path, k).sum(0)
                            expected = torch.cat((auto_w.flatten(), auto_b))
                            rel = float((computed - expected).norm() / expected.norm().clamp_min(1e-12))
                            checks.append(rel)
                        W.requires_grad_(False); bias.requires_grad_(False)
                        if rel > plan['score_parity_relative_tolerance']:
                            raise ValueError('Analytic readout score parity failed')
                    parity_error = max(checks)

                scores = [readout_score(path)]
                accepted_rows = []
                initial = path.clone()
                for sweep in range(1, plan['retained_paths']):
                    tick = time.monotonic()
                    accepted = []
                    for k in range(plan['steps']):
                        incoming = 1.0 if k == 0 else stds[k - 1]
                        epsilon = plan['step_scale'] * min(incoming, stds[k])
                        path, yes = engine.mala_step(path, observed, k, epsilon, generator=generator)
                        accepted.append(yes.cpu().numpy())
                    torch.cuda.synchronize()
                    sweep_seconds += time.monotonic() - tick
                    accepted_rows.append(np.stack(accepted, axis=1))
                    scores.append(readout_score(path))
                    if sweep % 4 == 0:
                        progress('sampling', context=context, path_offset=offset, sweep=sweep,
                                 retained_paths=plan['retained_paths'])
                if not torch.equal(observed, fixed_observed):
                    raise ValueError('Observed prefix changed')
                if torch.equal(path, initial):
                    raise ValueError('No latent movement')
                all_scores.append(np.stack(scores, axis=1))
                all_observed.append(observed.cpu().numpy())
                all_rewards.append(reward.cpu().numpy())
                all_acceptance.append(np.stack(accepted_rows, axis=1))
                del prepared, observation, path, scores, initial

            output = root / f'context_{context}.npz'
            np.savez_compressed(output, scores=np.concatenate(all_scores),
                                observed=np.concatenate(all_observed),
                                action_space_weights=np.concatenate(all_rewards),
                                acceptance=np.concatenate(all_acceptance))
            row = dict(context=context, paths=plan['paths_per_context'],
                       retained_paths=plan['retained_paths'], score_seconds=score_seconds,
                       sweep_seconds=sweep_seconds, score_parity_relative_error=parity_error,
                       file=output.name, sha256=sha(output))
            metadata.append(row)
            (root / f'context_{context}.json').write_text(json.dumps(row, indent=2) + '\n')
            progress('context_complete', **row)
        if not torch.equal(W, W_before) or not torch.equal(bias, bias_before):
            raise ValueError('Readout weights changed')
        result = dict(complete=True, elapsed_seconds=time.monotonic() - start,
                      plan_sha256=sha(pp), transition_calls=calls, contexts=metadata,
                      peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                      optimizer_steps=0, new_environment_episodes=0,
                      scope='Readout-score variance only; not full-actor or task-return variance')
        (root / 'collection_complete.json').write_text(json.dumps(result, indent=2) + '\n')
        progress('complete')
    except Exception as exc:
        progress('failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if hook is not None:
            hook.remove()


if __name__ == '__main__':
    main()
