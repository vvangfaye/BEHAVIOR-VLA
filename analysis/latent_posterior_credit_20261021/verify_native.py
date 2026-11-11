"""Bounded one-observation native SDE/posterior/backward feasibility check.

No optimizer step, simulator, download, or upload. Preserve all source files.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'analysis/relative_flow_20261021'))
from adapter import NativeAdapter
from vla_runtime import load_base
from vla_data import VlaPilotData
from rlinf.models.embodiment.openpi.sampling import rl_sampler
from native_transition import OpenPiGaussianTransition
from posterior import GaussianChainPosterior


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = HERE / 'native_feasibility'
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    if (root / 'result.json').exists():
        raise RuntimeError('A completed result exists; refusing silent rerun')
    start = time.monotonic()
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed: ' + path)
    calls = 0

    def progress(stage, **kwargs):
        payload = dict(stage=stage, pid=os.getpid(), observed_at=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.monotonic() - start, transition_calls=calls, **kwargs)
        (root / 'state.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload), flush=True)

    def deadline():
        if time.monotonic() - start > plan['max_seconds']:
            raise TimeoutError('Native feasibility wall limit')

    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        progress('loading')
        base = load_base(plan['checkpoint'], plan['tokenizer'], precision='fp32')
        adapter = NativeAdapter(base).cuda().eval()
        data = VlaPilotData(plan['data_directory'], base)
        observation, _, _ = data.batch([plan['context_index']])
        prepared = adapter.prepare(observation)
        trainable = [(name, p) for name, p in base.named_parameters() if p.requires_grad]
        for _, parameter in trainable:
            parameter.requires_grad_(False)
        callback = OpenPiGaussianTransition(adapter, prepared, rl_sampler.sample_mean_var,
                                            rl_sampler.get_timesteps, steps=plan['steps'],
                                            noise_level=plan['noise_level'])

        def transition(flat, step):
            nonlocal calls
            deadline()
            calls += 1
            return callback(flat, step)

        generator = torch.Generator(device='cuda').manual_seed(plan['seed'])
        dimension = callback.horizon * callback.dim
        mask = torch.zeros(callback.horizon, callback.dim, device='cuda', dtype=torch.bool)
        mask[:plan['executed_steps'], :plan['physical_dimensions']] = True
        mask = mask.flatten()
        engine = GaussianChainPosterior(transition, plan['steps'], mask)
        latent = []
        standard_deviations = []
        with torch.no_grad():
            x = torch.randn((1, dimension), device='cuda', generator=generator)
            for k in range(plan['steps']):
                latent.append(x.detach().clone())
                mean, std = transition(x, k)
                standard_deviations.append(float(std.min()))
                x = mean + std * torch.randn(x.shape, device=x.device, generator=generator)
            observed = x[:, mask].detach().clone()
            latent = torch.stack(latent, dim=1)
            original = latent.detach().clone()
            original_observed = observed.detach().clone()
            original_log_joint = engine.log_joint(latent, observed).detach()
        progress('sampled', latent_shape=list(latent.shape), observed_shape=list(observed.shape),
                 minimum_stds=standard_deviations, original_log_joint=float(original_log_joint))

        proposals = []
        for k in range(plan['steps']):
            incoming = 1.0 if k == 0 else standard_deviations[k - 1]
            epsilon = plan['step_scale'] * min(incoming, standard_deviations[k])
            before = latent[:, k].detach().clone()
            tick = time.monotonic()
            latent, accepted = engine.mala_step(latent, observed, k, epsilon, generator=generator)
            displacement = float((latent[:, k] - before).norm())
            proposals.append(dict(site=k, step_size=epsilon, accepted=bool(accepted.item()),
                                  displacement_l2=displacement, seconds=time.monotonic() - tick))
            progress('posterior_update', site=k, accepted=bool(accepted.item()),
                     displacement_l2=displacement)
        if not torch.equal(observed, original_observed):
            raise ValueError('Executed prefix mutated')
        for _, parameter in trainable:
            parameter.requires_grad_(True)
        base.zero_grad(set_to_none=True)
        # One full actor score backward, factor by factor to bound activation memory.
        score_values = []
        for k in range(plan['steps']):
            mean, std = transition(latent[:, k].detach(), k)
            if k == plan['steps'] - 1:
                from posterior import normal_logprob
                factor = normal_logprob(observed, mean[:, mask], std[:, mask]).sum()
            else:
                from posterior import normal_logprob
                factor = normal_logprob(latent[:, k + 1].detach(), mean, std).sum()
            if not torch.isfinite(factor):
                raise ValueError('Nonfinite log-density factor')
            factor.backward()
            score_values.append(float(factor.detach()))
            progress('actor_backward', factor=k)
        gradient_sq = 0.0
        grad_parameters = 0
        grad_elements = 0
        nonfinite = []
        for name, parameter in trainable:
            if parameter.grad is None:
                continue
            grad_parameters += 1
            grad_elements += parameter.numel()
            if not torch.isfinite(parameter.grad).all():
                nonfinite.append(name)
            gradient_sq += float(parameter.grad.double().square().sum())
        if nonfinite or not grad_elements or gradient_sq <= 0:
            raise ValueError('Invalid full action-expert gradient')
        result = dict(
            complete=True, scope='One native observation; no optimizer, simulator, task reward or success-rate check.',
            plan_sha256=sha(plan_path), original_context_index=plan['context_index'],
            elapsed_seconds=time.monotonic() - start, transition_calls=calls,
            observed_prefix_bitwise_preserved=torch.equal(observed, original_observed),
            latent_changed=not torch.equal(latent, original), proposals=proposals,
            score_factors=score_values, actor_parameters_with_gradient=grad_parameters,
            actor_elements_with_gradient=grad_elements, actor_gradient_norm=gradient_sq**0.5,
            nonfinite_gradient_parameters=nonfinite,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
            framework_cuda_device=torch.cuda.get_device_name(),
            new_environment_episodes=0, optimizer_steps=0,
            limits=['One sweep is not evidence of adequate mixing or lower gradient variance.',
                    'SDE action distribution differs from pure Euler checkpoint evaluation.',
                    'No closed-loop training benefit established.'])
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        progress('complete', result='result.json')
        base.zero_grad(set_to_none=True)
    except Exception as exc:
        progress('failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
