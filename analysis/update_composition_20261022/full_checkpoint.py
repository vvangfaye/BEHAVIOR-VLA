"""One bounded real-checkpoint meta-update; no environment or benefit claim."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from vla_runtime import load_base
from vla_data import VlaPilotData
from core import inner_sgd, inner_adam_first_step, on_policy_loss
from lora import install_action_lora
from native import NativeObjective


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest_frozen(module):
    h = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            h.update(name.encode())
            value = parameter.detach().contiguous().view(torch.uint8).flatten()
            for start in range(0, value.numel(), 8 * 1024**2):
                h.update(value[start:start + 8 * 1024**2].cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(plan['seed'])
    def event(stage, **info):
        value = dict(stage=stage, elapsed_seconds=time.monotonic()-started, **info)
        write(args.output / 'progress.json', value)
        print(json.dumps(value), flush=True)
    event('loading')
    base = load_base(plan['checkpoint'], plan['tokenizer'], precision=plan['precision'])
    base.rl_cfg = dataclasses.replace(base.rl_cfg, joint_logprob=True)
    installed = install_action_lora(base, rank=plan['rank'])
    wrapper = NativeObjective(base)
    params = {n: p for n, p in wrapper.named_parameters() if p.requires_grad}
    data = VlaPilotData(plan['data_directory'], base)
    # Predetermined existing examples from separate episodes; not selected by loss.
    indices = []
    for index in data.train_ids:
        if not indices or data.episode[index] != data.episode[indices[0]]:
            indices.append(int(index))
        if len(indices) == 2:
            break
    if len(indices) != 2:
        raise ValueError('Need two separate existing demonstration episodes')
    obs, target, _ = data.batch(indices)
    noise = torch.randn_like(target)
    fm_time = torch.tensor([.25, .75], device=target.device)
    original = {n: p.detach().clone() for n, p in params.items()}
    frozen_before = digest_frozen(wrapper)
    event('loaded', trainable_parameters=sum(p.numel() for p in params.values()),
          installed_modules=len(installed), peak_gpu_bytes=torch.cuda.max_memory_allocated())
    def demo_loss(q):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=plan['precision'] == 'bf16'):
            return functional_call(wrapper, q, ('fm', obs),
                                   dict(actions=target, noise=noise, time=fm_time))
    inner_before = demo_loss(params)
    noop_loss_error=None
    predicted_change=None
    if plan.get('check_noop',False):
        with torch.no_grad():
            copied={n:p.detach().clone() for n,p in params.items()}
            noop_loss_error=float((demo_loss(copied)-inner_before.detach()).abs())
        if noop_loss_error>1e-7:raise ValueError('No-op parameter replacement changed demonstration loss')
        inner_gradient=torch.autograd.grad(inner_before,tuple(params.values()))
    if plan.get('inner_optimizer','sgd')=='adam':
        adapted=inner_adam_first_step(params,demo_loss,lr=plan['inner_lr'],epsilon=plan['inner_epsilon'])
    else:
        adapted=inner_sgd(params, demo_loss, lr=plan['inner_lr'])
    inner_after = demo_loss(adapted)
    if plan.get('check_noop',False):
        predicted_change=float(sum((g.detach()*(adapted[n].detach()-params[n].detach())).sum()
                                   for n,g in zip(params,inner_gradient)))
    event('inner_update', before=float(inner_before.detach()), after=float(inner_after.detach()))
    class Sampler(nn.Module):
        def __init__(self, network):
            super().__init__(); self.network = network
        def forward(self):
            return self.network._predict_train(obs, noise=None,
                rng=torch.Generator(device=target.device).manual_seed(118), compute_values=False)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=plan['precision']=='bf16'):
        tuned_actions, rollout = functional_call(Sampler(base), {n: p.detach() for n,p in adapted.items()}, ())
        original_actions, _ = functional_call(Sampler(base), {n: p.detach() for n,p in params.items()}, ())
        if plan.get('check_noop',False):
            noop_actions,_=functional_call(Sampler(base),copied,())
            noop_action_error=float((noop_actions-original_actions).abs().max())
            if noop_action_error>1e-6:raise ValueError('No-op parameter replacement changed sampled actions')
        else:noop_action_error=None
    action_change=float((tuned_actions-original_actions).abs().max())
    chain = rollout['forward_inputs']['chains'].detach()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=plan['precision']=='bf16'):
        old_logp = functional_call(wrapper, params, ('path_logp', obs), dict(chain=chain))
        new_elements = functional_call(wrapper, adapted, ('path_logp_elements', obs), dict(chain=chain))
    expected = new_elements[:, :base.action_chunk, :base.action_env_dim] / (base.num_steps + 1)
    parity_error = float((expected.detach()-rollout['prev_logprobs']).abs().max())
    torch.testing.assert_close(expected.detach(), rollout['prev_logprobs'], atol=2e-3, rtol=2e-3)
    new_logp = new_elements.flatten(1).sum(-1)
    # Deliberately synthetic weights exercise both routes. They are NOT environment rewards.
    surrogate = on_policy_loss(old_logp[:,None], new_logp[:,None],
        torch.tensor([[False],[True]], device=target.device),
        torch.ones(2,1,device=target.device,dtype=torch.bool),
        torch.tensor([1.,-.4], device=target.device))
    grads = torch.autograd.grad(surrogate, tuple(params.values()))
    if not all(torch.isfinite(g).all() for g in grads):
        raise ValueError('Nonfinite outer gradient')
    norm = torch.sqrt(sum(g.square().sum() for g in grads))
    if not norm > 0:
        raise ValueError('Zero outer gradient')
    event('outer_gradient', norm=float(norm), parity_error=parity_error,
          peak_gpu_bytes=torch.cuda.max_memory_allocated())
    with torch.no_grad():
        for p, gradient in zip(params.values(), grads):
            p.add_(gradient, alpha=-plan['outer_lr']/max(float(norm),1.))
    changed = max(float((p-original[n]).abs().max()) for n,p in params.items())
    if not changed > 0:
        raise ValueError('Outer update changed no adapter weight')
    frozen_after = digest_frozen(wrapper)
    if frozen_before != frozen_after:
        raise ValueError('Frozen base weights changed')
    overlay = {n:p.detach().cpu() for n,p in params.items()}
    torch.save(overlay, args.output/'adapter.pt')
    saved = torch.load(args.output/'adapter.pt', map_location='cpu', weights_only=True)
    if not all(torch.equal(saved[n], value) for n,value in overlay.items()):
        raise ValueError('Adapter checkpoint round-trip failed')
    result = dict(passed=True, original_checkpoint=plan['checkpoint'],
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        actual_total_parameters=sum(p.numel() for p in wrapper.parameters()),
        trainable_parameters=sum(p.numel() for p in params.values()),
        rank=plan['rank'], modules=installed, indices=indices,
        episode_ids=[int(data.episode[i]) for i in indices],
        inner_loss_before=float(inner_before.detach()), inner_loss_after=float(inner_after.detach()),
        inner_optimizer=plan.get('inner_optimizer','sgd'),
        inner_update_loss_changed=bool(inner_before.detach()!=inner_after.detach()),
        inner_common_noise_action_max_change=action_change,
        adaptation_effective=bool(inner_after.detach()<inner_before.detach()) and action_change>0,
        precision=plan['precision'],inner_learning_rate=plan['inner_lr'],
        noop_loss_error=noop_loss_error,noop_action_error=noop_action_error,
        predicted_first_order_loss_change=predicted_change,
        outer_gradient_norm=float(norm), max_adapter_change=changed,
        sampling_replay_max_error=parity_error, frozen_weights_unchanged=True,
        checkpoint_round_trip=True, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        elapsed_seconds=time.monotonic()-started,
        scope='One real checkpoint update using existing expert-conditioned observations and synthetic outer weights; no environment, success benefit or sample-efficiency claim')
    if plan.get('require_effective_adaptation',False):
        result['passed']=result['adaptation_effective']
    write(args.output/'result.json', result)
    event('complete', passed=result['passed'], peak_gpu_bytes=result['peak_gpu_bytes'])


if __name__=='__main__':
    main()
