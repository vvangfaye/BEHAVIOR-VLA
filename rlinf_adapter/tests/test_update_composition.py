"""Verify the proposed training primitive on actual, reduced-width Pi0 modules."""
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import torch
import pytest
from torch import nn
from torch.func import functional_call
from torch.nn.utils import parametrize

from test_model import model, observation, pack_replay

ROOT = Path(__file__).resolve().parents[2] / "analysis/update_composition_20261022"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LowRankOffset(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.a = nn.Parameter(torch.randn(shape[0], 2) * .01)
        self.b = nn.Parameter(torch.randn(2, shape[1]) * .01)

    def forward(self, weight):
        return weight + self.a @ self.b


def test_native_differentiable_update(model):
    torch.manual_seed(11)
    for p in model.parameters():
        p.requires_grad_(False)
    parametrize.register_parametrization(model.action_out_proj, "weight",
                                         LowRankOffset(model.action_out_proj.weight.shape))
    model.rl_cfg = dataclasses.replace(model.rl_cfg, joint_logprob=True)
    wrapper = load("native").NativeObjective(model)
    core = load("core")
    params = {name: p for name, p in wrapper.named_parameters() if p.requires_grad}
    obs = observation(model)
    actions = torch.randn(2, model.action_horizon, model.model_action_dim) * .2
    noise = torch.randn_like(actions)
    time = torch.tensor([.25, .75])

    def loss(q):
        return functional_call(wrapper, q, ("fm", obs),
                               {"actions": actions, "noise": noise, "time": time})
    adapted = core.inner_sgd(params, loss, lr=.05)
    # Sample actual all-step Flow-SDE paths with the adapted native weights.
    class Sampler(nn.Module):
        def __init__(self, network):
            super().__init__(); self.network = network
        def forward(self):
            return self.network._predict_train(obs, noise=None,
                        rng=torch.Generator().manual_seed(12), compute_values=False)
    sampler_params = {k: v.detach() for k, v in adapted.items()}
    with torch.no_grad():
        _, rollout = functional_call(Sampler(model), sampler_params, ())
    chain = rollout["forward_inputs"]["chains"].detach()
    elements = functional_call(wrapper, adapted, ("path_logp_elements", obs), {"chain": chain})
    expected_prefix = elements[:, :model.action_chunk, :model.action_env_dim] / (model.num_steps + 1)
    torch.testing.assert_close(expected_prefix, rollout["prev_logprobs"], atol=2e-5, rtol=2e-5)
    def outer(q):
        updated = core.inner_sgd(q, loss, lr=.05)
        logp_old = functional_call(wrapper, q, ("path_logp", obs), {"chain": chain})[:, None]
        logp_new = functional_call(wrapper, updated, ("path_logp", obs), {"chain": chain})[:, None]
        # Numerical derivative check of a fixed-data mixed-policy surrogate.
        return core.on_policy_loss(logp_old, logp_new,
                    torch.tensor([[False], [True]]), torch.ones(2, 1, dtype=torch.bool),
                    torch.tensor([1., -.4]))
    result = outer(params)
    grads = torch.autograd.grad(result, tuple(params.values()))
    assert all(torch.isfinite(g).all() for g in grads)
    direction = [g / torch.sqrt(sum(h.square().sum() for h in grads)) for g in grads]
    derivative = sum((g*d).sum() for g, d in zip(grads, direction))
    # Float32 native modules require a moderate central-difference step.
    eps = .005
    plus = {n: (p.detach()+eps*d).requires_grad_() for (n,p),d in zip(params.items(),direction)}
    minus = {n: (p.detach()-eps*d).requires_grad_() for (n,p),d in zip(params.items(),direction)}
    fd = (outer(plus)-outer(minus))/(2*eps)
    relative_error = float((derivative-fd).abs() / derivative.abs().clamp_min(1e-6))
    assert relative_error < .05
    report = {"passed": True, "trainable_parameters": sum(p.numel() for p in params.values()),
              "sampling_replay_density_parity": True,
              "inner_loss_before": float(loss(params).detach()),
              "inner_loss_after": float(loss(adapted).detach()),
              "gradient_norm": float(torch.sqrt(sum(g.square().sum() for g in grads))),
              "finite_difference_relative_error": relative_error,
              "scope": "Native reduced-width Gemma/Flow-SDE plus rank-2 output LoRA; no real checkpoint or environment"}
    (ROOT / "native_check.json").write_text(json.dumps(report, indent=2)+"\n")


def test_all_attention_lora_preserves_initial_policy_and_meta_gradient(model):
    # The randomly initialized fixture has all-zero adaRMS residual gates,
    # making attention provably irrelevant. Activate them before either side
    # of the comparison; this does not modify the real checkpoint runner.
    with torch.no_grad():
        for layer in model.llm.layers:
            for norm in (layer.pre_attention_norms[1], layer.pre_ffw_norms[1]):
                norm.ada_modulation.bias[-norm.dim:].fill_(.2)
    obs = observation(model)
    model.rl_cfg = dataclasses.replace(model.rl_cfg, joint_logprob=True)
    with torch.no_grad():
        before, _ = model._predict_train(obs, noise=None,
            rng=torch.Generator().manual_seed(51), compute_values=False)
    installed = load("lora").install_action_lora(model, rank=4)
    with torch.no_grad():
        after, _ = model._predict_train(obs, noise=None,
            rng=torch.Generator().manual_seed(51), compute_values=False)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert len(installed) == 8
    wrapper = load("native").NativeObjective(model)
    params = {n:p for n,p in wrapper.named_parameters() if p.requires_grad}
    actions = torch.zeros(2,32,32)
    noise = torch.randn_like(actions)
    times = torch.tensor([.25,.75])
    def loss(q):
        return functional_call(wrapper, q, ("fm",obs),
            dict(actions=actions,noise=noise,time=times))
    adapted = load("core").inner_sgd(params,loss,lr=.01)
    outer = loss(adapted)
    gradients = torch.autograd.grad(outer,tuple(params.values()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.square().sum() for g in gradients) > 0


@pytest.mark.parametrize('update_kind',['sgd','adam'])
def test_native_streaming_equals_complete_meta_gradient(model,update_kind):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parametrize.register_parametrization(model.action_out_proj,'weight',
                                         LowRankOffset(model.action_out_proj.weight.shape))
    model.rl_cfg=dataclasses.replace(model.rl_cfg,joint_logprob=True)
    sys.path.insert(0,str(ROOT))
    try:
        runtime=load('closed_loop')
    finally:
        sys.path.remove(str(ROOT))
    wrapper=load('native').NativeObjective(model)
    params={n:p for n,p in wrapper.named_parameters() if p.requires_grad}
    obs=observation(model)
    actions=torch.zeros(2,32,32);noise=torch.randn_like(actions);times=torch.tensor([.25,.75])
    def demo_loss(q):
        return functional_call(wrapper,q,('fm',obs),dict(actions=actions,noise=noise,time=times))
    with torch.no_grad():
        _,result=model._predict_train(obs,noise=None,rng=torch.Generator().manual_seed(61),compute_values=False)
    replay=pack_replay(obs,result)
    episode_old=runtime.Episode([dict(updated=False,replay=replay)],True,1.,16,99,None)
    episode_new=runtime.Episode([dict(updated=True,replay=replay)],False,0.,16,0,0)
    core=load('core')
    def update(q,loss):
        if update_kind=='adam':return core.inner_adam_first_step(q,loss,lr=.001,epsilon=1e-6)
        return core.inner_sgd(q,loss,lr=.05)
    adapted=update(params,demo_loss)
    logp_old=functional_call(wrapper,params,('path_logp',obs),dict(chain=replay['chains'])).sum()
    logp_new=functional_call(wrapper,adapted,('path_logp',obs),dict(chain=replay['chains'])).sum()
    reference=torch.autograd.grad(-.5*.75*logp_old+.5*.25*logp_new,tuple(params.values()))
    streamed,_=runtime.batch_meta_gradient(model,params,demo_loss,[episode_old,episode_new],
            [.5,.5],[.25,.25],adapted_snapshot={n:p.detach() for n,p in adapted.items()},
            inner_lr=.05,bf16=False,inner_update=update)
    for (name,_),expected in zip(params.items(),reference):
        torch.testing.assert_close(streamed[name],expected,rtol=1e-5,atol=1e-5)
