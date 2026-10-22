"""Training primitives for actual uninterrupted before/after policy execution."""
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from core import inner_sgd
from native import NativeObjective
from streaming_gradient import compose_meta_gradient


def map_tensors(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {key:map_tensors(item,fn) for key,item in value.items()}
    if isinstance(value, list):
        return [map_tensors(item,fn) for item in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(item,fn) for item in value)
    return value


class FunctionalActor(nn.Module):
    """Use the same parameter names as NativeObjective: network.*."""
    def __init__(self, network):
        super().__init__(); self.network=network

    def forward(self, raw, generator):
        env={"states":raw['state'][None],"main_images":raw['images'][0][None],
             "wrist_images":np.stack(raw['images'][1:])[None],
             "task_descriptions":[raw['prompt']]}
        actions, result=self.network.predict_action_batch(env,mode='train',
            compute_values=False,rng=generator)
        replay=result['forward_inputs']
        length=int(replay['subtask_lengths'][0])
        text=self.network.joint_tokenizer.processor.decode(replay['subtask_tokens'][0,:length].tolist())
        return actions[0].detach().cpu().numpy(), text, replay


@dataclass
class Episode:
    decisions: list
    success: bool
    score: float
    control_steps: int
    cut_requested: int
    cut_executed: int | None


def collect(simulator, policy_step, *, max_steps, cut_step, before_is_updated,
            record_replay=True):
    """policy_step(updated: bool, raw_observation) returns action, text, replay.

    Simulator lifecycle belongs to the caller. This routine never resets it.
    A cut beyond termination is retained as an unswitched episode.
    """
    if max_steps<=0 or cut_step<0:
        raise ValueError('Invalid rollout horizon or cut')
    decisions=[];actual_cut=None
    while True:
        current=simulator.current
        controls=int(current['status']['steps'])
        if current.get('done',False) or controls>=max_steps:
            break
        crossed=controls>=cut_step
        if crossed and actual_cut is None:
            actual_cut=controls
        updated=not before_is_updated if crossed else before_is_updated
        actions,text,replay=policy_step(updated,current['observation'])
        actions=np.asarray(actions)
        if actions.ndim!=2 or actions.shape[1]!=23 or not np.isfinite(actions).all():
            raise ValueError('Invalid native action chunk')
        take=min(len(actions),max_steps-controls)
        if take<=0:
            raise ValueError('Empty action chunk')
        simulator.step(actions[:take],text)
        executed=int(simulator.current['status']['steps'])-controls
        if executed<=0 or executed>take:
            raise ValueError('Simulator did not advance')
        if record_replay:
            decisions.append(dict(updated=updated,control_start=controls,executed=executed,
                replay=map_tensors(replay,lambda t:t.detach().cpu().clone())))
    status=simulator.current['status']
    return Episode(decisions,bool(status['success']),float(status['score']),
                   int(status['steps']),cut_step,actual_cut)


def native_policy_step(network, initial, adapted, generator, *, bf16=True):
    if not network.rl_cfg.joint_logprob or network.rl_cfg.noise_method!='flow_sde':
        raise ValueError('Collection requires the full stochastic path policy')
    actor=FunctionalActor(network)
    versions=({n:p.detach() for n,p in initial.items()},
              {n:p.detach() for n,p in adapted.items()})
    def step(updated, observation):
        context=torch.autocast('cuda',dtype=torch.bfloat16) if bf16 else nullcontext()
        with torch.no_grad(),context:
            return functional_call(actor,versions[int(updated)],(observation,generator))
    return step


def batch_meta_gradient(network, params, demo_loss, episodes, episode_weights,
                        baselines, *, adapted_snapshot, inner_lr, bf16=True, inner_update=None):
    """Exact first-order outer score gradient including the inner-update VJP.

    Episode weights already contain mode probability / number of episodes of
    that mode. Baselines must be frozen before collection and must not use the
    episode's actions or outcome. This function does not fit a critic.
    """
    if not(len(episodes)==len(episode_weights)==len(baselines)) or not episodes:
        raise ValueError('Need aligned nonempty episode data')
    device=next(iter(params.values())).device
    objective=NativeObjective(network)
    if tuple(adapted_snapshot)!=tuple(params):
        raise ValueError('Collection snapshot parameter names differ')
    # Use exactly the numerical weights that generated the data, without
    # retaining an inner-update graph over all trajectory decisions.
    old_leaves={n:p.detach().requires_grad_() for n,p in params.items()}
    new_leaves={n:p.detach().requires_grad_() for n,p in adapted_snapshot.items()}
    accumulated=[{n:torch.zeros_like(p) for n,p in params.items()} for _ in range(2)]
    records=0
    for episode,weight,baseline in zip(episodes,episode_weights,baselines):
        multiplier=-float(weight)*(float(episode.success)-float(baseline))
        for decision in episode.decisions:
            route=int(decision['updated'])
            leaves=new_leaves if route else old_leaves
            replay=map_tensors(decision['replay'],lambda t:t.to(device))
            observation=network._observation_from_forward_inputs(replay)
            context=torch.autocast('cuda',dtype=torch.bfloat16) if bf16 else nullcontext()
            with context:
                logp=functional_call(objective,leaves,('path_logp',observation),
                                    {'chain':replay['chains'].detach()}).sum()
                loss=logp*multiplier
            gradients=torch.autograd.grad(loss,tuple(leaves.values()))
            for (name,_),gradient in zip(leaves.items(),gradients):
                if not torch.isfinite(gradient).all():
                    raise ValueError('Nonfinite per-decision score gradient')
                accumulated[route][name].add_(gradient.detach())
            records+=1
            del gradients,loss,logp,replay,observation
    # This backward is through the actual deterministic FM update, not an
    # identity/first-order approximation to adaptation.
    adapted=(inner_sgd(params,demo_loss,lr=inner_lr) if inner_update is None
             else inner_update(params,demo_loss))
    if not all(torch.equal(adapted[n].detach(),adapted_snapshot[n]) for n in params):
        raise ValueError('Recomputed demonstration update differs from collection; check noise/time or parameter mutation')
    result=compose_meta_gradient(params,adapted,accumulated[0],accumulated[1])
    return result,dict(decisions=records,episodes=len(episodes),
                      success_count=sum(ep.success for ep in episodes))
