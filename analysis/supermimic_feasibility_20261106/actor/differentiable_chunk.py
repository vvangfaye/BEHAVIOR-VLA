"""Differentiate a factual B1K action chunk, keeping vision/language fixed.

This is not an imagined observation adapter or a replacement for PPO. The
default law reproduces the current single-noisy-step Flow-SDE behavior policy.
``sampling_law='flow_ode'`` deliberately uses the deterministic evaluation law.
Both retain the entire denoising graph for the action expert. A world model
must consume *execution_actions*, not normalized actions or the PPO chain.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import Callable, Iterable, Mapping

import torch
from torch import nn
from rlinf.models.embodiment.openpi.sampling import rl_sampler


@dataclass
class ChunkSample:
    model_actions: torch.Tensor
    policy_actions: torch.Tensor
    execution_actions: torch.Tensor
    prefix_features: torch.Tensor
    sampling_law: str
    denoise_index: int | None
    initial_noise: torch.Tensor
    step_noises: torch.Tensor
    chains: torch.Tensor


def sample_chunk_with_grad(
    model: nn.Module,
    forward_inputs: Mapping[str, torch.Tensor],
    *,
    sampling_law: str = "flow_sde",
    noise: torch.Tensor | None = None,
    step_noises: torch.Tensor | None = None,
    denoise_index: int | None = None,
    generator: torch.Generator | None = None,
) -> ChunkSample:
    """Sample one action chunk conditioned on exact recorded observations.

    Inputs use JointPi0RL's recorded ``forward_inputs`` format. Existing
    ``chains`` are not treated as policy outputs: a new differentiable chain
    is computed. Supply ``noise`` and ``step_noises`` to reproduce a draw;
    the index uses Python random just like upstream unless explicitly given.

    The prefix is recomputed once under no_grad using these factual pixels,
    state tokens, and subtask tokens. It is constant only for this chunk.
    No subtask generation or image/state conversion through NumPy is run.
    """
    if not torch.is_grad_enabled():
        raise RuntimeError("The differentiable chunk must run with gradients enabled")
    if sampling_law not in {"flow_sde", "flow_ode"}:
        raise ValueError("sampling_law must be flow_sde (behavior) or flow_ode (evaluation)")
    if not model.rl_cfg.train_expert_only:
        raise ValueError("This prototype requires the existing frozen vision/language policy")
    if sampling_law == "flow_sde" and (
        model.rl_cfg.noise_method != "flow_sde" or model.rl_cfg.joint_logprob
    ):
        raise ValueError("Expected the deployed single-noisy-step Flow-SDE configuration")
    if sampling_law == "flow_ode" and denoise_index is not None:
        raise ValueError("The evaluation ODE law has no stochastic denoise index")

    # This intentionally bypasses predict_action_batch's no_grad decorator and
    # its NumPy/token generation path, but not the established replay contract.
    with torch.no_grad():
        replay = {key: value.to(model.device) if torch.is_tensor(value) else value
                  for key, value in forward_inputs.items()}
        observation = model._observation_from_forward_inputs(replay)
        prefix_out, prefix_mask, kv_cache = model._build_prefix_cache_for_actor(observation)
        mask = prefix_mask.to(prefix_out.dtype).unsqueeze(-1)
        prefix_features = (prefix_out * mask).sum(1) / mask.sum(1).clamp(min=1)

    device = observation.state.device
    batch = observation.state.shape[0]
    steps = model.num_steps
    shape = (batch, model.action_horizon, model.model_action_dim)
    if steps < 2:
        raise ValueError("At least two denoising steps are required")
    if noise is None:
        noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    if tuple(noise.shape) != shape:
        raise ValueError(f"Expected initial noise shape {shape}, got {tuple(noise.shape)}")
    x_t = noise.detach().to(device=device, dtype=torch.float32)
    initial_noise = x_t.clone()
    if step_noises is not None and tuple(step_noises.shape) != (steps, *shape):
        raise ValueError(f"Expected step noise shape {(steps, *shape)}")
    if sampling_law == "flow_sde":
        last = steps - 2 if model.rl_cfg.ignore_last else steps - 1
        if denoise_index is None:
            denoise_index = random.randint(0, last)
        if type(denoise_index) is not int or not 0 <= denoise_index <= last:
            raise ValueError(f"denoise_index must lie in [0, {last}]")

    timesteps = rl_sampler.get_timesteps(steps, device)
    chains = [x_t.detach()]
    used_noises = []
    for index in range(steps):
        # ODE follows the evaluation integrator exactly; Flow-SDE follows the
        # training sampler's time values and mean/variance computation exactly.
        time = 1.0 - index / steps if sampling_law == "flow_ode" else float(timesteps[index])
        t = torch.full((batch,), time, device=device, dtype=torch.float32)
        suffix = model.run_suffix(observation, x_t, t, kv_cache, prefix_mask)
        velocity = model.velocity_from_suffix(suffix).to(torch.float32)
        if sampling_law == "flow_ode":
            x_t = x_t + (-1.0 / steps) * velocity
            epsilon = torch.zeros(shape, device=device, dtype=torch.float32)
        else:
            method = "flow_sde" if index == denoise_index else "flow_ode"
            indices = torch.full((batch,), index, dtype=torch.long, device=device)
            mean, std = rl_sampler.sample_mean_var(
                x_t, velocity, indices, noise_method=method,
                noise_level=model.rl_cfg.noise_level, num_steps=steps,
            )
            epsilon = (
                torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
                if step_noises is None else step_noises[index].detach().to(device=device, dtype=torch.float32)
            )
            x_t = mean + epsilon * std
        used_noises.append(epsilon.detach())
        chains.append(x_t.detach())

    policy_actions = model.decode_actions(x_t, observation.state)
    execution_actions = model.execution_interface(policy_actions)
    return ChunkSample(
        model_actions=x_t, policy_actions=policy_actions,
        execution_actions=execution_actions, prefix_features=prefix_features.detach(),
        sampling_law=sampling_law, denoise_index=denoise_index,
        initial_noise=initial_noise, step_noises=torch.stack(used_noises),
        chains=torch.stack(chains, dim=1),
    )


@contextmanager
def _frozen_parameters(modules: Iterable[nn.Module]):
    parameters = {id(p): p for module in modules for p in module.parameters()}
    original = [(p, p.requires_grad) for p in parameters.values()]
    try:
        for parameter, _ in original:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in original:
            parameter.requires_grad_(flag)


def model_actor_loss(
    sample: ChunkSample,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    frozen_modules: Iterable[nn.Module] = (),
    reference_actions: torch.Tensor | None = None,
    trust_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Maximize a differentiable short model return, optionally with a trust loss.

    ``score_fn(execution_actions)`` returns one score per batch item. Its closure
    can compute model reward + gamma * continuation * terminal value, with the
    initial state and factual prefix detached. List its standalone world model
    and critic in ``frozen_modules`` to avoid accumulating parameter gradients;
    freezing does not use no_grad and therefore preserves d(score)/d(action).

    This MSE trust term is a local action constraint, not a PPO/KL guarantee.
    No synthetic transition is sent to the real PPO rollout buffer.
    """
    if trust_coef < 0 or not torch.isfinite(torch.tensor(trust_coef)):
        raise ValueError("trust_coef must be finite and nonnegative")
    with _frozen_parameters(frozen_modules):
        score = score_fn(sample.execution_actions)
    batch = sample.execution_actions.shape[0]
    if tuple(score.shape) not in {(batch,), (batch, 1)}:
        raise ValueError("score_fn must return [batch] or [batch, 1]")
    if not torch.isfinite(score).all() or not score.requires_grad:
        raise ValueError("The model score must be finite and retain its action gradient")
    trust = score.new_zeros(())
    if reference_actions is not None:
        if reference_actions.shape != sample.execution_actions.shape:
            raise ValueError("Reference and sampled execution actions must have the same shape")
        trust = (sample.execution_actions - reference_actions.detach().to(sample.execution_actions)).square().mean()
    elif trust_coef:
        raise ValueError("A positive trust coefficient requires reference_actions")
    loss = -score.mean() + trust_coef * trust
    return loss, {"model_score": score.mean().detach(), "action_trust_mse": trust.detach()}
