# Execution-Space Policy Optimization (ESPO), regression form, v1 (local addition, 2026-11-04; not upstream RLinf).
#
# Per PPO iteration (see EmbodiedFSDPActor._espo_prepare / train_micro_batch):
#   1. z_i  = pooled frozen-VLM prefix of recorded state i (the value head's input), u_i = executed action chunk
#            (normalized model space, first action_chunk steps x action_env_dim) of rollout sample i.
#   2. Fit a small return predictor Q(z, u) on (z_i, u_i, return_i); phi(z, u) = [hidden features, Q output].
#   3. With M fresh noise draws eps_ij per state, run the native K-step ODE sampler of the CURRENT policy:
#            Phi0_i = mean_j phi(z_i, P F_theta(h_i, eps_ij)),   b_i = A_i * (phi(z_i, u_i) - Phi0_i),
#            Y_i = Phi0_i + eta * b_i                      (desired change of executed-action features, from real returns)
#   4. Update epochs: minimize  sum_i w_i ||mean_j phi(z_i, P F_theta(h_i, eps_ij)) - Y_i||^2  through the sampler
#      (same eps_ij), plus the usual critic loss. w_i follows RLinf's per-trajectory normalization by default.
# Validated first on a toy flow policy with exact gradients: analysis/espo_toy_20261104 (RESULTS*.md).
from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import nn

from rlinf.models.embodiment.openpi.sampling import rl_sampler


class ReturnPredictor(nn.Module):
    def __init__(self, dz: int, du: int, width: int = 128, feat: int = 64):
        super().__init__()
        self.zproj = nn.Sequential(nn.Linear(dz, width), nn.Tanh())
        self.body = nn.Sequential(
            nn.Linear(width + du, width), nn.Tanh(), nn.Linear(width, feat), nn.Tanh()
        )
        self.head = nn.Linear(feat, 1)

    def features(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        f = self.body(torch.cat([self.zproj(z), u], dim=-1))
        return torch.cat([f, self.head(f)], dim=-1)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(torch.cat([self.zproj(z), u], dim=-1))).squeeze(-1)


def fit_return_predictor(
    z: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int = 400,
    lr: float = 1e-3,
    seed: int = 0,
    holdout: float = 0.1,
) -> tuple[ReturnPredictor, dict[str, float]]:
    """Full-batch Adam on standardized targets; returns the frozen predictor and fit diagnostics."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = z.shape[0]
    perm = torch.randperm(n, generator=g).to(z.device)
    n_val = max(1, int(holdout * n))
    val, tr = perm[:n_val], perm[n_val:]
    mu, sd = y[tr].mean(), y[tr].std().clamp_min(1e-6)
    yt = (y - mu) / sd
    torch.manual_seed(seed)
    q = ReturnPredictor(z.shape[1], u.shape[1]).to(z.device, torch.float32)
    opt = torch.optim.Adam(q.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((q(z[tr], u[tr]) - yt[tr]) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        def r2(idx):
            if idx.numel() < 2:
                return float("nan")
            return float(1.0 - ((q(z[idx], u[idx]) - yt[idx]) ** 2).mean() / yt[idx].var().clamp_min(1e-8))

        stats = {"q_train_r2": r2(tr), "q_val_r2": r2(val)}
    for p in q.parameters():
        p.requires_grad_(False)
    q.eval()
    return q, stats


class ActionMomentFeatures(nn.Module):
    """Fallback features phi(z, u) = [u, u^2] (generic moment projection, the toy's 'moments' variant)."""

    def features(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat([u, u * u], dim=-1)


def pooled_prefix(prefix_out: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean of the prefix hidden states (same pooling as value_from_prefix(mode='mean_token'))."""
    m = prefix_mask.to(prefix_out.dtype).unsqueeze(-1)
    return ((prefix_out * m).sum(1) / m.sum(1).clamp(min=1.0)).float()


def _repeat(x: Any, m: int) -> Any:
    if isinstance(x, torch.Tensor):
        return x.repeat_interleave(m, dim=0)
    if isinstance(x, dict):
        return {k: _repeat(v, m) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_repeat(v, m) for v in x)
    return x


def expand_observation(observation: Any, m: int) -> Any:
    """Repeat every tensor of an Observation dataclass m times along the batch dim (interleaved)."""
    if m == 1:
        return observation
    fields = {f.name: _repeat(getattr(observation, f.name), m) for f in dataclasses.fields(observation)}
    return dataclasses.replace(observation, **fields)


def ode_executed_actions(model, observation, kv_cache, prefix_mask, noise: torch.Tensor) -> torch.Tensor:
    """Native K-step flow ODE sampler of ``model`` (differentiable in the action expert) from initial noise
    [B, action_horizon, model_action_dim]. Returns the executed part, flattened: [B, action_chunk * action_env_dim]."""
    B = noise.shape[0]
    num_steps = model.num_steps
    x_t = noise.to(torch.float32)
    idx_step = torch.empty(B, device=noise.device, dtype=torch.long)
    timesteps = rl_sampler.get_timesteps(num_steps, noise.device)
    for idx in range(num_steps):
        t_tensor = torch.full((B,), float(timesteps[idx].item()), device=noise.device, dtype=torch.float32)
        suffix_act = model.run_suffix(observation, x_t, t_tensor, kv_cache, prefix_mask)
        v_t = model.velocity_from_suffix(suffix_act).to(torch.float32)
        idx_step.fill_(idx)
        x_t, _ = rl_sampler.sample_mean_var(
            x_t, v_t, idx_step, noise_method="flow_ode", noise_level=0.0, num_steps=num_steps
        )
    return x_t[:, : model.action_chunk, : model.action_env_dim].reshape(B, -1)


def executed_from_chains(model, chains: torch.Tensor) -> torch.Tensor:
    """Executed action chunk of the recorded rollout sample (final chain element), flattened."""
    x0 = chains[:, -1].to(torch.float32)
    return x0[:, : model.action_chunk, : model.action_env_dim].reshape(x0.shape[0], -1)
