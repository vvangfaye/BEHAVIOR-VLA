"""Differentiable action score backed by the fitted B1K world-model probe.

Only full 16-command chunks are supported. Normalization and PCA use torch
buffers, so execution_actions -> score remains differentiable. A frozen model
is not evaluated under no_grad: gradients through its inputs are intentional.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
from typing import Callable

import torch
from torch import nn


@lru_cache(maxsize=1)
def world_model_class():
    """Use the same architecture definition as the training checkpoint."""
    source = Path(__file__).resolve().parents[1] / "model" / "train_world_model.py"
    spec = importlib.util.spec_from_file_location("_b1k_fitted_world_model", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load world-model definition: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WorldModel


class FrozenWorldModelScore(nn.Module):
    """Load full.pt and score physical action commands, preserving action grads.

    Inputs are exported checkpoint-normalized proprioception, current Boolean
    goal bits, raw 576-dimensional [0,1] image-pool features, and physical
    execution commands shaped [B,16,23]. This adds the probe's training-only
    standardization internally; it does not normalize robot commands twice.

    Optional value_callback receives a dictionary from predict(). A module
    callback must already be frozen. Function callbacks must contain no
    trainable parameters or be covered by the caller's frozen_modules context.
    No termination model was trained, so continuation must be supplied explicitly
    when using a value callback; observed future done is not a valid actor input.
    """

    _BUFFER_NAMES = ("state_mean", "state_scale", "action_mean", "action_scale",
                     "reward_mean", "reward_scale", "visual_mean", "components", "visual_scale")

    def __init__(self, checkpoint: dict):
        super().__init__()
        if checkpoint.get("schema") != "b1k_small_world_model_probe_v1":
            raise ValueError("Unsupported fitted world-model checkpoint schema")
        config = dict(checkpoint["config"])
        if checkpoint.get("name") != "full" or not config.get("use_action"):
            raise ValueError("Actor scoring requires the action-conditioned full checkpoint")
        self.goal_dim = int(config["goals"])
        self.visual_dim = int(config["visual_dim"])
        self.use_visual = bool(config["use_visual"])
        for name in self._BUFFER_NAMES:
            value = checkpoint["transform"][name]
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise ValueError(f"Expected floating torch transform buffer: {name}")
            self.register_buffer(name, value.detach().clone().float())
        expected = {"state_mean": (23,), "state_scale": (23,), "action_mean": (368,),
                    "action_scale": (368,), "reward_mean": (1,), "reward_scale": (1,),
                    "visual_mean": (576,), "components": (576, self.visual_dim),
                    "visual_scale": (self.visual_dim,)}
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name} shape or values")
            if name.endswith("scale") and not (value > 0).all():
                raise ValueError(f"Expected positive {name}")
        model_type = world_model_class()
        self.models = nn.ModuleList()
        for parameters in checkpoint["model_states"]:
            model = model_type(**config)
            model.load_state_dict(parameters, strict=True)
            model.requires_grad_(False)
            model.eval()
            self.models.append(model)
        if not self.models:
            raise ValueError("Checkpoint ensemble is empty")
        self.data_sha256 = checkpoint.get("data_sha256")
        self.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device="cpu"):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(checkpoint).to(device)

    def train(self, mode: bool = True):
        # This wrapper is always frozen, including if a parent actor calls train().
        return super().train(False)

    def _input(self, value: torch.Tensor, shape: tuple, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be a torch Tensor shaped {shape}")
        return value.to(device=self.state_mean.device, dtype=self.state_mean.dtype)

    def predict(self, obs_state: torch.Tensor, goals: torch.Tensor,
                visual_features: torch.Tensor | None, execution_actions: torch.Tensor,
                *, goal_valid: torch.Tensor | None = None,
                visual_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if execution_actions.ndim != 3:
            raise ValueError("execution_actions must be [batch,16,23]")
        batch_size = execution_actions.shape[0]
        actions = self._input(execution_actions, (batch_size, 16, 23), "execution_actions")
        raw_state = self._input(obs_state, (batch_size, 23), "obs_state")
        current_goals = self._input(goals, (batch_size, self.goal_dim), "goals")
        if goal_valid is None:
            goal_valid = torch.ones_like(current_goals)
        goal_mask = self._input(goal_valid, (batch_size, self.goal_dim), "goal_valid")
        if visual_valid is None:
            visual_valid = torch.full((batch_size,), visual_features is not None,
                                      device=self.state_mean.device, dtype=torch.bool)
        valid = self._input(visual_valid, (batch_size,), "visual_valid").bool()
        if visual_features is None:
            if valid.any():
                raise ValueError("Visual features missing despite valid mask")
            raw_visual = self.visual_mean.expand(batch_size, -1)
        else:
            raw_visual = self._input(visual_features, (batch_size, 576), "visual_features")
        if not torch.isfinite(actions).all() or not torch.isfinite(raw_state).all():
            raise ValueError("Nonfinite state or action")
        if not torch.isfinite(current_goals).all() or not torch.isfinite(goal_mask).all():
            raise ValueError("Nonfinite current goals or masks")
        if not torch.isfinite(raw_visual[valid]).all():
            raise ValueError("Nonfinite valid visual features")
        filled_visual = torch.where(valid[:, None], raw_visual, self.visual_mean)
        visual = ((filled_visual - self.visual_mean) @ self.components) / self.visual_scale
        normalized_state = (raw_state - self.state_mean) / self.state_scale
        normalized_actions = (actions.reshape(batch_size, -1) - self.action_mean) / self.action_scale
        inputs = {"state": normalized_state, "goals": current_goals, "goal_valid": goal_mask,
                  "visual": visual, "visual_valid": valid[:, None].to(raw_state.dtype),
                  "actions": normalized_actions, "action_mask": torch.ones((batch_size, 16),
                                       device=actions.device, dtype=actions.dtype)}
        members = [model(inputs) for model in self.models]
        next_state = torch.stack([normalized_state + m["state_delta"] for m in members]).mean(0)
        next_visual = torch.stack([visual + m["visual_delta"] for m in members]).mean(0)
        next_goals = torch.stack([m["goal_logits"].sigmoid() for m in members]).mean(0)
        reward = torch.stack([m["reward"] for m in members]).mean(0)
        return {"next_obs_state": next_state * self.state_scale + self.state_mean,
                "next_visual_features": (next_visual * self.visual_scale) @ self.components.T + self.visual_mean,
                "next_goals": next_goals,
                "raw_reward": (reward * self.reward_scale + self.reward_mean).squeeze(-1),
                "next_normalized_state": next_state, "next_projected_visual": next_visual}

    def forward(self, obs_state: torch.Tensor, goals: torch.Tensor,
                visual_features: torch.Tensor | None, execution_actions: torch.Tensor,
                *, goal_valid: torch.Tensor | None = None,
                visual_valid: torch.Tensor | None = None,
                value_callback: Callable[[dict[str, torch.Tensor]], torch.Tensor] | None = None,
                gamma: float = .99,
                continuation: float | torch.Tensor | None = None) -> torch.Tensor:
        predictions = self.predict(obs_state, goals, visual_features, execution_actions,
                                   goal_valid=goal_valid, visual_valid=visual_valid)
        score = predictions["raw_reward"]
        if value_callback is None:
            if continuation is not None:
                raise ValueError("continuation has no meaning without value_callback")
            return score
        if not 0 <= gamma <= 1 or continuation is None:
            raise ValueError("Value scoring needs gamma in [0,1] and explicit continuation")
        if isinstance(value_callback, nn.Module) and any(p.requires_grad for p in value_callback.parameters()):
            raise ValueError("value_callback module must be frozen before scoring")
        value = value_callback(predictions)
        if not isinstance(value, torch.Tensor):
            raise ValueError("value_callback must return a torch Tensor, without detaching its inputs")
        if value.shape == (len(score), 1):
            value = value[:, 0]
        if value.shape != score.shape or not torch.isfinite(value).all():
            raise ValueError("value_callback must return a finite [batch] value")
        survival = torch.as_tensor(continuation, dtype=score.dtype, device=score.device)
        if survival.shape == (len(score), 1):
            survival = survival[:, 0]
        if survival.ndim > 0 and survival.shape != score.shape:
            raise ValueError("continuation must be a scalar or [batch]")
        if not torch.isfinite(survival).all() or not ((survival >= 0) & (survival <= 1)).all():
            raise ValueError("continuation must be finite and in [0,1]")
        return score + gamma * survival * value
