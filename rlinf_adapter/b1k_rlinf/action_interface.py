"""R1Pro execution semantics, separate from sampled policy actions and density."""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ExecutionActionInterface:
    """Align near-open gripper commands with the controller's open endpoint.

    R1Pro uses 23 environment actions; indices 14 and 22 are normalized
    gripper commands. The assisted-grasp controller distinguishes its exact
    fully-open target from a target just below it. This opt-in mapping changes
    only [1 - tolerance, 1), after policy sampling and action unnormalization.
    It does not change the raw samples used for PPO likelihood computation.
    """

    gripper_open_tolerance: float = 0.0

    def __post_init__(self):
        value = self.gripper_open_tolerance
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("gripper_open_tolerance must be finite and in [0, 1)")

    def __call__(self, actions):
        if not isinstance(actions, (torch.Tensor, np.ndarray)):
            raise TypeError("Execution actions must be a NumPy array or torch tensor")
        if actions.ndim < 1 or actions.shape[-1] != 23:
            raise ValueError("R1Pro execution actions must have 23 values in the last dimension")
        is_torch = isinstance(actions, torch.Tensor)
        if is_torch:
            if not actions.is_floating_point() or not bool(torch.isfinite(actions).all()):
                raise ValueError("Execution actions must be finite floating-point values")
        elif not np.issubdtype(actions.dtype, np.floating) or not np.isfinite(actions).all():
            raise ValueError("Execution actions must be finite floating-point values")
        tolerance = self.gripper_open_tolerance
        if tolerance == 0:
            return actions
        result = actions.clone() if is_torch else actions.copy()
        for index in (14, 22):
            value = actions[..., index]
            mask = (value >= 1.0 - tolerance) & (value < 1.0)
            if is_torch:
                result[..., index] = torch.where(mask, torch.ones_like(value), value)
            else:
                result[..., index] = np.where(mask, np.ones_like(value), value)
        return result

    def contract(self):
        return {
            "version": "r1pro_gripper_open_v1",
            "gripper_open_tolerance": self.gripper_open_tolerance,
            "gripper_indices": [14, 22],
            "stage": "after action unnormalization; before environment execution",
            "raw_policy_actions_preserved": True,
        }
