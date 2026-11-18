"""Direct LIBERO inference using RLinf's evaluation policy and transforms."""

import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

import rlinf
from rlinf.envs.action_utils import prepare_actions_for_libero
from rlinf.models import get_model


class Policy:
    def __init__(self, cfg_dir, model_dir, checkpoint_or_none, *, config_name):
        embodied_path = Path(rlinf.__file__).resolve().parent.parent / "examples/embodiment"
        os.environ.setdefault("EMBODIED_PATH", str(embodied_path))
        with initialize_config_dir(config_dir=str(Path(cfg_dir).resolve()), version_base="1.1"):
            cfg = compose(
                config_name=config_name,
                overrides=["rollout.model.num_action_chunks=10"],
            )
        model_cfg = cfg.rollout.model
        model_cfg.model_path = str(Path(model_dir).resolve())
        if (model_cfg.openpi.task, model_cfg.openpi.num_steps, model_cfg.openpi.action_chunk) != ("eval", 5, 10):
            raise ValueError("The pilot requires the official five-step ODE and ten-action LIBERO eval config.")
        self.model = get_model(model_cfg).eval()
        if checkpoint_or_none is not None:
            weights = torch.load(checkpoint_or_none, map_location="cpu", weights_only=True, mmap=True)
            # The official evaluator removes the training-only value head before strict loading.
            actor_weights = {key: value for key, value in weights.items() if not key.startswith("value_head.")}
            self.model.load_state_dict(actor_weights, strict=True)
        self.model.cuda().requires_grad_(False)

    @torch.inference_mode()
    def actions(self, list_obs, task_text, *, noise_seeds=None):
        """Predict a B x 10 x 7 action chunk, optionally pairing ODE initial noise by row."""
        size = len(list_obs)
        if size == 0:
            raise ValueError("An inference batch must contain at least one observation.")
        # DrawerEnv already applies the two native 180-degree camera rotations
        # and LIBERO's quaternion-to-axis-angle conversion before this call.
        wrapped = {
            "main_images": torch.from_numpy(np.stack([obs["main_images"] for obs in list_obs])),
            "wrist_images": torch.from_numpy(np.stack([obs["wrist_images"] for obs in list_obs])),
            "states": torch.from_numpy(np.stack([obs["states"] for obs in list_obs])).float(),
            "task_descriptions": [task_text] * size,
        }
        noise = None
        if noise_seeds is not None:
            if len(noise_seeds) != size:
                raise ValueError("Provide exactly one noise seed per observation.")
            device = self.model.device
            shape = (self.model.action_horizon, self.model.action_dim)
            noise = torch.stack([
                torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(int(seed)))
                for seed in noise_seeds
            ])
        actions, _ = self.model.predict_action_batch(wrapped, mode="eval", noise=noise)
        actions = prepare_actions_for_libero(actions.cpu().numpy(), model_type="openpi")
        if actions.shape != (size, 10, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid LIBERO action batch, shape={actions.shape}")
        return actions
