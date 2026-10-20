"""Checkpoint-specific R1Pro coordinates, normalization and camera order."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DELTA_INDICES = np.asarray(list(range(3, 14)) + list(range(15, 22)))
PROPRIO_KEYS = (
    "base_qvel",
    "trunk_qpos",
    "arm_left_qpos",
    "gripper_left_qpos",
    "arm_right_qpos",
    "gripper_right_qpos",
)


def as_numpy(value):
    if torch.is_tensor(value):
        return (
            value.detach().cpu().float().numpy()
            if value.is_floating_point()
            else value.detach().cpu().numpy()
        )
    return np.asarray(value)


def extract_state(raw, layout):
    """Return base, torso, left arm/gripper, right arm/gripper (23 values)."""
    raw = as_numpy(raw)
    if layout == "canonical25":
        if raw.shape[-1] != 25:
            raise ValueError(f"Expected canonical25 proprio, got {raw.shape}")
        pieces = (
            raw[..., :14],
            raw[..., 14:16].sum(-1, keepdims=True),
            raw[..., 16:23],
            raw[..., 23:25].sum(-1, keepdims=True),
        )
    elif layout in ("checkpoint57", "checkpoint61"):
        expected = 57 if layout == "checkpoint57" else 61
        if raw.shape[-1] != expected:
            raise ValueError(f"Expected {layout} proprio, got {raw.shape}")
        pieces = (
            raw[..., :3],
            raw[..., 53:57],
            raw[..., 3:10],
            raw[..., 24:26].sum(-1, keepdims=True),
            raw[..., 28:35],
            raw[..., 49:51].sum(-1, keepdims=True),
        )
    elif layout == "policy23":
        if raw.shape[-1] != 23:
            raise ValueError(f"Expected policy23 state, got {raw.shape}")
        pieces = (raw,)
    else:
        raise ValueError(f"Unknown explicit state layout: {layout}")
    state = np.concatenate(pieces, axis=-1).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Non-finite robot state")
    return state


def resize_image(image, size=224):
    """Match the existing OpenPI PIL bilinear resize-with-pad."""
    image = as_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if np.issubdtype(image.dtype, np.floating):
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Floating camera images must be in [0, 1]")
        image = (255 * image).astype(np.uint8)
    if image.dtype != np.uint8 or image.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB or floating RGB in [0, 1]")
    if image.shape[:2] == (size, size):
        return image.copy()
    pil = Image.fromarray(image)
    width, height = pil.size
    ratio = max(width / size, height / size)
    width, height = int(width / ratio), int(height / ratio)
    resized = pil.resize((width, height), resample=Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (size, size), 0)
    padded.paste(resized, ((size - width) // 2, (size - height) // 2))
    return np.asarray(padded).copy()


class CheckpointIO:
    def __init__(self, norm_path, state_layout="canonical25"):
        path = Path(norm_path)
        raw = path.read_bytes()
        stats = json.loads(raw)
        stats = stats.get("norm_stats", stats)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.state_layout = state_layout
        self.stats = {}
        for name in ("state", "actions"):
            self.stats[name] = {
                # JSON stats in the original OpenPI pipeline are float64. Keep
                # that precision until tokenization (bin boundaries matter).
                k: np.asarray(stats[name][k], dtype=np.float64)[:23]
                for k in ("mean", "std")
            }
            if any(v.shape != (23,) or not np.isfinite(v).all() for v in self.stats[name].values()):
                raise ValueError(f"Expected 23 finite normalization values for {name}")
            if np.any(self.stats[name]["std"] < 0):
                raise ValueError(f"Negative normalization std for {name}")

    def normalize_state(self, state):
        stats = self.stats["state"]
        return (state - stats["mean"]) / (stats["std"] + 1e-6)

    def prepare(self, env_obs):
        state = extract_state(env_obs["states"], self.state_layout)
        if state.ndim != 2:
            raise ValueError("Expected batched proprio [B, D]")
        batch = len(state)
        prompts = list(env_obs["task_descriptions"])
        if len(prompts) != batch or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("Expected one nonempty task description per environment")
        main = env_obs["main_images"]
        wrists = env_obs["wrist_images"]
        images = {
            IMAGE_KEYS[0]: np.stack([resize_image(main[i]) for i in range(batch)]),
            IMAGE_KEYS[1]: np.stack([resize_image(wrists[i][0]) for i in range(batch)]),
            IMAGE_KEYS[2]: np.stack([resize_image(wrists[i][1]) for i in range(batch)]),
        }
        normalized = self.normalize_state(state)
        return images, normalized, prompts

    def decode(self, model_actions, normalized_state, action_chunk):
        device = model_actions.device
        actions = model_actions[..., :23].double()
        state = normalized_state[..., :23].double()

        def tensor(value):
            return torch.as_tensor(value, device=device, dtype=torch.float64)

        actions = actions * (tensor(self.stats["actions"]["std"]) + 1e-6) + tensor(
            self.stats["actions"]["mean"]
        )
        state = state * (tensor(self.stats["state"]["std"]) + 1e-6) + tensor(self.stats["state"]["mean"])
        # Construct without in-place modification of values needed by autograd.
        mask = torch.zeros(23, device=device)
        mask[torch.as_tensor(DELTA_INDICES, device=device)] = 1
        return (actions + state[:, None] * mask)[:, :action_chunk].float().contiguous()
