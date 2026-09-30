"""Load inference artifacts without importing any training data reader."""
import torch
from .model import GoalTransitionModel, ModelConfig


def load_checkpoint(path, device="cpu"):
    artifact = torch.load(path, map_location=device, weights_only=True)
    if artifact.get("format") != "goalkeeper-v1":
        raise ValueError("Unknown checkpoint format")
    model = GoalTransitionModel(ModelConfig(**artifact["config"])).to(device)
    model.load_state_dict(artifact["state_dict"])
    return model.eval(), artifact
