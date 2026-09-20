"""Persist the joint model's conditioning contract independently of weights.

The two modes have identical parameter trees, so loading weights cannot detect
an accidental change of training objective or inference input format.
"""

import json

LEGACY = "task_and_subtask"
FACTORIZED = "subtask_only"
FILENAME = "subtask_conditioning.json"


def from_data_config(data_config):
    for transform in data_config.model_transforms.inputs:
        mode = getattr(transform, "subtask_action_conditioning", None)
        if mode is not None:
            return mode
    return None


def save(assets_dir, mode):
    if mode is None:
        return
    if mode not in (LEGACY, FACTORIZED):
        raise ValueError(f"Unknown subtask action conditioning: {mode}")
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / FILENAME).write_text(json.dumps({"version": 1, "action_conditioning": mode}) + "\n")


def load(assets_dir):
    path = assets_dir / FILENAME
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    mode = record.get("action_conditioning")
    if record.get("version") != 1 or mode not in (LEGACY, FACTORIZED):
        raise ValueError(f"Invalid joint conditioning metadata: {path}")
    return mode


def check_resume(assets_dir, current_mode):
    if current_mode is None:
        return
    saved_mode = load(assets_dir) or LEGACY
    if saved_mode != current_mode:
        raise ValueError(
            f"Cannot resume joint conditioning {saved_mode!r} as {current_mode!r}. "
            "Start a new fine-tuning experiment with the old weights instead."
        )
