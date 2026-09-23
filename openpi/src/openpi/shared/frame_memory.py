"""Shared causal frame sampling and checkpoint contracts; no simulator labels."""

import hashlib
import json

import numpy as np

FIELDS = ("memory_features", "memory_positions", "memory_mask")
FILENAME = "frame_memory.json"


def model_contract(config):
    if not getattr(config, "framesamp_modul", False):
        return None
    return {
        "version": 1,
        "method": "framesamp_modul",
        "pool": "mean",
        "position": "thw_sincos_768",
        "num_frames": config.memory_num_frames,
        "grid_size": config.memory_grid_size,
        "frame_stride": config.memory_frame_stride,
        "camera": config.memory_camera,
    }


def save(assets, contract):
    if contract is not None:
        assets.mkdir(parents=True, exist_ok=True)
        (assets / FILENAME).write_text(json.dumps(contract, indent=2) + "\n")


def load(assets):
    path = assets / FILENAME
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    valid = (
        result.get("version") == 1
        and result.get("method") == "framesamp_modul"
        and result.get("pool") == "mean"
        and result.get("position") == "thw_sincos_768"
        and result.get("grid_size") in (1, 2, 4, 8, 16)
        and result.get("camera") in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    )
    for key in ("num_frames", "frame_stride"):
        valid = valid and type(result.get(key)) is int and result[key] > 0
    if not valid:
        raise ValueError(f"Invalid FrameSamp checkpoint contract: {path}")
    return result


def check_resume(assets, contract):
    if load(assets) != contract:
        raise ValueError(
            "Cannot resume with a different FrameSamp memory/cache contract. Start a new experiment from weights."
        )


def sample_indices(count, num_frames):
    if count < 1 or num_frames < 1:
        raise ValueError("History and frame budget must be positive.")
    return np.linspace(0, count - 1, min(count, num_frames), dtype=np.int64)


def pack_history(features, frame_ids, *, num_frames, grid_size):
    """Pack already selected frames and right-pad. frame_ids are original control steps."""
    features = np.asarray(features, dtype=np.float32)
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    count, patches, width = features.shape
    if not 0 < count <= num_frames or patches != grid_size**2 or frame_ids.shape != (count,):
        raise ValueError("Invalid sampled history shape.")
    if np.any(frame_ids < 0) or np.any(np.diff(frame_ids) <= 0) or not np.isfinite(features).all():
        raise ValueError("History must be finite, causal and strictly ordered.")
    budget = num_frames * patches
    output = np.zeros((budget, width), dtype=np.float32)
    positions = np.zeros((budget, 3), dtype=np.int32)
    mask = np.arange(budget) < count * patches
    output[: count * patches] = features.reshape(-1, width)
    positions[: count * patches, 0] = np.repeat(frame_ids, patches)
    positions[: count * patches, 1] = np.tile(np.repeat(np.arange(grid_size), grid_size), count)
    positions[: count * patches, 2] = np.tile(np.arange(grid_size), count * grid_size)
    return dict(zip(FIELDS, (output, positions, mask), strict=True))


def encoder_fingerprint(params):
    """Hash canonical BF16 vision weights, independent of checkpoint storage dtype."""
    from flax import traverse_util
    import ml_dtypes

    digest = hashlib.sha256()
    for key, value in sorted(traverse_util.flatten_dict(params).items()):
        array = np.asarray(value, dtype=ml_dtypes.bfloat16)
        digest.update("/".join(key).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
