"""Input provenance and fidelity checks for saved paired drawer continuations."""
import hashlib
import json
import math
import re
from pathlib import Path


HORIZON = 520
BLOCK = 10
NOISE_VERSION = "drawer-block-20261118-v1"


def is_sha256(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def load_manifest(path):
    """Read trusted digest pins supplied separately from the replay artifacts."""
    manifest = json.loads(path.read_text())
    if not is_sha256(manifest.get("ppo_actor_sha256")):
        raise ValueError("Source manifest requires a PPO actor digest")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("Source manifest requires at least one scene")
    for seed, source in scenes.items():
        if not seed.isdigit() or str(int(seed)) != seed or not isinstance(source, dict):
            raise ValueError(f"Invalid source manifest entry for scene {seed}")
        if type(source.get("landmark")) is not int or not 80 < source["landmark"] < HORIZON:
            raise ValueError(f"Invalid landmark for scene {seed}")
        if not all(is_sha256(source.get(key)) for key in ("prefix_sha256", "record_sha256")):
            raise ValueError(f"Missing source digests for scene {seed}")
        references = source.get("reference_sha256s", {})
        if not isinstance(references, dict) or any(
            not name or Path(name).name != name or name.startswith(".") or not is_sha256(value)
            for name, value in references.items()
        ):
            raise ValueError(f"Invalid reference digests for scene {seed}")
    return manifest


def cross_host_gate(prefix, record, first, second):
    """Pre-specified numerical and RGB agreement plus exact within-node replay."""
    opening_limit = 0.0001
    rgb_limits = {"decoded_rgb_mae": 0.5, "max_tile32_mae": 3.0, "fraction_pixels_above_16": 0.001}
    first_step, landmark = record["first_in_step"], prefix["step"]
    anchors = {str(i) for i in (10, 100, 200, first_step)}
    grid = {str(i) for i in range(first_step, landmark + 1, BLOCK)}
    checks = {
        "original_initial_state": first["initial_hash"] == second["initial_hash"] == prefix["initial_hash"],
        "same_node_repeat": (first["grid"] == second["grid"]
                             and first["native_window"] == second["native_window"]
                             and set(first["grid"]) == grid),
        "matching_first_open_and_no_success": all(
            r["first_native_bowl_in_open"] == first_step and r["first_success"] is None for r in (first, second)),
        "handoff_geometry": all(
            set(r.get("handoff_deltas_m", {})) == {"signed_front_m", "drawer_open_m"}
            and all(abs(value) <= opening_limit for value in r["handoff_deltas_m"].values())
            for r in (first, second)),
        "four_native_window_states": len(first["native_window"]) == 4 and all(
            row["step"] == original["step"] and row["source"] == original
            and row["measured"]["step"] == original["step"]
            and all(row["measured"][k] == original[k] for k in ("bowl_in", "drawer_closed"))
            and abs(row["measured"]["drawer_open_m"] - original["drawer_open_m"]) <= opening_limit
            for row, original in zip(first["native_window"], record["window"])),
        "four_original_main_camera_anchors": set(first["image_anchors"]) == anchors and all(
            math.isfinite(scores[key]) and 0 <= scores[key] <= limit
            for scores in first["image_anchors"].values() for key, limit in rgb_limits.items()),
    }
    return {"passed": all(checks.values()), "checks": checks, "geometry_limit_m": opening_limit,
            "image_limits_uint8": rgb_limits, "mode": "cross_host_equivalent_v1"}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(8 * 1024**2), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_source(directory, seed, source):
    landmark = source["landmark"]
    prefix_sha = source["prefix_sha256"]
    record_sha = source["record_sha256"]
    prefix_path = directory / f"prefix_{seed}.json"
    record_path = directory / f"paired_{seed}.json"
    if digest(prefix_path) != prefix_sha or digest(record_path) != record_sha:
        raise ValueError(f"Scene {seed} does not match the verified PPO source files")
    prefix = json.loads(prefix_path.read_text())
    record = json.loads(record_path.read_text())
    if (not record["flagged"] or not record["eligible"] or record["root_new_success"]
            or record["split"] != "formal_holdout" or record["seed"] != seed
            or record["landmark_step"] != landmark or record["first_in_step"] != landmark - 80
            or prefix["seed"] != seed or prefix["step"] != landmark or prefix["success_once"]
            or len(prefix["actions"]) != landmark
            or any(prefix[key] != record[key] for key in ("initial_hash", "state_hash", "observation_hash"))):
        raise ValueError(f"Scene {seed} is inconsistent with its verified landmark record")
    return prefix, record


def stream_name(seed, phase, repeat):
    return f"{NOISE_VERSION}/{seed}/{phase}/{repeat}"
