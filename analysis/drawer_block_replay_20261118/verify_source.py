"""Verify alternate-host replay fidelity independently from continuation outcomes."""
import hashlib
import io
import json
import math
import struct

from source_inputs import cross_host_gate, digest


def jpeg_anchor_scores(image_path, reference_path):
    from PIL import Image, ImageChops, ImageStat

    with Image.open(image_path) as raw, Image.open(reference_path) as original:
        if raw.mode != "RGB" or original.mode != "RGB" or raw.size != (256, 256) or original.size != raw.size:
            raise ValueError("Invalid main-camera frame format")
        buffer = io.BytesIO()
        raw.save(buffer, format="JPEG", quality=83)
        encoded = buffer.getvalue()
        with Image.open(io.BytesIO(encoded)) as recoded:
            diff = ImageChops.difference(recoded, original)
    channels = diff.split()
    max_channel = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    tiles = [diff.crop((x, y, x + 32, y + 32))
             for y in range(0, diff.height, 32) for x in range(0, diff.width, 32)]
    return {"jpeg_bytes_equal": encoded == reference_path.read_bytes(),
            "decoded_rgb_mae": sum(ImageStat.Stat(diff).mean) / 3,
            "max_tile32_mae": max(sum(ImageStat.Stat(tile).mean) / 3 for tile in tiles),
            "fraction_pixels_above_16": sum(max_channel.histogram()[17:]) / (diff.width * diff.height)}


def observation_digest(directory, step, proprio):
    """Recreate the canonical float32/arbitrary-byte RGB hash from exported PNGs and state."""
    from PIL import Image

    hasher = hashlib.sha256()
    hasher.update(str(("<f4", (len(proprio),))).encode())
    hasher.update(struct.pack(f"<{len(proprio)}f", *proprio))
    for name in ("main", "wrist"):
        with Image.open(directory / f"{name}_{step}.png") as image:
            if image.mode != "RGB" or image.size != (256, 256):
                raise ValueError("Invalid saved raw observation")
            hasher.update(str(("|u1", (image.height, image.width, 3))).encode())
            hasher.update(image.tobytes())
    return hasher.hexdigest()


def verify(source, directory, references, prefix, record, reference_digests):
    mode = source.get("reconstruction_mode", "exact")
    if mode == "exact":
        return mode, {}
    if mode != "cross_host_equivalent_v1":
        raise ValueError(f"Unrecognized source reconstruction mode: {mode}")
    scene = record["seed"]
    if references is None:
        raise ValueError("Cross-host replay requires --anchor-dir")
    ref_root = references
    expected_images = {f"{scene}_new_root_{suffix}.jpg" for suffix in ("000", "090", "190", "handoff")}
    root_files = set(reference_digests) - expected_images
    if (len(root_files) != 1 or not expected_images.issubset(reference_digests)
            or not next(iter(root_files)).startswith("batch_")
            or not next(iter(root_files)).endswith(".json")):
        raise ValueError(f"Cross-host replay requires four JPEG pins and a root batch pin for {scene}")
    hashes = {name: digest(ref_root / name) for name in reference_digests}
    if hashes != reference_digests or hashes != source.get("reference_sha256s"):
        raise ValueError(f"Archived PPO replay evidence changed for scene {scene}")
    diag = directory / "replay_diagnostics"
    read = lambda path: json.loads(path.read_text())
    first, second = (read(diag / f"pass_{i}.json") for i in (1, 2))
    gate_saved = read(diag / "gate.json")
    reconstructed = cross_host_gate(prefix, record, first, second)
    if not reconstructed["passed"] or reconstructed != source.get("cross_host_gate") or reconstructed != gate_saved["cross_host_gate"]:
        raise ValueError(f"Fidelity gate failed for scene {scene}")
    if gate_saved["reconstruction_mode"] != mode or gate_saved["reference_sha256s"] != hashes:
        raise ValueError(f"Fidelity gate did not use the documented mode or references for scene {scene}")

    handoff, landmark = record["first_in_step"], record["landmark_step"]
    root_file = next(name for name in hashes if name.startswith("batch_"))
    roots = read(ref_root / root_file)["root_new"]
    measured_handoff = first["grid"][str(handoff)]["metrics"]
    original_handoff = next(row["handoff_metrics"] for row in roots if row["seed"] == scene)
    deltas = {k: measured_handoff[k] - original_handoff[k] for k in ("signed_front_m", "drawer_open_m")}
    if first["handoff_deltas_m"] != deltas or second["handoff_deltas_m"] != deltas:
        raise ValueError(f"Reported handoff errors disagree with original root measurements for scene {scene}")
    for row in first["native_window"]:
        if row["measured"] != first["grid"][str(row["step"])]["metrics"]:
            raise ValueError(f"Native window and exported grid disagree at {scene}:{row['step']}")
    for step, suffix in ((10, "000"), (100, "090"), (200, "190"), (handoff, "handoff")):
        score = jpeg_anchor_scores(diag / f"main_{step}.png", ref_root / f"{scene}_new_root_{suffix}.jpg")
        reported = first["image_anchors"][str(step)]
        for key, limit in reconstructed["image_limits_uint8"].items():
            if score[key] > limit or not math.isclose(score[key], reported[key], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"Published main-camera anchor scores disagree at {scene}:{step}")
    for step in {handoff, landmark, 200}:
        row = first["grid"][str(step)]
        if observation_digest(diag, step, row["proprio"]) != row["observation_hash"]:
            raise ValueError(f"Raw cameras and proprio do not reproduce the paired observation at {scene}:{step}")
    for step, measured in first["grid"].items():
        state = source["blocks"][step]
        if any(state[k] != measured[k] for k in ("state_hash", "observation_hash")) or state["initial_hash"] != prefix["initial_hash"]:
            raise ValueError(f"Wrong physical state in worker manifest for {scene}:{step}")
    return mode, {step: data["metrics"] for step, data in first["grid"].items()}
