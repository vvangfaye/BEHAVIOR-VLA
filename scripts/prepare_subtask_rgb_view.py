#!/usr/bin/env python3
"""Freeze an expert RGB metadata view; keep original videos and parquet in place.

Only the copied meta/info.json changes. This deliberately excludes collected
exports, whose original acceptance evidence must keep its original identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


RGB_KEYS = frozenset({"observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0"})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root):
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted((root / "meta").rglob("*")) if path.is_file()}


def prepare_rgb_view(dataset_root, output):
    source = Path(dataset_root).expanduser().resolve()
    target_argument = Path(output).expanduser()
    if target_argument.exists() or target_argument.is_symlink():
        raise ValueError("RGB metadata view output already exists")
    target = target_argument.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("RGB metadata view must be separate from the original dataset")
    info_path = source / "meta/info.json"
    if not info_path.is_file() or not (source / "data").is_dir() or not (source / "videos").is_dir():
        raise ValueError("Original expert metadata, data and videos directories are required")
    if (source / "meta/collection_provenance.json").exists() or (source / "_VERIFIED_EXPORT").exists():
        raise ValueError("RGB metadata views support original expert datasets only; preserve collected export identities")
    info = json.loads(info_path.read_text())
    features = info.get("features")
    if not isinstance(features, dict): raise ValueError("Missing original feature metadata")
    removed = []
    for key, feature in features.items():
        if not isinstance(feature, dict): raise ValueError("Feature metadata must be objects")
        details = feature.get("info") or {}
        if not isinstance(details, dict): raise ValueError("Video feature info must be an object")
        depth = details.get("is_depth_map", False)
        if type(depth) is not bool: raise ValueError("is_depth_map must be a boolean")
        if feature.get("dtype") == "video" and depth: removed.append(key)
    rgb_features = {key: feature for key, feature in features.items() if key not in removed}
    remaining_videos = {key for key, feature in rgb_features.items() if feature.get("dtype") == "video"}
    if remaining_videos != RGB_KEYS:
        raise ValueError("RGB metadata view requires exactly the three B1K R1Pro RGB video keys")
    for key in RGB_KEYS:
        shape = rgb_features[key].get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("RGB video feature must declare an HWC three-channel shape")
    source_files = _inventory(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrite atomically, including another preparer racing this one.
    target.mkdir()
    try:
        shutil.copytree(source / "meta", target / "meta")
        if removed:
            info["features"] = rgb_features
            (target / "meta/info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        (target / "data").symlink_to(source / "data", target_is_directory=True)
        (target / "videos").symlink_to(source / "videos", target_is_directory=True)
        derived_files = _inventory(target)
        if _inventory(source) != source_files:
            raise ValueError("Original metadata changed during RGB-view preparation")
        if {key: value for key, value in source_files.items() if key != "meta/info.json"} != {
            key: value for key, value in derived_files.items() if key != "meta/info.json"}:
            raise ValueError("RGB metadata view changed original episode/task metadata")
        manifest = dict(schema_version=1, kind="b1k_expert_rgb_metadata_view", source_dataset_root=str(source),
            dataset_root=str(target), source_info_sha256=source_files["meta/info.json"],
            derived_info_sha256=derived_files["meta/info.json"], removed_depth_video_keys=sorted(removed),
            source_metadata_files=source_files, derived_metadata_files=derived_files,
            data_target=str(source / "data"), videos_target=str(source / "videos"),
            source_metadata_modified=False, videos_reencoded=False, raw_parquet_copied=False,
            original_episode_ids_and_offsets_preserved=True)
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        manifest["fingerprint"] = hashlib.sha256(encoded).hexdigest()
        (target / "rgb_view_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        return manifest
    except BaseException:
        # This directory was exclusively created here; unlinking symlinks never
        # touches original data/video targets.
        shutil.rmtree(target)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_rgb_view(args.dataset_root, args.output)
    print(json.dumps({key: manifest[key] for key in ("fingerprint", "dataset_root", "removed_depth_video_keys",
        "source_info_sha256", "derived_info_sha256")}))


if __name__ == "__main__": main()
