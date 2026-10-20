#!/usr/bin/env python
"""Offline demonstration action error for a B1K π0.5 policy.

This is not the BEHAVIOR benchmark score. It compares predicted action chunks with
recorded demonstration actions; closed-loop task success still requires simulator
rollouts.

Examples:
  ./scripts/eval_openloop.py --model pi05 --samples-per-task 2 --out eval_results/openloop_pi05.json
  ./scripts/eval_openloop.py --model pi05 --subtask-conditioning provided \
      --subtask-prompts /path/to/subtask_segments.parquet --episodes-file /path/to/val_episodes.txt \
      --samples-per-subtask 2 --out eval_results/openloop_joint_provided.json
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import pathlib
import time
from collections.abc import Callable, Collection, Sequence
from typing import Any

import numpy as np

DATASET = "/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/dataset_views/pi05/2026-challenge-demos"
EVAL_HORIZON = 32
# Raw R1Pro action layout in the BEHAVIOR demonstration dataset.
ACTION_GROUPS = {
    "base": (0, 3),
    "torso": (3, 7),
    "left_arm": (7, 14),
    "left_gripper": (14, 15),
    "right_arm": (15, 22),
    "right_gripper": (22, 23),
}
# State indices matching each action group, for the "hold current state" baseline.
STATE_FOR_ACTION = {
    "torso": (53, 57),
    "left_arm": (3, 10),
    "left_gripper": (24, 26),
    "right_arm": (28, 35),
    "right_gripper": (49, 51),
}
ERROR_METHODS = ("model", "baseline_hold", "baseline_zero")


@dataclasses.dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    task_index: int
    dataset_from_index: int
    dataset_to_index: int
    relative_from_index: int = 0

    @property
    def length(self) -> int:
        return self.dataset_to_index - self.dataset_from_index


@dataclasses.dataclass(frozen=True)
class EvalSample:
    task_index: int
    episode_index: int
    frame_index: int
    global_frame_index: int
    dataset_index: int
    subtask: str | None = None
    segment_end: int | None = None
    probe_metadata: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class ErrorRecord:
    sample: EvalSample
    errors: dict[str, np.ndarray]
    valid: np.ndarray


def read_episode_ids(path: pathlib.Path) -> list[int]:
    """Read unique original LeRobot episode IDs from a whitespace-delimited file."""
    try:
        episode_ids = [int(token) for token in path.read_text().split()]
    except ValueError as error:
        raise ValueError(f"Episode file contains a non-integer ID: {path}") from error
    if not episode_ids:
        raise ValueError(f"Episode file is empty: {path}")
    duplicates = sorted(episode for episode, count in collections.Counter(episode_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Episode file contains duplicate IDs: {duplicates[:10]}")
    return episode_ids


def load_episode_meta(dataset: str | pathlib.Path) -> list[EpisodeMeta]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    root = pathlib.Path(dataset)
    files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise ValueError(f"No episode metadata found under {root / 'meta' / 'episodes'}")
    rows = []
    for path in files:
        columns = [
            "episode_index",
            "task_index",
            "dataset_from_index",
            "dataset_to_index",
        ]
        missing = set(columns) - set(pq.read_schema(path).names)
        if missing:
            raise ValueError(f"Episode metadata {path} is missing columns: {sorted(missing)}")
        rows.extend(
            EpisodeMeta(
                episode_index=int(row["episode_index"]),
                task_index=int(row["task_index"]),
                dataset_from_index=int(row["dataset_from_index"]),
                dataset_to_index=int(row["dataset_to_index"]),
            )
            for row in pq.read_table(path, columns=columns).to_pylist()
        )
    return rows


def filter_episode_meta(
    episodes_meta: Sequence[EpisodeMeta], allowed_episode_ids: Collection[int] | None
) -> list[EpisodeMeta]:
    """Filter original episodes and assign rows in the filtered LeRobot view.

    LeRobot preserves source parquet order after episode filtering. Therefore a
    selected episode's dataset row starts at the sum of lengths of preceding
    selected episodes, not at its original ``dataset_from_index``.
    """
    ordered = sorted(episodes_meta, key=lambda episode: episode.dataset_from_index)
    by_id = {}
    previous_end = None
    for episode in ordered:
        if episode.episode_index in by_id:
            raise ValueError(f"Duplicate episode_index={episode.episode_index} in episode metadata")
        if episode.length <= 0:
            raise ValueError(f"Episode {episode.episode_index} has invalid frame range")
        if previous_end is not None and episode.dataset_from_index < previous_end:
            raise ValueError("Episode metadata contains overlapping global frame ranges")
        by_id[episode.episode_index] = episode
        previous_end = episode.dataset_to_index

    allowed = None if allowed_episode_ids is None else set(allowed_episode_ids)
    if allowed is not None:
        unknown = sorted(allowed - by_id.keys())
        if unknown:
            raise ValueError(f"Unknown episode IDs in episode file: {unknown[:10]}")

    selected = []
    relative_start = 0
    for episode in ordered:
        if allowed is not None and episode.episode_index not in allowed:
            continue
        selected.append(dataclasses.replace(episode, relative_from_index=relative_start))
        relative_start += episode.length
    if not selected:
        raise ValueError("No episodes remain after filtering")
    return selected


def task_names(dataset: str | pathlib.Path) -> dict[int, str]:
    out = {}
    path = pathlib.Path(dataset) / "meta" / "tasks.jsonl"
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            out[int(row["task_index"])] = str(row.get("task_name") or row.get("task") or row["task_index"])
    return out


def _pick_offsets(ranges: Sequence[tuple[EpisodeMeta, int, int]], count: int, rng: np.random.Generator):
    lengths = np.asarray([end - start for _, start, end in ranges], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("Sampling ranges must be non-empty half-open intervals")
    cumulative = np.cumsum(lengths)
    total = int(cumulative[-1])
    offsets = np.sort(rng.choice(total, size=min(count, total), replace=False))
    for offset in offsets.tolist():
        range_index = int(np.searchsorted(cumulative, offset, side="right"))
        preceding = 0 if range_index == 0 else int(cumulative[range_index - 1])
        episode, start, _ = ranges[range_index]
        yield episode, start + offset - preceding


def select_samples(
    episodes_meta: Sequence[EpisodeMeta],
    per_group: int,
    seed: int,
    segments: Any | None = None,
) -> list[EvalSample]:
    """Deterministically sample per task, or per canonical (task, subtask)."""
    if per_group < 1:
        raise ValueError("The per-task/per-subtask sample count must be positive")
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[int, str | None], list[tuple[EpisodeMeta, int, int]]] = collections.defaultdict(list)

    if segments is None:
        for episode in episodes_meta:
            grouped[(episode.task_index, None)].append((episode, 0, episode.length))
    else:
        lengths = {episode.episode_index: episode.length for episode in episodes_meta}
        segments.validate_coverage(lengths)
        for episode in episodes_meta:
            starts, ends, labels = segments.segments[episode.episode_index]
            annotated = segments.annotation_flags(episode.episode_index)
            for start, end, label, is_annotated in zip(starts, ends, labels, annotated, strict=True):
                if not is_annotated:
                    continue
                grouped[(episode.task_index, label)].append((episode, int(start), int(end)))

    if not grouped:
        raise ValueError("No eligible evaluation anchors were found")

    picked = []
    for (task_index, subtask), ranges in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        for episode, frame in _pick_offsets(ranges, per_group, rng):
            segment_end = None
            if segments is not None:
                strict_label, segment_end = segments.lookup(episode.episode_index, frame, require_annotated=True)
                if strict_label != subtask:
                    raise RuntimeError("Subtask sampling and strict segment lookup disagree")
            picked.append(
                EvalSample(
                    task_index=task_index,
                    episode_index=episode.episode_index,
                    frame_index=frame,
                    global_frame_index=episode.dataset_from_index + frame,
                    dataset_index=episode.relative_from_index + frame,
                    subtask=subtask,
                    segment_end=segment_end,
                )
            )
    return picked


def replay_samples(
    path: pathlib.Path,
    episodes_meta: Sequence[EpisodeMeta],
    segments: Any,
    *,
    names: dict[int, str] | None = None,
    repo_id: str | None = None,
    horizon: int | None = None,
    action_noise_seed: int | None = None,
) -> list[EvalSample]:
    """Reuse exact frames and valid horizons while resolving labels from the current segment table."""
    payload = json.loads(path.read_text())
    rows = payload.get("sample_keys") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Sample key file has no non-empty sample_keys list: {path}")

    is_boundary_probe = isinstance(payload, dict) and payload.get("kind") == "matched_training_boundary_probe"
    if is_boundary_probe:
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported boundary probe schema")
        if repo_id is not None and payload.get("repo_id") != repo_id:
            raise ValueError("Boundary probe repo_id differs from the evaluation repo_id")
        if horizon is not None and payload.get("model_horizon_H") != horizon:
            raise ValueError("Boundary probe model horizon differs from --horizon")
        required_seed = payload.get("pairing_contract", {}).get("action_noise_seed")
        if type(required_seed) is not int or required_seed < 0 or action_noise_seed != required_seed:
            raise ValueError("Boundary probe requires its explicit --action-noise-seed for paired inference")

    by_episode = {episode.episode_index: episode for episode in episodes_meta}
    samples = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Sample key entries must be objects: {path}")
        try:
            episode_index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
            valid_action_steps = int(row["valid_action_steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid sample key entry in {path}: {row!r}") from error
        key = (episode_index, frame_index)
        if key in seen:
            raise ValueError(f"Duplicate sample key episode/frame={key} in {path}")
        seen.add(key)
        episode = by_episode.get(episode_index)
        if episode is None:
            raise ValueError(f"Sample key episode {episode_index} is outside the selected episode split")
        if frame_index < 0 or frame_index >= episode.length:
            raise ValueError(f"Sample key frame {frame_index} is outside episode {episode_index}")
        if "task_index" in row and int(row["task_index"]) != episode.task_index:
            raise ValueError(f"Sample key task does not match metadata for episode {episode_index}")
        if valid_action_steps < 1:
            raise ValueError(f"Sample key has no valid action steps for episode/frame={key}")

        subtask, current_segment_end = segments.lookup(episode_index, frame_index, require_annotated=True)
        if row.get("subtask") is not None and row["subtask"] != subtask:
            raise ValueError(f"Replayed subtask differs from current annotation for episode/frame={key}")
        reference_segment_end = frame_index + valid_action_steps
        if current_segment_end < reference_segment_end:
            raise ValueError(
                f"Current segment ends before the replayed valid horizon for episode/frame={key}: "
                f"{current_segment_end} < {reference_segment_end}"
            )
        probe_metadata = None
        if is_boundary_probe:
            probe_metadata = _validated_probe_metadata(row, payload, frame_index, valid_action_steps,
                                                       current_segment_end)
            if names is not None and probe_metadata["task_name"] != names.get(episode.task_index):
                raise ValueError(f"Boundary probe task name differs from current metadata for episode/frame={key}")
        samples.append(
            EvalSample(
                task_index=episode.task_index,
                episode_index=episode_index,
                frame_index=frame_index,
                global_frame_index=episode.dataset_from_index + frame_index,
                dataset_index=episode.relative_from_index + frame_index,
                subtask=subtask,
                segment_end=reference_segment_end,
                probe_metadata=probe_metadata,
            )
        )
    return samples


def _validated_probe_metadata(row, manifest, frame, valid_steps, current_segment_end):
    """Retain only known probe fields after checking their type and geometry.

    Root/instance IDs remain claims of the frozen source index. The current
    dataset row, label, task name and segment bounds are checked independently.
    """
    string_fields = ("root_id", "source_kind", "repo_id", "task_name", "boundary_id", "sample_id",
                     "bucket", "next_subtask")
    integer_fields = ("task_instance_id", "segment_start", "segment_end", "next_segment_end",
                      "distance_to_boundary_frames")
    boolean_fields = ("full_H_legal_before", "executed_K_within_original_label",
                      "recorded_full_H_future_available_by_index", "training_seen_source")
    result = {}
    for field in string_fields:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Boundary probe field {field!r} must be a nonempty string")
        result[field] = value
    for field in integer_fields:
        value = row.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"Boundary probe field {field!r} must be a nonnegative integer")
        result[field] = value
    for field in boolean_fields:
        value = row.get(field)
        if type(value) is not bool:
            raise ValueError(f"Boundary probe field {field!r} must be boolean")
        result[field] = value
    for field in ("root_id", "repo_id", "source_kind"):
        if result[field] != manifest.get(field):
            raise ValueError(f"Boundary probe sample {field} differs from its manifest")
    distance = result["distance_to_boundary_frames"]
    model_horizon, executed = manifest.get("model_horizon_H"), manifest.get("executed_prefix_K")
    if type(model_horizon) is not int or type(executed) is not int or not 1 < executed < model_horizon:
        raise ValueError("Boundary probe has invalid model/execute horizons")
    expected_distances = {"far_2H": 2 * model_horizon, "edge_H": model_horizon,
                          "inside_K": executed, "last_1": 1}
    if result["bucket"] not in expected_distances or expected_distances[result["bucket"]] != distance:
        raise ValueError("Boundary probe bucket disagrees with its distance")
    if not result["segment_start"] <= frame < result["segment_end"] == current_segment_end:
        raise ValueError("Boundary probe source interval differs from the current annotation")
    if distance != current_segment_end - frame or valid_steps != min(model_horizon, distance):
        raise ValueError("Boundary probe valid horizon/distance disagrees with its interval")
    if frame + model_horizon > result["next_segment_end"]:
        raise ValueError("Boundary probe lacks its declared full future")
    if result["full_H_legal_before"] != (distance >= model_horizon):
        raise ValueError("Boundary probe full-H flag disagrees with geometry")
    if result["executed_K_within_original_label"] != (distance >= executed):
        raise ValueError("Boundary probe prefix-K flag disagrees with geometry")
    return result


def valid_action_mask(
    is_pad: np.ndarray,
    *,
    frame_index: int,
    segment_end: int | None = None,
    horizon: int | None = None,
) -> np.ndarray:
    """Mask episode padding and, when given, offsets crossing a [start, end) segment."""
    is_pad = np.asarray(is_pad, dtype=np.bool_)
    if is_pad.ndim != 1:
        raise ValueError(f"Action padding mask must be one-dimensional, got {is_pad.shape}")
    length = len(is_pad) if horizon is None else min(horizon, len(is_pad))
    valid = ~is_pad[:length]
    if segment_end is not None:
        if segment_end <= frame_index:
            raise ValueError(f"Anchor frame {frame_index} is outside segment ending at {segment_end}")
        valid &= np.arange(length) < segment_end - frame_index
    return valid


def aggregate_errors(
    records: Sequence[ErrorRecord],
    *,
    action_slice: slice | None = None,
    position: int | None = None,
) -> dict[str, dict[str, float | int]]:
    """Aggregate masked error arrays without counting padding or boundary crossings."""
    result = {}
    selected_slice = action_slice or slice(None)
    for method in ERROR_METHODS:
        chunks = []
        valid_steps = 0
        for record in records:
            if method not in record.errors:
                continue
            error = np.asarray(record.errors[method])
            valid = np.asarray(record.valid, dtype=np.bool_)[: len(error)]
            if position is None:
                selected = error[: len(valid)][valid, selected_slice]
                valid_steps += int(valid.sum())
            elif position < len(valid) and valid[position]:
                selected = np.asarray(error[position, selected_slice]).reshape(1, -1)
                valid_steps += 1
            else:
                continue
            if selected.size:
                chunks.append(selected.reshape(-1))
        if not chunks:
            continue
        values = np.concatenate(chunks).astype(np.float64, copy=False)
        result[method] = {
            "l1": float(np.abs(values).mean()),
            "rmse": float(np.sqrt(np.square(values).mean())),
            "valid_action_steps": valid_steps,
            "scalar_values": int(values.size),
        }
    return result


def complete_prefix_records(records: Sequence[ErrorRecord], steps: int) -> list[ErrorRecord]:
    """Use an equal-length prefix, dropping samples without all requested targets."""
    if steps < 1:
        raise ValueError("Prefix length must be positive")
    return [
        ErrorRecord(record.sample, {name: values[:steps] for name, values in record.errors.items()},
                    record.valid[:steps])
        for record in records
        if len(record.valid) >= steps and bool(np.all(record.valid[:steps]))
        and all(len(values) >= steps for values in record.errors.values())
    ]


def comparable_error_metrics(records: Sequence[ErrorRecord]) -> dict[str, Any]:
    """Keep fixed-position and full 16-step estimates separate from masked means."""
    prefix = complete_prefix_records(records, 16)
    return {
        "first_action": {
            "metrics": aggregate_errors(records, position=0),
            "per_action_group": {
                name: aggregate_errors(records, position=0, action_slice=slice(start, end))
                for name, (start, end) in ACTION_GROUPS.items()
            },
        },
        "complete_prefix_16": {
            "samples": len(prefix),
            "metrics": aggregate_errors(prefix),
            "per_action_group": {
                name: aggregate_errors(prefix, action_slice=slice(start, end))
                for name, (start, end) in ACTION_GROUPS.items()
            },
        },
    }


def sample_error_diagnostics(record: ErrorRecord) -> dict[str, Any]:
    return {
        "metrics": aggregate_errors([record]),
        "per_action_group": {
            name: aggregate_errors([record], action_slice=slice(start, end))
            for name, (start, end) in ACTION_GROUPS.items()
        },
        **comparable_error_metrics([record]),
        "per_chunk_position": [
            {"position": position, "valid": bool(valid),
             "metrics": aggregate_errors([record], position=position)}
            for position, valid in enumerate(record.valid)
        ],
    }


def sha256_path(path: str | pathlib.Path) -> tuple[str, str]:
    """Hash a file, or a directory as sorted relative paths plus file contents."""
    source = pathlib.Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Cannot hash missing artifact: {source}")
    digest = hashlib.sha256()
    if source.is_file():
        with source.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), "file"

    files = sorted(candidate for candidate in source.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Cannot hash empty checkpoint directory: {source}")
    for candidate in files:
        digest.update(candidate.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        with candidate.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), "recursive_file_tree"


def artifact_record(path: str | pathlib.Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    digest, scope = sha256_path(path)
    return {
        "path": str(pathlib.Path(path).expanduser().resolve()),
        "sha256": digest,
        "sha256_scope": scope,
    }


def _scalar_int(sample: dict, key: str) -> int:
    if key not in sample:
        raise RuntimeError(f"LeRobot sample is missing required index field {key!r}")
    value = np.asarray(sample[key])
    if value.size != 1:
        raise RuntimeError(f"LeRobot sample field {key!r} is not scalar: {value.shape}")
    return int(value.item())


def paired_action_noise(seed: int, episode: int, frame: int, horizon: int, action_dim: int) -> np.ndarray:
    """Explicit float32 policy noise independent of checkpoint and sample order.

    The selected episode/frame IDs refer to one dataset root. Compare the same
    root and shape across runs, and verify the saved per-sample byte hashes.
    """
    values = (seed, episode, frame, horizon, action_dim)
    if any(type(value) is not int for value in values) or min(values[:3]) < 0 or min(values[3:]) < 1:
        raise ValueError("Noise requires nonnegative integer seed/IDs and positive integer shape")
    identity = f"b1k-openloop-noise-v1:{seed}:{episode}:{frame}:{horizon}:{action_dim}".encode()
    entropy = int.from_bytes(hashlib.sha256(identity).digest(), "big")
    rng = np.random.Generator(np.random.PCG64(entropy))
    return rng.standard_normal((horizon, action_dim), dtype=np.float32)


def input_content_sha256(value: Any) -> str:
    """Hash raw policy inputs with structure, shape and dtype included."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item):
                if not isinstance(key, str):
                    raise ValueError("Policy input dictionaries require string keys for hashing")
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}\0".encode())
            for child in item:
                visit(child)
        elif isinstance(item, str):
            raw = item.encode("utf-8")
            digest.update(f"str:{len(raw)}\0".encode())
            digest.update(raw)
        elif item is None:
            digest.update(b"none\0")
        else:
            array = np.asarray(item)
            if array.dtype.hasobject:
                raise ValueError("Object arrays cannot provide a stable policy input hash")
            digest.update(f"array:{array.dtype.str}:{array.shape}\0".encode())
            digest.update(np.ascontiguousarray(array).tobytes())

    visit(value)
    return digest.hexdigest()


def build_pi05(args, episode_ids: list[int] | None,
               *, sample_provider: Callable[[EvalSample], dict] | None = None):
    import openpi.policies.policy_config as _policy_config  # noqa: PLC0415
    import openpi.training.config as _config  # noqa: PLC0415
    import openpi.training.data_loader as _dl  # noqa: PLC0415
    import openpi.transforms as _tf  # noqa: PLC0415
    from openpi.shared import frame_memory  # noqa: PLC0415
    from openpi.shared.download import maybe_download  # noqa: PLC0415
    from openpi.training.frame_memory_data import read_manifest  # noqa: PLC0415

    cfg = _config.get_config("pi05_b1k")
    model = dataclasses.replace(
        cfg.model,
        train_subtask_prediction=args.subtask_conditioning == "provided",
        max_token_len=args.max_token_len,
    )
    checkpoint_dir = maybe_download(str(args.pi05_ckpt))
    memory = frame_memory.load(checkpoint_dir / "assets")
    cache_dir = getattr(args, "memory_cache_dir", None)
    if memory is not None:
        if cache_dir is None:
            raise ValueError("Random-frame evaluation of a memory checkpoint requires --memory-cache-dir.")
        manifest = read_manifest(cache_dir)
        for key in ("encoder_sha256", "dataset_sha256"):
            if manifest[key] != memory.get(key):
                raise ValueError(f"Evaluation memory cache {key} disagrees with the checkpoint.")
        model = dataclasses.replace(model, framesamp_modul=True, memory_num_frames=memory["num_frames"],
                                    memory_grid_size=memory["grid_size"], memory_frame_stride=memory["frame_stride"],
                                    memory_camera=memory["camera"])
    elif cache_dir is not None:
        raise ValueError("--memory-cache-dir requires a FrameSamp checkpoint.")

    if args.horizon > model.action_horizon:
        raise ValueError(f"--horizon cannot exceed the model action horizon ({model.action_horizon})")

    base = dataclasses.replace(cfg.data.base_config, dataset_root=args.dataset, episodes_index=episode_ids)
    # A dataset identifier need not be the name used when the checkpoint's
    # normalization assets were saved. Always resolve statistics inside this
    # checkpoint, before loading its model or opening the dataset.
    asset_kwargs = {"assets_dir": str(checkpoint_dir / "assets")}
    norm_asset_id = getattr(args, "norm_asset_id", None)
    if norm_asset_id is not None:
        asset_path = pathlib.PurePosixPath(norm_asset_id)
        if not norm_asset_id.strip() or asset_path.is_absolute() or ".." in asset_path.parts:
            raise ValueError("--norm-asset-id must be a nonempty relative checkpoint asset ID")
        asset_kwargs["asset_id"] = norm_asset_id
    data_factory = dataclasses.replace(
        cfg.data,
        repo_id=args.repo_id,
        assets=dataclasses.replace(cfg.data.assets, **asset_kwargs),
        base_config=base,
        episodes_index=episode_ids,
        frame_stride=1,
        subtask_prompts_path=str(args.subtask_prompts) if args.subtask_prompts else None,
        subtask_whole_task_prob=0.0,
        memory_cache_dir=str(cache_dir) if cache_dir else None,
    )
    cfg = dataclasses.replace(cfg, data=data_factory, model=model)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    if data_config.norm_stats is None:
        raise ValueError(
            f"No normalization statistics for asset {data_config.asset_id!r} under {checkpoint_dir / 'assets'}. "
            "Set --norm-asset-id to the normalization asset saved in this checkpoint; --repo-id identifies the dataset."
        )
    args._resolved_norm_asset_id = data_config.asset_id
    args._use_quantile_norm = data_config.use_quantile_norm
    args._norm_stats_path = checkpoint_dir / "assets" / data_config.asset_id / "norm_stats.json"

    dataset = _dl.create_b1k_dataset(data_config, cfg.model.action_horizon) if sample_provider is None else None
    repack = _tf.compose(data_config.repack_transforms.inputs)
    policy = _policy_config.create_trained_policy(cfg, args.pi05_ckpt, norm_stats=data_config.norm_stats)

    def get_sample(key: EvalSample):
        sample = dataset[key.dataset_index] if sample_provider is None else sample_provider(key)
        observed = (
            _scalar_int(sample, "episode_index"),
            _scalar_int(sample, "frame_index"),
            _scalar_int(sample, "index"),
        )
        expected = (key.episode_index, key.frame_index, key.global_frame_index)
        if observed != expected:
            raise RuntimeError(
                "Filtered LeRobot row mapping mismatch: "
                f"expected episode/frame/global={expected}, got {observed} at relative row {key.dataset_index}"
            )

        if args.subtask_conditioning == "provided" and sample.get("subtask") != key.subtask:
            raise RuntimeError("Joint evaluation sample did not receive the current subtask")

        packed = repack(sample)
        is_pad = np.asarray(sample[f"{data_config.action_sequence_keys[0]}_is_pad"], dtype=np.bool_)
        valid = valid_action_mask(
            is_pad,
            frame_index=key.frame_index,
            segment_end=key.segment_end,
            horizon=args.horizon,
        )
        if not valid.any():
            raise RuntimeError(f"Selected anchor has no valid action targets: {key}")
        item = {
            "policy_input": {
                name: np.asarray(value) if hasattr(value, "shape") else value for name, value in packed.items()
            },
            "gt_action": np.asarray(sample[data_config.action_sequence_keys[0]], dtype=np.float64)[: args.horizon],
            "valid": valid,
            "hold_action": hold_from_state(np.asarray(sample["observation.state"], dtype=np.float64)),
        }
        item["policy_input_sha256"] = input_content_sha256(item["policy_input"])
        item["ground_truth_sha256"] = input_content_sha256(item["gt_action"])
        noise_seed = getattr(args, "action_noise_seed", None)
        if noise_seed is not None:
            noise = paired_action_noise(noise_seed, key.episode_index, key.frame_index,
                                        model.action_horizon, model.action_dim)
            item["action_noise"] = noise
            item["action_noise_sha256"] = hashlib.sha256(noise.astype("<f4", copy=False).tobytes()).hexdigest()
        return item

    def predict(item):
        kwargs = {"noise": item["action_noise"]} if "action_noise" in item else {}
        result = policy.infer(item["policy_input"], **kwargs)
        if args.subtask_conditioning == "provided":
            if result.get("subtask_source") != "provided":
                raise AssertionError(
                    "Provided-subtask evaluation requires every policy output to have subtask_source='provided'; "
                    f"got {result.get('subtask_source')!r}"
                )
        elif "subtask_source" in result:
            raise AssertionError("A non-joint policy unexpectedly returned subtask_source")
        return np.asarray(result["actions"], dtype=np.float64)

    return predict, get_sample


def _sample_record(sample: EvalSample, names: dict[int, str], valid_steps: int) -> dict[str, Any]:
    result = {
        "task_index": sample.task_index,
        "task": names[sample.task_index],
        "subtask": sample.subtask,
        "episode_index": sample.episode_index,
        "frame_index": sample.frame_index,
        "global_frame_index": sample.global_frame_index,
        "filtered_dataset_index": sample.dataset_index,
        "segment_end": sample.segment_end,
        "valid_action_steps": valid_steps,
    }
    if sample.probe_metadata is not None:
        result["probe_metadata"] = sample.probe_metadata
    return result


def _group_records(records: Sequence[ErrorRecord], key: Callable[[ErrorRecord], Any]) -> dict[Any, list[ErrorRecord]]:
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)
    return grouped


def main(argv: Sequence[str] | None = None, *, build_policy: Callable | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="pi05", choices=["pi05"])
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--samples-per-subtask", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-token-len",
        type=int,
        help="language token budget (default: 512 for provided joint subtasks, 200 for whole task)",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--memory-cache-dir", type=pathlib.Path,
                        help="matching frozen vision cache; required for FrameSamp checkpoints")
    parser.add_argument(
        "--repo-id",
        default="behavior-1k/2026-challenge-demos",
        help="dataset identity; also the default normalization asset ID unless --norm-asset-id is set",
    )
    parser.add_argument("--norm-asset-id",
                        help="normalization asset directory under this checkpoint's assets/; independent of dataset repo ID")
    parser.add_argument(
        "--pi05-ckpt",
        default=(
            "/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/openpi/outputs/checkpoints/"
            "pi05_b1k/pi05_b1k_full_20260803_162620/49999"
        ),
    )
    parser.add_argument(
        "--subtask-conditioning",
        choices=("none", "provided"),
        default="none",
        help="none=whole task; provided=joint model with an explicit oracle subtask",
    )
    parser.add_argument(
        "--subtask-prompts",
        type=pathlib.Path,
        help="strict half-open subtask segment parquet (required for provided)",
    )
    parser.add_argument(
        "--episodes-file",
        type=pathlib.Path,
        help="whitespace-delimited ORIGINAL episode_index values; does not establish a held-out split",
    )
    parser.add_argument(
        "--split-label",
        choices=("in_sample", "held_out", "unspecified"),
        default="unspecified",
        help="user-declared split status; training-episode overlap is not checked by this script",
    )
    parser.add_argument(
        "--sample-keys-file",
        type=pathlib.Path,
        help="prior result JSON whose exact episode/frame keys and valid horizons should be replayed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stop after N selected samples (smoke test)",
    )
    parser.add_argument("--horizon", type=int, default=EVAL_HORIZON, help="action-chunk steps to score")
    parser.add_argument("--action-noise-seed", type=int,
                        help="explicit episode/frame-keyed policy noise for paired checkpoint comparisons; "
                             "independent of --seed, which selects frames")
    parser.add_argument(
        "--save-raw-actions", action="store_true",
        help="save finite physical action arrays after evaluation horizon truncation (no images)",
    )
    parser.add_argument("--dataset", default=DATASET)
    args = parser.parse_args(argv)

    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.action_noise_seed is not None and args.action_noise_seed < 0:
        parser.error("--action-noise-seed must be nonnegative")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.samples_per_task < 1 or args.samples_per_subtask < 1:
        parser.error("sample counts must be positive")
    if args.max_token_len is None:
        args.max_token_len = 512 if args.subtask_conditioning == "provided" else 200
    if args.max_token_len < 1:
        parser.error("--max-token-len must be positive")
    if args.split_label == "held_out" and args.episodes_file is None:
        parser.error("--split-label held_out requires --episodes-file")
    if args.subtask_conditioning != "none" and args.subtask_prompts is None:
        parser.error("--subtask-prompts is required for provided conditioning")
    if args.subtask_conditioning == "none" and args.subtask_prompts is not None:
        parser.error("--subtask-prompts requires explicit --subtask-conditioning provided")
    if args.sample_keys_file is not None and (args.subtask_conditioning == "none" or args.episodes_file is None):
        parser.error("--sample-keys-file requires subtask conditioning and --episodes-file")

    try:
        requested_episode_ids = read_episode_ids(args.episodes_file) if args.episodes_file else None
        metadata = filter_episode_meta(load_episode_meta(args.dataset), requested_episode_ids)
        names = task_names(args.dataset)
        missing_task_names = sorted({episode.task_index for episode in metadata} - names.keys())
        if missing_task_names:
            raise ValueError(f"Missing task names for task indices: {missing_task_names[:10]}")

        segments = None
        if args.subtask_conditioning != "none":
            from openpi.training.subtask_data import (
                SubtaskFromSegments,  # noqa: PLC0415
            )

            segments = SubtaskFromSegments.from_parquet(
                args.subtask_prompts,
                episodes_index=[episode.episode_index for episode in metadata],
            )
        if args.sample_keys_file is not None:
            samples = replay_samples(args.sample_keys_file, metadata, segments, names=names,
                                     repo_id=args.repo_id, horizon=args.horizon,
                                     action_noise_seed=args.action_noise_seed)
        else:
            per_group = args.samples_per_task if segments is None else args.samples_per_subtask
            samples = select_samples(metadata, per_group, args.seed, segments)
        if args.limit:
            samples = samples[: args.limit]
        if not samples:
            raise ValueError("No evaluation samples selected")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    canonical_subtasks = {(sample.task_index, sample.subtask) for sample in samples if sample.subtask is not None}
    print(
        f"evaluating {len(samples)} samples over {len({sample.task_index for sample in samples})} tasks"
        + (f" and {len(canonical_subtasks)} canonical subtasks" if canonical_subtasks else ""),
        flush=True,
    )

    try:
        policy_builder = build_pi05 if build_policy is None else build_policy
        predict, get_sample = policy_builder(args, requested_episode_ids)
    except ValueError as error:
        parser.error(str(error))

    records = []
    evaluated_samples = []
    started = time.time()
    for index, sample in enumerate(samples):
        item = get_sample(sample)
        ground_truth = item["gt_action"]
        prediction = predict(item)
        horizon = min(len(ground_truth), len(prediction), len(item["valid"]))
        if horizon == 0:
            raise RuntimeError(f"Policy returned an empty action chunk for {sample}")
        ground_truth = ground_truth[:horizon]
        prediction = prediction[:horizon]
        valid = item["valid"][:horizon]
        if ground_truth.shape != prediction.shape:
            raise RuntimeError(
                f"Action shape mismatch for episode={sample.episode_index}, frame={sample.frame_index}: "
                f"ground truth {ground_truth.shape}, prediction {prediction.shape}"
            )
        if not np.isfinite(prediction).all() or not np.isfinite(ground_truth).all():
            raise RuntimeError(f"Nonfinite prediction or target for {sample}")
        hold = np.tile(item["hold_action"], (horizon, 1))
        errors = {
            "model": prediction - ground_truth,
            "baseline_hold": hold - ground_truth,
            "baseline_zero": -ground_truth,
        }
        record = ErrorRecord(sample=sample, errors=errors, valid=valid)
        records.append(record)
        sample_record = _sample_record(sample, names, int(valid.sum()))
        sample_record["error_diagnostics"] = sample_error_diagnostics(record)
        for digest_field in ("action_noise_sha256", "policy_input_sha256", "ground_truth_sha256"):
            if digest_field in item:
                sample_record[digest_field] = item[digest_field]
        if args.save_raw_actions:
            raw_arrays = {
                "prediction": np.asarray(prediction),
                "ground_truth": np.asarray(ground_truth),
                "valid_mask": np.asarray(valid, dtype=np.bool_),
                "hold_action": np.asarray(hold),
            }
            if any(not np.isfinite(value).all() for value in raw_arrays.values()):
                raise RuntimeError(f"Nonfinite raw action export for {sample}")
            sample_record["raw_actions"] = {
                "schema_version": 1,
                **{name: value.tolist() for name, value in raw_arrays.items()},
                # JSON does not retain dtype; readers need it to reproduce the content hash.
                "dtypes": {name: value.dtype.str for name, value in raw_arrays.items()},
                "sha256": input_content_sha256(raw_arrays),
            }
        evaluated_samples.append(sample_record)
        if (index + 1) % 20 == 0:
            print(
                f"  {index + 1}/{len(samples)}  {time.time() - started:.0f}s",
                flush=True,
            )

    if not records:
        raise RuntimeError("No samples had valid action targets")

    by_task = _group_records(
        records,
        lambda record: (record.sample.task_index, names[record.sample.task_index]),
    )
    per_task = [
        {
            "task_index": task_index,
            "task": task,
            "samples": len(group_records),
            "metrics": aggregate_errors(group_records),
        }
        for (task_index, task), group_records in sorted(by_task.items())
    ]

    per_subtask = []
    if segments is not None:
        by_subtask = _group_records(
            records,
            lambda record: (
                record.sample.task_index,
                names[record.sample.task_index],
                record.sample.subtask,
            ),
        )
        per_subtask = [
            {
                "task_index": task_index,
                "task": task,
                "subtask": subtask,
                "samples": len(group_records),
                "metrics": aggregate_errors(group_records),
            }
            for (task_index, task, subtask), group_records in sorted(by_subtask.items())
        ]

    per_action_group = {
        group: aggregate_errors(records, action_slice=slice(start, end))
        for group, (start, end) in ACTION_GROUPS.items()
    }
    per_chunk_position = []
    for position in range(args.horizon):
        valid_samples = sum(position < len(record.valid) and bool(record.valid[position]) for record in records)
        per_chunk_position.append(
            {
                "position": position,
                "valid_samples": valid_samples,
                "metrics": aggregate_errors(records, position=position),
            }
        )

    by_bucket = _group_records(records, lambda record: (record.sample.probe_metadata or {}).get("bucket"))
    per_probe_bucket = [
        {"bucket": bucket, "samples": len(bucket_records),
         "masked_metrics": aggregate_errors(bucket_records),
         **comparable_error_metrics(bucket_records)}
        for bucket, bucket_records in sorted(by_bucket.items(), key=lambda item: item[0] or "")
        if bucket is not None
    ]

    split_notes = {
        "in_sample": "In-sample demonstration action error",
        "held_out": "User-declared held-out demonstration action error",
        "unspecified": "Demonstration action error with unspecified train/evaluation split",
    }
    result = {
        "model": args.model,
        "repo_id": args.repo_id,
        "normalization": {
            "asset_id": getattr(args, "_resolved_norm_asset_id", args.norm_asset_id or args.repo_id),
            "use_quantile_norm": getattr(args, "_use_quantile_norm", None),
        },
        "dataset": str(pathlib.Path(args.dataset).expanduser().resolve()),
        "conditioning_mode": args.subtask_conditioning,
        "max_token_len": args.max_token_len,
        "horizon": args.horizon,
        "samples": len(records),
        "seed": args.seed,
        "action_noise": {
            "seed": args.action_noise_seed,
            "algorithm": "b1k-openloop-noise-v1/sha256/PCG64/float32" if args.action_noise_seed is not None else None,
            "numpy_version": np.__version__,
            "scope": "same dataset root, episode/frame and model action shape; not all runtime randomness",
        },
        "sampling": {
            "samples_per_task": args.samples_per_task if segments is None and args.sample_keys_file is None else None,
            "samples_per_subtask": (
                args.samples_per_subtask if segments is not None and args.sample_keys_file is None else None
            ),
            "replayed_sample_keys": args.sample_keys_file is not None,
            "tasks": len(by_task),
            "canonical_task_subtasks": len(per_subtask),
            "allowed_episodes": len(metadata),
            "episode_filter": args.episodes_file is not None,
            "held_out_episode_filter": args.split_label == "held_out",
            "split_label": args.split_label,
            "training_overlap_checked": False,
        },
        "artifacts": {
            "checkpoint": artifact_record(args.pi05_ckpt),
            "segments_file": artifact_record(args.subtask_prompts),
            "episodes_file": artifact_record(args.episodes_file),
            "sample_keys_file": artifact_record(args.sample_keys_file),
            "norm_stats": artifact_record(getattr(args, "_norm_stats_path", None)),
            "memory_cache_manifest": artifact_record(args.memory_cache_dir / "manifest.json") if args.memory_cache_dir else None,
        },
        "episode_ids": sorted(requested_episode_ids) if requested_episode_ids is not None else None,
        "note": (
            f"{split_notes[args.split_label]}; training-episode overlap is not checked by this script. "
            "Not closed-loop success or BEHAVIOR q_score."
        ),
        "error_metric_notes": {
            "masked_metrics": "Means over available current-subtask targets; unequal valid lengths are not directly comparable across buckets.",
            "first_action": "Position zero only; common target position for all four boundary-distance buckets.",
            "complete_prefix_16": "Only samples with all first 16 current-subtask targets valid; last_1 probes are excluded.",
            "probe_metadata": "Whitelisted frozen-index claims; row indices, label, source segment bounds and task name checked against current data.",
            "policy_input_sha256": "Hash of raw repacked input including images, state and prompts; does not by itself verify checkpoint-dependent preprocessing.",
        },
        "sample_keys": evaluated_samples,
        "overall": aggregate_errors(records),
        "comparable_errors": comparable_error_metrics(records),
        "per_probe_bucket": per_probe_bucket,
        "per_task": per_task,
        "per_subtask": per_subtask,
        "per_action_group": per_action_group,
        "per_chunk_position": per_chunk_position,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2))
    report(result)
    worst = sorted(per_task, key=lambda item: -item["metrics"]["model"]["l1"])[:10]
    print("\nworst tasks by L1:")
    for row in worst:
        print(f"  {row['metrics']['model']['l1']:.4f}  {row['task']}")
    print(f"\nwrote {args.out}")


def report(result: dict) -> None:
    """Print concise action-group and chunk-position diagnostics."""
    print(f"\n=== per action group (horizon {result['horizon']}) ===")
    print(f"{'group':16s} {'model L1':>10s} {'model RMSE':>11s} {'hold L1':>10s} {'hold RMSE':>11s}")
    for group, metrics in result["per_action_group"].items():
        model = metrics["model"]
        hold = metrics["baseline_hold"]
        print(f"{group:16s} {model['l1']:10.4f} {model['rmse']:11.4f} {hold['l1']:10.4f} {hold['rmse']:11.4f}")

    print("\n=== error across the action chunk ===")
    print(f"{'step':>5s} {'valid':>7s} {'model L1':>10s} {'model RMSE':>11s} {'hold L1':>10s}")
    positions = result["per_chunk_position"]
    for row in positions:
        position = row["position"]
        if position % 4 != 0 and position != len(positions) - 1:
            continue
        model = row["metrics"].get("model")
        hold = row["metrics"].get("baseline_hold")
        if model is None or hold is None:
            print(f"{position:5d} {row['valid_samples']:7d} {'n/a':>10s} {'n/a':>11s} {'n/a':>10s}")
        else:
            print(
                f"{position:5d} {row['valid_samples']:7d} {model['l1']:10.4f} {model['rmse']:11.4f} {hold['l1']:10.4f}"
            )


def hold_from_state(state: np.ndarray) -> np.ndarray:
    """Baseline: keep the current joint configuration and zero base velocity."""
    action = np.zeros(23)
    for group, (action_start, action_end) in ACTION_GROUPS.items():
        if group in STATE_FOR_ACTION:
            state_start, state_end = STATE_FOR_ACTION[group]
            action[action_start:action_end] = state[state_start:state_end][: action_end - action_start]
    return action


if __name__ == "__main__":
    main()
