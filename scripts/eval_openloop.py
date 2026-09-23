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
from collections.abc import Callable, Collection, Sequence
import dataclasses
import hashlib
import json
import pathlib
import time
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
) -> list[EvalSample]:
    """Reuse exact frames and valid horizons while resolving labels from the current segment table."""
    payload = json.loads(path.read_text())
    rows = payload.get("sample_keys") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Sample key file has no non-empty sample_keys list: {path}")

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
        reference_segment_end = frame_index + valid_action_steps
        if current_segment_end < reference_segment_end:
            raise ValueError(
                f"Current segment ends before the replayed valid horizon for episode/frame={key}: "
                f"{current_segment_end} < {reference_segment_end}"
            )
        samples.append(
            EvalSample(
                task_index=episode.task_index,
                episode_index=episode_index,
                frame_index=frame_index,
                global_frame_index=episode.dataset_from_index + frame_index,
                dataset_index=episode.relative_from_index + frame_index,
                subtask=subtask,
                segment_end=reference_segment_end,
            )
        )
    return samples


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


def build_pi05(args, episode_ids: list[int] | None):
    import openpi.policies.policy_config as _policy_config  # noqa: PLC0415
    import openpi.training.config as _config  # noqa: PLC0415
    import openpi.training.data_loader as _dl  # noqa: PLC0415
    import openpi.transforms as _tf  # noqa: PLC0415

    cfg = _config.get_config("pi05_b1k")
    model = dataclasses.replace(
        cfg.model,
        train_subtask_prediction=args.subtask_conditioning == "provided",
    )

    if args.horizon > model.action_horizon:
        raise ValueError(f"--horizon cannot exceed the model action horizon ({model.action_horizon})")

    base = dataclasses.replace(cfg.data.base_config, dataset_root=args.dataset, episodes_index=episode_ids)
    data_factory = dataclasses.replace(
        cfg.data,
        repo_id=args.repo_id,
        base_config=base,
        episodes_index=episode_ids,
        frame_stride=1,
        subtask_prompts_path=str(args.subtask_prompts) if args.subtask_prompts else None,
        subtask_whole_task_prob=0.0,
    )
    cfg = dataclasses.replace(cfg, data=data_factory, model=model)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

    dataset = _dl.create_b1k_dataset(data_config, cfg.model.action_horizon)
    repack = _tf.compose(data_config.repack_transforms.inputs)
    policy = _policy_config.create_trained_policy(cfg, args.pi05_ckpt)

    def get_sample(key: EvalSample):
        sample = dataset[key.dataset_index]
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
        return {
            "policy_input": {
                name: np.asarray(value) if hasattr(value, "shape") else value for name, value in packed.items()
            },
            "gt_action": np.asarray(sample[data_config.action_sequence_keys[0]], dtype=np.float64)[: args.horizon],
            "valid": valid,
            "hold_action": hold_from_state(np.asarray(sample["observation.state"], dtype=np.float64)),
        }

    def predict(item):
        result = policy.infer(item["policy_input"])
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
    return {
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


def _group_records(records: Sequence[ErrorRecord], key: Callable[[ErrorRecord], Any]) -> dict[Any, list[ErrorRecord]]:
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="pi05", choices=["pi05"])
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--samples-per-subtask", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="behavior-1k/2026-challenge-demos",
        help="checkpoint asset ID used to load normalization statistics",
    )
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
        help="whitespace-delimited held-out ORIGINAL episode_index values",
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
    parser.add_argument("--dataset", default=DATASET)
    args = parser.parse_args()

    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.samples_per_task < 1 or args.samples_per_subtask < 1:
        parser.error("sample counts must be positive")
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
            from openpi.training.subtask_data import SubtaskFromSegments  # noqa: PLC0415

            segments = SubtaskFromSegments.from_parquet(
                args.subtask_prompts,
                episodes_index=[episode.episode_index for episode in metadata],
            )
        if args.sample_keys_file is not None:
            samples = replay_samples(args.sample_keys_file, metadata, segments)
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
        predict, get_sample = build_pi05(args, requested_episode_ids)
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
        hold = np.tile(item["hold_action"], (horizon, 1))
        errors = {
            "model": prediction - ground_truth,
            "baseline_hold": hold - ground_truth,
            "baseline_zero": -ground_truth,
        }
        records.append(ErrorRecord(sample=sample, errors=errors, valid=valid))
        evaluated_samples.append(_sample_record(sample, names, int(valid.sum())))
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

    held_out = args.episodes_file is not None
    result = {
        "model": args.model,
        "repo_id": args.repo_id,
        "dataset": str(pathlib.Path(args.dataset).expanduser().resolve()),
        "conditioning_mode": args.subtask_conditioning,
        "horizon": args.horizon,
        "samples": len(records),
        "seed": args.seed,
        "sampling": {
            "samples_per_task": args.samples_per_task if segments is None and args.sample_keys_file is None else None,
            "samples_per_subtask": (
                args.samples_per_subtask if segments is not None and args.sample_keys_file is None else None
            ),
            "replayed_sample_keys": args.sample_keys_file is not None,
            "tasks": len(by_task),
            "canonical_task_subtasks": len(per_subtask),
            "allowed_episodes": len(metadata),
            "held_out_episode_filter": held_out,
        },
        "artifacts": {
            "checkpoint": artifact_record(args.pi05_ckpt),
            "segments_file": artifact_record(args.subtask_prompts),
            "episodes_file": artifact_record(args.episodes_file),
            "sample_keys_file": artifact_record(args.sample_keys_file),
        },
        "episode_ids": sorted(requested_episode_ids) if requested_episode_ids is not None else None,
        "note": (
            "Held-out demonstration action error, not closed-loop success or BEHAVIOR q_score."
            if held_out
            else "Demonstration action error on all episodes (legacy mode; not held out), not closed-loop success or "
            "BEHAVIOR q_score."
        ),
        "sample_keys": evaluated_samples,
        "overall": aggregate_errors(records),
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
