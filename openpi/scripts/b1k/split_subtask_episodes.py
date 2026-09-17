"""Create deterministic task-stratified episode splits without task-instance leakage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

_REQUIRED_COLUMNS = ("episode_index", "task_index", "task_instance_id")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_retained_episode_ids(path: Path) -> list[int]:
    try:
        episode_ids = [int(token) for token in path.read_text().split()]
    except ValueError as error:
        raise ValueError(f"Retained episode file contains a non-integer ID: {path}") from error
    if not episode_ids:
        raise ValueError(f"Retained episode file is empty: {path}")
    duplicates = sorted(episode_id for episode_id, count in Counter(episode_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate requested episode IDs: {duplicates[:10]}")
    return episode_ids


def read_episode_metadata(dataset_root: Path) -> tuple[dict[int, dict], list[Path]]:
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise ValueError(f"No episode metadata found under {dataset_root / 'meta' / 'episodes'}.")

    metadata = {}
    for path in paths:
        missing = set(_REQUIRED_COLUMNS) - set(pq.read_schema(path).names)
        if missing:
            raise ValueError(f"Episode metadata {path} is missing columns: {sorted(missing)}")
        for row in pq.read_table(path, columns=list(_REQUIRED_COLUMNS)).to_pylist():
            episode_id = int(row["episode_index"])
            if episode_id in metadata:
                raise ValueError(f"Duplicate episode_index={episode_id} in episode metadata.")
            metadata[episode_id] = row
    return metadata, paths


def _group_sort_key(seed: int, task_index: int, task_instance_id) -> tuple[bytes, str]:
    canonical_id = json.dumps(task_instance_id, sort_keys=True, separators=(",", ":"))
    payload = f"{seed}\0{task_index}\0{canonical_id}".encode()
    return hashlib.sha256(payload).digest(), canonical_id


def validate_episode_split(train_ids: list[int], val_ids: list[int], retained_episode_ids: list[int]) -> None:
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Duplicate episode IDs detected within a split side.")
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"Episode split overlap detected: {sorted(overlap)[:10]}")
    if set(train_ids) | set(val_ids) != set(retained_episode_ids):
        raise ValueError("Split does not contain every retained episode exactly once.")
    if not train_ids or not val_ids:
        raise ValueError("Episode split has an empty train or validation side.")


def split_episode_groups(
    metadata: dict[int, dict],
    retained_episode_ids: list[int],
    task_indices: list[int],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int], list[dict], dict[str, dict]]:
    """Split retained IDs by task-instance groups and return IDs plus audit statistics."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    if not task_indices:
        raise ValueError("At least one task index is required.")
    if len(set(task_indices)) != len(task_indices):
        raise ValueError("Duplicate task indices are not allowed.")
    if len(set(retained_episode_ids)) != len(retained_episode_ids):
        raise ValueError("Duplicate requested episode IDs are not allowed.")

    unknown_ids = sorted(set(retained_episode_ids) - metadata.keys())
    if unknown_ids:
        raise ValueError(f"Unknown requested episode IDs: {unknown_ids[:10]}")

    requested_tasks = set(task_indices)
    selected_tasks = {int(metadata[episode_id]["task_index"]) for episode_id in retained_episode_ids}
    unexpected_tasks = sorted(selected_tasks - requested_tasks)
    if unexpected_tasks:
        raise ValueError(f"Retained episodes include unrequested task indices: {unexpected_tasks}")
    missing_tasks = sorted(requested_tasks - selected_tasks)
    if missing_tasks:
        raise ValueError(f"Requested task indices have no retained episodes: {missing_tasks}")

    groups_by_task = {task_index: {} for task_index in task_indices}
    for episode_id in retained_episode_ids:
        row = metadata[episode_id]
        task_index = int(row["task_index"])
        group_id = row["task_instance_id"]
        missing_group_id = group_id is None or (isinstance(group_id, str) and not group_id.strip())
        missing_group_id |= isinstance(group_id, float) and math.isnan(group_id)
        if missing_group_id:
            raise ValueError(f"Missing task_instance_id for episode_index={episode_id}.")
        groups_by_task[task_index].setdefault(group_id, []).append(episode_id)

    train_ids = []
    val_ids = []
    assignments = []
    per_task = {}
    for task_index in sorted(task_indices):
        groups = groups_by_task[task_index]
        if len(groups) < 2:
            raise ValueError(f"Task {task_index} needs at least two task-instance groups for a nonempty split.")
        ordered_group_ids = sorted(groups, key=lambda group_id: _group_sort_key(seed, task_index, group_id))
        val_group_count = min(len(groups) - 1, max(1, math.floor(len(groups) * val_fraction + 0.5)))
        val_groups = set(ordered_group_ids[:val_group_count])

        task_train_ids = sorted(
            episode_id
            for group_id, episode_ids in groups.items()
            if group_id not in val_groups
            for episode_id in episode_ids
        )
        task_val_ids = sorted(
            episode_id
            for group_id, episode_ids in groups.items()
            if group_id in val_groups
            for episode_id in episode_ids
        )
        if not task_train_ids or not task_val_ids:
            raise ValueError(f"Task {task_index} produced an empty train or validation side.")
        train_ids.extend(task_train_ids)
        val_ids.extend(task_val_ids)

        assignments.extend(
            {
                "task_index": task_index,
                "task_instance_id": group_id,
                "split": "val" if group_id in val_groups else "train",
                "episode_indices": sorted(groups[group_id]),
            }
            for group_id in sorted(groups, key=lambda value: json.dumps(value, sort_keys=True))
        )
        per_task[str(task_index)] = {
            "episodes": len(task_train_ids) + len(task_val_ids),
            "groups": len(groups),
            "train_episodes": len(task_train_ids),
            "train_groups": len(groups) - len(val_groups),
            "val_episodes": len(task_val_ids),
            "val_groups": len(val_groups),
        }

    train_ids.sort()
    val_ids.sort()
    validate_episode_split(train_ids, val_ids, retained_episode_ids)

    side_by_group = {}
    for assignment in assignments:
        key = (assignment["task_index"], assignment["task_instance_id"])
        previous = side_by_group.setdefault(key, assignment["split"])
        if previous != assignment["split"]:
            raise ValueError(f"Task-instance group leakage detected for {key}.")
    return train_ids, val_ids, assignments, per_task


def create_split(
    dataset_root: Path,
    retained_episode_file: Path,
    task_indices: list[int],
    output_dir: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> dict:
    """Validate inputs, create the split files, and return the written manifest."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")

    retained_episode_ids = read_retained_episode_ids(retained_episode_file)
    metadata, metadata_paths = read_episode_metadata(dataset_root)
    train_ids, val_ids, assignments, per_task = split_episode_groups(
        metadata,
        retained_episode_ids,
        task_indices,
        val_fraction=val_fraction,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_episodes.txt"
    val_path = output_dir / "val_episodes.txt"
    manifest_path = output_dir / "split_manifest.json"
    train_path.write_text("".join(f"{episode_id}\n" for episode_id in train_ids))
    val_path.write_text("".join(f"{episode_id}\n" for episode_id in val_ids))

    manifest = {
        "schema_version": 1,
        "inputs": {
            "dataset_root": str(dataset_root.resolve()),
            "retained_episode_file": str(retained_episode_file.resolve()),
            "retained_episode_file_sha256": _file_sha256(retained_episode_file),
            "episode_metadata_sha256": _metadata_sha256(metadata_paths, dataset_root),
        },
        "parameters": {
            "task_indices": sorted(task_indices),
            "val_fraction": val_fraction,
            "seed": seed,
        },
        "outputs": {
            "train_episodes": {
                "path": train_path.name,
                "sha256": _file_sha256(train_path),
                "episodes": len(train_ids),
            },
            "val_episodes": {
                "path": val_path.name,
                "sha256": _file_sha256(val_path),
                "episodes": len(val_ids),
            },
        },
        "stats": {
            "episodes": len(retained_episode_ids),
            "groups": len(assignments),
            "tasks": len(task_indices),
            "train_episodes": len(train_ids),
            "train_groups": sum(stats["train_groups"] for stats in per_task.values()),
            "val_episodes": len(val_ids),
            "val_groups": sum(stats["val_groups"] for stats in per_task.values()),
            "per_task": per_task,
        },
        "group_assignments": assignments,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--retained-episode-file", type=Path, required=True)
    parser.add_argument("--task-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = create_split(
        args.dataset_root,
        args.retained_episode_file,
        args.task_indices,
        args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest["stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
