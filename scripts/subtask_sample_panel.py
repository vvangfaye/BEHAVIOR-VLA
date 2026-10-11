#!/usr/bin/env python3
"""Build and validate deterministic single-subtask evaluation panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASK_INDEX_TO_NAME = {
    14: "carrying_in_groceries",
    56: "make_rose_centerpieces",
}
SELECTION_VERSION = "subtask_closed_loop_panel_v1"
MIN_SEGMENT_FRAMES = 32
FROZEN_V1_PANEL_FINGERPRINT = (
    "42f0b8d946f54015d796641237ad244c1f508d95c14078f8482588e043157d40"
)
FROZEN_PRODUCTION_ANCHORS = (
    (
        "carrying_in_groceries-instance-24-episode-2823-segment-15-interval-0",
        "0008647c892bee4bc221435b7544c2d8c0ac9f8abd87f433402b9cb1c4801992",
    ),
    (
        "carrying_in_groceries-instance-174-episode-2905-segment-7-interval-0",
        "00246339a3380ca6f3d509b5dff89dfd414780ae1b340d7955d87a81e9d87ded",
    ),
    (
        "carrying_in_groceries-instance-147-episode-2881-segment-4-interval-0",
        "002a4ee4d5ed2ad11c0c4f11146bb9c38c8f746dde6e2b3cc8783071c84d8310",
    ),
    (
        "carrying_in_groceries-instance-248-episode-2972-segment-18-interval-0",
        "002dc1dfa5812cab84fbfddd180fac9d5877b02e476843927bc54b44e683701d",
    ),
    (
        "carrying_in_groceries-instance-225-episode-2951-segment-9-interval-0",
        "00465f7b35daa8d157bc3e255da180e10b9a96c565b5cdd4cb0291c2faf1575e",
    ),
    (
        "carrying_in_groceries-instance-84-episode-2843-segment-13-interval-0",
        "0056c7a26c3f09b52281e51abafb67982d740cd8bb789d76ba1ef75a5774c581",
    ),
    (
        "carrying_in_groceries-instance-188-episode-2919-segment-10-interval-0",
        "008a62abb430a842f09f07900087fc78f34fb11470eba8d6783c6a4129a79fcc",
    ),
    (
        "carrying_in_groceries-instance-222-episode-2948-segment-6-interval-0",
        "00ad11234468b0b64f6cbb48321fbd11315318ac8332b989f6b6c7c67f7006e5",
    ),
    (
        "carrying_in_groceries-instance-169-episode-2901-segment-12-interval-0",
        "00b07c765605bbb5be5e1c332f9ae0ca1a6a000a1ea89aa8589b5efa453a1880",
    ),
    (
        "carrying_in_groceries-instance-10-episode-2809-segment-17-interval-0",
        "00bc3066a427f9a0def1a5327aa7d1eb6cd22efd2d549e5ff5ece969d503c2a0",
    ),
    (
        "make_rose_centerpieces-instance-250-episode-11368-segment-13-interval-0",
        "006e2665dc131564e5460c789392ce475889d499e038799ff691a3b05583a98d",
    ),
    (
        "make_rose_centerpieces-instance-262-episode-11375-segment-4-interval-0",
        "008db8b063cb3d2abd6015ceba9e33d6559a708224695b5a9d55786879a5ec29",
    ),
    (
        "make_rose_centerpieces-instance-159-episode-11305-segment-3-interval-0",
        "00a427ba0877fd0fa2bea32f0920cea1fb43da4b8a4930f16f7bcca7f0281b14",
    ),
    (
        "make_rose_centerpieces-instance-32-episode-11216-segment-8-interval-0",
        "00bed082a3c3548510da21c623b3f690e593c0f456143fe3247f7091306babd0",
    ),
    (
        "make_rose_centerpieces-instance-79-episode-11247-segment-1-interval-0",
        "00c57025d344f2260ec42ac1132d5ff041d9ff8af4dd1cc6f51781b851d96b6b",
    ),
    (
        "make_rose_centerpieces-instance-152-episode-11299-segment-9-interval-0",
        "00f1772ba05aac8956560b93f329bf8d9dba915cd673c4ca1a5488901d7098e2",
    ),
    (
        "make_rose_centerpieces-instance-185-episode-11322-segment-5-interval-0",
        "00f6559f80dd00d097060cdb1288b7a2242fca9e6f0b97c1a3d6e3c7479f5ea4",
    ),
    (
        "make_rose_centerpieces-instance-25-episode-11212-segment-11-interval-0",
        "011cf16f31d6ed0a4f92a216d129a768771aa775e84f891bd6cffc100cee626d",
    ),
    (
        "make_rose_centerpieces-instance-2-episode-11200-segment-6-interval-0",
        "01242caed8d04e76ed790dcf659c8dde15f43b1b0b82612370854357d5e1347a",
    ),
    (
        "make_rose_centerpieces-instance-206-episode-11337-segment-12-interval-0",
        "02d6e6247bd9fa7e8edd65b54caf0943b1463a5ed0ddddcd5667d7cc7dece802",
    ),
)
PRODUCTION_COUNTS = {
    "episode_anchors": 20,
    "episodes": 20,
    "annotated_segments": 348,
    "samples": 344,
    "exclusions": 4,
}


def challenge_tasks() -> set[str]:
    path = Path(__file__).with_name("b1k_tasks.txt")
    return {line.split()[0] for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalized(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("_", " ").lower().split())


def _contains(value: str, *needles: str) -> bool:
    return any(needle in value for needle in needles)


def infer_predicate(task: str, row: dict[str, Any]) -> dict[str, Any] | None:
    """Map one role-aware annotation row to an observable completion predicate."""
    if row.get("completion_contract") == "annotation_roles_v1":
        from subtask_completion import infer_role_predicate
        return infer_role_predicate(row)
    description = _normalized(row.get("skill_description"))
    text = _normalized(row.get("subtask"))
    object_text = " ".join(
        _normalized(item)
        for name in ("manipulating_object_ids", "reference_object_ids", "object_ids")
        for item in row.get(name, [])
    )
    semantic_text = f"{text} {object_text}"
    if description == "hand over":
        return None

    if task == "carrying_in_groceries":
        if description == "move to":
            if _contains(semantic_text, "paper bag"):
                return {"kind": "near", "target": "sack"}
            if _contains(semantic_text, "breakfast table"):
                return {"kind": "near", "target": "breakfast_table"}
            if _contains(semantic_text, "fridge"):
                return {"kind": "near", "target": "refrigerator"}
            if _contains(semantic_text, "door"):
                return {"kind": "near", "target": "passage_door"}
        if description == "pick up from":
            if _contains(semantic_text, "paper bag") and not _contains(
                semantic_text, "tomato", "milk"
            ):
                return {"kind": "grasped", "target": "sack"}
            if _contains(semantic_text, "tomato"):
                return {"kind": "grasped", "target": "tomato"}
            if _contains(semantic_text, "milk"):
                return {"kind": "grasped", "target": "milk"}
        if description == "close lid" and _contains(semantic_text, "car"):
            return {"kind": "open", "target": "car", "value": False}
        if description == "open door":
            target = (
                "refrigerator" if _contains(semantic_text, "fridge") else "passage_door"
            )
            return {"kind": "open", "target": target, "value": True}
        if description == "close door" and _contains(semantic_text, "fridge"):
            return {"kind": "open", "target": "refrigerator", "value": False}
        if description == "place on" and _contains(
            semantic_text, "paper bag", "breakfast table"
        ):
            return {"kind": "on_top", "target": "sack", "reference": "breakfast_table"}
        if description == "place in" and _contains(semantic_text, "fridge"):
            if _contains(semantic_text, "tomato"):
                return {
                    "kind": "inside",
                    "target": "tomato",
                    "reference": "refrigerator",
                }
            if _contains(semantic_text, "milk"):
                return {"kind": "inside", "target": "milk", "reference": "refrigerator"}
        if description == "tip over" and _contains(semantic_text, "paper bag"):
            return {"kind": "tilted", "target": "sack"}
        if description == "pour" and _contains(semantic_text, "paper bag"):
            return {
                "kind": "not_inside",
                "target": "milk",
                "reference": "sack",
            }
        return None

    if task == "make_rose_centerpieces":
        if description == "move to":
            if _contains(semantic_text, "coffee table"):
                return {"kind": "near", "target": "coffee_table"}
            if _contains(semantic_text, "vase"):
                return {"kind": "near", "target": "vase"}
            if _contains(semantic_text, "rose", "bottom cabinet"):
                return {"kind": "near", "target": "active_rose"}
        if description == "pick up from":
            if _contains(semantic_text, "vase"):
                return {"kind": "grasped", "target": "vase"}
            if _contains(semantic_text, "rose"):
                return {"kind": "grasped", "target": "active_rose"}
        if description == "place in" and _contains(semantic_text, "rose", "vase"):
            return {"kind": "rose_resolved_increment", "value": 1}
        if description == "place on" and _contains(
            semantic_text, "vase", "coffee table"
        ):
            return {"kind": "on_top", "target": "vase", "reference": "coffee_table"}
        return None

    raise ValueError(f"Unsupported task: {task}")


def _read_episode_metadata(dataset_root: Path) -> tuple[dict[int, dict], list[Path]]:
    import pyarrow.parquet as pq

    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise ValueError(
            f"No episode metadata under {dataset_root / 'meta' / 'episodes'}"
        )
    required = {
        "episode_index",
        "task_index",
        "task_instance_id",
        "raw_episode_id",
        "annotation_path",
        "length",
        "tasks",
    }
    metadata: dict[int, dict] = {}
    for path in paths:
        missing = required - set(pq.read_schema(path).names)
        if missing:
            raise ValueError(f"{path} is missing metadata columns: {sorted(missing)}")
        for row in pq.read_table(path, columns=sorted(required)).to_pylist():
            episode = int(row["episode_index"])
            if episode in metadata:
                raise ValueError(f"Duplicate episode_index={episode}")
            metadata[episode] = row
    return metadata, paths


def _read_episode_ids(path: Path) -> list[int]:
    try:
        values = [int(token) for token in path.read_text().split()]
    except ValueError as error:
        raise ValueError(f"Non-integer episode ID in {path}") from error
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Expected unique episode IDs in {path}")
    return values


def _candidate_rank(seed: int, candidate: dict[str, Any]) -> tuple[str, str]:
    identity = {
        "version": SELECTION_VERSION,
        "seed": seed,
        "task": candidate["task"],
        "episode_index": candidate["episode_index"],
        "task_instance_id": candidate["task_instance_id"],
        "source_segment_index": candidate["source_segment_index"],
        "source_interval_index": candidate["source_interval_index"],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), canonical


def _sample_id(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['task']}-instance-{candidate['task_instance_id']}-"
        f"episode-{candidate['episode_index']}-segment-{candidate['source_segment_index']}-"
        f"interval-{candidate['source_interval_index']}"
    )


def _select_unique(
    candidates: list[dict[str, Any]], count: int, *, unique_segments: bool = True
) -> list[dict[str, Any]]:
    selected = []
    episodes: set[int] = set()
    instances: set[int] = set()
    source_segments: set[int] = set()
    for candidate in candidates:
        episode = candidate["episode_index"]
        instance = candidate["task_instance_id"]
        source_segment = candidate["source_segment_index"]
        if (
            episode in episodes
            or instance in instances
            or (unique_segments and source_segment in source_segments)
        ):
            continue
        selected.append(candidate)
        episodes.add(episode)
        instances.add(instance)
        source_segments.add(source_segment)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} unique candidates available; need {count}")


def _segment_record(
    row: dict[str, Any], episode_metadata: dict[str, Any], task: str
) -> dict[str, Any]:
    episode = int(row["episode_index"])
    start = int(row["start"])
    end = int(row["end"])
    source_segment = row["source_segment_index"]
    source_interval = row["source_interval_index"]
    if source_segment is None or source_interval is None:
        raise ValueError(f"Annotated episode {episode} has a missing source index")
    source_segment = int(source_segment)
    source_interval = int(source_interval)
    source_key = (episode, source_segment, source_interval)
    episode_length = int(episode_metadata["length"])
    if start < 0 or end <= start or end > episode_length:
        raise ValueError(
            f"Segment {source_key} has invalid frame range [{start}, {end}) "
            f"for episode length {episode_length}"
        )
    task_index = int(episode_metadata["task_index"])
    task_values = episode_metadata["tasks"]
    if task_values != [task]:
        raise ValueError(
            f"Episode {episode} tasks are {task_values!r}, expected {[task]!r}"
        )
    subtask = row["subtask"]
    record = {
        "task": task,
        "task_index": task_index,
        "episode_index": episode,
        "raw_episode_id": int(episode_metadata["raw_episode_id"]),
        "task_instance_id": int(episode_metadata["task_instance_id"]),
        "episode_length": episode_length,
        "annotation_path": row["annotation_path"],
        "source_segment_index": source_segment,
        "source_interval_index": source_interval,
        "source_skill_index": (
            None
            if row["source_skill_index"] is None
            else int(row["source_skill_index"])
        ),
        "start": start,
        "end": end,
        "segment_frames": end - start,
        "subtask": subtask.strip() if isinstance(subtask, str) else subtask,
        "skill_description": row["skill_description"],
        "manipulating_object_ids": list(row["manipulating_object_ids"] or []),
        "reference_object_ids": list(row["reference_object_ids"] or []),
        "object_ids": list(row["object_ids"] or []),
        "memory_prefix": row["memory_prefix"],
        "spatial_prefix": row["spatial_prefix"],
        "provenance": row["provenance"],
    }

    if "completion_contract" in row:
        record["completion_contract"] = row["completion_contract"]
    return record


def _segment_order(value: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(value["start"]),
        int(value["end"]),
        int(value["source_segment_index"]),
        int(value["source_interval_index"]),
        value["sample_id"],
    )


def build_panel(
    dataset_root: str | Path,
    segments_path: str | Path,
    train_episodes_path: str | Path,
    split_manifest_path: str | Path,
    *,
    seed: int = 1,
    samples_per_task: int | None = None,
    episodes_per_task: int | None = None,
    reserve_per_task: int = 100,
    task_scope: str = "two_tasks",
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if episodes_per_task is None:
        episodes_per_task = 10 if samples_per_task is None else samples_per_task
    elif samples_per_task is not None and samples_per_task != episodes_per_task:
        raise ValueError("samples_per_task and episodes_per_task disagree")
    if episodes_per_task <= 0 or reserve_per_task < 0:
        raise ValueError(
            "episode counts must be non-negative and episodes_per_task must be positive"
        )
    dataset_root = Path(dataset_root).expanduser().resolve()
    segments_path = Path(segments_path).expanduser().resolve()
    train_episodes_path = Path(train_episodes_path).expanduser().resolve()
    split_manifest_path = Path(split_manifest_path).expanduser().resolve()
    train_episodes = set(_read_episode_ids(train_episodes_path))
    metadata, metadata_paths = _read_episode_metadata(dataset_root)

    required_columns = {
        "episode_index",
        "start",
        "end",
        "subtask",
        "annotation_path",
        "source_segment_index",
        "source_interval_index",
        "source_skill_index",
        "skill_description",
        "manipulating_object_ids",
        "reference_object_ids",
        "object_ids",
        "memory_prefix",
        "spatial_prefix",
        "is_annotated",
        "provenance",
    }
    missing = required_columns - set(pq.read_schema(segments_path).names)
    if missing:
        raise ValueError(f"Segment table is missing columns: {sorted(missing)}")
    columns = required_columns | ({"completion_contract"} if "completion_contract" in pq.read_schema(segments_path).names else set())
    segment_rows = pq.read_table(segments_path, columns=sorted(columns)).to_pylist()
    if task_scope == "full100" and any(row.get("completion_contract") != "annotation_roles_v1" for row in segment_rows):
        raise ValueError("Full100 requires role-aware completion contracts")

    task_map = dict(TASK_INDEX_TO_NAME)
    if task_scope == "full100":
        task_map = {}
        for item in metadata.values():
            index, name = int(item["task_index"]), item["tasks"][0]
            if index in task_map and task_map[index] != name:
                raise ValueError("Inconsistent task index in dataset metadata")
            task_map[index] = name
        if set(task_map) != set(range(100)) or set(task_map.values()) != challenge_tasks():
            raise ValueError("Full100 requires all 100 challenge tasks in dataset metadata")
    elif task_scope != "two_tasks":
        raise ValueError("task_scope must be two_tasks or full100")

    # This is the frozen v1 selection path. Keep its filters, ranking identity, and
    # uniqueness semantics unchanged so schema v2 selects the same episode set.
    candidates_by_task = {task: [] for task in task_map.values()}
    annotated_rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    seen_anchor_keys: set[tuple[int, int, int]] = set()
    for row in segment_rows:
        episode = int(row["episode_index"])
        if episode not in train_episodes or not row["is_annotated"]:
            continue
        annotated_rows_by_episode.setdefault(episode, []).append(row)
        start = int(row["start"])
        end = int(row["end"])
        if end - start < MIN_SEGMENT_FRAMES or not _normalized(row["subtask"]):
            continue
        source_segment = row["source_segment_index"]
        source_interval = row["source_interval_index"]
        if source_segment is None or source_interval is None:
            continue
        source_segment = int(source_segment)
        source_interval = int(source_interval)
        source_key = (episode, source_segment, source_interval)
        if source_key in seen_anchor_keys:
            raise ValueError(f"Duplicate source segment row: {source_key}")
        seen_anchor_keys.add(source_key)

        episode_metadata = metadata.get(episode)
        if episode_metadata is None:
            raise ValueError(f"Segment references unknown episode {episode}")
        task_index = int(episode_metadata["task_index"])
        task = task_map.get(task_index)
        if task is None:
            continue
        candidate = _segment_record(row, episode_metadata, task)
        predicate = infer_predicate(task, candidate)
        if predicate is None:
            continue
        candidate["predicate"] = predicate
        candidate["sample_id"] = _sample_id(candidate)
        candidate["selection_rank"] = _candidate_rank(seed, candidate)[0]
        candidates_by_task[task].append(candidate)

    episode_anchors = []
    for task in sorted(candidates_by_task):
        ordered = sorted(
            candidates_by_task[task], key=lambda value: _candidate_rank(seed, value)
        )
        episode_anchors.extend(_select_unique(ordered, episodes_per_task, unique_segments=task_scope != "full100"))
    episode_anchors.sort(key=lambda value: (value["task"], value["selection_rank"]))

    samples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    seen_scheduled_keys: set[tuple[int, int, int]] = set()
    for anchor in episode_anchors:
        episode = int(anchor["episode_index"])
        episode_metadata = metadata[episode]
        task = anchor["task"]
        scheduled = []
        for row in annotated_rows_by_episode.get(episode, []):
            record = _segment_record(row, episode_metadata, task)
            source_key = (
                record["episode_index"],
                record["source_segment_index"],
                record["source_interval_index"],
            )
            if source_key in seen_scheduled_keys:
                raise ValueError(f"Duplicate source segment row: {source_key}")
            seen_scheduled_keys.add(source_key)
            record["sample_id"] = _sample_id(record)
            predicate = infer_predicate(task, record)
            if predicate is None:
                record["reason"] = "unsupported_predicate"
            else:
                record["predicate"] = predicate
            scheduled.append(record)

        scheduled.sort(key=_segment_order)
        episode_samples = []
        episode_exclusions = []
        for record in scheduled:
            if "predicate" not in record:
                episode_exclusions.append(record)
                continue
            record["rollout_id"] = len(episode_samples)
            episode_samples.append(record)

        sample_ids = [sample["sample_id"] for sample in episode_samples]
        exclusion_ids = [item["sample_id"] for item in episode_exclusions]
        episodes.append(
            {
                "task": task,
                "task_index": int(anchor["task_index"]),
                "episode_index": episode,
                "raw_episode_id": int(anchor["raw_episode_id"]),
                "task_instance_id": int(anchor["task_instance_id"]),
                "episode_length": int(anchor["episode_length"]),
                "annotation_path": episode_metadata["annotation_path"],
                "anchor_sample_id": anchor["sample_id"],
                "sample_ids": sample_ids,
                "exclusion_ids": exclusion_ids,
                "counts": {
                    "annotated_segments": len(scheduled),
                    "samples": len(episode_samples),
                    "exclusions": len(episode_exclusions),
                },
            }
        )
        samples.extend(episode_samples)
        exclusions.extend(episode_exclusions)

    tasks = sorted(task_map.values())
    by_task = {}
    for task in tasks:
        task_episodes = [episode for episode in episodes if episode["task"] == task]
        task_samples = [sample for sample in samples if sample["task"] == task]
        task_exclusions = [item for item in exclusions if item["task"] == task]
        by_task[task] = {
            "episode_anchors": sum(
                anchor["task"] == task for anchor in episode_anchors
            ),
            "episodes": len(task_episodes),
            "annotated_segments": len(task_samples) + len(task_exclusions),
            "samples": len(task_samples),
            "exclusions": len(task_exclusions),
        }
    exclusion_reasons = sorted({item["reason"] for item in exclusions})
    counts = {
        "episode_anchors": len(episode_anchors),
        "episodes": len(episodes),
        "annotated_segments": len(samples) + len(exclusions),
        "samples": len(samples),
        "exclusions": len(exclusions),
        "exclusions_by_reason": {
            reason: sum(item["reason"] == reason for item in exclusions)
            for reason in exclusion_reasons
        },
        "by_task": by_task,
    }
    panel = {
        "schema_version": 2,
        "kind": "training_subtask_closed_loop_panel",
        "selection": {
            "tasks": tasks,
            "episode_anchors": {
                "version": SELECTION_VERSION,
                "seed": seed,
                "episodes_per_task": episodes_per_task,
                "minimum_segment_frames": MIN_SEGMENT_FRAMES,
                "uniqueness": [
                    "episode_index",
                    "task_instance_id",
                    "source_segment_index",
                ],
            },
            "scheduled_segments": {
                "scope": "all_annotated_segments_in_selected_episodes",
                "order": [
                    "start",
                    "end",
                    "source_segment_index",
                    "source_interval_index",
                    "sample_id",
                ],
                "unsupported_predicate": "exclude",
            },
        },
        "inputs": {
            "dataset_root": str(dataset_root),
            "segments_path": str(segments_path),
            "segments_sha256": file_sha256(segments_path),
            "train_episodes_path": str(train_episodes_path),
            "train_episodes_sha256": file_sha256(train_episodes_path),
            "split_manifest_path": str(split_manifest_path),
            "split_manifest_sha256": file_sha256(split_manifest_path),
            "episode_metadata_sha256": metadata_sha256(metadata_paths, dataset_root),
        },
        "episode_anchors": episode_anchors,
        "episodes": episodes,
        "samples": samples,
        "exclusions": exclusions,
        "counts": counts,
    }
    if task_scope == "full100":
        panel["selection"].update(task_scope="full100", task_index_to_name={str(k): v for k, v in sorted(task_map.items())})
        panel["selection"]["episode_anchors"].update(version="subtask_closed_loop_full100_v1", uniqueness=["episode_index", "task_instance_id"])
    panel["fingerprint"] = _canonical_hash(panel)
    return panel


def validate_frozen_production_panel(panel: dict[str, Any]) -> dict[str, Any]:
    anchors = tuple(
        (anchor.get("sample_id"), anchor.get("selection_rank"))
        for anchor in panel.get("episode_anchors", [])
    )
    if anchors != FROZEN_PRODUCTION_ANCHORS:
        raise ValueError(
            "production episode anchors differ from frozen v1 panel "
            f"{FROZEN_V1_PANEL_FINGERPRINT}"
        )
    counts = panel.get("counts", {})
    mismatches = {
        name: (counts.get(name), value)
        for name, value in PRODUCTION_COUNTS.items()
        if counts.get(name) != value
    }
    exclusions = panel.get("exclusions", [])
    if mismatches:
        raise ValueError(f"production panel counts are invalid: {mismatches}")
    if (
        counts.get("exclusions_by_reason") != {"unsupported_predicate": 4}
        or any(item.get("skill_description") != "hand over" for item in exclusions)
    ):
        raise ValueError("production exclusions must be four unsupported hand-over rows")
    return panel


def raw_hdf5_path(raw_root: str | Path, sample: dict[str, Any]) -> Path:
    return (
        Path(raw_root).expanduser().resolve()
        / "2026-challenge-rawdata"
        / f"task-{int(sample['task_index']):04d}"
        / f"episode_{int(sample['raw_episode_id']):08d}.hdf5"
    )


def _raw_episode_structure(
    file: Any, path: Path, expected_length: int
) -> tuple[dict[str, Any], Any, Any, tuple[int, ...]]:
    if "data" not in file:
        raise ValueError(f"{path} has no data group")
    data = file["data"]
    for attribute in ("config", "scene_file"):
        if attribute not in data.attrs:
            raise ValueError(f"{path} data group has no {attribute!r} attribute")
        try:
            json.loads(data.attrs[attribute])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path} has invalid data.{attribute} JSON") from error
    raw_git_hash = data.attrs.get("omnigibson_git_hash")
    omnigibson_git_hash = (
        None
        if raw_git_hash is None or str(raw_git_hash).strip().lower() == "null"
        else str(raw_git_hash).strip()
    )
    demo_ids = sorted(
        int(name.removeprefix("demo_"))
        for name in data
        if name.startswith("demo_") and name.removeprefix("demo_").isdigit()
    )
    if not demo_ids:
        raise ValueError(f"{path} has no numeric data/demo_* groups")
    demo_id = demo_ids[-1]
    group = data[f"demo_{demo_id}"]
    for name in ("state", "state_size", "action", "init_metadata"):
        if name not in group:
            raise ValueError(f"{path} demo_{demo_id} has no {name}")
    if "transitions" not in group.attrs:
        raise ValueError(f"{path} demo_{demo_id} has no transitions attribute")
    try:
        transitions = json.loads(group.attrs["transitions"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} demo_{demo_id} has invalid transitions JSON") from error
    if not isinstance(transitions, dict):
        raise TypeError(f"{path} demo_{demo_id} transitions must be an object")
    try:
        transition_frames = tuple(int(index) for index in transitions)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} demo_{demo_id} has invalid transition frames") from error

    state = group["state"]
    state_size = group["state_size"]
    action = group["action"]
    num_samples = int(group.attrs.get("num_samples", len(state)))
    state_samples = len(state)
    action_samples = len(action)
    if state.ndim != 2:
        raise ValueError(f"{path} demo_{demo_id} state must be a 2D array")
    if action_samples != expected_length:
        raise ValueError(
            f"{path} demo_{demo_id} has {action_samples} actions, expected {expected_length}"
        )
    if state_samples not in (expected_length, expected_length + 1):
        raise ValueError(
            f"{path} demo_{demo_id} has {state_samples} states, expected "
            f"{expected_length} or {expected_length + 1}"
        )
    if len(state_size) != state_samples or num_samples != state_samples:
        raise ValueError(
            f"{path} demo_{demo_id} state arrays and num_samples disagree: "
            f"state={state_samples}, state_size={len(state_size)}, num_samples={num_samples}"
        )
    return (
        {
            "path": str(path),
            "omnigibson_git_hash": omnigibson_git_hash,
            "demo_group": f"demo_{demo_id}",
            "num_samples": num_samples,
            "action_samples": action_samples,
            "state_samples": state_samples,
            "state_layout": (
                "aligned" if state_samples == expected_length else "trailing_state"
            ),
            "state_width": int(state.shape[1]),
        },
        state,
        state_size,
        transition_frames,
    )


def _raw_frame_evidence(
    sample: dict[str, Any],
    state: Any,
    state_size: Any,
    transition_frames: tuple[int, ...],
    np: Any,
) -> dict[str, int]:
    expected_length = int(sample["episode_length"])
    frame = int(sample["start"])
    if frame < 0 or frame >= expected_length:
        raise ValueError(
            f"Sample frame {frame} is outside LeRobot episode length {expected_length}"
        )
    serialized_size = int(state_size[frame])
    if serialized_size <= 0 or state.ndim != 2 or serialized_size > state.shape[1]:
        raise ValueError(f"Invalid serialized state size {serialized_size} at frame {frame}")
    selected_state = np.asarray(state[frame, :serialized_size])
    if not np.isfinite(selected_state).all():
        raise ValueError(f"Serialized state at frame {frame} contains non-finite values")
    return {
        "selected_frame": frame,
        "selected_state_size": serialized_size,
        "selected_state_sha256": hashlib.sha256(selected_state.tobytes()).hexdigest(),
        "transition_count_before_frame": sum(
            transition_frame < frame for transition_frame in transition_frames
        ),
    }


def validate_raw_hdf5(
    path: str | Path,
    sample: dict[str, Any],
    *,
    include_sha256: bool = True,
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    import h5py
    import numpy as np

    with h5py.File(path, "r", locking=False) as file:
        episode, state, state_size, transition_frames = _raw_episode_structure(
            file, path, int(sample["episode_length"])
        )
        frame = _raw_frame_evidence(
            sample, state, state_size, transition_frames, np
        )
    result = {**episode, **frame}
    if include_sha256:
        result["sha256"] = file_sha256(path)
    return result


def _preflight_report(
    panel: dict[str, Any],
    expected_episodes: int,
    episodes: list[dict[str, Any]],
    expected_samples: int,
    samples: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "panel_fingerprint": panel.get("fingerprint"),
        "valid": (
            not errors
            and len(episodes) == expected_episodes
            and len(samples) == expected_samples
        ),
        "counts": {
            "expected_episodes": expected_episodes,
            "validated_episodes": len(episodes),
            "expected_samples": expected_samples,
            "validated_samples": len(samples),
        },
        "episodes": episodes,
        "samples": samples,
        "errors": errors,
    }


def preflight_panel(
    panel: dict[str, Any], raw_root: str | Path, *, include_sha256: bool = True
) -> dict[str, Any]:
    panel_samples = panel.get("samples", [])
    if not isinstance(panel_samples, list):
        raise TypeError("panel samples must be a list")
    grouped: dict[Path, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for sample in panel_samples:
        try:
            path = raw_hdf5_path(raw_root, sample)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(
                {
                    "scope": "sample",
                    "sample_id": (
                        sample.get("sample_id") if isinstance(sample, dict) else None
                    ),
                    "path": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        grouped.setdefault(path, []).append(sample)

    panel_episodes = panel.get("episodes")
    expected_episodes = (
        len(panel_episodes) if isinstance(panel_episodes, list) else len(grouped)
    )
    expected_samples = len(panel_samples)
    episode_results: list[dict[str, Any]] = []
    sample_results: list[dict[str, Any]] = []
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        errors.append(
            {"scope": "preflight", "path": None, "error": f"ImportError: {error}"}
        )
        return _preflight_report(
            panel,
            expected_episodes,
            episode_results,
            expected_samples,
            sample_results,
            errors,
        )

    identity_fields = (
        "task",
        "task_index",
        "episode_index",
        "raw_episode_id",
        "task_instance_id",
        "episode_length",
    )
    for path, episode_samples in grouped.items():
        first = episode_samples[0]
        sample_ids = [sample.get("sample_id") for sample in episode_samples]
        try:
            for sample in episode_samples[1:]:
                if any(sample.get(name) != first.get(name) for name in identity_fields):
                    raise ValueError("samples sharing a raw path disagree on episode identity")
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = file_sha256(path) if include_sha256 else None
            with h5py.File(path, "r", locking=False) as file:
                details, state, state_size, transition_frames = _raw_episode_structure(
                    file, path, int(first["episode_length"])
                )
                episode_result = {
                    name: first[name] for name in identity_fields
                }
                episode_result.update(details)
                episode_result["sample_ids"] = sample_ids
                if digest is not None:
                    episode_result["sha256"] = digest
                episode_results.append(episode_result)
                for sample in episode_samples:
                    try:
                        evidence = _raw_frame_evidence(
                            sample, state, state_size, transition_frames, np
                        )
                    except (KeyError, RuntimeError, TypeError, ValueError) as error:
                        errors.append(
                            {
                                "scope": "sample",
                                "sample_id": sample.get("sample_id"),
                                "episode_index": sample.get("episode_index"),
                                "path": str(path),
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    else:
                        sample_results.append(
                            {
                                "sample_id": sample["sample_id"],
                                "episode_index": sample["episode_index"],
                                "rollout_id": sample.get("rollout_id"),
                                **evidence,
                            }
                        )
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            errors.append(
                {
                    "scope": "episode",
                    "episode_index": first.get("episode_index"),
                    "path": str(path),
                    "sample_ids": sample_ids,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return _preflight_report(
        panel,
        expected_episodes,
        episode_results,
        expected_samples,
        sample_results,
        errors,
    )


def _panel_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_panel_segment(
    value: object,
    episode_by_id: dict[int, dict[str, Any]],
    *,
    exclusion: bool,
) -> tuple[str, tuple[int, int, int], tuple[str, int, int] | None]:
    label = "exclusion" if exclusion else "sample"
    if not isinstance(value, dict):
        raise ValueError(f"Panel {label} is not an object")
    integer_fields = (
        "task_index",
        "episode_index",
        "raw_episode_id",
        "task_instance_id",
        "episode_length",
        "source_segment_index",
        "source_interval_index",
        "start",
        "end",
        "segment_frames",
    )
    if any(not _panel_int(value.get(name)) for name in integer_fields):
        raise ValueError(f"Panel {label} has invalid integer fields")
    sample_id = value.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"Panel {label} has an invalid sample ID")
    if sample_id != _sample_id(value):
        raise ValueError(f"Panel {label} has inconsistent sample ID {sample_id!r}")
    episode_index = value["episode_index"]
    episode = episode_by_id.get(episode_index)
    if episode is None:
        raise ValueError(f"Panel {label} references unselected episode {episode_index}")
    for name in (
        "task",
        "task_index",
        "raw_episode_id",
        "task_instance_id",
        "episode_length",
        "annotation_path",
    ):
        if value.get(name) != episode.get(name):
            raise ValueError(
                f"Panel {label} {sample_id!r} disagrees with episode {episode_index} on {name}"
            )
    if (
        value["start"] < 0
        or value["end"] <= value["start"]
        or value["end"] > value["episode_length"]
        or value["segment_frames"] != value["end"] - value["start"]
    ):
        raise ValueError(f"Panel {label} {sample_id!r} has invalid frame bounds")

    source_key = (
        episode_index,
        value["source_segment_index"],
        value["source_interval_index"],
    )
    if exclusion:
        if value.get("reason") != "unsupported_predicate":
            raise ValueError(f"Panel exclusion {sample_id!r} has an invalid reason")
        if "predicate" in value or infer_predicate(value["task"], value) is not None:
            raise ValueError(f"Panel exclusion {sample_id!r} has a supported predicate")
        if "rollout_id" in value:
            raise ValueError(f"Panel exclusion {sample_id!r} has a rollout ID")
        return sample_id, source_key, None

    rollout_id = value.get("rollout_id")
    if not _panel_int(rollout_id) or rollout_id < 0:
        raise ValueError(f"Panel sample {sample_id!r} has an invalid rollout ID")
    predicate = value.get("predicate")
    if not isinstance(predicate, dict) or infer_predicate(value["task"], value) != predicate:
        raise ValueError(f"Panel sample {sample_id!r} has an invalid predicate")
    job_key = (value["task"], value["task_instance_id"], rollout_id)
    return sample_id, source_key, job_key


def load_panel(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    panel = json.loads(path.read_text())
    if isinstance(panel, dict) and panel.get('schema_version') == 3:
        from additional_training_panel import validate
        return validate(panel)
    if not isinstance(panel, dict) or panel.get("schema_version") != 2:
        raise ValueError(f"Unsupported panel schema in {path}")
    expected_fingerprint = panel.get("fingerprint")
    actual_fingerprint = _canonical_hash(
        {name: value for name, value in panel.items() if name != "fingerprint"}
    )
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            f"Panel fingerprint mismatch: stored={expected_fingerprint}, actual={actual_fingerprint}"
        )
    if panel.get("kind") != "training_subtask_closed_loop_panel":
        raise ValueError(f"Unsupported panel kind in {path}")

    selection = panel.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"Panel selection metadata is invalid: {path}")
    tasks = selection.get("tasks")
    task_map = dict(TASK_INDEX_TO_NAME)
    if selection.get("task_scope") == "full100":
        raw_map = selection.get("task_index_to_name")
        if not isinstance(raw_map, dict) or set(raw_map) != {str(i) for i in range(100)}:
            raise ValueError("Full100 panel has an invalid task mapping")
        task_map = {int(k): v for k, v in raw_map.items()}
        if set(task_map.values()) != challenge_tasks():
            raise ValueError("Full100 panel does not cover challenge tasks")
    elif selection.get("task_scope") is not None:
        raise ValueError("Unknown panel task scope")
    expected_tasks = sorted(task_map.values())
    if tasks != expected_tasks:
        raise ValueError(f"Panel tasks are invalid: {path}")
    anchor_selection = selection.get("episode_anchors")
    scheduled_selection = selection.get("scheduled_segments")
    if not isinstance(anchor_selection, dict) or not isinstance(
        scheduled_selection, dict
    ):
        raise ValueError(f"Panel selection stages are invalid: {path}")
    episodes_per_task = anchor_selection.get("episodes_per_task")
    seed = anchor_selection.get("seed")
    if not _panel_int(episodes_per_task) or episodes_per_task <= 0:
        raise ValueError(f"Panel episodes_per_task is invalid: {path}")
    if not _panel_int(seed):
        raise ValueError(f"Panel anchor seed is invalid: {path}")
    if (
        anchor_selection.get("version") != ("subtask_closed_loop_full100_v1" if selection.get("task_scope") == "full100" else SELECTION_VERSION)
        or anchor_selection.get("minimum_segment_frames") != MIN_SEGMENT_FRAMES
        or anchor_selection.get("uniqueness")
        != (["episode_index", "task_instance_id"] if selection.get("task_scope") == "full100" else ["episode_index", "task_instance_id", "source_segment_index"])
    ):
        raise ValueError(f"Panel anchor selection contract is invalid: {path}")
    if scheduled_selection != {
        "scope": "all_annotated_segments_in_selected_episodes",
        "order": [
            "start",
            "end",
            "source_segment_index",
            "source_interval_index",
            "sample_id",
        ],
        "unsupported_predicate": "exclude",
    }:
        raise ValueError(f"Panel scheduled-segment contract is invalid: {path}")

    inputs = panel.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"Panel inputs are invalid: {path}")
    for name in (
        "segments_sha256",
        "train_episodes_sha256",
        "split_manifest_sha256",
        "episode_metadata_sha256",
    ):
        digest = inputs.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Panel input hash {name} is invalid: {path}")
    for name in (
        "dataset_root",
        "segments_path",
        "train_episodes_path",
        "split_manifest_path",
    ):
        if not isinstance(inputs.get(name), str) or not inputs[name]:
            raise ValueError(f"Panel input path {name} is invalid: {path}")

    if selection.get("task_scope") == "full100":
        records = [*panel.get("episode_anchors", []), *panel.get("samples", []), *panel.get("exclusions", [])]
        if any(item.get("completion_contract") != "annotation_roles_v1" for item in records):
            raise ValueError("Full100 panel is missing role-bound completion contracts")
    expected_episode_count = len(tasks) * episodes_per_task
    anchors = panel.get("episode_anchors")
    episodes = panel.get("episodes")
    samples = panel.get("samples")
    exclusions = panel.get("exclusions")
    counts = panel.get("counts")
    if not isinstance(anchors, list) or len(anchors) != expected_episode_count:
        raise ValueError(
            f"Panel must contain {expected_episode_count} episode anchors: {path}"
        )
    if not isinstance(episodes, list) or len(episodes) != expected_episode_count:
        raise ValueError(f"Panel must contain {expected_episode_count} episodes: {path}")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Panel has no samples: {path}")
    if not isinstance(exclusions, list) or not isinstance(counts, dict):
        raise ValueError(f"Panel exclusions or counts are invalid: {path}")

    anchor_ids = []
    anchors_by_episode = {}
    anchor_instances: dict[str, set[int]] = {task: set() for task in tasks}
    anchor_segments: dict[str, set[int]] = {task: set() for task in tasks}
    anchor_task_counts = {task: 0 for task in tasks}
    for anchor in anchors:
        if not isinstance(anchor, dict) or anchor.get("task") not in tasks:
            raise ValueError(f"Panel contains an invalid episode anchor: {path}")
        integer_fields = (
            "task_index",
            "episode_index",
            "raw_episode_id",
            "task_instance_id",
            "episode_length",
            "source_segment_index",
            "source_interval_index",
            "start",
            "end",
            "segment_frames",
        )
        if any(not _panel_int(anchor.get(name)) for name in integer_fields):
            raise ValueError(f"Panel episode anchor has invalid integer fields: {path}")
        sample_id = anchor.get("sample_id")
        if not isinstance(sample_id, str) or sample_id != _sample_id(anchor):
            raise ValueError(f"Panel episode anchor has an invalid sample ID: {path}")
        if task_map.get(anchor["task_index"]) != anchor["task"]:
            raise ValueError(f"Panel episode anchor has an invalid task index: {path}")
        if (
            anchor["start"] < 0
            or anchor["end"] <= anchor["start"]
            or anchor["end"] > anchor["episode_length"]
            or anchor["segment_frames"] != anchor["end"] - anchor["start"]
        ):
            raise ValueError(f"Panel episode anchor has invalid frame bounds: {path}")
        if (
            anchor.get("selection_rank") != _candidate_rank(seed, anchor)[0]
            or infer_predicate(anchor["task"], anchor) != anchor.get("predicate")
        ):
            raise ValueError(f"Panel episode anchor does not match v1 selection: {path}")
        episode_index = anchor["episode_index"]
        if episode_index in anchors_by_episode:
            raise ValueError(f"Panel contains duplicate anchor episodes: {path}")
        task = anchor["task"]
        instance = anchor["task_instance_id"]
        source_segment = anchor["source_segment_index"]
        if (
            instance in anchor_instances[task]
            or (selection.get("task_scope") != "full100" and source_segment in anchor_segments[task])
        ):
            raise ValueError(f"Panel episode anchors violate v1 uniqueness: {path}")
        anchor_instances[task].add(instance)
        anchor_segments[task].add(source_segment)
        anchor_task_counts[task] += 1
        anchor_ids.append(episode_index)
        anchors_by_episode[episode_index] = anchor
    if any(count != episodes_per_task for count in anchor_task_counts.values()):
        raise ValueError(f"Panel episode anchors are not balanced by task: {path}")
    if anchors != sorted(
        anchors, key=lambda value: (value["task"], value["selection_rank"])
    ):
        raise ValueError(f"Panel episode anchors are not deterministically ordered: {path}")

    episode_by_id = {}
    episode_task_counts = {task: 0 for task in tasks}
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get("task") not in tasks:
            raise ValueError(f"Panel contains an invalid episode record: {path}")
        episode_index = episode.get("episode_index")
        if not _panel_int(episode_index) or episode_index in episode_by_id:
            raise ValueError(f"Panel contains duplicate or invalid episodes: {path}")
        anchor = anchors_by_episode.get(episode_index)
        if anchor is None:
            raise ValueError(f"Panel episode has no anchor: {path}")
        for name in (
            "task",
            "task_index",
            "raw_episode_id",
            "task_instance_id",
            "episode_length",
            "annotation_path",
        ):
            if episode.get(name) != anchor.get(name):
                raise ValueError(
                    f"Panel episode {episode_index} disagrees with its anchor on {name}"
                )
        if episode.get("anchor_sample_id") != anchor["sample_id"]:
            raise ValueError(f"Panel episode {episode_index} has the wrong anchor sample")
        sample_ids = episode.get("sample_ids")
        exclusion_ids = episode.get("exclusion_ids")
        episode_counts = episode.get("counts")
        if (
            not isinstance(sample_ids, list)
            or not isinstance(exclusion_ids, list)
            or not isinstance(episode_counts, dict)
            or any(not isinstance(value, str) or not value for value in sample_ids)
            or any(not isinstance(value, str) or not value for value in exclusion_ids)
            or len(sample_ids) != len(set(sample_ids))
            or len(exclusion_ids) != len(set(exclusion_ids))
        ):
            raise ValueError(f"Panel episode {episode_index} membership is invalid")
        expected_counts = {
            "annotated_segments": len(sample_ids) + len(exclusion_ids),
            "samples": len(sample_ids),
            "exclusions": len(exclusion_ids),
        }
        if episode_counts != expected_counts:
            raise ValueError(f"Panel episode {episode_index} counts are invalid")
        episode_by_id[episode_index] = episode
        episode_task_counts[episode["task"]] += 1
    if anchor_ids != [episode["episode_index"] for episode in episodes]:
        raise ValueError(f"Panel episode records do not match anchor order: {path}")
    if any(count != episodes_per_task for count in episode_task_counts.values()):
        raise ValueError(f"Panel episodes are not balanced by task: {path}")

    sample_ids = []
    source_keys = []
    job_keys = []
    samples_by_episode: dict[int, list[dict[str, Any]]] = {
        episode: [] for episode in episode_by_id
    }
    for sample in samples:
        sample_id, source_key, job_key = _validate_panel_segment(
            sample, episode_by_id, exclusion=False
        )
        sample_ids.append(sample_id)
        source_keys.append(source_key)
        job_keys.append(job_key)
        samples_by_episode[sample["episode_index"]].append(sample)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Panel contains duplicate sample IDs: {path}")
    if len(job_keys) != len(set(job_keys)):
        raise ValueError(f"Panel contains duplicate queue identities: {path}")

    exclusion_ids = []
    exclusions_by_episode: dict[int, list[dict[str, Any]]] = {
        episode: [] for episode in episode_by_id
    }
    for exclusion in exclusions:
        exclusion_id, source_key, _ = _validate_panel_segment(
            exclusion, episode_by_id, exclusion=True
        )
        exclusion_ids.append(exclusion_id)
        source_keys.append(source_key)
        exclusions_by_episode[exclusion["episode_index"]].append(exclusion)
    if len(sample_ids) + len(exclusion_ids) != len(set(sample_ids + exclusion_ids)):
        raise ValueError(f"Panel contains duplicate segment IDs: {path}")
    if len(source_keys) != len(set(source_keys)):
        raise ValueError(f"Panel contains duplicate source keys: {path}")
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    for anchor in anchors:
        scheduled_anchor = sample_by_id.get(anchor["sample_id"])
        anchor_payload = {
            name: value
            for name, value in anchor.items()
            if name != "selection_rank"
        }
        scheduled_payload = (
            None
            if scheduled_anchor is None
            else {
                name: value
                for name, value in scheduled_anchor.items()
                if name != "rollout_id"
            }
        )
        if scheduled_payload != anchor_payload:
            raise ValueError(
                f"Panel anchor {anchor['sample_id']!r} does not match its scheduled sample"
            )

    expected_sample_order = []
    expected_exclusion_order = []
    for episode in episodes:
        episode_index = episode["episode_index"]
        episode_samples = sorted(samples_by_episode[episode_index], key=_segment_order)
        episode_exclusions = sorted(
            exclusions_by_episode[episode_index], key=_segment_order
        )
        rollout_ids = [sample["rollout_id"] for sample in episode_samples]
        if rollout_ids != list(range(len(episode_samples))):
            raise ValueError(
                f"Panel episode {episode_index} rollout IDs are not dense and ordered"
            )
        ordered_sample_ids = [sample["sample_id"] for sample in episode_samples]
        ordered_exclusion_ids = [item["sample_id"] for item in episode_exclusions]
        if episode["sample_ids"] != ordered_sample_ids:
            raise ValueError(f"Panel episode {episode_index} sample IDs are inconsistent")
        if episode["exclusion_ids"] != ordered_exclusion_ids:
            raise ValueError(
                f"Panel episode {episode_index} exclusion IDs are inconsistent"
            )
        if episode["anchor_sample_id"] not in ordered_sample_ids:
            raise ValueError(f"Panel episode {episode_index} anchor is not scheduled")
        expected_sample_order.extend(ordered_sample_ids)
        expected_exclusion_order.extend(ordered_exclusion_ids)
    if sample_ids != expected_sample_order or exclusion_ids != expected_exclusion_order:
        raise ValueError(f"Panel segments are not deterministically ordered: {path}")

    actual_counts = {
        "episode_anchors": len(anchors),
        "episodes": len(episodes),
        "annotated_segments": len(samples) + len(exclusions),
        "samples": len(samples),
        "exclusions": len(exclusions),
        "exclusions_by_reason": {
            reason: sum(item["reason"] == reason for item in exclusions)
            for reason in sorted({item["reason"] for item in exclusions})
        },
        "by_task": {},
    }
    for task in tasks:
        task_anchors = sum(anchor["task"] == task for anchor in anchors)
        task_episodes = sum(episode["task"] == task for episode in episodes)
        task_samples = sum(sample["task"] == task for sample in samples)
        task_exclusions = sum(item["task"] == task for item in exclusions)
        actual_counts["by_task"][task] = {
            "episode_anchors": task_anchors,
            "episodes": task_episodes,
            "annotated_segments": task_samples + task_exclusions,
            "samples": task_samples,
            "exclusions": task_exclusions,
        }
    if counts != actual_counts:
        raise ValueError(f"Panel counts are inconsistent: {path}")
    return panel


def _write_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--dataset-root", type=Path, required=True)
    create.add_argument("--segments", type=Path, required=True)
    create.add_argument("--train-episodes", type=Path, required=True)
    create.add_argument("--split-manifest", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--seed", type=int, default=1)
    create.add_argument("--task-scope", choices=("two_tasks", "full100"), default="two_tasks")
    create.add_argument(
        "--episodes-per-task",
        "--samples-per-task",
        dest="episodes_per_task",
        type=int,
        default=10,
    )
    create.add_argument("--reserve-per-task", type=int, default=100)

    check = subparsers.add_parser("check")
    check.add_argument("--panel", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--panel", type=Path, required=True)
    preflight.add_argument("--raw-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path)
    preflight.add_argument("--skip-sha256", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "create":
        panel = build_panel(
            args.dataset_root,
            args.segments,
            args.train_episodes,
            args.split_manifest,
            seed=args.seed,
            episodes_per_task=args.episodes_per_task,
            reserve_per_task=args.reserve_per_task,
            task_scope=args.task_scope,
        )
        _write_new_json(args.output, panel)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "fingerprint": panel["fingerprint"],
                    "samples": len(panel["samples"]),
                }
            )
        )
        return 0
    panel = load_panel(args.panel)
    if args.command == "check":
        print(
            json.dumps(
                {
                    "valid": True,
                    "fingerprint": panel["fingerprint"],
                    "samples": len(panel["samples"]),
                }
            )
        )
        return 0
    report = preflight_panel(panel, args.raw_root, include_sha256=not args.skip_sha256)
    if args.output is not None:
        _write_new_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
