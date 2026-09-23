#!/usr/bin/env python
"""Build a per-frame sub-task language table for BEHAVIOR-1K from the dataset's
skill annotations, so training/inference can condition on the CURRENT sub-task
instead of one whole-episode prompt.

Verified alignment:
  * Each dataset episode carries an authoritative ``annotation_path`` column in
    meta/episodes/*.parquet that names its annotation JSON directly, so no
    positional pairing heuristic is needed (an earlier per-task rank pairing was
    checked to agree with it on all 20000 episodes).
  * A dataset episode shares the annotation's frame origin (frame 0 == frame 0),
    so an annotation skill segment [start,end) maps onto dataset LOCAL frame
    indices, and frames at/after task_duration inherit the last skill.
  * Usually the dataset episode is >= the annotated duration (extra trailing
    settle frames), but 53 of 20000 episodes are SHORTER than their annotation
    (worst: episode_index 544, length 148 vs task_duration 8790). Intervals are
    clipped to the episode length, so those demos are labelled only up to where
    the recorded data actually ends.

Output: a compact run-length-encoded segment table, one row per contiguous sub-task
run, written to <dataset>/meta/subtask_segments.parquet:
  episode_index (int64), start (int64), end (int64, exclusive), subtask (str)
The runs tile [0, episode_length) so a frame lookup is an interval search. This is
~1000x smaller than a per-frame table, so every dataloader worker can hold it.
The opt-in role_aware_v1 format also retains source boundaries and audit columns;
the default legacy format keeps the historical four-column schema and labels.

  openpi/.venv/bin/python scripts/gen_subtask_prompts.py \
      --dataset /var/tmp/behavior-1k/2026-challenge-demos \
      --annotations /var/tmp/behavior-1k/2026-challenge-demos/annotations
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq


def _obj_phrase(seg: dict) -> str:
    """Human-readable object phrase from a skill/primitive segment."""

    def leaves(x):
        out = []
        if isinstance(x, (list, tuple)):
            for e in x:
                out.extend(leaves(e))
        elif x is not None:
            out.append(str(x))
        return out

    manip = leaves(seg.get("manipulating_object_id"))
    objs = leaves(seg.get("object_id"))

    def clean(x):
        s = re.sub(r"_[0-9]+$", "", str(x))  # drop trailing instance number
        return s.replace("_", " ").strip()

    target = manip[0] if manip else (objs[0] if objs else None)
    ct = clean(target) if target else ""
    # a reference object only if it is genuinely different from the target
    ref = None
    for o in objs:
        if o != target and clean(o) and clean(o) != ct:
            ref = o
            break
    if ct and ref:
        return f"{ct} to {clean(ref)}"
    return ct


def _subtask_text(seg: dict) -> str:
    """Return the historical label text. Keep this unchanged for legacy output."""
    desc = seg.get("skill_description") or seg.get("primitive_description") or []
    desc = desc[0] if isinstance(desc, list) and desc else (desc if isinstance(desc, str) else "")
    obj = _obj_phrase(seg)
    return f"{desc} {obj}".strip() if obj else str(desc).strip()


def _leaves(value) -> list[str]:
    result = []
    if isinstance(value, (list, tuple)):
        for item in value:
            result.extend(_leaves(item))
    elif value is not None:
        result.append(str(value))
    return result


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _raw_description(segment: dict) -> str:
    descriptions = segment.get("skill_description") or segment.get("primitive_description") or []
    if isinstance(descriptions, str):
        return descriptions
    values = _leaves(descriptions)
    return values[0] if values else ""


def _description(segment: dict) -> str:
    return " ".join(_raw_description(segment).replace("_", " ").split()).lower()


def _object_base(raw_id: str) -> str:
    # Scene fixtures use category_modelhash_0; task objects use category_instance.
    match = re.fullmatch(r"(.+)_([a-z]{6})_(\d+)", raw_id)
    value = match.group(1) if match and match.group(3) == "0" else re.sub(r"_[0-9]+$", "", raw_id)
    return " ".join(value.replace("_", " ").split()).lower()


def stable_object_aliases(segments: list[dict]) -> dict[str, str]:
    """Build deterministic aliases, adding ordinals only for distinct raw IDs."""
    raw_ids = set()
    for segment in segments:
        raw_ids.update(_leaves(segment.get("manipulating_object_id")))
        raw_ids.update(_leaves(segment.get("object_id")))

    by_base = {}
    for raw_id in sorted(raw_ids):
        by_base.setdefault(_object_base(raw_id), []).append(raw_id)

    aliases = {}
    for base, ids in by_base.items():
        if len(ids) == 1:
            aliases[ids[0]] = base
        else:
            aliases.update({raw_id: f"{base} {index}" for index, raw_id in enumerate(ids, start=1)})
    return aliases


def _normalized_prefix(value, aliases: dict[str, str]) -> str:
    parts = []
    for raw_part in _leaves(value):
        part = aliases.get(raw_part, " ".join(raw_part.replace("_", " ").split()).lower())
        if part:
            parts.append(part)
    return " ".join(parts)


def _segment_roles(segment: dict) -> tuple[list[str], list[str], list[str]]:
    manipulating_ids = _unique(_leaves(segment.get("manipulating_object_id")))
    object_ids = _unique(_leaves(segment.get("object_id")))
    target_ids = manipulating_ids or object_ids[:1]
    target_set = set(target_ids)
    reference_ids = [raw_id for raw_id in object_ids if raw_id not in target_set]
    return manipulating_ids, object_ids, reference_ids


def role_aware_subtask_text(segment: dict, aliases: dict[str, str]) -> str:
    """Render a label from annotation descriptions and object-role fields only."""
    description = _description(segment)
    manipulating_ids, object_ids, reference_ids = _segment_roles(segment)
    target_id = manipulating_ids[0] if manipulating_ids else (object_ids[0] if object_ids else None)
    target = aliases.get(target_id, _object_base(target_id)) if target_id is not None else ""
    references = [aliases.get(raw_id, _object_base(raw_id)) for raw_id in reference_ids]

    if description == "move to":
        core = f"move to {target}" if target else description
    elif description == "pick up from":
        core = f"pick up {target}" if target else "pick up"
        if references:
            core = f"{core} from {references[0]}"
    elif description.startswith("place "):
        relation = description.removeprefix("place ").strip()
        core = f"place {target}" if target else "place"
        if relation and references:
            if relation.endswith(" next to") and len(references) > 1:
                primary_relation = relation.removesuffix(" next to")
                core = f"{core} {primary_relation} {references[0]} next to {references[1]}"
            else:
                core = f"{core} {relation} {references[0]}"
    elif description in {"open door", "close door", "open lid", "close lid"}:
        action, part = description.split()
        container = references[0] if references else target
        core = f"{action} {part}"
        if container and container != part:
            core = f"{core} of {container}"
    else:
        core = f"{description} {target}".strip() if target else description

    memory_prefix = _normalized_prefix(segment.get("memory_prefix"), aliases)
    spatial_prefix = _normalized_prefix(segment.get("spatial_prefix"), aliases)
    return " ".join(part for part in (memory_prefix, spatial_prefix, core) if part)


def _intervals(segment: dict) -> list[list[int]]:
    duration = segment.get("frame_duration")
    if not duration:
        return []
    return duration if isinstance(duration[0], list) else [duration]


def build_legacy_segments(
    annotation: dict, episode_index: int, length: int, segment_key: str
) -> tuple[list[dict], int, set[str]]:
    """Build legacy rows with the same text, overwrite, fill, and RLE semantics."""
    labels = [None] * length
    vocabulary = set()
    for segment in annotation.get(segment_key) or []:
        intervals = _intervals(segment)
        if not intervals:
            continue
        text = _subtask_text(segment)
        vocabulary.add(text)
        for interval in intervals:
            start, end = int(interval[0]), int(interval[1])
            for frame in range(max(0, start), min(length, end)):
                labels[frame] = text

    uncovered = 0
    first = next((label for label in labels if label is not None), None)
    last = None
    for frame in range(length):
        if labels[frame] is None:
            labels[frame] = last if last is not None else (first if first is not None else "")
            uncovered += 1
        else:
            last = labels[frame]

    rows = []
    run_start = 0
    for frame in range(1, length + 1):
        if frame == length or labels[frame] != labels[run_start]:
            rows.append(
                {"episode_index": episode_index, "start": run_start, "end": frame, "subtask": labels[run_start]}
            )
            run_start = frame
    return rows, uncovered, vocabulary


def _source_index(segment: dict, segment_key: str) -> int | None:
    key = "primitive_idx" if segment_key == "primitive_annotation" else "skill_idx"
    values = _leaves(segment.get(key))
    if not values:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


def _role_record(
    segment: dict,
    *,
    aliases: dict[str, str],
    annotation_path: str,
    segment_key: str,
    source_segment_index: int,
    source_interval_index: int,
) -> dict:
    manipulating_ids, object_ids, reference_ids = _segment_roles(segment)
    return {
        "subtask": role_aware_subtask_text(segment, aliases),
        "annotation_path": annotation_path,
        "source_segment_index": source_segment_index,
        "source_interval_index": source_interval_index,
        "source_skill_index": _source_index(segment, segment_key),
        "skill_description": _raw_description(segment),
        "manipulating_object_ids": manipulating_ids,
        "reference_object_ids": reference_ids,
        "object_ids": object_ids,
        "memory_prefix": _normalized_prefix(segment.get("memory_prefix"), aliases),
        "spatial_prefix": _normalized_prefix(segment.get("spatial_prefix"), aliases),
        "provenance": "annotated",
    }


def build_role_aware_segments(
    annotation: dict, episode_index: int, length: int, annotation_path: str, segment_key: str = "skill_annotation"
) -> tuple[list[dict], int, set[str]]:
    """Build auditable role-aware rows while retaining source annotation boundaries."""
    segments = annotation.get(segment_key) or []
    aliases = stable_object_aliases(segments)
    frames = [None] * length
    vocabulary = set()

    for segment_index, segment in enumerate(segments):
        for interval_index, interval in enumerate(_intervals(segment)):
            record = _role_record(
                segment,
                aliases=aliases,
                annotation_path=annotation_path,
                segment_key=segment_key,
                source_segment_index=segment_index,
                source_interval_index=interval_index,
            )
            vocabulary.add(record["subtask"])
            start, end = int(interval[0]), int(interval[1])
            for frame in range(max(0, start), min(length, end)):
                frames[frame] = record

    annotated_frames = [frame for frame, record in enumerate(frames) if record is not None]
    first_annotated = frames[annotated_frames[0]] if annotated_frames else None
    last_annotated_frame = annotated_frames[-1] if annotated_frames else None
    uncovered = 0
    last = None
    for frame, record in enumerate(frames):
        if record is not None:
            last = record
            continue
        uncovered += 1
        if last is None:
            if first_annotated is None:
                frames[frame] = {
                    "subtask": "",
                    "annotation_path": annotation_path,
                    "source_segment_index": None,
                    "source_interval_index": None,
                    "source_skill_index": None,
                    "skill_description": "",
                    "manipulating_object_ids": [],
                    "reference_object_ids": [],
                    "object_ids": [],
                    "memory_prefix": "",
                    "spatial_prefix": "",
                    "provenance": "imputed_leading",
                }
            else:
                frames[frame] = {**first_annotated, "provenance": "imputed_leading"}
        else:
            provenance = "imputed_tail" if frame > last_annotated_frame else "imputed_gap"
            frames[frame] = {**last, "provenance": provenance}

    rows = []
    run_start = 0
    for frame in range(1, length + 1):
        if frame == length or frames[frame] != frames[run_start]:
            rows.append(
                {
                    "episode_index": episode_index,
                    "start": run_start,
                    "end": frame,
                    **frames[run_start],
                }
            )
            run_start = frame
    return rows, uncovered, vocabulary


def load_episode_meta(dataset: str, task_indices: set[int] | None = None):
    """[(episode_index, length, annotation_path)] sorted by episode_index."""
    rows = []
    for path in sorted(glob.glob(f"{dataset}/meta/episodes/chunk-*/*.parquet")):
        columns = ["episode_index", "length", "annotation_path"]
        if task_indices is not None:
            columns.append("task_index")
        table = pq.read_table(path, columns=columns).to_pydict()
        for index, episode in enumerate(table["episode_index"]):
            if task_indices is not None and int(table["task_index"][index]) not in task_indices:
                continue
            rows.append((int(episode), int(table["length"][index]), table["annotation_path"][index]))
    rows.sort(key=lambda row: row[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/var/tmp/behavior-1k/2026-challenge-demos")
    ap.add_argument("--annotations", default=None, help="annotations root; default <dataset>/annotations")
    ap.add_argument("--use-primitive", action="store_true", help="use primitive_annotation instead of skill_annotation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--task-indices", type=int, nargs="+", help="only generate episodes for these task indices")
    ap.add_argument(
        "--label-format",
        choices=("legacy", "role_aware_v1"),
        default="legacy",
        help="label renderer and output schema; legacy preserves the historical four-column output",
    )
    ap.add_argument(
        "--min-annotated-coverage",
        type=float,
        default=0.0,
        help=(
            "skip episodes whose recorded length covers less than this fraction of the annotation's task_duration "
            "(0 = keep all). e.g. 0.5 drops episode 544 (148 frames vs 8790 annotated)"
        ),
    )
    args = ap.parse_args()
    out = args.out or f"{args.dataset}/meta/subtask_segments.parquet"
    seg_key = "primitive_annotation" if args.use_primitive else "skill_annotation"

    task_indices = None if args.task_indices is None else set(args.task_indices)
    episodes = load_episode_meta(args.dataset, task_indices)
    if not episodes:
        raise ValueError("No episodes match the requested task indices.")
    print(f"dataset episodes: {len(episodes)}")

    rows = []
    n_uncovered = 0
    n_ep_done = 0
    n_missing_files = 0
    n_skipped_coverage = 0
    vocab = set()

    for episode_index, length, annotation_path in episodes:
        # annotation_path is relative to the dataset root; allow overriding the
        # annotations root while keeping the task-XXXX/episode_*.json tail.
        if args.annotations:
            path = os.path.join(args.annotations, *annotation_path.split("/")[1:])
        else:
            path = os.path.join(args.dataset, annotation_path)
        if not os.path.exists(path):
            n_missing_files += 1
            continue
        with open(path) as annotation_file:
            annotation = json.load(annotation_file)
        if args.min_annotated_coverage > 0:
            duration = (annotation.get("meta_data") or {}).get("task_duration") or 0
            if duration > 0 and length / duration < args.min_annotated_coverage:
                n_skipped_coverage += 1
                continue
        if args.label_format == "role_aware_v1":
            episode_rows, uncovered, episode_vocab = build_role_aware_segments(
                annotation, episode_index, length, annotation_path, seg_key
            )
        else:
            episode_rows, uncovered, episode_vocab = build_legacy_segments(annotation, episode_index, length, seg_key)
        rows.extend(episode_rows)
        n_uncovered += uncovered
        vocab.update(episode_vocab)
        n_ep_done += 1

    if args.label_format == "role_aware_v1":
        tbl = pa.table(
            {
                "episode_index": pa.array([row["episode_index"] for row in rows], pa.int64()),
                "start": pa.array([row["start"] for row in rows], pa.int64()),
                "end": pa.array([row["end"] for row in rows], pa.int64()),
                "subtask": pa.array([row["subtask"] for row in rows], pa.string()),
                "annotation_path": pa.array([row["annotation_path"] for row in rows], pa.string()),
                "source_segment_index": pa.array([row["source_segment_index"] for row in rows], pa.int64()),
                "source_interval_index": pa.array([row["source_interval_index"] for row in rows], pa.int64()),
                "source_skill_index": pa.array([row["source_skill_index"] for row in rows], pa.int64()),
                "skill_description": pa.array([row["skill_description"] for row in rows], pa.string()),
                "manipulating_object_ids": pa.array(
                    [row["manipulating_object_ids"] for row in rows], pa.list_(pa.string())
                ),
                "reference_object_ids": pa.array([row["reference_object_ids"] for row in rows], pa.list_(pa.string())),
                "object_ids": pa.array([row["object_ids"] for row in rows], pa.list_(pa.string())),
                "memory_prefix": pa.array([row["memory_prefix"] for row in rows], pa.string()),
                "spatial_prefix": pa.array([row["spatial_prefix"] for row in rows], pa.string()),
                "is_annotated": pa.array([row["provenance"] == "annotated" for row in rows], pa.bool_()),
                "provenance": pa.array([row["provenance"] for row in rows], pa.string()),
            }
        )
    else:
        tbl = pa.table(
            {
                "episode_index": pa.array([row["episode_index"] for row in rows], pa.int64()),
                "start": pa.array([row["start"] for row in rows], pa.int64()),
                "end": pa.array([row["end"] for row in rows], pa.int64()),
                "subtask": pa.array([row["subtask"] for row in rows], pa.string()),
            }
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pq.write_table(tbl, out)

    total_frames = sum(r[1] for r in episodes)
    print(f"\nwrote {out}")
    print(
        f"  episodes covered: {n_ep_done}/{len(episodes)}   missing annotation files: {n_missing_files}"
        f"   skipped(low coverage): {n_skipped_coverage}"
    )
    print(f"  segments: {len(rows):,}   (avg {len(rows) / max(n_ep_done, 1):.1f}/episode)")
    print(
        f"  frames: {total_frames:,}   uncovered/tail-filled: {n_uncovered:,} "
        f"({n_uncovered / max(total_frames, 1) * 100:.1f}%)"
    )
    print(f"  distinct sub-task strings: {len(vocab)}")
    print("  sample sub-tasks:")
    for v in sorted(vocab)[:10]:
        print(f"    - {v!r}")


if __name__ == "__main__":
    main()
