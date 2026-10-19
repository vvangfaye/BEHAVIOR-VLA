#!/usr/bin/env python3
"""Audit legal observation/action support of frozen interval indexes (stdlib only).

An interval is [start, end), a model target window has H actions, and deployment
executes its first K actions. This is geometry, not a measured model-error test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_digest(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode())


def empty_group():
    return {"length_histogram": Counter(), "episode_keys": set(),
            "boundary_types": Counter(), "interval_count": 0,
            "indexed_frames": 0, "legal_H_anchors": 0}


def validate_index(document, expected_horizon):
    index = document if document.get("kind") == "b1k_valid_intervals" else document.get("index", {})
    if index.get("kind") != "b1k_valid_intervals" or index.get("schema_version") != 1:
        raise ValueError("Expected schema 1 b1k_valid_intervals index")
    if index.get("action_horizon") != expected_horizon:
        raise ValueError("Frozen index horizon differs from requested model horizon")
    if index.get("fingerprint") != canonical_digest({k: v for k, v in index.items() if k != "fingerprint"}):
        raise ValueError("Index semantic fingerprint failed")
    episodes = {}
    for episode in index["episodes"]:
        number = episode["episode_index"]
        if type(number) is not int or number < 0 or number in episodes:
            raise ValueError("Invalid or duplicate episode")
        if type(episode["length"]) is not int or episode["length"] <= 0:
            raise ValueError("Invalid episode length")
        if not isinstance(episode.get("task_name"), str) or not episode["task_name"]:
            raise ValueError("Missing task name")
        episodes[number] = episode
    previous = {}
    for row in sorted(index["intervals"], key=lambda r: (r["episode_index"], r["start"])):
        ep, start, end = row["episode_index"], row["start"], row["end"]
        if any(type(v) is not int for v in (ep, start, end)):
            raise ValueError("Noninteger interval identity")
        if ep not in episodes or not 0 <= start < end <= episodes[ep]["length"]:
            raise ValueError("Invalid interval bounds")
        if end - start < expected_horizon or start < previous.get(ep, 0):
            raise ValueError("Short or overlapping frozen interval")
        if not isinstance(row.get("subtask"), str) or not row["subtask"].strip():
            raise ValueError("Missing conditioning label")
        previous[ep] = end
    if set(previous) != set(episodes):
        raise ValueError("Indexed episode has no eligible interval")
    return index, episodes


def coverage_for_length(length, horizon, execute, distance):
    """Exact counts at target/observation frame t = end - distance in one segment."""
    if distance > length:
        return (0, 0, 0, 0)
    frame = length - distance
    latest_anchor = length - horizon

    def target_occurrences(slots):
        return max(0, min(frame, latest_anchor) - max(0, frame - slots + 1) + 1)

    return (1, int(frame <= latest_anchor), target_occurrences(horizon),
            target_occurrences(execute))


def summarize(group, horizon, execute):
    histogram = group["length_histogram"]
    segments = group["interval_count"]
    frames = group["indexed_frames"]
    anchors = group["legal_H_anchors"]
    by_distance = []
    for distance in range(1, horizon * 2 + 1):
        totals = [0, 0, 0, 0]
        for length, count in histogram.items():
            row = coverage_for_length(length, horizon, execute, distance)
            for i, value in enumerate(row):
                totals[i] += value * count
        positions, legal, all_targets, executed_targets = totals
        by_distance.append({"distance_to_interval_end_frames": distance,
            "observation_positions": positions, "legal_observation_anchors": legal,
            "positions_ever_targeted_in_any_H_slot": positions,
            "positions_ever_targeted_in_executed_K_slots": positions if executed_targets else 0,
            "all_H_anchor_target_pairs": all_targets,
            "executed_K_anchor_target_pairs": executed_targets})
    # At one fixed distance, prefix support is either present for all intervals
    # long enough to contain that frame, or absent for all. This is tested below.
    near_frames = sum(r["observation_positions"] for r in by_distance)
    near_anchors = sum(r["legal_observation_anchors"] for r in by_distance)
    near_all_targets = sum(r["all_H_anchor_target_pairs"] for r in by_distance)
    near_executed_targets = sum(r["executed_K_anchor_target_pairs"] for r in by_distance)
    no_prefix_positions = (horizon - execute) * segments
    near_prefix_positions = sum(r["positions_ever_targeted_in_executed_K_slots"] for r in by_distance)
    by_distance.append({"distance_to_interval_end_frames": f">{2*horizon}",
        "observation_positions": frames - near_frames,
        "legal_observation_anchors": anchors - near_anchors,
        "positions_ever_targeted_in_any_H_slot": frames - near_frames,
        "positions_ever_targeted_in_executed_K_slots": frames - no_prefix_positions - near_prefix_positions,
        "all_H_anchor_target_pairs": anchors * horizon - near_all_targets,
        "executed_K_anchor_target_pairs": anchors * execute - near_executed_targets})
    excluded = (horizon - 1) * segments
    extra_k = (horizon - execute) * segments
    return {"episodes": len(group["episode_keys"]), "intervals": segments,
        "indexed_frames": frames, "legal_H_anchors": anchors,
        "excluded_observation_anchor_positions": excluded,
        "excluded_observation_anchor_fraction": excluded / frames if frames else 0,
        "legal_K_anchors_in_retained_intervals": anchors + extra_k,
        "K_legal_anchors_excluded_by_H_requirement": extra_k,
        "K_legal_anchor_exclusion_fraction": extra_k / (anchors + extra_k) if anchors + extra_k else 0,
        "action_positions_never_targeted_in_executed_K_slots": no_prefix_positions,
        "action_positions_never_targeted_in_any_H_slot": 0,
        "boundary_types": dict(sorted(group["boundary_types"].items())),
        "distance_to_interval_end_coverage": by_distance}


def audit_cohort(paths, horizon=32, execute=16):
    if type(horizon) is not int or type(execute) is not int or not 1 <= execute <= horizon:
        raise ValueError("Require integers 1 <= execute <= horizon")
    sources, tasks = defaultdict(empty_group), defaultdict(empty_group)
    identities, inputs = set(), []
    for path in sorted(map(Path, paths)):
        raw = path.read_bytes()
        index, episodes = validate_index(json.loads(raw), horizon)
        kind, root = index["source_kind"], index["root_id"]
        for number in episodes:
            identity = (root, number)
            if identity in identities:
                raise ValueError(f"Duplicate episode within cohort: {identity}")
            identities.add(identity)
        rows_by_episode = defaultdict(list)
        for row in index["intervals"]:
            rows_by_episode[row["episode_index"]].append(row)
        for number, rows in rows_by_episode.items():
            rows.sort(key=lambda r: r["start"])
            episode = episodes[number]
            for pos, row in enumerate(rows):
                if row["end"] == episode["length"]:
                    boundary_type = "episode_end"
                elif pos + 1 == len(rows):
                    boundary_type = "unindexed_episode_tail"
                elif rows[pos + 1]["start"] > row["end"]:
                    boundary_type = "unindexed_gap"
                elif rows[pos + 1]["subtask"] == row["subtask"]:
                    boundary_type = "contiguous_same_label"
                else:
                    boundary_type = "contiguous_different_label"
                for group in (sources[kind], tasks[(kind, episode["task_name"])]):
                    length = row["end"] - row["start"]
                    group["length_histogram"][length] += 1
                    group["episode_keys"].add((root, number))
                    group["boundary_types"][boundary_type] += 1
                    group["interval_count"] += 1
                    group["indexed_frames"] += length
                    group["legal_H_anchors"] += length - horizon + 1
        stats = index.get("provenance", {}).get("statistics", {})
        inputs.append({"path": str(path.resolve()), "file_sha256": digest_bytes(raw),
            "index_fingerprint": index["fingerprint"], "fingerprint_verified": True,
            "source_kind": kind, "root_id": root, "episodes": len(episodes),
            "indexed_intervals": len(index["intervals"]),
            "all_declared_episode_frames": sum(e["length"] for e in episodes.values()),
            "indexed_frames": sum(r["end"]-r["start"] for r in index["intervals"]),
            "short_segments_reported_by_source_metadata": stats.get("short_segments"),
            "short_segment_frames_reported_by_source_metadata": stats.get("short_segment_frames"),
            "provenance_action_boundary_policy": index.get("provenance", {}).get("action_boundary_policy")})
    return {"inputs": inputs,
        "by_source": {kind: summarize(g, horizon, execute) for kind, g in sorted(sources.items())},
        "by_source_and_task": {f"{kind}/{task}": summarize(g, horizon, execute)
                               for (kind, task), g in sorted(tasks.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", required=True, metavar="NAME=PATH",
                        help="Repeatable; PATH is an index or directory of *.json indexes. Same-name inputs combine.")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--execute", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = defaultdict(list)
    for value in args.cohort:
        name, separator, location = value.partition("=")
        if not separator or not name or not location:
            parser.error("--cohort must be NAME=PATH")
        path = Path(location)
        paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
        if not paths:
            parser.error(f"No indexes in {path}")
        cohorts[name].extend(paths)
    result = {"schema_version": 1, "analysis": "frozen_interval_window_support",
        "implementation_sha256": digest_bytes(Path(__file__).read_bytes()),
        "model_horizon_H": args.horizon, "executed_prefix_K": args.execute,
        "definitions": {"interval": "[start,end), from the frozen eligible index",
            "distance": "end-frame, so distance 1 is the final frame",
            "legal_anchor": "start <= frame <= end-H",
            "prefix_support": "Whether an action frame ever appears in slots 0..K-1 of any legal H window",
            "counts": "Unweighted structural support, not realized sampler exposure or model errors"},
        "limitations": [
            "Intervals are index boundaries; they need not all be physical events or semantic task changes.",
            "Excluded observation anchors can still appear as action targets in earlier windows.",
            "Every retained interval has length >= H. Short discarded intervals are absent; aggregate metadata cannot locate or classify them per task.",
            "K-legal support recovered by relaxing H is a lower bound for the original dataset because discarded K..H-1 length intervals are unavailable.",
            "Unindexed gaps/tails may represent omitted short segments, missing labels or filtering; this audit cannot distinguish them.",
            "Cohorts may overlap; do not sum cohorts as independent unique data.",
            "This audit does not establish causation, policy performance, an optimal training target, or algorithmic novelty."],
        "cohorts": {name: audit_cohort(paths, args.horizon, args.execute)
                    for name, paths in sorted(cohorts.items())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output": str(args.output.resolve()),
        "cohorts": {name: {kind: {k: v for k, v in summary.items()
                     if k != "distance_to_interval_end_coverage"} for kind, summary in cohort["by_source"].items()}
                    for name, cohort in result["cohorts"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
