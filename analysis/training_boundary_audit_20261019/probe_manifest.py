#!/usr/bin/env python3
"""Prepare matched boundary probes from a frozen index; never run a policy.

Output sample_keys are accepted by scripts/eval_openloop.py. Its explicit action
noise seed is part of the pairing contract; this script never runs inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from audit import canonical_digest, validate_index


def hash_order(seed, value):
    return canonical_digest({"seed": seed, "identity": value})


def build_probe(document, *, source_path, source_file_sha256, horizon=32,
                execute=16, max_per_task=3, seed=0, action_noise_seed=0):
    if any(type(x) is not int for x in (horizon, execute, max_per_task, seed, action_noise_seed)):
        raise ValueError("Horizon, execute, max-per-task and seed must be integers")
    if not 1 < execute < horizon or max_per_task < 1 or action_noise_seed < 0:
        raise ValueError("Four distinct anchors require 1 < execute < horizon, max-per-task >= 1")
    index, episodes = validate_index(document, horizon)
    grouped = defaultdict(list)
    for row in index["intervals"]:
        grouped[row["episode_index"]].append(row)
    candidates = defaultdict(list)
    rejected = Counter()
    adjacency_count = 0
    for episode_index, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["start"])
        episode = episodes[episode_index]
        if type(episode.get("task_instance_id")) is not int:
            raise ValueError("Probe requires integer task_instance_id for each episode")
        for left, right in zip(rows, rows[1:]):
            adjacency_count += 1
            if left["end"] != right["start"]:
                rejected["noncontiguous_gap"] += 1
                continue
            if left["subtask"] == right["subtask"]:
                rejected["same_label"] += 1
                continue
            if left["end"] - left["start"] < 2*horizon:
                rejected["left_shorter_than_2H"] += 1
                continue
            if right["end"] - right["start"] < horizon:
                rejected["right_shorter_than_H"] += 1
                continue
            identity = {"source_kind": index["source_kind"], "root_id": index["root_id"],
                "task_name": episode["task_name"], "task_instance_id": episode["task_instance_id"],
                "episode_index": episode_index, "left_start": left["start"],
                "boundary_frame": left["end"], "right_end": right["end"],
                "left_subtask": left["subtask"], "right_subtask": right["subtask"]}
            candidate = {**identity, "boundary_id": canonical_digest(identity),
                         "selection_hash": hash_order(seed, identity)}
            candidates[episode["task_name"]].append(candidate)
    selected = []
    by_task = {}
    for task in sorted({e["task_name"] for e in episodes.values()}):
        available = sorted(candidates[task], key=lambda item: (item["selection_hash"], item["boundary_id"]))
        seen = set()
        chosen = []
        for boundary in available:
            identity = (boundary["task_instance_id"], boundary["episode_index"])
            if identity in seen:
                continue
            seen.add(identity)
            chosen.append(boundary)
            if len(chosen) == max_per_task:
                break
        selected.extend(chosen)
        by_task[task] = {"eligible_boundaries": len(available),
            "eligible_instance_episode_pairs": len({(b["task_instance_id"], b["episode_index"]) for b in available}),
            "selected_boundaries": len(chosen), "selected_samples": len(chosen)*4,
            "under_requested_boundary_limit": len(chosen) < max_per_task}
    sample_keys = []
    seen_samples = set()
    distances = [("far_2H", 2*horizon), ("edge_H", horizon),
                 ("inside_K", execute), ("last_1", 1)]
    for boundary in selected:
        for bucket, distance in distances:
            frame = boundary["boundary_frame"] - distance
            sample_key = (index["root_id"], boundary["episode_index"], frame)
            if sample_key in seen_samples:
                raise ValueError(f"Duplicate sample key: {sample_key}")
            seen_samples.add(sample_key)
            if not boundary["left_start"] <= frame < boundary["boundary_frame"]:
                raise ValueError("Selected observation lies outside its left interval")
            if frame + horizon > boundary["right_end"]:
                raise ValueError("Probe lacks a full H-step future in the adjacent recorded intervals")
            sample_identity = {"root_id": index["root_id"],
                "episode_index": boundary["episode_index"], "frame_index": frame}
            sample_id = canonical_digest(sample_identity)
            sample_keys.append({**sample_identity, "source_kind": index["source_kind"],
                "repo_id": index["repo_id"], "task_name": boundary["task_name"],
                "task_instance_id": boundary["task_instance_id"],
                "valid_action_steps": min(horizon, distance),
                "subtask": boundary["left_subtask"], "segment_start": boundary["left_start"],
                "segment_end": boundary["boundary_frame"],
                "next_subtask": boundary["right_subtask"],
                "next_segment_end": boundary["right_end"],
                "boundary_id": boundary["boundary_id"], "sample_id": sample_id,
                "bucket": bucket, "distance_to_boundary_frames": distance,
                "full_H_legal_before": distance >= horizon,
                "executed_K_within_original_label": distance >= execute,
                "recorded_full_H_future_available_by_index": True,
                "training_seen_source": True})
    if not sample_keys:
        raise ValueError("No eligible boundaries remain")
    buckets = dict(sorted(Counter(row["bucket"] for row in sample_keys).items()))
    if len(set(buckets.values())) != 1:
        raise AssertionError("Boundary buckets are not matched")
    selected_ids = sorted({b["episode_index"] for b in selected})
    statistics = {"indexed_episodes": len(episodes), "indexed_tasks": len(by_task),
        "adjacent_interval_pairs_examined": adjacency_count,
        "rejected_pair_counts_exclusive": dict(sorted(rejected.items())),
        "eligible_boundaries": sum(len(v) for v in candidates.values()),
        "selected_boundaries": len(selected), "selected_samples": len(sample_keys),
        "selected_episodes": len(selected_ids), "tasks_with_selected_boundaries": sum(bool(v["selected_boundaries"]) for v in by_task.values()),
        "bucket_counts": buckets, "by_task": by_task}
    common = {"source_index_path": source_path, "source_index_file_sha256": source_file_sha256,
        "source_index_fingerprint": index["fingerprint"], "source_index_fingerprint_verified": True,
        "source_kind": index["source_kind"], "root_id": index["root_id"],
        "repo_id": index["repo_id"], "dataset_root_in_frozen_index": index["dataset_root"],
        "training_seen_source": True, "independent_test": False,
        "inference_executed": False, "new_supervision_collected": False}
    manifest = {"schema_version": 1, "kind": "matched_training_boundary_probe", **common,
        "selection": {"seed": seed, "max_boundaries_per_task": max_per_task,
            "distinct_within_task": "(task_instance_id,episode_index)",
            "order": "stable SHA-256 of seed and boundary semantic identity",
            "left_minimum_frames": 2*horizon, "right_minimum_frames": horizon,
            "boundary_type": "same episode, contiguous, differing nonempty labels"},
        "model_horizon_H": horizon, "executed_prefix_K": execute,
        "interpretation": "Training-seen offline diagnostic; not closed-loop success evidence or new supervision.",
        "pairing_contract": {
            "same_sample_keys_and_order_for_all_checkpoints": True,
            "same_observation_and_preprocessing_for_all_checkpoints": True,
            "same_explicit_initial_action_noise_for_each_sample_across_checkpoints": True,
            "eval_openloop_action_noise_argument": f"--action-noise-seed {action_noise_seed}",
            "action_noise_seed": action_noise_seed,
            "action_noise_algorithm": "b1k-openloop-noise-v1/sha256/PCG64/float32",
            "noise_identity": "b1k-openloop-noise-v1:{seed}:{episode}:{frame}:{H}:{D}",
            "expected_inference_evidence": "Matching per-sample action_noise_sha256 across checkpoints",
            "matching_action_noise_does_not_control_all_runtime_randomness": True,
            "valid_action_steps_masks_only_the_current_label_prefix": True,
            "full_H_future_in_index_does_not_mean_cross_label_actions_are_valid_targets_for_current_subtask": True},
        "statistics": statistics, "boundaries": selected, "sample_keys": sample_keys}
    episode_manifest = {"schema_version": 1, "kind": "boundary_probe_selected_episodes", **common,
        "episode_indices": selected_ids, "episodes": [episodes[i] for i in selected_ids],
        "selection_seed": seed}
    return manifest, episode_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-task", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--execute", type=int, default=16)
    parser.add_argument("--action-noise-seed", type=int, default=0)
    args = parser.parse_args()
    names = ("probe_manifest.json", "selected_episodes.json", "selected_episodes.txt", "probe_sampling_statistics.json")
    for name in names:
        if (args.output_dir/name).exists():
            parser.error(f"Refusing to overwrite existing artifact: {args.output_dir/name}")
    raw = args.index.read_bytes()
    manifest, episodes = build_probe(json.loads(raw), source_path=str(args.index.resolve()),
        source_file_sha256=hashlib.sha256(raw).hexdigest(), horizon=args.horizon,
        execute=args.execute, max_per_task=args.max_per_task, seed=args.seed,
        action_noise_seed=args.action_noise_seed)
    manifest["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    products = {"probe_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2)+"\n",
        "selected_episodes.json": json.dumps(episodes, ensure_ascii=False, indent=2)+"\n",
        "selected_episodes.txt": "\n".join(map(str, episodes["episode_indices"]))+"\n",
        "probe_sampling_statistics.json": json.dumps(manifest["statistics"], ensure_ascii=False, indent=2)+"\n"}
    for name, content in products.items():
        with (args.output_dir/name).open("x") as stream:
            stream.write(content)
    print(json.dumps({key: value for key, value in manifest["statistics"].items() if key != "by_task"}, indent=2))


if __name__ == "__main__":
    main()
