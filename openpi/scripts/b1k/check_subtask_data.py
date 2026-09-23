"""Check BEHAVIOR subtask coverage and token budgets without loading videos or model weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq

from openpi.configs import ROBOT_REGISTRY
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.subtask_data import SubtaskFromSegments
from openpi.training.subtask_data import episode_lengths_in_dataset_order


def episode_metadata(root: Path) -> tuple[dict[int, int], dict[int, int]]:
    lengths = {}
    task_indices = {}
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        schema_names = pq.read_schema(path).names
        columns = ["episode_index", "length"]
        if "task_index" in schema_names:
            columns.append("task_index")
        for row in pq.read_table(path, columns=columns).to_pylist():
            ep, length = int(row["episode_index"]), int(row["length"])
            if ep in lengths:
                raise ValueError(f"Duplicate episode_index={ep} in episode metadata.")
            lengths[ep] = length
            if "task_index" in row:
                task_indices[ep] = int(row["task_index"])
    if not lengths:
        raise ValueError(f"No LeRobot v3 episode metadata found under {root / 'meta' / 'episodes'}.")
    return lengths, task_indices


def episode_lengths(root: Path) -> dict[int, int]:
    return episode_metadata(root)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_list_sha256(episodes: list[int]) -> str:
    return hashlib.sha256("".join(f"{episode}\n" for episode in episodes).encode()).hexdigest()


def task_texts(root: Path) -> set[str]:
    # Include both formats as a conservative bound: the loader prefers the
    # natural-language JSONL descriptions and falls back to LeRobot metadata.
    texts = set()
    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                text = json.loads(line).get("task")
                if isinstance(text, str) and text.strip():
                    texts.add(text)
    parquet = root / "meta" / "tasks.parquet"
    if parquet.is_file():
        frame = pq.read_table(parquet).to_pandas()
        for index, row in frame.iterrows():
            text = row.get("task", index)
            if isinstance(text, str) and text.strip():
                texts.add(text)
    if not texts:
        raise ValueError("No task instruction text found in meta/tasks.jsonl or meta/tasks.parquet.")
    return texts


def check_dataset(
    dataset_root: Path,
    segments_path: Path,
    *,
    episodes_index: list[int] | None = None,
    robot: str = "b1k/R1Pro",
    max_token_len: int = 512,
    max_subtask_len: int = 64,
    empty_label_policy: Literal["error", "drop_episode"] = "error",
    frame_stride: int = 1,
    global_batch_size: int | None = None,
    process_count: int = 1,
    epochs: int | None = None,
    expected_tasks: int | None = None,
    expected_task_indices: list[int] | None = None,
) -> dict:
    all_lengths, task_indices = episode_metadata(dataset_root)
    lengths = episode_lengths_in_dataset_order(dataset_root, episodes_index)
    if episodes_index is not None:
        task_indices = {ep: task_indices[ep] for ep in lengths if ep in task_indices}
    elif set(lengths) != set(all_lengths):
        raise ValueError("Episode metadata readers disagree on available episode IDs.")
    requested_lengths = lengths.copy()
    labels = SubtaskFromSegments.from_parquet(
        segments_path,
        episodes_index=list(lengths),
        empty_label_policy=empty_label_policy,
    )
    lengths = {ep: length for ep, length in lengths.items() if ep not in labels.dropped_episodes}
    if not lengths:
        raise ValueError(
            f"No episodes remain after dropping {len(labels.dropped_episodes)} episodes with empty subtask labels."
        )
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive.")
    if process_count < 1:
        raise ValueError("process_count must be positive.")
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive.")
    if global_batch_size is None and epochs is not None:
        raise ValueError("global_batch_size is required when epochs is set.")
    if global_batch_size is not None and (global_batch_size < 1 or global_batch_size % process_count):
        raise ValueError("global_batch_size must be positive and divisible by process_count.")

    labels.validate_coverage(lengths)
    selected_episodes = list(lengths)
    retained_task_indices = sorted({task_indices[ep] for ep in selected_episodes if ep in task_indices})
    if (expected_tasks is not None or expected_task_indices is not None) and len(task_indices) != len(
        requested_lengths
    ):
        raise ValueError("Episode metadata must include task_index when expected tasks are set.")
    if expected_tasks is not None and len(retained_task_indices) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} retained tasks, found {len(retained_task_indices)}.")
    if expected_task_indices is not None:
        expected_indices = sorted(set(expected_task_indices))
        if retained_task_indices != expected_indices:
            raise ValueError(f"Expected retained task indices {expected_indices}, found {retained_task_indices}.")

    vocabulary = {text for ep in lengths for text in labels.segments[ep][2]}
    if not vocabulary:
        raise ValueError("No annotated subtask text remains for preflight.")
    robot_config = ROBOT_REGISTRY[robot]
    state_dim = sum(1 if p.is_eef else len(p.indices) for p in robot_config.proprio)
    state = np.ones(state_dim, dtype=np.float32)
    tokenizer = PaligemmaTokenizer(max_token_len)
    instructions = task_texts(dataset_root)
    longest_task = max(instructions, key=lambda text: tokenizer.subtask_token_lengths(text, "x", state)[0])
    longest_label = max(vocabulary, key=lambda text: tokenizer.subtask_token_lengths("x", text, state)[1])
    prefix_len, label_len, cue_len = tokenizer.subtask_token_lengths(longest_task, longest_label, state)
    tokenizer.tokenize_subtask_training(longest_task, longest_label, state, max_subtask_len=max_subtask_len)
    tokenizer.tokenize_subtask_prefix(longest_task, state, max_subtask_len=max_subtask_len)
    token_report = {
        "max_prefix_tokens": prefix_len,
        "max_label_tokens_including_eos": label_len,
        "action_cue_tokens": cue_len,
        "conservative_sequence_tokens": prefix_len + label_len + cue_len,
        "generation_sequence_budget": prefix_len + max_subtask_len + cue_len,
        "max_subtask_len": max_subtask_len,
    }

    frames = sum(lengths.values())
    strided_samples = math.ceil(frames / frame_stride)
    contract = {
        "annotation_sha256": _sha256(segments_path),
        "episode_list_sha256": _episode_list_sha256(selected_episodes),
        "frame_stride": frame_stride,
        "process_count": process_count,
        "retained_episodes": len(selected_episodes),
        "retained_frames": frames,
        "retained_tasks": len(retained_task_indices),
        "strided_samples": strided_samples,
    }
    role_counts = labels.role_counts(selected_episodes)
    provenance_counts = labels.provenance_counts(selected_episodes)
    if global_batch_size is not None:
        local_batch_size = global_batch_size // process_count
        samples_per_process = strided_samples // process_count
        batches_per_epoch = samples_per_process // local_batch_size
        contract.update(
            global_batch_size=global_batch_size,
            local_batch_size=local_batch_size,
            samples_per_process=samples_per_process,
            batches_per_epoch=batches_per_epoch,
            dropped_samples_per_epoch=strided_samples - batches_per_epoch * global_batch_size,
        )
        if epochs is not None:
            contract.update(epochs=epochs, train_steps=epochs * batches_per_epoch)
    contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    return {
        "dataset_root": str(dataset_root.resolve()),
        "segments_path": str(segments_path.resolve()),
        "empty_label_policy": empty_label_policy,
        "requested_episodes": len(requested_lengths),
        "requested_frames": sum(requested_lengths.values()),
        "selected_episode_indices": selected_episodes,
        "selected_episode_indices_sha256": contract["episode_list_sha256"],
        "dropped_episode_indices": sorted(labels.dropped_episodes),
        "dropped_episode_frames": sum(requested_lengths[ep] for ep in labels.dropped_episodes),
        "empty_label_segments": sum(len(intervals) for intervals in labels.dropped_episodes.values()),
        "dropped_episodes": [
            {"episode_index": ep, "reason": "empty_subtask_label", "empty_intervals": intervals}
            for ep, intervals in sorted(labels.dropped_episodes.items())
        ],
        "episodes": len(lengths),
        "frames": frames,
        "retained_task_indices": retained_task_indices,
        "tasks": len(retained_task_indices),
        "segments": sum(len(labels.segments[ep][0]) for ep in lengths),
        "unique_subtasks": len(vocabulary),
        "annotation_role_column": labels.annotation_column,
        "provenance_column": labels.provenance_column,
        "provenance_counts": provenance_counts,
        **role_counts,
        "state_dim": state_dim,
        "max_token_len": max_token_len,
        **token_report,
        "training_contract": contract,
        "training_contract_sha256": contract_sha256,
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--segments-path", type=Path)
    parser.add_argument("--episodes-index", type=int, nargs="+")
    parser.add_argument("--robot", default="b1k/R1Pro")
    parser.add_argument("--max-token-len", type=int, default=512)
    parser.add_argument("--max-subtask-len", type=int, default=64)
    parser.add_argument("--empty-label-policy", choices=("error", "drop_episode"), default="error")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--process-count", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--expected-task-indices", type=int, nargs="+")
    parser.add_argument("--report-file", type=Path)
    parser.add_argument(
        "--write-episodes-file", type=Path, help="Write the validated original episode IDs for training."
    )
    args = parser.parse_args()
    report = check_dataset(
        args.dataset_root,
        args.segments_path or args.dataset_root / "meta" / "subtask_segments.parquet",
        episodes_index=args.episodes_index,
        robot=args.robot,
        max_token_len=args.max_token_len,
        max_subtask_len=args.max_subtask_len,
        empty_label_policy=args.empty_label_policy,
        frame_stride=args.frame_stride,
        global_batch_size=args.global_batch_size,
        process_count=args.process_count,
        epochs=args.epochs,
        expected_tasks=args.expected_tasks,
        expected_task_indices=args.expected_task_indices,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(text + "\n")
    if args.write_episodes_file:
        args.write_episodes_file.parent.mkdir(parents=True, exist_ok=True)
        args.write_episodes_file.write_text("".join(f"{ep}\n" for ep in report["selected_episode_indices"]))
    # Keep launch logs compact; the JSON report retains every ID and interval.
    summary = {
        k: v
        for k, v in report.items()
        if k not in ("selected_episode_indices", "dropped_episode_indices", "dropped_episodes")
    }
    summary["dropped_episodes"] = len(report["dropped_episode_indices"])
    summary["report_file"] = str(args.report_file) if args.report_file else None
    summary["episodes_file"] = str(args.write_episodes_file) if args.write_episodes_file else None
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
