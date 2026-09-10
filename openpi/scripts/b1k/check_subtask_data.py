"""Check BEHAVIOR subtask coverage and token budgets without loading videos or model weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.configs import ROBOT_REGISTRY
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.subtask_data import SubtaskFromSegments


def episode_lengths(root: Path) -> dict[int, int]:
    lengths = {}
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist():
            ep, length = int(row["episode_index"]), int(row["length"])
            if ep in lengths:
                raise ValueError(f"Duplicate episode_index={ep} in episode metadata.")
            lengths[ep] = length
    if not lengths:
        raise ValueError(f"No LeRobot v3 episode metadata found under {root / 'meta' / 'episodes'}.")
    return lengths


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
) -> dict:
    lengths = episode_lengths(dataset_root)
    if episodes_index is not None:
        unknown = set(episodes_index) - lengths.keys()
        if unknown:
            raise ValueError(f"Unknown selected episode ids: {sorted(unknown)[:10]}")
        lengths = {ep: lengths[ep] for ep in episodes_index}
    labels = SubtaskFromSegments.from_parquet(segments_path)
    labels.validate_coverage(lengths)
    instructions = task_texts(dataset_root)
    vocabulary = {text for ep in lengths for text in labels.segments[ep][2]}
    robot_config = ROBOT_REGISTRY[robot]
    state_dim = sum(1 if p.is_eef else len(p.indices) for p in robot_config.proprio)
    # Every finite normalized state discretizes to [-1,255]. Three-digit
    # entries give the longest numeric representation with PaliGemma's tokenizer.
    state = np.ones(state_dim, dtype=np.float32)
    tokenizer = PaligemmaTokenizer(max_token_len)
    longest_task = max(instructions, key=lambda text: tokenizer.subtask_token_lengths(text, "x", state)[0])
    longest_label = max(vocabulary, key=lambda text: tokenizer.subtask_token_lengths("x", text)[1])
    prefix_len, label_len, cue_len = tokenizer.subtask_token_lengths(longest_task, longest_label, state)
    # Calls the same strict validation used in the training data transform.
    tokenizer.tokenize_subtask_training(longest_task, longest_label, state, max_subtask_len=max_subtask_len)
    # The same checkpoint must have room for the configured generation budget,
    # even when the longest label observed during training is shorter.
    tokenizer.tokenize_subtask_prefix(longest_task, state, max_subtask_len=max_subtask_len)
    return {
        "dataset_root": str(dataset_root.resolve()),
        "segments_path": str(segments_path.resolve()),
        "episodes": len(lengths),
        "frames": sum(lengths.values()),
        "segments": sum(len(labels.segments[ep][0]) for ep in lengths),
        "unique_subtasks": len(vocabulary),
        "state_dim": state_dim,
        "max_prefix_tokens": prefix_len,
        "max_label_tokens_including_eos": label_len,
        "action_cue_tokens": cue_len,
        "conservative_sequence_tokens": prefix_len + label_len + cue_len,
        "generation_sequence_budget": prefix_len + max_subtask_len + cue_len,
        "max_token_len": max_token_len,
        "max_subtask_len": max_subtask_len,
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
    parser.add_argument("--report-file", type=Path)
    args = parser.parse_args()
    report = check_dataset(
        args.dataset_root,
        args.segments_path or args.dataset_root / "meta" / "subtask_segments.parquet",
        episodes_index=args.episodes_index,
        robot=args.robot,
        max_token_len=args.max_token_len,
        max_subtask_len=args.max_subtask_len,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(text + "\n")


if __name__ == "__main__":
    main()
