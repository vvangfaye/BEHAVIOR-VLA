"""Compatibility imports for LeRobot dataset APIs."""

from typing import Any

try:
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets import MultiLeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

__all__ = [
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
    "b1k_task_prompts",
    "tasks_from_metadata",
]


def tasks_from_metadata(metadata: Any) -> dict[int, str]:
    tasks = metadata.tasks
    if isinstance(tasks, dict):
        return tasks
    # New LeRobot DataFrame, index is task name, task_index is in column
    import pandas as pd
    if isinstance(tasks, pd.DataFrame):
        return {int(row["task_index"]): str(task_name) for task_name, row in tasks.iterrows()}
    raise TypeError(f"Unsupported tasks type: {type(tasks)}")


def b1k_task_prompts(dataset_root: str | None, fallback: dict[int, str]) -> dict[int, str]:
    """Natural-language task prompts for B1K, keyed by task_index.

    B1K's meta/tasks.parquet stores the snake_case ``task_name`` in its ``task``
    column, and that parquet is what LeRobot loads. meta/tasks.jsonl additionally
    carries a natural-language ``task`` sentence, which is what the closed-loop
    server's TASK_REGISTRY was generated from. Training off the parquet therefore
    conditions on a different string than serving supplies, so prefer the jsonl.
    """
    import json
    import os

    if not dataset_root:
        return fallback
    path = os.path.join(dataset_root, "meta", "tasks.jsonl")
    if not os.path.exists(path):
        return fallback
    prompts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("task")
            if "task_index" not in rec or not text:
                continue
            # Only an improvement when the jsonl distinguishes name from prompt.
            if rec.get("task_name") is None or text == rec["task_name"]:
                continue
            prompts[int(rec["task_index"])] = str(text)
    return {**fallback, **prompts}