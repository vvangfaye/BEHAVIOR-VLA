"""Strict frame-to-subtask supervision from BEHAVIOR's compact segment table."""

import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class SubtaskFromSegments:
    # Episode ids remain the ORIGINAL LeRobot ids, including when using a subset.
    segments: dict[int, tuple[np.ndarray, np.ndarray, list[str]]]

    @classmethod
    def from_parquet(cls, path: str | Path) -> "SubtaskFromSegments":
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Subtask segments not found: {path}. Run scripts/gen_subtask_prompts.py first.")
        table = pq.read_table(path, columns=["episode_index", "start", "end", "subtask"])
        if table.num_rows == 0:
            raise ValueError(f"Subtask segment table is empty: {path}")
        for name in table.column_names:
            if table[name].null_count:
                raise ValueError(f"Null values in subtask segment column {name!r}.")
        for name in ("episode_index", "start", "end"):
            if not pa.types.is_integer(table.schema.field(name).type):
                raise ValueError(f"Subtask segment column {name!r} must contain integer frame/episode indices.")
        grouped = {}
        for row in table.to_pylist():
            ep, start, end, text = (row[k] for k in ("episode_index", "start", "end", "subtask"))
            if ep < 0 or start < 0 or end <= start:
                raise ValueError(f"Invalid subtask interval: episode={ep}, [{start}, {end}).")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty subtask label: episode={ep}, [{start}, {end}).")
            grouped.setdefault(ep, []).append((start, end, text.strip()))
        segments = {}
        for ep, runs in grouped.items():
            runs.sort(key=lambda r: r[0])
            starts = np.asarray([r[0] for r in runs], dtype=np.int64)
            ends = np.asarray([r[1] for r in runs], dtype=np.int64)
            if np.any(starts[1:] < ends[:-1]):
                raise ValueError(f"Overlapping/duplicate subtask intervals in episode={ep}.")
            segments[ep] = (starts, ends, [r[2] for r in runs])
        return cls(segments)

    def validate_coverage(self, episode_lengths: dict[int, int]) -> None:
        """Require an exact tiling of each selected episode's [0, length)."""
        if not episode_lengths:
            raise ValueError("No episodes selected for subtask training.")
        for ep, length in episode_lengths.items():
            if ep not in self.segments:
                raise ValueError(f"Missing subtask annotations for episode={ep}.")
            starts, ends, _ = self.segments[ep]
            if length <= 0 or starts[0] != 0 or ends[-1] != length or np.any(starts[1:] != ends[:-1]):
                raise ValueError(f"Subtask intervals must exactly cover episode={ep}, [0, {length}) without gaps.")

    def __call__(self, data: dict) -> dict:
        indices = []
        for key in ("episode_index", "frame_index"):
            if key not in data:
                raise ValueError(f"Subtask supervision requires {key!r} from the original dataset.")
            value = np.asarray(data[key])
            if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"{key} must be a single integer, got shape={value.shape}, dtype={value.dtype}.")
            indices.append(int(value.item()))
        ep, frame = indices
        if ep in self.segments:
            starts, ends, labels = self.segments[ep]
            index = int(np.searchsorted(starts, frame, side="right")) - 1
            if index >= 0 and frame < ends[index]:
                return {**data, "subtask": labels[index]}
        # Never use the high-level task as a substitute target or extend the last
        # label beyond its annotated end. Tail settling belongs in the segment table.
        raise ValueError(f"No subtask annotation for episode={ep}, frame={frame}.")
