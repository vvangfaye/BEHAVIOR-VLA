"""Frame-to-subtask supervision from BEHAVIOR's compact segment table."""

from collections.abc import Collection
import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np


def episode_lengths_in_dataset_order(
    dataset_root: str | Path, episodes_index: Collection[int] | None = None
) -> dict[int, int]:
    """Load episode lengths in the same source order as the LeRobot parquet rows."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    entries = []
    seen = set()
    ordinal = 0
    paths = sorted((Path(dataset_root) / "meta" / "episodes").glob("chunk-*/*.parquet"))
    for path in paths:
        schema_names = pq.read_schema(path).names
        columns = ["episode_index", "length"]
        if "dataset_from_index" in schema_names:
            columns.append("dataset_from_index")
        for row in pq.read_table(path, columns=columns).to_pylist():
            episode = int(row["episode_index"])
            if episode in seen:
                raise ValueError(f"Duplicate episode_index={episode} in episode metadata.")
            seen.add(episode)
            length = int(row["length"])
            if length <= 0:
                raise ValueError(f"Episode {episode} has invalid length {length}.")
            source_index = row.get("dataset_from_index")
            entries.append((None if source_index is None else int(source_index), ordinal, episode, length))
            ordinal += 1
    if not entries:
        raise ValueError(f"No LeRobot v3 episode metadata found under {Path(dataset_root) / 'meta' / 'episodes'}.")
    if all(entry[0] is not None for entry in entries):
        entries.sort(key=lambda entry: (entry[0], entry[1]))
    if episodes_index is not None:
        selected = set(episodes_index)
        unknown = selected - seen
        if unknown:
            raise ValueError(f"Unknown selected episode ids: {sorted(unknown)[:10]}")
        entries = [entry for entry in entries if entry[2] in selected]
    if not entries:
        raise ValueError("No episodes selected for subtask training.")
    return {episode: length for _, _, episode, length in entries}


@dataclasses.dataclass(frozen=True)
class SubtaskFromSegments:
    # Episode ids remain the ORIGINAL LeRobot ids, including when using a subset.
    segments: dict[int, tuple[np.ndarray, np.ndarray, list[str]]]
    annotations: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    provenances: dict[int, list[str]] = dataclasses.field(default_factory=dict)
    annotation_column: str | None = None
    provenance_column: str | None = None
    dropped_episodes: dict[int, list[tuple[int, int]]] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        episodes_index: Collection[int] | None = None,
        empty_label_policy: Literal["error", "drop_episode"] = "error",
    ) -> "SubtaskFromSegments":
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.compute as pc  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        if empty_label_policy not in ("error", "drop_episode"):
            raise ValueError(f"Unknown empty_label_policy: {empty_label_policy!r}")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Subtask segments not found: {path}. Run scripts/gen_subtask_prompts.py first.")
        schema = pq.read_schema(path)
        role_columns = [name for name in ("annotated", "is_annotated") if name in schema.names]
        provenance_column = "provenance" if "provenance" in schema.names else None
        columns = ["episode_index", "start", "end", "subtask", *role_columns]
        if provenance_column is not None:
            columns.append(provenance_column)
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            raise ValueError(f"Subtask segment table is empty: {path}")
        for name in ("episode_index", "start", "end"):
            if not pa.types.is_integer(table.schema.field(name).type):
                raise ValueError(f"Subtask segment column {name!r} must contain integer frame/episode indices.")
        for name in role_columns:
            if not pa.types.is_boolean(table.schema.field(name).type):
                raise ValueError(f"Subtask segment column {name!r} must contain booleans.")
            if table[name].null_count:
                raise ValueError(f"Null values in subtask segment column {name!r}.")
        if provenance_column is not None:
            if not pa.types.is_string(table.schema.field(provenance_column).type):
                raise ValueError("Subtask segment column 'provenance' must contain strings.")
            if table[provenance_column].null_count:
                raise ValueError("Null values in subtask segment column 'provenance'.")
        if table["episode_index"].null_count:
            raise ValueError("Null values in subtask segment column 'episode_index'.")
        if episodes_index is not None:
            if len(episodes_index) == 0:
                raise ValueError("No episodes selected for subtask training.")
            # Filter before validating labels: excluded episodes must not block a valid subset.
            table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(list(episodes_index))))
        for name in ("start", "end"):
            if table[name].null_count:
                raise ValueError(f"Null values in subtask segment column {name!r}.")

        grouped = {}
        empty_intervals = {}
        valid_provenances = {"annotated", "imputed_leading", "imputed_gap", "imputed_tail"}
        for row in table.to_pylist():
            ep, start, end, text = (row[k] for k in ("episode_index", "start", "end", "subtask"))
            if ep < 0 or start < 0 or end <= start:
                raise ValueError(f"Invalid subtask interval: episode={ep}, [{start}, {end}).")
            flags = [row[name] for name in role_columns]
            if len(flags) == 2 and flags[0] != flags[1]:
                raise ValueError(f"Conflicting annotated/is_annotated values: episode={ep}, [{start}, {end}).")
            is_annotated = True if not flags else bool(flags[0])
            provenance = row.get("provenance")
            if provenance is not None:
                if provenance not in valid_provenances:
                    raise ValueError(f"Unknown provenance {provenance!r}: episode={ep}, [{start}, {end}).")
                if is_annotated != (provenance == "annotated"):
                    raise ValueError(f"Conflicting is_annotated/provenance values: episode={ep}, [{start}, {end}).")

            empty_text = text is None or (isinstance(text, str) and not text.strip())
            if empty_text:
                if empty_label_policy == "error":
                    raise ValueError(
                        f"Empty subtask label: episode={ep}, [{start}, {end}). "
                        "Run the subtask preflight with --empty-label-policy drop_episode "
                        "to select fully annotated episodes."
                    )
                empty_intervals.setdefault(ep, []).append((start, end))
                continue
            if not isinstance(text, str):
                raise ValueError(f"Subtask label must be a string: episode={ep}, [{start}, {end}).")
            grouped.setdefault(ep, []).append(
                (
                    start,
                    end,
                    text.strip(),
                    is_annotated,
                    provenance or ("annotated" if is_annotated else "imputed_unspecified"),
                )
            )
        # Joint training samples every retained frame, including imputed intervals.
        for ep in empty_intervals:
            grouped.pop(ep, None)

        segments = {}
        annotations = {}
        provenances = {}
        for ep, runs in grouped.items():
            runs.sort(key=lambda r: r[0])
            starts = np.asarray([r[0] for r in runs], dtype=np.int64)
            ends = np.asarray([r[1] for r in runs], dtype=np.int64)
            if np.any(starts[1:] < ends[:-1]):
                raise ValueError(f"Overlapping/duplicate subtask intervals in episode={ep}.")
            segments[ep] = (starts, ends, [r[2] for r in runs])
            annotations[ep] = np.asarray([r[3] for r in runs], dtype=np.bool_)
            provenances[ep] = [r[4] for r in runs]
        annotation_column = (
            role_columns[0] if len(role_columns) == 1 else ("annotated/is_annotated" if role_columns else None)
        )
        return cls(
            segments,
            annotations=annotations,
            provenances=provenances,
            annotation_column=annotation_column,
            provenance_column=provenance_column,
            dropped_episodes=empty_intervals,
        )

    def annotation_flags(self, episode: int) -> np.ndarray:
        starts = self.segments[episode][0]
        return self.annotations.get(episode, np.ones(len(starts), dtype=np.bool_))

    def role_counts(self, episodes: Collection[int] | None = None) -> dict[str, int]:
        selected = self.segments if episodes is None else {ep: self.segments[ep] for ep in episodes}
        counts = {"annotated_segments": 0, "imputed_segments": 0, "annotated_frames": 0, "imputed_frames": 0}
        for episode, (starts, ends, _) in selected.items():
            flags = self.annotation_flags(episode)
            lengths = ends - starts
            counts["annotated_segments"] += int(flags.sum())
            counts["imputed_segments"] += int((~flags).sum())
            counts["annotated_frames"] += int(lengths[flags].sum())
            counts["imputed_frames"] += int(lengths[~flags].sum())
        return counts

    def provenance_counts(self, episodes: Collection[int] | None = None) -> dict[str, dict[str, int]]:
        selected = self.segments.keys() if episodes is None else episodes
        segment_counts: dict[str, int] = {}
        frame_counts: dict[str, int] = {}
        for episode in selected:
            starts, ends, _ = self.segments[episode]
            for provenance, length in zip(self.provenances[episode], ends - starts, strict=True):
                segment_counts[provenance] = segment_counts.get(provenance, 0) + 1
                frame_counts[provenance] = frame_counts.get(provenance, 0) + int(length)
        return {"segments": segment_counts, "frames": frame_counts}

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

    def lookup(self, episode: int, frame: int, *, require_annotated: bool = False) -> tuple[str, int]:
        if episode in self.segments:
            starts, ends, labels = self.segments[episode]
            index = int(np.searchsorted(starts, frame, side="right")) - 1
            if index >= 0 and frame < ends[index]:
                if require_annotated and not self.annotation_flags(episode)[index]:
                    raise ValueError(f"Subtask segment is imputed for episode={episode}, frame={frame}.")
                return labels[index], int(ends[index])
        raise ValueError(f"No subtask annotation for episode={episode}, frame={frame}.")

    @staticmethod
    def indices(data: dict) -> tuple[int, int]:
        indices = []
        for key in ("episode_index", "frame_index"):
            if key not in data:
                raise ValueError(f"Subtask supervision requires {key!r} from the original dataset.")
            value = np.asarray(data[key])
            if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"{key} must be a single integer, got shape={value.shape}, dtype={value.dtype}.")
            indices.append(int(value.item()))
        return indices[0], indices[1]

    def __call__(self, data: dict) -> dict:
        episode, frame = self.indices(data)
        label, _ = self.lookup(episode, frame)
        return {**data, "subtask": label}
