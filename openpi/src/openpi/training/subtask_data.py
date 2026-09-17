"""Strict frame-to-subtask supervision from BEHAVIOR's compact segment table."""

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


def _evenly_spaced_indices(length: int, limit: int | None) -> np.ndarray:
    if limit is None or length <= limit:
        return np.arange(length, dtype=np.int64)
    if limit < 1:
        raise ValueError(f"max_samples_per_segment must be positive, got {limit}")
    if limit == 1:
        return np.asarray([length - 1], dtype=np.int64)
    result = np.rint(np.linspace(0, length - 1, num=limit)).astype(np.int64)
    if len(np.unique(result)) != limit:
        raise RuntimeError("Evenly spaced anchor selection produced duplicate indices.")
    return result


@dataclasses.dataclass(frozen=True)
class SubtaskActionIndex:
    """A deterministic indexable view of annotated subtask anchors."""

    dataset_indices: np.ndarray
    episode_indices: np.ndarray
    frame_indices: np.ndarray
    segment_ends: np.ndarray
    raw_frames: int
    eligible_anchors: int
    action_horizon: int
    frame_stride: int
    max_samples_per_segment: int | None

    @classmethod
    def from_segments(
        cls,
        labels: "SubtaskFromSegments",
        episode_lengths: dict[int, int],
        *,
        action_horizon: int,
        frame_stride: int = 1,
        max_samples_per_segment: int | None = None,
    ) -> "SubtaskActionIndex":
        dataset_offset = 0
        episode_rows = []
        for episode, length in episode_lengths.items():
            episode_rows.append((int(episode), dataset_offset, int(length)))
            dataset_offset += int(length)
        return cls._from_episode_rows(
            labels,
            episode_rows,
            raw_frames=dataset_offset,
            action_horizon=action_horizon,
            frame_stride=frame_stride,
            max_samples_per_segment=max_samples_per_segment,
        )

    @classmethod
    def from_episode_frame_indices(
        cls,
        labels: "SubtaskFromSegments",
        episode_indices: np.ndarray,
        frame_indices: np.ndarray,
        *,
        action_horizon: int,
        frame_stride: int = 1,
        max_samples_per_segment: int | None = None,
    ) -> "SubtaskActionIndex":
        """Build from the selected HF dataset's scalar index columns without decoding media."""
        episode_indices = np.asarray(episode_indices)
        frame_indices = np.asarray(frame_indices)
        if episode_indices.ndim != 1 or frame_indices.ndim != 1 or episode_indices.shape != frame_indices.shape:
            raise ValueError("HF episode_index and frame_index columns must be same-length one-dimensional arrays.")
        if not np.issubdtype(episode_indices.dtype, np.integer) or not np.issubdtype(frame_indices.dtype, np.integer):
            raise ValueError("HF episode_index and frame_index columns must contain integers.")
        if not len(episode_indices):
            raise ValueError("The selected HF dataset contains no frames.")
        episode_indices = episode_indices.astype(np.int64, copy=False)
        frame_indices = frame_indices.astype(np.int64, copy=False)

        run_starts = np.flatnonzero(np.r_[True, episode_indices[1:] != episode_indices[:-1]])
        run_ends = np.r_[run_starts[1:], len(episode_indices)]
        run_episodes = episode_indices[run_starts]
        if len(np.unique(run_episodes)) != len(run_episodes):
            raise ValueError("Each selected episode must occupy exactly one contiguous HF row range.")
        if np.any(frame_indices[run_starts] != 0):
            raise ValueError("Each selected HF episode must start at frame_index=0.")
        same_episode = episode_indices[1:] == episode_indices[:-1]
        if np.any(same_episode & (frame_indices[1:] != frame_indices[:-1] + 1)):
            raise ValueError("Selected HF episode rows must contain consecutive frame_index values in order.")

        episode_rows = [
            (int(episode), int(start), int(end - start))
            for episode, start, end in zip(run_episodes, run_starts, run_ends, strict=True)
        ]
        return cls._from_episode_rows(
            labels,
            episode_rows,
            raw_frames=len(episode_indices),
            action_horizon=action_horizon,
            frame_stride=frame_stride,
            max_samples_per_segment=max_samples_per_segment,
        )

    @classmethod
    def _from_episode_rows(
        cls,
        labels: "SubtaskFromSegments",
        episode_rows: list[tuple[int, int, int]],
        *,
        raw_frames: int,
        action_horizon: int,
        frame_stride: int,
        max_samples_per_segment: int | None,
    ) -> "SubtaskActionIndex":
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        if max_samples_per_segment is not None and max_samples_per_segment < 1:
            raise ValueError("max_samples_per_segment must be positive when set.")

        lengths = {episode: length for episode, _, length in episode_rows}
        if len(lengths) != len(episode_rows):
            raise ValueError("Each selected episode must occur exactly once in dataset order.")
        labels.validate_coverage(lengths)

        dataset_chunks = []
        episode_chunks = []
        frame_chunks = []
        segment_end_chunks = []
        eligible_anchors = 0
        for episode, dataset_start, _ in episode_rows:
            starts, ends, _ = labels.segments[episode]
            annotated = labels.annotation_flags(episode)
            for start, end, is_annotated in zip(starts, ends, annotated, strict=True):
                if not is_annotated:
                    continue
                frames = np.arange(int(start), int(end), frame_stride, dtype=np.int64)
                final_frame = int(end) - 1
                if frames[-1] != final_frame:
                    frames = np.append(frames, final_frame)
                eligible_anchors += len(frames)
                keep = _evenly_spaced_indices(len(frames), max_samples_per_segment)
                frames = frames[keep]
                dataset_chunks.append(dataset_start + frames)
                episode_chunks.append(np.full(len(frames), episode, dtype=np.int64))
                frame_chunks.append(frames)
                segment_end_chunks.append(np.full(len(frames), int(end), dtype=np.int64))

        if not dataset_chunks:
            raise ValueError("No annotated subtask anchors remain after stride/cap selection.")
        return cls(
            dataset_indices=np.concatenate(dataset_chunks).astype(np.int64, copy=False),
            episode_indices=np.concatenate(episode_chunks),
            frame_indices=np.concatenate(frame_chunks).astype(np.int64, copy=False),
            segment_ends=np.concatenate(segment_end_chunks),
            raw_frames=int(raw_frames),
            eligible_anchors=eligible_anchors,
            action_horizon=int(action_horizon),
            frame_stride=int(frame_stride),
            max_samples_per_segment=max_samples_per_segment,
        )

    def __len__(self) -> int:
        return len(self.dataset_indices)

    @property
    def capped_sample_count(self) -> int:
        return len(self)

    def resolve(self, index: int) -> tuple[int, int, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return (
            int(self.dataset_indices[index]),
            int(self.episode_indices[index]),
            int(self.frame_indices[index]),
        )

    def valid_action_steps(self) -> np.ndarray:
        return np.minimum(self.action_horizon, self.segment_ends - self.frame_indices)

    def valid_action_step_distribution(self) -> dict[int, int]:
        values, counts = np.unique(self.valid_action_steps(), return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


@dataclasses.dataclass(frozen=True)
class StrictSubtaskActionIndex:
    dataset_starts: np.ndarray
    frame_starts: np.ndarray
    episode_indices: np.ndarray
    counts: np.ndarray
    cumulative_counts: np.ndarray
    stride: int
    raw_frames: int
    eligible_anchors: int
    boundary_rejected_anchors: int

    @classmethod
    def from_segments(
        cls,
        labels: "SubtaskFromSegments",
        episode_lengths: dict[int, int],
        *,
        action_horizon: int,
        frame_stride: int = 1,
    ) -> "StrictSubtaskActionIndex":
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        labels.validate_coverage(episode_lengths)

        dataset_starts = []
        frame_starts = []
        episode_indices = []
        counts = []
        dataset_offset = 0
        eligible_anchors = 0
        for episode, length in episode_lengths.items():
            starts, ends, _ = labels.segments[episode]
            annotated = labels.annotation_flags(episode)
            for start, end, is_annotated in zip(starts.tolist(), ends.tolist(), annotated.tolist(), strict=True):
                if not is_annotated:
                    continue
                valid_end = end - action_horizon + 1
                if valid_end <= start:
                    continue
                eligible_anchors += valid_end - start
                raw_start = dataset_offset + start
                aligned_start = raw_start + (-raw_start % frame_stride)
                raw_end = dataset_offset + valid_end
                if aligned_start >= raw_end:
                    continue
                count = (raw_end - 1 - aligned_start) // frame_stride + 1
                dataset_starts.append(aligned_start)
                frame_starts.append(start + aligned_start - raw_start)
                episode_indices.append(episode)
                counts.append(count)
            dataset_offset += length

        if not counts:
            raise ValueError("No action anchors remain after strict subtask-boundary filtering.")
        counts_array = np.asarray(counts, dtype=np.int64)
        return cls(
            dataset_starts=np.asarray(dataset_starts, dtype=np.int64),
            frame_starts=np.asarray(frame_starts, dtype=np.int64),
            episode_indices=np.asarray(episode_indices, dtype=np.int64),
            counts=counts_array,
            cumulative_counts=np.cumsum(counts_array, dtype=np.int64),
            stride=int(frame_stride),
            raw_frames=dataset_offset,
            eligible_anchors=eligible_anchors,
            boundary_rejected_anchors=dataset_offset - eligible_anchors,
        )

    def __len__(self) -> int:
        return int(self.cumulative_counts[-1])

    def resolve(self, index: int) -> tuple[int, int, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        range_index = int(np.searchsorted(self.cumulative_counts, index, side="right"))
        previous_count = 0 if range_index == 0 else int(self.cumulative_counts[range_index - 1])
        offset = index - previous_count
        return (
            int(self.dataset_starts[range_index] + offset * self.stride),
            int(self.episode_indices[range_index]),
            int(self.frame_starts[range_index] + offset * self.stride),
        )


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
        require_role_provenance: bool = False,
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
        if require_role_provenance and ("is_annotated" not in role_columns or provenance_column is None):
            raise ValueError("Pure subtask executor requires role_aware_v1 columns is_annotated and provenance.")
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
            elif require_role_provenance:
                raise ValueError(f"Missing provenance: episode={ep}, [{start}, {end}).")

            empty_text = text is None or (isinstance(text, str) and not text.strip())
            if empty_text and is_annotated:
                if empty_label_policy == "error":
                    raise ValueError(
                        f"Empty subtask label: episode={ep}, [{start}, {end}). "
                        "Run the subtask preflight with --empty-label-policy drop_episode "
                        "to select fully annotated episodes."
                    )
                empty_intervals.setdefault(ep, []).append((start, end))
                continue
            if not empty_text and not isinstance(text, str):
                raise ValueError(f"Subtask label must be a string: episode={ep}, [{start}, {end}).")
            grouped.setdefault(ep, []).append(
                (
                    start,
                    end,
                    "" if empty_text else text.strip(),
                    is_annotated,
                    provenance or ("annotated" if is_annotated else "imputed_unspecified"),
                )
            )
        # Drop whole episodes with empty annotated targets; empty imputed intervals remain coverage-only.
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


@dataclasses.dataclass(frozen=True)
class PureSubtaskExecutor:
    """Set a strict GT subtask prompt and temporal action mask for one anchor."""

    segments: SubtaskFromSegments
    action_horizon: int
    action_pad_key: str

    def __call__(self, data: dict) -> dict:
        episode, frame = self.segments.indices(data)
        prompt, segment_end = self.segments.lookup(episode, frame, require_annotated=True)
        if self.action_pad_key not in data:
            raise ValueError(f"Pure subtask executor requires LeRobot padding field {self.action_pad_key!r}.")
        is_pad = np.asarray(data[self.action_pad_key])
        if is_pad.shape != (self.action_horizon,):
            raise ValueError(f"{self.action_pad_key} must have shape ({self.action_horizon},), got {is_pad.shape}.")
        boundary_mask = np.arange(self.action_horizon) < segment_end - frame
        action_loss_mask = boundary_mask & ~is_pad.astype(np.bool_)
        if not action_loss_mask[0]:
            raise ValueError(f"Executor anchor has no valid offset-0 action: episode={episode}, frame={frame}.")
        return {**data, "prompt": prompt, "action_loss_mask": action_loss_mask}
