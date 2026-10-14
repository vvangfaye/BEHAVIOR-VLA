"""Read frozen valid intervals from existing LeRobot roots without rewriting videos.

This module deliberately has no JAX dependency. The source/frame selection can
be tested with a lightweight reader before a GPU environment is available.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from openpi.training.subtask_data import episode_lengths_in_dataset_order
from openpi.training.subtask_mixture import MixturePlan, SOURCE_IDS


RGB_KEYS = {f"observation.rgb.{camera}_camera_0" for camera in (
    "zed_link", "left_realsense_link", "right_realsense_link")}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_contract(batch_size, process_count, seed, norm_stats):
    if norm_stats is None:
        raise ValueError("Three-source fine-tuning must reuse explicit normalization statistics.")
    stats = {}
    for key, value in sorted(norm_stats.items()):
        stats[key] = {}
        for name in ("mean", "std", "q01", "q99"):
            array = getattr(value, name, None)
            if array is not None:
                array = np.asarray(array)
                if not np.isfinite(array).all():
                    raise ValueError("Normalization statistics must be finite.")
                stats[key][name] = array.tolist()
            else:
                stats[key][name] = None
    encoded = json.dumps(stats, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"global_batch_size": batch_size, "process_count": process_count, "training_seed": seed,
            "norm_stats_sha256": hashlib.sha256(encoded).hexdigest()}


def resolve_roots(plan, roots_path=None):
    overrides = {} if roots_path is None else json.loads(Path(roots_path).read_text())
    if not isinstance(overrides, dict) or set(overrides) - set(plan.roots):
        raise ValueError("Root overrides must map known root IDs to local dataset directories.")
    paths = {}
    for root_id, descriptor in plan.roots.items():
        value = overrides.get(root_id, descriptor["dataset_root"])
        if not isinstance(value, str) or not value.strip() or "://" in value:
            raise ValueError(f"Dataset {root_id} must be staged as a local or mounted directory.")
        paths[root_id] = Path(value).expanduser().resolve()
    if len(set(paths.values())) != len(paths):
        raise ValueError("Distinct root IDs must not alias the same physical dataset.")
    return paths


def verify_metadata(plan, roots):
    """Verify original metadata under relocated roots, without accessing video bytes."""
    verified = set()
    for index in plan.indexes:
        root = roots[index["root_id"]]
        entries = index.get("provenance", {}).get("source_files", [])
        metadata = [entry for entry in entries if entry.get("role") in {"metadata", "collection_evidence", "labels"}]
        if not any(entry.get("role") == "metadata" for entry in metadata):
            raise ValueError("Training requires metadata file hashes; prepare an index from real metadata first.")
        for entry in metadata:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Metadata evidence paths must stay relative to the dataset root.")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Metadata evidence must not escape the dataset root through a symlink.")
            identity = (path, entry["sha256"])
            if identity not in verified:
                if _sha256(path) != entry["sha256"]:
                    raise ValueError(f"Dataset metadata changed: {path}")
                verified.add(identity)


class LeRobotIntervalReader:
    """A root-specific reader; all offsets refer to its selected parquet rows."""

    def __init__(self, root, descriptor, episodes, records, action_horizon, action_keys,
                 dataset_cls, metadata_cls, dataset_kwargs):
        self.root = Path(root)
        self.records = records
        self.action_horizon = action_horizon
        self.action_keys = action_keys
        self.offsets = {}
        info = json.loads((self.root / "meta/info.json").read_text())
        features = info.get("features", {})
        self.raw_state_shape = tuple(features.get("observation.state", {}).get("shape", []))
        if len(self.raw_state_shape) != 1 or self.raw_state_shape[0] < 57:
            raise ValueError("Source metadata must preserve the original R1Pro proprioception (at least 57 dimensions).")
        self.video_shapes = {key: tuple(feature["shape"]) for key, feature in features.items()
                             if feature.get("dtype") == "video"}
        if set(self.video_shapes) != RGB_KEYS or any(
                len(shape) != 3 or min(shape) <= 0 or 3 not in (shape[0], shape[-1])
                for shape in self.video_shapes.values()):
            raise ValueError("B1K requires exactly its three RGB streams; prepare a frozen RGB metadata view before indexing depth datasets.")
        lengths = episode_lengths_in_dataset_order(root, episodes)
        offset = 0
        for episode, length in lengths.items():
            if length != records[episode]["length"]:
                raise ValueError(f"Episode length changed for {descriptor['repo_id']}:{episode}.")
            self.offsets[episode] = offset
            offset += length

        self.tasks = {}
        for line in (self.root / "meta/tasks.jsonl").read_text().splitlines():
            row = json.loads(line)
            task_id = row["task_index"]
            if task_id in self.tasks or not row.get("task_name") or not row.get("task"):
                raise ValueError("Every source must provide unambiguous canonical and natural-language task names.")
            self.tasks[task_id] = row

        meta = metadata_cls(repo_id=descriptor["repo_id"], root=str(root))
        fps = float(meta.fps)
        if not np.isfinite(fps) or not np.isclose(fps, 30.0):
            raise ValueError("All mixture roots must preserve the baseline's 30 Hz action timebase.")
        conflicting = {"episodes", "repo_id", "root", "delta_timestamps"} & set(dataset_kwargs)
        if conflicting:
            raise ValueError(f"Dataset kwargs override indexed sample identity: {sorted(conflicting)}")
        # A full expert library contains hundreds of millions of rows. Opening
        # every root eagerly duplicates that table in each loader worker. Keep
        # the frozen identity checks, but open only an episode actually sampled.
        self._dataset_cls = dataset_cls
        self._dataset_args = dict(repo_id=descriptor["repo_id"], root=str(root),
            delta_timestamps={key: [i / fps for i in range(action_horizon)] for key in action_keys},
            **dataset_kwargs)
        self._lengths = lengths
        self._datasets = OrderedDict()

    def _episode_dataset(self, episode):
        if episode not in self._datasets:
            dataset = self._dataset_cls(episodes=[episode], **self._dataset_args)
            if len(dataset) != self._lengths[episode]:
                raise ValueError("LeRobot selected row count does not match frozen episode metadata.")
            self._datasets[episode] = dataset
            while len(self._datasets) > 2:
                self._datasets.popitem(last=False)
        self._datasets.move_to_end(episode)
        return self._datasets[episode]

    def read(self, episode, frame):
        data = dict(self._episode_dataset(episode)[frame])
        actual_episode = int(np.asarray(data["episode_index"]).item())
        actual_frame = int(np.asarray(data["frame_index"]).item())
        if (actual_episode, actual_frame) != (episode, frame):
            raise ValueError("LeRobot row order differs from the frozen original episode/frame identity.")
        task_id = int(np.asarray(data["task_index"]).item())
        task = self.tasks[task_id]
        if task["task_name"] != self.records[episode]["task_name"]:
            raise ValueError("LeRobot task identity differs from the indexed source instance.")
        for key in self.action_keys:
            actions = np.asarray(data[key])
            if actions.shape != (self.action_horizon, 23) or not np.isfinite(actions).all():
                raise ValueError("Indexed B1K samples require finite horizon x 23 action chunks.")
            padding = data.get(key + "_is_pad")
            if padding is not None:
                padding = np.asarray(padding)
                if padding.shape != (self.action_horizon,) or np.any(padding):
                    raise ValueError("A valid indexed action window must not contain padded targets.")
        state = np.asarray(data["observation.state"])
        # Raw expert proprioception has 57/61 dimensions. B1KInputs extracts the
        # model's 23-dimensional state after this reader; do not truncate here.
        if state.shape != self.raw_state_shape or not np.isfinite(state).all():
            raise ValueError("Raw state does not match the source feature shape or contains nonfinite values.")
        for key, shape in self.video_shapes.items():
            pixels = np.asarray(data[key])
            # LeRobot decodes videos as CHW; some readers expose source HWC.
            alternatives = {shape, (shape[2], shape[0], shape[1])} if shape[-1] == 3 else {shape}
            if pixels.shape not in alternatives or not np.isfinite(pixels).all():
                raise ValueError(f"Video frame is missing, malformed or nonfinite: {key}")
        data["prompt"] = task["task"]
        return data


class IndexedMixtureDataset:
    def __init__(self, plan, readers):
        self.mixture_plan = plan
        self.readers = readers

    def __len__(self):
        return self.mixture_plan.num_samples

    def __getitem__(self, index):
        reference = self.mixture_plan.reference(int(index))
        sample = self.readers[reference["root_id"]].read(reference["episode_index"], reference["frame"])
        # Each root keeps its original IDs. The index provides the exact label
        # only after the raw reader has checked the original row identity.
        sample["subtask"] = reference["subtask"]
        sample["has_subtask_target"] = np.bool_(reference["source_kind"] != "full_episode")
        sample["source_kind_id"] = np.int32(SOURCE_IDS[reference["source_kind"]])
        return sample


def create_dataset(data_config, action_horizon, *, metadata_cls=None):
    if not data_config.subtask_prediction or not data_config.prompt_from_task:
        raise ValueError("Indexed mixtures require joint subtask prediction with full task prompts.")
    if data_config.subtask_whole_task_prob != 0:
        raise ValueError("Indexed joint training requires subtask_whole_task_prob=0.")
    if data_config.frame_stride != 1 or data_config.episodes_index is not None:
        raise ValueError("Mixture manifests own frame/episode selection; do not add stride or an episode filter.")
    if data_config.memory_contract is not None:
        raise ValueError("Indexed mixtures do not yet support cached frame memory.")
    plan = MixturePlan.load(data_config.subtask_mixture_path)
    if plan.action_horizon != action_horizon:
        raise ValueError("Model action horizon differs from the frozen mixture.")
    roots = resolve_roots(plan, data_config.subtask_mixture_roots_path)
    verify_metadata(plan, roots)
    if metadata_cls is None:
        from openpi.training.lerobot_compat import LeRobotDatasetMetadata
        metadata_cls = LeRobotDatasetMetadata
    readers = {
        root_id: LeRobotIntervalReader(
            roots[root_id], descriptor, plan.episodes_by_root[root_id], plan.episode_records_by_root[root_id],
            action_horizon, data_config.action_sequence_keys, data_config.data_cls, metadata_cls,
            data_config.dataset_kwargs,
        )
        for root_id, descriptor in plan.roots.items()
    }
    return IndexedMixtureDataset(plan, readers)
