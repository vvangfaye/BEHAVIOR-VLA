"""Read pooled frozen vision features with original episode/frame indexing."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.shared.frame_memory import pack_history
from openpi.shared.frame_memory import sample_indices
from openpi.training.subtask_data import episode_lengths_in_dataset_order


def dataset_fingerprint(root):
    digest = hashlib.sha256()
    paths = sorted((Path(root) / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise ValueError("Memory cache requires LeRobot v3 episode metadata.")
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_manifest(root):
    path = Path(root) / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or manifest.get("pool") != "mean" or manifest.get("dtype") != "float16":
        raise ValueError(f"Unsupported memory cache: {path}")
    if not isinstance(manifest.get("encoder_sha256"), str) or len(manifest["encoder_sha256"]) != 64:
        raise ValueError("Memory cache is missing a vision encoder fingerprint.")
    return manifest


class CachedFrameMemory:
    def __init__(self, root, contract, dataset_root, episodes=None):
        self.root = Path(root)
        self.contract = contract
        self.manifest = read_manifest(root)
        for key in ("grid_size", "frame_stride", "camera"):
            if self.manifest[key] != contract[key]:
                raise ValueError(f"Memory cache {key} disagrees with the model.")
        if self.manifest["dataset_sha256"] != dataset_fingerprint(dataset_root):
            raise ValueError("Memory cache belongs to different episode metadata.")
        self.lengths = episode_lengths_in_dataset_order(dataset_root, episodes)
        self._arrays = OrderedDict()
        # Check all selected episodes before launching a long distributed job.
        for episode, length in self.lengths.items():
            record = json.loads((self.root / f"episode_{episode:08d}.json").read_text())
            if record != {
                "episode_index": episode,
                "length": length,
                "encoder_sha256": self.manifest["encoder_sha256"],
            }:
                raise ValueError(f"Incomplete or mismatched memory cache episode {episode}.")
            if not (self.root / f"episode_{episode:08d}.npy").is_file():
                raise FileNotFoundError(f"Missing memory features for episode {episode}.")

    def __getstate__(self):
        return {**self.__dict__, "_arrays": OrderedDict()}

    def __call__(self, sample):
        episode = int(np.asarray(sample["episode_index"]).item())
        frame = int(np.asarray(sample["frame_index"]).item())
        if episode not in self.lengths or not 0 <= frame < self.lengths[episode]:
            raise ValueError("Memory requested with an unknown episode or out-of-range original frame index.")
        stride = self.contract["frame_stride"]
        if episode not in self._arrays:
            array = np.load(self.root / f"episode_{episode:08d}.npy", mmap_mode="r", allow_pickle=False)
            expected = (
                (self.lengths[episode] + stride - 1) // stride,
                self.contract["grid_size"] ** 2,
                self.manifest["feature_width"],
            )
            if array.shape != expected or array.dtype != np.float16:
                raise ValueError(f"Invalid memory cache shape/dtype for episode {episode}.")
            self._arrays[episode] = array
            if len(self._arrays) > 8:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(episode)
        # A prefix slice prevents future features from entering uniform sampling.
        indices = sample_indices(frame // stride + 1, self.contract["num_frames"])
        history = pack_history(
            self._arrays[episode][indices],
            indices * stride,
            num_frames=self.contract["num_frames"],
            grid_size=self.contract["grid_size"],
        )
        return {**sample, **history}
