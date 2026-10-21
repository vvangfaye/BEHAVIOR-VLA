"""Fixed episode-split data for the support-flow VLA pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from b1k_rlinf.io import DELTA_INDICES, IMAGE_KEYS, extract_state


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


class VlaPilotData:
    def __init__(self, directory, base, *, verify_images=True):
        self.directory = Path(directory)
        self.base = base
        self.plan = json.loads((self.directory / 'data_plan.json').read_text())
        self.receipt = json.loads((self.directory / 'image_receipt.json').read_text())
        if not self.receipt['complete']:
            raise ValueError('Image cache is incomplete')
        if digest(self.directory / 'data_plan.json') != self.receipt['data_plan_sha256']:
            raise ValueError('Image/data plan mismatch')
        target = self.directory / self.plan['raw_targets']['path']
        if digest(target) != self.plan['raw_targets']['sha256']:
            raise ValueError('Raw targets have changed')
        image_path = self.directory / self.receipt['image_file']
        if verify_images and digest(image_path) != self.receipt['image_sha256']:
            raise ValueError('Fixed image cache has changed')
        self.images = np.load(image_path, mmap_mode='r')
        if list(self.images.shape) != self.receipt['shape'] or self.images.dtype != np.uint8:
            raise ValueError('Image shape or dtype mismatch')
        raw = np.load(target)
        self.active = torch.from_numpy(raw['active'].copy()).float()
        self.fold = raw['fold'].copy()
        self.episode = raw['episode'].copy()
        self.frame = raw['frame'].copy()
        self.state = extract_state(raw['state'], 'checkpoint61').astype(np.float64)
        self.normalized_state = base.checkpoint_io.normalize_state(self.state)
        physical_mask = np.zeros(23)
        physical_mask[DELTA_INDICES] = 1
        stats = base.checkpoint_io.stats['actions']
        values = (raw['actions'].astype(np.float64) - self.state[:, None] * physical_mask - stats['mean']) / (stats['std'] + 1e-6)
        self.actions = torch.from_numpy(np.pad(values.astype(np.float32), ((0, 0), (0, 0), (0, 9))))
        self.tokens = []
        for i, sample in enumerate(self.plan['samples']):
            if (sample['episode_index'], sample['frame_index']) != (self.episode[i], self.frame[i]):
                raise ValueError('Anchor ordering changed')
            label = base.joint_tokenizer.processor.encode(sample['subtask'], add_eos=True)
            self.tokens.append(base.joint_tokenizer.actions(sample['prompt'], self.normalized_state[i], label))
        self.train_ids = np.flatnonzero(np.isin(self.fold, [2, 3, 4]))
        self.validation_ids = np.flatnonzero(self.fold == 1)
        self.test_ids = np.flatnonzero(self.fold == 0)

    def batch(self, indices):
        indices = np.asarray(indices, dtype=int)
        # Advanced indexing copies from read-only memmap before torch conversion.
        images = {key: np.ascontiguousarray(self.images[indices, j]) for j, key in enumerate(IMAGE_KEYS)}
        obs = self.base._token_observation(images, self.normalized_state[indices],
                                           [self.tokens[i] for i in indices])
        device = obs.state.device
        target = self.actions[indices].to(device)
        active = self.active[indices].to(device)
        return obs, target, active
