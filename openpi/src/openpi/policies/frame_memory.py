"""Episode-local feature buffer. Retains the complete observed prefix on CPU."""

import numpy as np
from openpi_client import image_tools

from openpi.policies.b1k_policy import _parse_image
from openpi.shared.frame_memory import pack_history
from openpi.shared.frame_memory import sample_indices
from openpi.shared.nnx_utils import module_jit


class FrameMemoryBuffer:
    def __init__(self, model):
        self.encode = module_jit(model.encode_memory_images)
        self.num_frames = model.memory_num_frames
        self.grid_size = model.memory_grid_size
        self.stride = model.memory_frame_stride
        self.camera = model.memory_camera
        self.reset()

    def reset(self):
        self.features = []
        self.frame_ids = []
        self.next_step = 0

    def add(self, image, step):
        if step < 0:
            raise ValueError("Memory control step must be nonnegative.")
        if step % self.stride:
            return
        if self.frame_ids and step == self.frame_ids[-1]:
            return  # A retry of the same observation must not append twice.
        expected = 0 if not self.frame_ids else self.frame_ids[-1] + self.stride
        if step != expected:
            raise ValueError(
                "Missing or reordered history observations; reset between episodes and observe every memory stride."
            )
        # Match the cache exporter, including PIL interpolation and uint8 rounding.
        image = image_tools.resize_with_pad(_parse_image(image), 224, 224)
        feature = np.asarray(self.encode(image[None]))[0].astype(np.float16)
        if not np.isfinite(feature).all():
            raise ValueError("Non-finite historical vision features.")
        self.features.append(feature)
        self.frame_ids.append(step)
        self.next_step = step + self.stride

    def pack(self):
        indices = sample_indices(len(self.features), self.num_frames)
        return pack_history(
            np.stack([self.features[i] for i in indices]),
            np.asarray(self.frame_ids)[indices],
            num_frames=self.num_frames,
            grid_size=self.grid_size,
        )
