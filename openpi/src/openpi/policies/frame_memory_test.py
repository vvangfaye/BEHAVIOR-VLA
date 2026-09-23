from types import SimpleNamespace

import numpy as np
import pytest

from openpi.configs import ROBOT_REGISTRY
from openpi.policies import frame_memory
from openpi.policies.policy import PolicyRecorder
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper


def test_online_buffer_uniform_sampling_causal_reset_and_encoder_reuse(monkeypatch):
    monkeypatch.setattr(frame_memory, "module_jit", lambda fn: fn)
    calls = []

    def encode(images):
        calls.append(images.copy())
        return np.ones((len(images), 4, 8), dtype=np.float32) * images.mean()

    model = SimpleNamespace(
        encode_memory_images=encode,
        memory_num_frames=3,
        memory_grid_size=2,
        memory_frame_stride=16,
        memory_camera="base_0_rgb",
    )
    buffer = frame_memory.FrameMemoryBuffer(model)
    for step in range(81):
        buffer.add(np.full((2, 2, 3), step, dtype=np.uint8), step)
    packed = buffer.pack()
    assert len(calls) == 6
    np.testing.assert_array_equal(packed["memory_positions"][::4, 0], [0, 32, 80])
    np.testing.assert_array_equal(packed["memory_features"][::4, 0], [0, 32, 80])
    buffer.add(np.zeros((2, 2, 3), dtype=np.uint8), 80)
    assert len(calls) == 6
    with pytest.raises(ValueError, match="Missing or reordered"):
        buffer.add(np.zeros((2, 2, 3), dtype=np.uint8), 112)
    buffer.reset()
    buffer.add(np.zeros((2, 2, 3), dtype=np.uint8), 0)
    assert buffer.pack()["memory_mask"].sum() == 4


def test_b1k_observes_between_chunks_preserves_memory_across_subtasks_and_resets(tmp_path):
    class RecordingPolicy:
        def __init__(self):
            self.metadata = {"framesamp_modul": True, "subtask_interval": 2}
            self.observed = []
            self.inferences = 0
            self.resets = 0
            self.subtask_resets = 0

        def observe_memory(self, obs, *, step):
            self.observed.append(step)

        def infer(self, obs):
            self.inferences += 1
            return {"actions": np.zeros((4, 23)), "subtask": "move"}

        def reset(self):
            self.resets += 1

        def reset_subtask(self):
            self.subtask_resets += 1

    policy = RecordingPolicy()
    wrapper = B1KPolicyWrapper(PolicyRecorder(policy, str(tmp_path)), "b1k/R1Pro", "tidy", "receding_horizon", 4, 4)
    robot = ROBOT_REGISTRY["b1k/R1Pro"]
    obs = {f"{robot.name}::proprio": np.zeros(57)}
    obs.update({camera.obs_key: np.zeros((224, 224, 3), dtype=np.uint8) for camera in robot.observations.values()})
    for _ in range(6):
        wrapper.act(obs)
    assert policy.observed == list(range(6))
    assert policy.inferences == 2
    wrapper.act({**obs, "subtask": "place"})
    assert policy.observed[-1] == 6
    assert policy.resets == 0
    assert policy.subtask_resets == 1
    wrapper.reset()
    wrapper.act(obs)
    assert policy.observed[-1] == 0
    assert policy.resets == 1
    with pytest.raises(ValueError, match="separate policy"):
        wrapper.process_input({**obs, f"{robot.name}::proprio": np.zeros((2, 57))})
