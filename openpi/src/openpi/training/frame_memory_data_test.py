import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.configs import ROBOT_REGISTRY
from openpi.models import tokenizer as tokenizer_lib
from openpi.models.pi0_config import Pi0Config
from openpi.policies import policy_config
from openpi.shared import frame_memory
from openpi.training import data_loader
from openpi.training.frame_memory_data import CachedFrameMemory
from openpi.training.frame_memory_data import dataset_fingerprint


@pytest.fixture
def cached(tmp_path):
    data_root = tmp_path / "data"
    meta = data_root / "meta/episodes/chunk-000/file-000.parquet"
    meta.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"episode_index": 37, "length": 81}, {"episode_index": 92, "length": 17}]), meta
    )
    root = tmp_path / "cache"
    root.mkdir()
    config = Pi0Config(pi05=True, framesamp_modul=True, memory_num_frames=3, memory_grid_size=2)
    contract = frame_memory.model_contract(config)
    manifest = {
        "version": 1,
        "pool": "mean",
        "dtype": "float16",
        "feature_width": 8,
        "grid_size": 2,
        "frame_stride": 16,
        "camera": "base_0_rgb",
        "dataset_camera": "rgb",
        "encoder_sha256": "a" * 64,
        "dataset_sha256": dataset_fingerprint(data_root),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for ep, length in ((37, 81), (92, 17)):
        features = np.broadcast_to(np.arange((length + 15) // 16)[:, None, None], ((length + 15) // 16, 4, 8)).astype(
            np.float16
        )
        np.save(root / f"episode_{ep:08d}.npy", features)
        (root / f"episode_{ep:08d}.json").write_text(
            json.dumps({"episode_index": ep, "length": length, "encoder_sha256": "a" * 64})
        )
    return root, data_root, config, contract, manifest


def test_original_indices_no_future_or_cross_episode_and_padding(cached):
    root, data_root, _, contract, _ = cached
    reader = CachedFrameMemory(root, contract, data_root, [37])
    early = reader({"episode_index": np.array(37), "frame_index": np.array(7)})
    assert early["memory_mask"].sum() == 4
    assert early["memory_positions"][:, 0].max() == 0
    middle = reader({"episode_index": 37, "frame_index": 33})
    np.testing.assert_array_equal(middle["memory_positions"][::4, 0], [0, 16, 32])
    late = reader({"episode_index": 37, "frame_index": 80})
    np.testing.assert_array_equal(late["memory_positions"][::4, 0], [0, 32, 80])
    with pytest.raises(ValueError, match="unknown episode"):
        reader({"episode_index": 92, "frame_index": 0})
    with pytest.raises(ValueError, match="out-of-range"):
        reader({"episode_index": 37, "frame_index": 81})


def test_contract_and_cache_mismatch_fail(cached, tmp_path):
    root, data_root, config, contract, _ = cached
    with pytest.raises(ValueError, match="frame_stride"):
        CachedFrameMemory(root, {**contract, "frame_stride": 8}, data_root)
    assets = tmp_path / "assets"
    frame_memory.save(assets, contract)
    frame_memory.check_resume(assets, contract)
    with pytest.raises(ValueError, match="Cannot resume"):
        frame_memory.check_resume(assets, None)
    with pytest.raises(ValueError, match="Cannot resume"):
        frame_memory.check_resume(assets, frame_memory.model_contract(dataclasses.replace(config, memory_num_frames=2)))


def test_cache_exporter_checks_original_indices_and_atomic_completion(cached):
    root, _, _, _, manifest = cached
    path = Path(__file__).parents[3] / "scripts/b1k/cache_frame_memory.py"
    spec = importlib.util.spec_from_file_location("cache_frame_memory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Dataset:
        def __getitem__(self, index):
            return {"episode_index": 121, "frame_index": index, "rgb": np.zeros((3, 224, 224), dtype=np.float32)}

    module.write_episode(root, 121, 33, manifest, Dataset(), lambda x: np.ones((len(x), 4, 8)), 2)
    array = np.load(root / "episode_00000121.npy")
    assert array.shape == (3, 4, 8)
    assert (root / "episode_00000121.json").is_file()
    assert not list(root.glob("*.tmp.*"))
    # Completion is reusable without running the encoder again.
    module.write_episode(root, 121, 33, manifest, Dataset(), lambda _: pytest.fail("encoded twice"), 2)


def test_openloop_random_frames_receive_their_own_causal_history(cached, tmp_path, monkeypatch):
    root, data_root, _, contract, manifest = cached
    path = Path(__file__).parents[4] / "scripts/eval_openloop.py"
    spec = importlib.util.spec_from_file_location("eval_openloop_memory_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "checkpoint"
    frame_memory.save(
        checkpoint / "assets",
        {**contract, "encoder_sha256": manifest["encoder_sha256"], "dataset_sha256": manifest["dataset_sha256"]},
    )
    args = SimpleNamespace(
        subtask_conditioning="none",
        horizon=2,
        dataset=str(data_root),
        repo_id="test/memory",
        subtask_prompts=None,
        pi05_ckpt=str(checkpoint),
        memory_cache_dir=None,
    )
    with pytest.raises(ValueError, match="requires --memory-cache-dir"):
        module.build_pi05(args, [37])
    args.memory_cache_dir = root
    monkeypatch.setattr(tokenizer_lib, "PaligemmaTokenizer", lambda *a: object())
    robot = ROBOT_REGISTRY["b1k/R1Pro"]

    class Dataset:
        def __init__(self, config):
            self.reader = CachedFrameMemory(config.memory_cache_dir, config.memory_contract, config.dataset_root, [37])

        def __getitem__(self, index):
            sample = {
                "episode_index": 37,
                "frame_index": index,
                "index": index,
                "prompt": "tidy",
                "observation.state": np.zeros(57),
                "action": np.zeros((2, 23)),
                "action_is_pad": np.zeros(2, dtype=bool),
            }
            sample.update({camera.dataset_key: np.zeros((3, 8, 8)) for camera in robot.observations.values()})
            return self.reader(sample)

    monkeypatch.setattr(data_loader, "create_b1k_dataset", lambda config, horizon: Dataset(config))
    received = []

    class Policy:
        def infer(self, obs):
            received.append(obs)
            return {"actions": np.zeros((2, 23))}

    monkeypatch.setattr(policy_config, "create_trained_policy", lambda *a, **kw: Policy())
    predict, get_sample = module.build_pi05(args, [37])
    for frame in (80, 7):
        item = get_sample(module.EvalSample(0, 37, frame, frame, frame))
        predict(item)
    assert received[0]["memory_positions"][:, 0].max() == 80
    assert received[1]["memory_positions"][:, 0].max() == 0
    assert received[1]["memory_mask"].sum() == 4
    bad = {**manifest, "encoder_sha256": "b" * 64}
    (root / "manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="encoder_sha256"):
        module.build_pi05(args, [37])
