import numpy as np
import pytest
import json
from types import SimpleNamespace
from etils import epath

from openpi.training import checkpoints


class _Manager:
    def __init__(self, steps):
        self._steps = steps

    def all_steps(self):
        return self._steps


def _mock_distributed(monkeypatch, *, gathered, steps=(25,), latest_gathered=None):
    gathered_values = iter((gathered, latest_gathered or [max(steps, default=-1)] * 2))
    monkeypatch.setattr(checkpoints.jax, "process_count", lambda: 2)
    monkeypatch.setattr(checkpoints.jax, "process_index", lambda: 0)
    monkeypatch.setattr(
        checkpoints.multihost_utils, "process_allgather", lambda value: np.asarray(next(gathered_values))
    )
    monkeypatch.setattr(checkpoints.multihost_utils, "broadcast_one_to_all", lambda value: value)
    monkeypatch.setattr(checkpoints.multihost_utils, "sync_global_devices", lambda name: None)
    monkeypatch.setattr(checkpoints.ocp, "CheckpointManager", lambda *args, **kwargs: _Manager(steps))


def test_multihost_resume_uses_existing_shared_directory(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _mock_distributed(monkeypatch, gathered=[1, 1])

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        checkpoint_dir, keep_period=250, overwrite=False, resume=True
    )

    assert resuming
    assert manager.all_steps() == (25,)


def test_multihost_resume_requires_checkpoint(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _mock_distributed(monkeypatch, gathered=[1, 1], steps=())

    with pytest.raises(FileNotFoundError, match="No resumable checkpoint"):
        checkpoints.initialize_checkpoint_dir(checkpoint_dir, keep_period=250, overwrite=False, resume=True)


def test_multihost_rejects_inconsistent_shared_directory(tmp_path, monkeypatch):
    _mock_distributed(monkeypatch, gathered=[0, 1])

    with pytest.raises(RuntimeError, match="disagree"):
        checkpoints.initialize_checkpoint_dir(
            tmp_path / "checkpoint", keep_period=250, overwrite=False, resume=False
        )


def test_multihost_resume_rejects_inconsistent_latest_step(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _mock_distributed(monkeypatch, gathered=[1, 1], latest_gathered=[25, 50])

    with pytest.raises(RuntimeError, match="latest checkpoint"):
        checkpoints.initialize_checkpoint_dir(checkpoint_dir, keep_period=250, overwrite=False, resume=True)


@pytest.mark.parametrize("filename,attribute", [("formal_run_contract.json", "formal_run_contract"),
                                               ("training_rng_contract.json", "training_rng_contract")])
def test_restore_rejects_changed_formal_provenance_before_state_load(tmp_path, filename, attribute):
    directory = tmp_path / "2/assets"
    directory.mkdir(parents=True)
    (directory / filename).write_text(json.dumps({"identity": "old"}))
    manager = SimpleNamespace(directory=epath.Path(tmp_path), latest_step=lambda: 2)
    loader = SimpleNamespace(learning_rate_contract=None, **{attribute: {"identity": "changed"}})
    with pytest.raises(ValueError, match="provenance or training RNG changed"):
        checkpoints.restore_state(manager, None, loader)
