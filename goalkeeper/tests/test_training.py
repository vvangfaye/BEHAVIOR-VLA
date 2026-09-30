import json
from pathlib import Path
import numpy as np
import pytest
import torch
from goalkeeper.synthetic import make_dataset
from goalkeeper.data import load_manifest, load_record, collate
from goalkeeper.model import ModelConfig, GoalTransitionModel, supervision_loss
from goalkeeper.training import train, calibrate
from goalkeeper.artifacts import load_checkpoint
from goalkeeper.calibration import ReliabilityCalibrator
from goalkeeper.policy import LearnedPredictor


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("synthetic"), groups=40, windows=1)


def test_group_splits_and_reject_eval_instances(dataset, tmp_path):
    rows = load_manifest(dataset)
    groups = {name: {r["group_id"] for r in rows if r["split"] == name} for name in ["train", "dev", "calibration"]}
    assert all(groups.values())
    assert not groups["train"] & groups["calibration"]
    row = json.loads(Path(dataset).read_text().splitlines()[0]); row["source"] = "public_test"
    bad = tmp_path / "bad.jsonl"; bad.write_text(json.dumps(row))
    with pytest.raises(ValueError): load_manifest(bad)


def test_action_conditioned_model_has_finite_gradients(dataset):
    torch.set_num_threads(1); torch.manual_seed(7)
    rows = load_manifest(dataset)
    batch = collate([load_record(r) for r in rows[:4]])
    model = GoalTransitionModel(ModelConfig(visual_dim=8, text_dim=8, proprio_dim=2, action_dim=2, horizon=4, hidden_dim=32))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    losses = []
    for _ in range(45):
        optimizer.zero_grad()
        output = model(batch["visual"], batch["proprio"], batch["queries"], batch["actions"])
        loss = supervision_loss(output, batch)["loss"]
        loss.backward(); optimizer.step(); losses.append(loss.item())
    assert losses[-1] < losses[0] * .55
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    pred = output["future_logits"].sigmoid().detach()
    assert pred[:, 3].mean() > pred[:, 0].mean() + .3


def test_training_checkpoint_calibration_end_to_end(dataset, tmp_path):
    torch.set_num_threads(1)
    history = train(dataset, tmp_path, epochs=2, batch_size=8, hidden_dim=32, device="cpu")
    assert len(history) == 2
    model, artifact = load_checkpoint(tmp_path / "best.pt")
    assert artifact["synthetic"] is True
    result = calibrate(dataset, tmp_path / "best.pt", tmp_path / "calibration.json", min_count=2)
    assert not set(result["groups"]) & set(artifact["train_groups"])
    assert 0 <= result["future_brier"] <= 1
    with pytest.raises(ValueError, match="Synthetic"):
        LearnedPredictor(None, tmp_path / "best.pt", tmp_path / "calibration.json", None, "synthetic-binary-v1")


def test_calibration_does_not_invent_confidence_for_sparse_bins():
    cal = ReliabilityCalibrator(min_count=30).fit([.99] * 2000, [1] * 2000)
    assert cal.estimate(.99).lower > .9
    assert cal.estimate(.5).lower == 0 and cal.estimate(.5).upper == 1
    clustered = ReliabilityCalibrator(min_count=30).fit([.99] * 2000, [1] * 2000, ["same-state"] * 2000)
    assert clustered.estimate(.99).lower == 0  # 2000 correlated labels are not 2000 independent states
    with pytest.raises(ValueError): ReliabilityCalibrator(counts=[1] * 10, positives=[2] * 10)
