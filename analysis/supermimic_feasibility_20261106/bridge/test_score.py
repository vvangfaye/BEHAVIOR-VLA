"""CPU bridge checks using the actual WorldModel architecture and transforms."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import FrozenWorldModelScore, world_model_class
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from actor import model_actor_loss
except ModuleNotFoundError:
    model_actor_loss = None

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "train_world_model.py"
spec = importlib.util.spec_from_file_location("bridge_test_training", MODEL_PATH)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class FrozenValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(23, 1)
        self.requires_grad_(False)

    def forward(self, prediction):
        return self.linear(prediction["next_obs_state"]).squeeze(-1) + prediction["next_goals"].mean(-1)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        rng = np.random.default_rng(11)
        n = 12
        obs = rng.normal(size=(n, 23)).astype(np.float32)
        visual = rng.uniform(size=(n, 576)).astype(np.float32)
        actions = rng.normal(size=(n, 16, 23)).astype(np.float32)
        goals = (rng.uniform(size=(n, 3)) > .5).astype(np.float32)
        self.data = {"obs_state": obs, "next_obs_state": obs + .1, "next_obs_valid": np.ones(n, bool),
            "execution_actions": actions, "action_mask": np.ones((n, 16), bool),
            "visual_features": visual, "next_visual_features": visual + .1,
            "visual_valid": np.ones(n, bool), "next_visual_valid": np.ones(n, bool),
            "goals": goals, "next_goals": goals, "goal_valid": np.ones((n, 3), bool),
            "next_goal_valid": np.ones((n, 3), bool), "reward": rng.normal(size=n).astype(np.float32),
            "split": np.array([0]*8+[1]*2+[2]*2)}
        self.transform = training.Transform(self.data, visual_dim=4)
        self.table = training.tensors(self.data, self.transform)
        self.config = {"goals": 3, "visual_dim": 4, "use_action": True, "use_visual": True, "hidden": 16}
        self.models = []
        for seed in (3, 5):
            torch.manual_seed(seed)
            model = world_model_class()(**self.config)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            for _ in range(2):
                optimizer.zero_grad()
                pred = model(self.table)
                loss = pred["reward"].square().mean() + pred["state_delta"].square().mean()
                loss.backward()
                optimizer.step()
            model.eval()
            self.models.append(model)
        self.checkpoint = {"schema": "b1k_small_world_model_probe_v1", "name": "full",
            "config": self.config, "model_states": [model.state_dict() for model in self.models],
            "transform": self.transform.save(), "data_sha256": "synthetic-test-fixture"}
        self.path = Path(self.temp.name) / "full.pt"
        torch.save(self.checkpoint, self.path)
        self.score = FrozenWorldModelScore.from_checkpoint(self.path)

    def inputs(self):
        return [torch.from_numpy(self.data[key].copy()) for key in
                ("obs_state", "goals", "visual_features", "execution_actions")]

    def test_loaded_checkpoint_matches_offline_transform_and_ensemble(self):
        obs, goals, visual, actions = self.inputs()
        actual = self.score.predict(obs, goals, visual, actions)
        offline = training.predict(self.models, self.table)
        t = self.transform
        expected_reward = offline["reward"].ravel() * t.reward_scale[0] + t.reward_mean[0]
        expected_state = offline["state"] * t.state_scale + t.state_mean
        np.testing.assert_allclose(actual["raw_reward"].detach().numpy(), expected_reward, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual["next_obs_state"].detach().numpy(), expected_state, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual["next_visual_features"].detach().numpy(), t.decode(offline["visual"]), rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual["next_goals"].detach().numpy(), offline["goals"], rtol=2e-5, atol=2e-6)
        self.assertIn("action_scale", dict(self.score.named_buffers()))

    def test_score_retains_action_gradients_without_world_model_gradients(self):
        obs, goals, visual, actions = self.inputs()
        actions.requires_grad_(True)
        self.score.train()  # A parent actor cannot turn the frozen model to train.
        self.assertFalse(self.score.training)
        self.score(obs, goals, visual, actions).mean().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(float(actions.grad.abs().sum()), 0.)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.score.parameters()))

    def test_frozen_value_callback_preserves_next_state_gradient(self):
        obs, goals, visual, actions = self.inputs()
        actions.requires_grad_(True)
        value = FrozenValue()
        prediction = self.score.predict(obs, goals, visual, actions)
        expected = prediction["raw_reward"] + .9 * .75 * value(prediction)
        actual = self.score(obs, goals, visual, actions, value_callback=value, gamma=.9, continuation=.75)
        torch.testing.assert_close(actual, expected)
        # Specifically test the continuation-value path, excluding reward.
        value_grad = torch.autograd.grad((actual - prediction["raw_reward"]).sum(), actions)[0]
        self.assertGreater(float(value_grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(value_grad).all())
        self.assertTrue(all(p.grad is None for p in value.parameters()))
        with self.assertRaisesRegex(ValueError, "explicit continuation"):
            self.score(obs, goals, visual, actions, value_callback=value)
        value.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "must be frozen"):
            self.score(obs, goals, visual, actions, value_callback=value, continuation=1.)

    def test_missing_visual_and_invalid_shapes(self):
        obs, goals, visual, actions = self.inputs()
        visual[0] = float("nan")
        valid = torch.ones(len(obs), dtype=torch.bool)
        valid[0] = False
        score = self.score(obs, goals, visual, actions, visual_valid=valid)
        self.assertTrue(torch.isfinite(score).all())
        with self.assertRaisesRegex(ValueError, "Nonfinite valid"):
            self.score(obs, goals, visual, actions)
        with self.assertRaisesRegex(ValueError, "execution_actions"):
            self.score(obs, goals, visual, actions[:, :8])

    def test_non_action_checkpoint_rejected(self):
        self.checkpoint["name"] = "no_action"
        with self.assertRaisesRegex(ValueError, "action-conditioned"):
            FrozenWorldModelScore(self.checkpoint)

    @unittest.skipIf(model_actor_loss is None, "RLinf must be available on PYTHONPATH for actor integration")
    def test_real_actor_loss_connects_to_loaded_world_model(self):
        obs, goals, visual, actions = self.inputs()
        gain = nn.Parameter(torch.tensor(1.))
        sample = SimpleNamespace(execution_actions=actions * gain)
        loss, metrics = model_actor_loss(
            sample, lambda act: self.score(obs.detach(), goals.detach(), visual.detach(), act),
            frozen_modules=[self.score], reference_actions=actions, trust_coef=.01,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(gain.grad))
        self.assertGreater(float(gain.grad.abs()), 0.)
        self.assertTrue(torch.isfinite(metrics["model_score"]))
        self.assertTrue(all(p.grad is None for p in self.score.parameters()))


if __name__ == "__main__":
    unittest.main()
