import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("flow_endpoint_probe", Path(__file__).with_name("flow_endpoint_probe.py"))
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class FakeObservation:
    def __init__(self, state):
        self.state = state.copy()

    def to_dict(self):
        return {"state": self.state}


class FakePolicy:
    def __init__(self):
        self._model = SimpleNamespace(action_horizon=32, action_dim=32, subtask_loss_weight=1.0)
        self.noise = None

    def _observation_transform(self, packed):
        packed["actions"] -= packed["state"][None, :]
        return packed

    def _tokenizer_transform(self, packed):
        return packed

    def _pad(self, packed):
        packed["actions"] = np.pad(packed["actions"], ((0, 0), (0, 9)))
        packed["state"] = np.pad(packed["state"], (0, 9))
        return packed

    def _output_transform(self, prepared):
        return {"actions": prepared["actions"][:, :23] + prepared["state"][None, :23]}

    def infer(self, packed, *, noise):
        self.noise = noise.copy()
        answer = packed["actions"].copy()
        packed["actions"][:] = -999  # Deliberately mutate this inference copy.
        return {"actions": answer, "subtask_source": "provided"}


def fixture():
    policy = FakePolicy()
    actions = np.arange(32 * 23, dtype=np.float32).reshape(32, 23) / 100
    state = np.arange(23, dtype=np.float32) / 10
    raw = {"action": actions, "state": state}
    packed = {"actions": actions.copy(), "state": state.copy()}
    item = {"policy_input": packed, "policy_input_sha256": probe.ev.input_content_sha256(packed),
            "valid": np.arange(32) < 7}

    def transform_dataset(dataset, _config):
        one = {"actions": dataset[0]["action"], "state": dataset[0]["state"]}
        return [policy._pad(policy._tokenizer_transform(policy._observation_transform(one)))]

    return policy, raw, item, transform_dataset


class ProbeTest(unittest.TestCase):
    def test_native_split_noise_time_and_nonuniform_beta(self):
        # Only small random arrays: no model creation, loading, or inference.
        import jax

        root = jax.random.key(19)
        noise, time = probe.cfm_draw(jax.random, root, (1, 32, 32))
        _, noise_key, time_key = jax.random.split(root, 3)
        np.testing.assert_array_equal(noise, jax.random.normal(noise_key, (1, 32, 32)))
        np.testing.assert_array_equal(time, jax.random.beta(time_key, 1.5, 1, (1,)) * 0.999 + 0.001)
        self.assertEqual(str(noise.dtype), "float32")
        self.assertEqual(str(time.dtype), "float32")

    def test_targets_survive_training_transform_and_noise_is_shared_without_mutating_raw(self):
        policy, raw, item, transforms = fixture()
        saved_raw, saved_item = copy.deepcopy(raw), copy.deepcopy(item)
        expected_target = np.pad(raw["action"] - raw["state"][None, :], ((0, 0), (0, 9)))[None]
        calls = []

        def batch(prepared):
            return FakeObservation(prepared["state"]), prepared["actions"][None]

        def compute(root, observation, actions, *, train):
            calls.append((root, observation, train))
            np.testing.assert_array_equal(actions, expected_target)
            return np.full((1, 32), 3., np.float32), {"action_loss": np.float32(2), "subtask_loss": np.float32(1)}

        noise = np.ones((1, 32, 32), dtype=np.float32)
        result = probe.evaluate_sample(policy=policy, raw=raw, item=item, data_config=None,
            transform_dataset=transforms, batch_inputs=batch, compute_cfm=compute,
            root_rng="root", root_key_data=np.array([1, 2], np.uint32), noise=noise,
            time=np.array([.75], np.float32))
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0][0], calls[0][2]), ("root", False))
        np.testing.assert_array_equal(policy.noise, noise[0])
        np.testing.assert_array_equal(raw["action"], saved_raw["action"])
        np.testing.assert_array_equal(item["policy_input"]["actions"], saved_item["policy_input"]["actions"])
        self.assertEqual(result["cfm"]["metrics"]["action_loss"]["values"], 2)
        self.assertEqual(result["cfm"]["metrics"]["subtask_loss"]["values"], 1)
        self.assertEqual(result["cfm"]["joint_loss_identity_residual"], 0)
        self.assertTrue(result["target_leakage_control"]["changed_normalized_action_target"])
        self.assertTrue(result["target_leakage_control"]["observation_identical"])
        self.assertEqual(result["cfm"]["native_returned_loss_including_weighted_language_ce"]["values"][0][0], 3)
        self.assertEqual(result["endpoint"]["gt_error_valid_prefix16"]["valid_steps"], 7)
        self.assertEqual(result["endpoint"]["gt_error_valid_horizon"]["groups"]["base"]["mse_per_coordinate"], [0, 0, 0])

    def test_missing_or_changed_training_targets_fail_closed(self):
        policy, raw, item, transforms = fixture()
        del item["policy_input"]["actions"]
        with self.assertRaisesRegex(ValueError, "targets are missing"):
            probe.prepare_targets(raw, item["policy_input"], None, policy, transforms)
        policy, raw, item, transforms = fixture()

        def broken(dataset, config):
            output = transforms(dataset, config)
            output[0]["actions"] += 1
            return output

        with self.assertRaisesRegex(ValueError, "transforms disagree"):
            probe.prepare_targets(raw, item["policy_input"], None, policy, broken)

    def test_joint_loss_identity_is_checked_before_endpoint_inference(self):
        policy, raw, item, transforms = fixture()
        with self.assertRaisesRegex(ValueError, "joint loss disagrees"):
            probe.evaluate_sample(policy=policy, raw=raw, item=item, data_config=None,
                transform_dataset=transforms,
                batch_inputs=lambda prepared: (FakeObservation(prepared["state"]), prepared["actions"][None]),
                compute_cfm=lambda *args, **kwargs: (np.zeros((1, 32), np.float32),
                    {"action_loss": np.float32(2), "subtask_loss": np.float32(1)}),
                root_rng="key", root_key_data=np.array([1, 2], np.uint32),
                noise=np.ones((1, 32, 32), np.float32), time=np.array([.75], np.float32))
        self.assertIsNone(policy.noise)

    def test_target_leakage_control_rejects_actions_in_observation(self):
        policy, raw, item, transforms = fixture()
        with self.assertRaisesRegex(ValueError, "leaked into"):
            probe.evaluate_sample(policy=policy, raw=raw, item=item, data_config=None,
                transform_dataset=transforms,
                batch_inputs=lambda prepared: (FakeObservation(prepared["actions"]), prepared["actions"][None]),
                compute_cfm=lambda *args, **kwargs: self.fail("No forward is allowed after target leakage"),
                root_rng="key", root_key_data=np.array([1, 2], np.uint32),
                noise=np.ones((1, 32, 32), np.float32), time=np.array([.75], np.float32))
        self.assertIsNone(policy.noise)

    def test_repeat_receipt_records_tolerance_and_detects_disagreement(self):
        first = {"cfm": {"native_returned_loss_including_weighted_language_ce": probe.numeric(np.ones((1, 32), np.float32)),
                         "metrics": {"action_loss": probe.numeric(np.float32(1)), "subtask_loss": probe.numeric(np.float32(0))}},
                 "endpoint": {"actions": probe.numeric(np.ones((32, 23), np.float32))},
                 "transformed_training_sample_sha256": "same",
                 "rng": {"root_key_data": [0, 1], "noise_sha256": "noise", "time_sha256": "time"}}
        repeated = copy.deepcopy(first)
        repeated["endpoint"]["actions"]["values"][0][0] += 1e-6
        checked = probe.repeat_checks(first, repeated)
        self.assertTrue(checked["all_passed"])
        self.assertEqual(checked["rtol"], 1e-5)
        self.assertEqual(checked["atol"], 1e-6)
        repeated["endpoint"]["actions"]["values"][0][0] += .01
        checked = probe.repeat_checks(first, repeated)
        self.assertFalse(checked["all_passed"])
        self.assertGreater(checked["checks"]["endpoint_actions"]["max_abs_difference"], .01)

    def test_local_assets_never_fall_back_to_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            token = root / "big_vision" / "token.model"
            token.parent.mkdir()
            token.write_bytes(b"already cached")
            self.assertEqual(probe.local_asset("gs://big_vision/token.model", cache_root=root), token)
            with self.assertRaisesRegex(FileNotFoundError, "downloads are disabled"):
                probe.local_asset("gs://big_vision/missing", cache_root=root)
            with self.assertRaisesRegex(ValueError, "force_download"):
                probe.local_asset(str(token), cache_root=root, force_download=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                probe.local_asset("gs://big_vision/../../outside", cache_root=root)

    def test_error_groups_keep_coordinates_and_masks_separate(self):
        target = np.zeros((32, 23))
        prediction = np.ones((32, 23))
        prediction[1:, :] = 100
        prediction[0, :3] = [1, 2, 3]
        result = probe.grouped_error(prediction, target, np.arange(32) == 0)
        self.assertEqual(result["groups"]["base"]["mse_per_coordinate"], [1, 4, 9])
        self.assertNotIn("mse", result)
        self.assertEqual(result["valid_steps"], 1)


if __name__ == "__main__":
    unittest.main()
