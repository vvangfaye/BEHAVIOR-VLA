import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


_SPEC = importlib.util.spec_from_file_location("eval_cached_probe", Path(__file__).with_name("eval_cached_probe.py"))
cached = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cached
_SPEC.loader.exec_module(cached)


def make_cache(root):
    files = {"selected_episodes.txt": b"7\n", "dataset/meta/tasks.jsonl": b'{"task_index":3,"task":"task"}\n',
             "dataset/meta/episodes/chunk-000/file-000.parquet": b"fake metadata, never parsed by helper tests",
             "dataset/meta/original_subtask_segments.parquet": b"fake segment metadata"}
    probe = {"kind": "matched_training_boundary_probe", "model_horizon_H": 32,
             "pairing_contract": {"action_noise_seed": 0}, "repo_id": "repo",
             "sample_keys": [{"episode_index": 7, "frame_index": 5, "sample_id": "sample"}]}
    files["probe_manifest.json"] = json.dumps(probe).encode()
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    raw = {"episode_index": np.array(7, dtype=np.int64), "frame_index": np.array(5, dtype=np.int64),
           "index": np.array(105, dtype=np.int64), "task_index": np.array(3, dtype=np.int64),
           "subtask": "grasp", "prompt": "task", "action": np.ones((32, 23), dtype=np.float32),
           "action_is_pad": np.zeros(32, dtype=bool), "observation.state": np.zeros(61, dtype=np.float32)}
    npz_path = root / "samples" / "sample.npz"
    npz_path.parent.mkdir()
    np.savez_compressed(npz_path, **raw)
    record = {"sample_id": "sample", "episode_index": 7, "frame_index": 5, "global_frame_index": 105,
              "task_index": 3, "npz_path": "samples/sample.npz", "npz_sha256": cached.file_sha256(npz_path),
              "raw_sample_sha256": cached.eval_openloop.input_content_sha256(raw)}
    manifest = {"kind": "raw_boundary_probe_cache", "schema_version": 1, "complete": True,
                "dataset_root": "dataset", "episodes_file": "selected_episodes.txt",
                "subtask_prompts": "dataset/meta/original_subtask_segments.parquet",
                "sample_keys_file": "probe_manifest.json", "repo_id": "repo", "records": [record],
                "artifact_hashes": {relative: cached.file_sha256(root / relative) for relative in files}}
    path = root / "cache_manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


class VerifiedRawCacheTest(unittest.TestCase):
    def test_exact_cached_frame_hash_and_string_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = make_cache(Path(folder))
            cache = cached.VerifiedRawCache(path)
            sample = cached.eval_openloop.EvalSample(3, 7, 5, 105, 0, "grasp", 6)
            raw = cache(sample)
            self.assertEqual(raw["subtask"], "grasp")
            self.assertIsInstance(raw["subtask"], str)
            self.assertEqual(len(cache.receipts), 1)
            with self.assertRaisesRegex(ValueError, "not explicitly cached"):
                cache(cached.eval_openloop.EvalSample(3, 7, 4, 104, 0, "grasp", 6))
            with self.assertRaisesRegex(ValueError, "index differs"):
                cache(cached.eval_openloop.EvalSample(3, 7, 5, 999, 0, "grasp", 6))

    def test_incomplete_missing_hash_and_wrong_raw_digest_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, original = make_cache(Path(folder))
            changed = copy.deepcopy(original)
            changed["complete"] = False
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                cached.VerifiedRawCache(path)
            changed = copy.deepcopy(original)
            del changed["artifact_hashes"]["dataset/meta/tasks.jsonl"]
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Unhashed"):
                cached.VerifiedRawCache(path)
            changed = copy.deepcopy(original)
            changed["records"][0]["raw_sample_sha256"] = "0" * 64
            path.write_text(json.dumps(changed))
            cache = cached.VerifiedRawCache(path)
            with self.assertRaisesRegex(ValueError, "Raw sample content"):
                cache(cached.eval_openloop.EvalSample(3, 7, 5, 105, 0, "grasp", 6))

    def test_npz_tamper_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manifest = make_cache(Path(folder))
            cache = cached.VerifiedRawCache(path)
            (Path(folder) / "samples/sample.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                cache(cached.eval_openloop.EvalSample(3, 7, 5, 105, 0, "grasp", 6))
            manifest["records"][0]["npz_path"] = "../not-part-of-cache.npz"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "escapes"):
                cached.VerifiedRawCache(path)

    def test_wrapper_calls_original_main_and_builder_without_reimplementing_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, _ = make_cache(root)
            output = root / "output.json"
            calls = []

            def original_main(argv, *, build_policy):
                calls.append(argv)
                args = SimpleNamespace(out=output)
                build_policy(args, [7])
                output.write_text(json.dumps({"sample_keys": [{"episode_index": 7, "frame_index": 5}],
                                              "original_metric_marker": 37}))

            def builder(args, episodes, *, sample_provider):
                self.assertEqual(episodes, [7])
                sample_provider(cached.eval_openloop.EvalSample(3, 7, 5, 105, 0, "grasp", 6))
                return None, None

            with mock.patch.object(cached.eval_openloop, "main", side_effect=original_main), \
                    mock.patch.object(cached.eval_openloop, "build_pi05", side_effect=builder):
                cached.main(["--cache-manifest", str(path), "--pi05-ckpt", "/checkpoint", "--out", str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(result["original_metric_marker"], 37)
            self.assertEqual(result["raw_cache"]["samples_read"], 1)
            self.assertFalse(result["raw_cache"]["unselected_raw_frames_read"])
            self.assertIn("--action-noise-seed", calls[0])
            self.assertIn("--sample-keys-file", calls[0])


if __name__ == "__main__":
    unittest.main()
