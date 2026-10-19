"""Small source-identity, transfer-bound and cache-resume tests; no network."""
from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import prepare_probe_cache as cache


class FakeResponse:
    status = 206

    def __init__(self, data, start, end, *, honor_range=True, etag="source-etag"):
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}", "ETag": etag}
        self.status = 206 if honor_range else 200
        self.stream = io.BytesIO(data[start:end + 1])
        self.resp = type("Resp", (), {"response": self.stream})()

    def read(self, count):
        return self.stream.read(count)


class FakeBucket:
    def __init__(self, data, honor_range=True):
        self.data, self.requests, self.honor_range = data, [], honor_range

    def get_object(self, key, byte_range, headers):
        self.requests.append((key, byte_range, headers))
        return FakeResponse(self.data, *byte_range, honor_range=self.honor_range)


class ProbeCacheTests(unittest.TestCase):
    def test_task_subset_keeps_exact_source_order_and_identity(self):
        samples = [{"task_name": task, "sample_id": str(i), "boundary_id": str(i), "episode_index": i}
                   for i, task in enumerate(("b", "a", "b", "c"))]
        source = {"kind": "matched_training_boundary_probe", "model_horizon_H": 32, "sample_keys": samples,
                  "boundaries": [{"boundary_id": str(i)} for i in range(4)]}
        selected = cache.select_manifest(source, ["b"])
        self.assertEqual(selected["sample_keys"], [samples[0], samples[2]])
        self.assertEqual(selected["boundaries"], [{"boundary_id": "0"}, {"boundary_id": "2"}])
        self.assertEqual(len(source["sample_keys"]), 4)
        with self.assertRaises(ValueError):
            cache.select_manifest(source, ["missing"])

    def test_reader_pins_identity_and_enforces_byte_bound(self):
        bucket = FakeBucket(b"0123456789")
        budget = cache.TransferBudget(5, 60)
        stream = cache.ObjectReader(bucket, {"key": "selected", "bytes": 10, "etag": "source-etag"}, budget)
        stream.seek(3)
        self.assertEqual(stream.read(4), b"3456")
        self.assertEqual(bucket.requests, [("selected", (3, 6), {"If-Match": "source-etag"})])
        with self.assertRaises(ValueError):
            stream.read(2)
        self.assertEqual(len(bucket.requests), 1)
        self.assertEqual(budget.used_bytes, 4)

    def test_reader_rejects_whole_object_fallback(self):
        bucket = FakeBucket(b"0123456789", honor_range=False)
        stream = cache.ObjectReader(bucket, {"key": "selected", "bytes": 10, "etag": "source-etag"},
                                    cache.TransferBudget(20, 60))
        with self.assertRaises(ValueError):
            stream.read(4)

    def test_raw_sample_uses_real_indices_and_full_action_horizon(self):
        spec = {"episode_index": 7, "frame_index": 3, "global_frame_index": 103, "task_index": 8,
                "prompt": "test task", "subtask": "test subtask"}
        columns = {"episode_index": [7] * 40, "frame_index": list(range(40)), "index": list(range(100, 140)),
                   "task_index": [8] * 40, "timestamp": [i / 30 for i in range(40)],
                   "action": [[i] * 23 for i in range(40)], "observation.state": [[i] * 61 for i in range(40)]}
        sample = cache.make_raw_sample(spec, columns, {"observation.rgb.camera": np.zeros((3, 2, 2), np.float32)})
        np.testing.assert_array_equal(sample["action"][:, 0], np.arange(3, 35))
        self.assertFalse(sample["action_is_pad"].any())
        self.assertEqual(sample["timestamp"].item(), 0.1)
        bad = {**columns, "index": list(range(101, 141))}
        with self.assertRaises(ValueError):
            cache.make_raw_sample(spec, bad, {})

    def test_npz_round_trip_and_resume_rejects_corruption(self):
        sample = {"episode_index": np.asarray(7, dtype=np.int64), "frame_index": np.asarray(3, dtype=np.int64),
                  "index": np.asarray(103, dtype=np.int64), "task_index": np.asarray(8, dtype=np.int64),
                  "prompt": "将鞋擦亮", "subtask": "polish shoe", "action": np.zeros((32, 23), np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.npz"
            np.savez_compressed(path, **sample)
            self.assertEqual(cache.raw_sample_hash(sample), cache.raw_sample_hash(cache.load_npz(path)))
            row = {"sample_id": "a", "episode_index": 7, "frame_index": 3, "global_frame_index": 103, "task_index": 8}
            plan = {"samples": [row]}
            plan["contract_sha256"] = __import__("hashlib").sha256(cache.canonical(plan)).hexdigest()
            record = {**row, "npz_path": "sample.npz", "npz_sha256": cache.sha_file(path),
                      "raw_sample_sha256": cache.raw_sample_hash(sample)}
            manifest = {"kind": cache.KIND, "contract_sha256": plan["contract_sha256"], "artifact_hashes": {},
                        "records": [record], "complete": True}
            cache.validate_cache(root, plan, manifest)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                cache.validate_cache(root, plan, manifest)

    def test_plan_projects_selected_episode_rows_without_video(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = {"episode_index": [1, 2, 1], "frame_index": [0, 0, 1], "index": [0, 1, 2],
                   "task_index": [0] * 3, "timestamp": [0., 0., 1 / 30],
                   "action": [[0.] * 23] * 3, "observation.state": [[0.] * 61] * 3,
                   "unrequested": [b"x" * 10000] * 3}
        buffer = io.BytesIO()
        pq.write_table(pa.table(columns), buffer)
        data = buffer.getvalue()
        bucket = FakeBucket(data)
        selected = cache.read_selected_parquet(bucket, {"key": "parquet", "bytes": len(data), "etag": "source-etag"},
                                               cache.TransferBudget(1024 * 1024, 60), {1})
        self.assertEqual(selected["episode_index"], [1, 1])
        self.assertNotIn("unrequested", selected)
        self.assertTrue(all(key == "parquet" for key, _, _ in bucket.requests))

    def test_relocated_file_only_plan_prepares_real_parquet_group_and_npz(self):
        """A file-only deployment must not need original paths or empty dirs.

        Arrow reads the actual small parquet fixture through ObjectReader. Only
        authentication and video transport/decoding are mocked. The ordinary
        prepare loop must create all four NPZs and their verifiable receipts.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        def parquet_bytes(value):
            buffer = io.BytesIO()
            pq.write_table(pa.table(value), buffer)
            return buffer.getvalue()

        task = "polishing_shoes"
        cameras = ["observation.rgb." + name for name in ("left", "right", "zed")]
        frame_ids = [0, 32, 48, 63]
        samples = [{"sample_id": f"{i:064x}", "episode_index": 7, "frame_index": frame,
                    "task_name": task, "task_instance_id": 8, "boundary_id": "boundary",
                    "segment_start": 0, "segment_end": 64, "subtask": "polish shoe"}
                   for i, frame in enumerate(frame_ids)]
        info = {"features": {camera: {"dtype": "video", "shape": [2, 3, 3]} for camera in cameras},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
        episode = {"episode_index": 7, "task_index": 3, "task_instance_id": 8, "length": 128,
                   "dataset_from_index": 100, "dataset_to_index": 228, "data/chunk_index": 0, "data/file_index": 0}
        data_relative = "data/chunk-000/file-000.parquet"
        payload = parquet_bytes({"episode_index": [7] * 128, "frame_index": list(range(128)),
                                 "index": list(range(100, 228)), "task_index": [3] * 128,
                                 "timestamp": [i / 30 for i in range(128)],
                                 "action": [[float(i)] * 23 for i in range(128)],
                                 "observation.state": [[float(i)] * 61 for i in range(128)]})
        objects = [{"relative_path": data_relative, "key": "source/" + data_relative,
                    "bytes": len(payload), "etag": "source-etag"}]
        offsets = {}
        for i, camera in enumerate(cameras, 1):
            prefix = "videos/" + camera
            offsets[camera] = i * 10.0
            episode.update({prefix + "/chunk_index": 0, prefix + "/file_index": 0,
                            prefix + "/from_timestamp": offsets[camera], prefix + "/to_timestamp": offsets[camera] + 128 / 30})
            relative = prefix + "/chunk-000/file-000.mp4"
            objects.append({"relative_path": relative, "key": "source/" + relative, "bytes": 16, "etag": "source-etag"})
        members = {"source_objects.json": cache.canonical({"bucket": "fixture", "prefix": "source/", "objects": objects}),
                   "expert_original/meta/info.json": cache.canonical(info),
                   "expert_original/meta/tasks.jsonl": cache.canonical({"task_index": 3, "task_name": task, "task": "Polish the shoes."}) + b"\n",
                   "expert_original/meta/episodes/chunk-000/file-000.parquet": parquet_bytes({key: [value] for key, value in episode.items()}),
                   "expert_original/meta/original_subtask_segments.parquet": parquet_bytes({"episode_index": [7], "start": [0], "end": [128], "subtask": ["polish shoe"]})}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin, deployed = root / "origin", root / "different-host" / "deployed"
            origin.mkdir()
            archive = origin / "metadata.tar.gz"
            with tarfile.open(archive, "w:gz") as packed:
                for name, data in members.items():
                    entry = tarfile.TarInfo(name)
                    entry.size = len(data)
                    packed.addfile(entry, io.BytesIO(data))
            source_manifest = origin / "probe.json"
            source_manifest.write_bytes(cache.canonical({"kind": "matched_training_boundary_probe", "model_horizon_H": 32,
                "repo_id": "fixture/repo", "source_index_path": str(origin / "informational-index.json"),
                "sample_keys": samples, "boundaries": [{"boundary_id": "boundary"}]}))
            plan_dir = origin / "plan"
            cache.make_plan(SimpleNamespace(manifest=source_manifest, metadata_archive=archive,
                metadata_sha256=cache.sha_file(archive), tasks=[task], output=plan_dir, max_bytes=1024**2, max_seconds=60))
            for path in plan_dir.rglob("*"):
                if path.is_file():
                    target = deployed / path.relative_to(plan_dir)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            # Mimic the archive uploader that carried files but no empty dirs.
            self.assertFalse((deployed / "samples").exists())
            shutil.rmtree(origin)
            bucket = FakeBucket(payload)
            decoded_calls = []

            def decode_fixture(_bucket, record, budget, timestamps, shape):
                self.assertIs(_bucket, bucket)
                self.assertTrue((deployed / "samples").is_dir())
                camera = next(camera for camera in cameras if camera in record["key"])
                self.assertEqual(shape, [2, 3, 3])
                np.testing.assert_allclose(timestamps, [offsets[camera] + frame / 30 for frame in frame_ids])
                decoded_calls.append(camera)
                budget.reserve(16)
                return ([np.full((3, 2, 3), i / 10, np.float32) for i in range(4)], timestamps)

            with mock.patch.object(cache, "open_source_bucket", return_value=bucket), \
                 mock.patch.object(cache, "decode_rgb", side_effect=decode_fixture):
                with mock.patch("sys.stdout", new=io.StringIO()):
                    result = cache.prepare(deployed, None)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["records"]), 4)
            self.assertEqual(decoded_calls, cameras)
            self.assertGreater(len(bucket.requests), 0)
            for i, record in enumerate(result["records"]):
                self.assertFalse(Path(record["npz_path"]).is_absolute())
                raw = cache.load_npz(deployed / record["npz_path"])
                np.testing.assert_array_equal(raw["action"][:, 0], np.arange(frame_ids[i], frame_ids[i] + 32))
                self.assertEqual(raw["prompt"], "Polish the shoes.")
                self.assertEqual(raw["subtask"], "polish shoe")
                self.assertFalse(raw["action_is_pad"].any())
                self.assertEqual(cache.raw_sample_hash(raw), record["raw_sample_sha256"])
            cache.validate_cache(deployed, json.loads((deployed / "cache_plan.json").read_text()), result)
            # Verified completion must make a relocated resume read-only.
            with mock.patch.object(cache, "open_source_bucket", side_effect=AssertionError("unexpected remote access")):
                self.assertEqual(cache.prepare(deployed, None), result)


if __name__ == "__main__":
    unittest.main()
