"""Offline-only tests: fake S3 responses and a local synthetic tar/HDF5."""
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import sparse_native_frames as sparse


class FakeS3:
    def __init__(self, data):
        self.data, self.etag = data, '"frozen-etag"'
        self.requests, self.bodies = [], []
        self.change = None
        self.head_change = {}

    def head_object(self, **kwargs):
        return {"ResponseMetadata": {"HTTPStatusCode": 200}, "ContentLength": len(self.data),
                "ETag": self.etag, **self.head_change}

    def get_object(self, **kwargs):
        self.requests.append(kwargs)
        assert kwargs["IfMatch"] == self.etag
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", kwargs["Range"])
        assert match is not None
        begin, end = map(int, match.groups())
        response = {"ResponseMetadata": {"HTTPStatusCode": 206}, "ContentLength": end - begin + 1,
                    "ContentRange": f"bytes {begin}-{end}/{len(self.data)}", "ETag": self.etag}
        payload = self.data[begin:end + 1]
        if self.change is not None:
            response, payload = self.change(response, payload)
        response["Body"] = io.BytesIO(payload)
        self.bodies.append(response["Body"])
        return response


def archive_record(client, key="episode.tar"):
    return {"bytes": len(client.data), "etag": client.etag, "object_key": key, "sha256": "a" * 64}


def synthetic_episode(*, external=False, duplicate=False):
    count = 12
    receipt = {"task": "tidying_bathroom", "instance_id": 301, "attempt_id": "test-attempt", "run_id": "test-run",
               "source_manifest_sha256": "b" * 64, "steps": count, "content_sha256": "c" * 64,
               "subtask_sha256": "d" * 64,
               "result": {"task": "tidying_bathroom", "instance_id": 301, "steps": count, "success": True}}
    expected = {"observation.state": np.arange(count * 57, dtype=np.float32).reshape(count, 57) / 100,
                "action": np.arange(count * 23, dtype=np.float32).reshape(count, 23) / 10}
    for index, name in enumerate(sparse.CAMERAS):
        expected[name] = (np.arange(count * 8 * 9 * 3).reshape(count, 8, 9, 3) + index).astype(np.uint8)
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as file:
        file.attrs.update(kind="native_full_episode_collection", schema_version=3, complete=True, control_steps=count,
                          metadata=json.dumps({key: receipt[key] for key in
                                               ("task", "instance_id", "attempt_id", "run_id", "source_manifest_sha256")}),
                          result=json.dumps(receipt["result"]), content_sha256=receipt["content_sha256"],
                          subtask_sha256=receipt["subtask_sha256"])
        for name, value in expected.items():
            file.create_dataset(name, data=value, chunks=(1, *value.shape[1:]), compression="gzip")
        file.create_dataset("frame_index", data=np.arange(count, dtype=np.int64))
        # Keep the archive substantially larger than the selected data so the
        # test can demonstrate skipped payload, rather than a tiny whole read.
        file.create_dataset("unused_payload", data=np.zeros(1024**2, dtype=np.uint8))
        if external:
            del file[sparse.CAMERAS[0]]
            file[sparse.CAMERAS[0]] = h5py.ExternalLink("must_not_be_opened.h5", "/rgb")
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w", format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo("episode.hdf5")
        payload = buffer.getvalue()
        member.size = len(payload)
        member.pax_headers = {"mtime": "0.25"}
        archive.addfile(member, io.BytesIO(payload))
        if duplicate:
            archive.addfile(member, io.BytesIO(payload))
        result = json.dumps(receipt["result"]).encode()
        member = tarfile.TarInfo("result.json")
        member.size = len(result)
        archive.addfile(member, io.BytesIO(result))
    client = FakeS3(packed.getvalue())
    receipt["archive"] = archive_record(client)
    return client, receipt, expected


class SparseNativeFramesTest(unittest.TestCase):
    def test_budgets_precede_get_and_count_shared_archives(self):
        client = FakeS3(bytes(range(128)))
        reader = sparse.FrozenRanges(client, "bucket", archive_record(client),
                                     budget=sparse.RangeBudget(per_archive=8, total=100))
        with self.assertRaisesRegex(ValueError, "Per-archive"):
            reader.exact(0, 8)  # The extra response-validation byte is reserved.
        self.assertEqual(client.requests, [])
        budget = sparse.RangeBudget(per_archive=20, total=17)
        first = sparse.FrozenRanges(client, "bucket", archive_record(client, "first"), budget=budget)
        second = sparse.FrozenRanges(client, "bucket", archive_record(client, "second"), budget=budget)
        self.assertEqual(first.exact(0, 8), bytes(range(8)))
        with self.assertRaisesRegex(ValueError, "Total"):
            second.exact(0, 8)
        self.assertEqual(len(client.requests), 1)

    def test_cache_eof_unbounded_and_head_guards(self):
        client = FakeS3(bytes(range(128)))
        reader = sparse.FrozenRanges(client, "bucket", archive_record(client), budget=sparse.RangeBudget(), block_size=16)
        self.assertEqual(reader.read(4), bytes(range(4)))
        reader.seek(2)
        self.assertEqual(reader.read(4), bytes(range(2, 6)))
        self.assertEqual(len(client.requests), 1)
        reader.seek(0, io.SEEK_END)
        self.assertEqual(reader.read(4), b"")
        self.assertEqual(reader.exact(0, 0), b"")
        with self.assertRaisesRegex(ValueError, "Unbounded"):
            reader.read()
        with self.assertRaisesRegex(ValueError, "bounds"):
            reader.seek(-1)
        client.head_change = {"ETag": '"changed"'}
        with self.assertRaisesRegex(ValueError, "HEAD"):
            reader.finish()

    def test_every_range_response_contract_is_enforced(self):
        changes = [lambda r, p: ({**r, "ResponseMetadata": {"HTTPStatusCode": 200}}, p),
                   lambda r, p: ({**r, "ContentRange": "bytes 1-8/128"}, p),
                   lambda r, p: ({**r, "ContentLength": 9}, p),
                   lambda r, p: ({**r, "ETag": '"changed"'}, p),
                   lambda r, p: (r, p[:-1]), lambda r, p: (r, p + b"extra")]
        for change in changes:
            with self.subTest(change=change):
                client = FakeS3(bytes(range(128)))
                client.change = change
                reader = sparse.FrozenRanges(client, "bucket", archive_record(client), budget=sparse.RangeBudget())
                with self.assertRaisesRegex(ValueError, "Rejected"):
                    reader.exact(0, 8)
                self.assertTrue(client.bodies[-1].closed)
                self.assertEqual(len(client.requests), 1)

    def test_tar_hdf5_random_reads_and_export_are_exact_and_sparse(self):
        client, receipt, expected = synthetic_episode()
        frames = [1, 9]
        entry = {"id": "synthetic", "frames": frames, "receipt_file_sha256": "e" * 64}
        manifest = {"bucket": "bucket", "selection_note": "synthetic offline fixture"}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "selected"
            with mock.patch.object(sparse, "load_manifest", return_value=(manifest, [(entry, receipt)], "f" * 64)):
                report = sparse.run(client, "unused.json", output)
            with np.load(output / "synthetic/frames.npz", allow_pickle=False) as packed:
                np.testing.assert_array_equal(packed["frame_index"], frames)
                for name, values in expected.items():
                    np.testing.assert_array_equal(packed[name], values[frames])
            for name in sparse.CAMERAS:
                path = output / "synthetic" / f"frame_{frames[0]:06d}_{name.removeprefix('observation.rgb.')}.png"
                np.testing.assert_array_equal(np.asarray(Image.open(path)), expected[name][frames[0]])
            provenance = report["episodes"][0]["provenance"]
            self.assertFalse(provenance["whole_archive_sha256_verified"])
            self.assertFalse(provenance["whole_hdf5_sha256_verified"])
            self.assertFalse(provenance["terminal_observation_group_read"])
            self.assertIn("pre-action", provenance["observation_alignment"])
            self.assertLess(provenance["downloaded_body_bytes"], len(client.data) / 2)
            self.assertTrue(all("Range" in request and "IfMatch" in request for request in client.requests))

    def test_bounds_external_links_and_duplicate_members_are_rejected(self):
        client, receipt, _ = synthetic_episode()
        with self.assertRaisesRegex(ValueError, "exact HDF5 bounds"):
            sparse.read_episode(client, "bucket", receipt, [12], budget=sparse.RangeBudget())
        client, receipt, _ = synthetic_episode(external=True)
        with self.assertRaisesRegex(ValueError, "hard-linked"):
            sparse.read_episode(client, "bucket", receipt, [1], budget=sparse.RangeBudget())
        client, receipt, _ = synthetic_episode(duplicate=True)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            sparse.read_episode(client, "bucket", receipt, [1], budget=sparse.RangeBudget())

    def test_real_local_manifest_is_receipt_bound_without_network(self):
        manifest, selected, digest = sparse.load_manifest(Path(__file__).with_name("selected_frames_manifest.json"))
        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(len(entry["frames"]) for entry, _ in selected), 6)
        self.assertEqual(manifest["total_limit_bytes"], 64 * 1024**2)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
