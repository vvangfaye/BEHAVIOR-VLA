#!/usr/bin/env python3
"""Read only manifest-selected native HDF5 frames through pinned S3 ranges.

Run with the existing numeric Python environment and boto3 on PYTHONPATH:
  python sparse_native_frames.py --manifest selected_frames_manifest.json \
      --output selected_frames

No tar member is extracted to disk. Only the two frozen bathroom receipts and
their six explicitly selected frames are accepted. The 32 MiB per-archive and
64 MiB total budgets include metadata, cache misses/re-reads and the extra byte
reserved for response-length validation. There is no full-object fallback.
ETag pinning and range hashes DO NOT recompute either whole-object SHA256.
All selected images/states precede their indexed recorded action. In particular,
steps-1 is not a post-terminal observation; terminal_observation is not read.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import io
import json
import numbers
from pathlib import Path
import re
import sys
import tarfile
import threading


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
KIND = "sparse_native_episode_frames_v1"
BUCKET = "260010165"
PREFIX = "behavior1k/full-episode-collection/native-full1000-1688977-20261014-r3/ddea3ef13b17/episodes/"
PER_ARCHIVE_LIMIT = 32 * 1024**2
TOTAL_LIMIT = 64 * 1024**2
BLOCK_SIZE = 64 * 1024
CAMERAS = ("observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
           "observation.rgb.right_realsense_link_camera_0")
FIELDS = ("observation.state", "action", *CAMERAS)
ALLOWED = {"tidying_bathroom_301_success": (301, True, [8272, 8991, 14761]),
           "tidying_bathroom_304_failure": (304, False, [7920, 8367, 19504])}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require_int(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid integer: {name}")
    return value


class RangeBudget:
    def __init__(self, per_archive=PER_ARCHIVE_LIMIT, total=TOTAL_LIMIT):
        self.per_archive, self.total = per_archive, total
        self.claimed = 0
        self.by_archive = {}
        self.lock = threading.Lock()

    def reserve(self, key, size):
        require_int(size, "range reservation", minimum=1)
        with self.lock:
            previous = self.by_archive.get(key, 0)
            if previous + size > self.per_archive:
                raise ValueError("Per-archive range budget exceeded")
            if self.claimed + size > self.total:
                raise ValueError("Total range budget exceeded")
            self.by_archive[key] = previous + size
            self.claimed += size


class FrozenRanges(io.RawIOBase):
    """Seekable, bounded, cached S3 reader. Every GET has Range and IfMatch."""
    def __init__(self, client, bucket, archive, *, budget, block_size=BLOCK_SIZE, cache_blocks=16):
        super().__init__()
        self.client, self.bucket, self.archive, self.budget = client, bucket, archive, budget
        self.size = require_int(archive["bytes"], "archive bytes", minimum=1)
        self.key, self.etag = archive["object_key"], archive["etag"]
        if not isinstance(self.key, str) or not isinstance(self.etag, str) or not self.etag:
            raise ValueError("Archive key and ETag must be explicit")
        self.block_size = require_int(block_size, "block size", minimum=1)
        self.cache_blocks = require_int(cache_blocks, "cache blocks", minimum=1)
        self.position = self.received = self.cache_hits = 0
        self.cache, self.receipts = OrderedDict(), []
        self._check_head()

    def _check_head(self):
        response = self.client.head_object(Bucket=self.bucket, Key=self.key)
        if (response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 200 or
                type(response.get("ContentLength")) is not int or response["ContentLength"] != self.size or
                response.get("ETag") != self.etag):
            raise ValueError("HEAD differs from receipt bytes/ETag")

    def exact(self, offset, size):
        require_int(offset, "range offset")
        require_int(size, "range size")
        if offset + size > self.size:
            raise ValueError("Range exceeds archive bounds")
        if size == 0:
            return b""
        # Reserve the largest possible body read before contacting the server.
        self.budget.reserve((self.bucket, self.key), size + 1)
        end = offset + size - 1
        response = self.client.get_object(Bucket=self.bucket, Key=self.key,
                                          Range=f"bytes={offset}-{end}", IfMatch=self.etag)
        body = response.get("Body")
        try:
            if (response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 206 or
                    response.get("ContentRange") != f"bytes {offset}-{end}/{self.size}" or
                    type(response.get("ContentLength")) is not int or response["ContentLength"] != size or
                    response.get("ETag") != self.etag or body is None):
                raise ValueError("Rejected range response status/ContentRange/length/ETag")
            data = body.read(size + 1)
            self.received += len(data)
            if len(data) != size:
                raise ValueError("Rejected truncated or oversized range body")
        finally:
            if body is not None:
                body.close()
        self.receipts.append({"offset": offset, "bytes": size, "http_status": 206,
                              "content_range": response["ContentRange"], "etag": response["ETag"],
                              "sha256": sha256(data), "reserved_budget_bytes": size + 1})
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if not isinstance(offset, numbers.Integral) or isinstance(offset, bool):
            raise ValueError("Invalid seek offset")
        if whence == io.SEEK_SET:
            target = int(offset)
        elif whence == io.SEEK_CUR:
            target = self.position + int(offset)
        elif whence == io.SEEK_END:
            target = self.size + int(offset)
        else:
            raise ValueError("Invalid seek mode")
        if target < 0 or target > self.size:
            raise ValueError("Seek exceeds archive bounds")
        self.position = target
        return target

    def read(self, size=-1):
        if not isinstance(size, numbers.Integral) or size < 0:
            raise ValueError("Unbounded archive reads are forbidden")
        size = min(int(size), self.size - self.position)
        if size > self.budget.per_archive:
            raise ValueError("Single read exceeds per-archive budget")
        buffer = bytearray(size)
        self.readinto(buffer)
        return bytes(buffer)

    def readinto(self, buffer):
        view = memoryview(buffer).cast("B")
        size = min(len(view), self.size - self.position)
        written = 0
        while written < size:
            block = self.position // self.block_size
            if block not in self.cache:
                start = block * self.block_size
                self.cache[block] = self.exact(start, min(self.block_size, self.size - start))
                if len(self.cache) > self.cache_blocks:
                    self.cache.popitem(last=False)
            else:
                self.cache_hits += 1
            self.cache.move_to_end(block)
            data = self.cache[block]
            within = self.position % self.block_size
            count = min(size - written, len(data) - within)
            view[written:written + count] = data[within:within + count]
            written += count
            self.position += count
        return size

    def finish(self):
        self._check_head()
        return {"archive_object_key": self.key, "archive_bytes": self.size, "etag": self.etag,
                "head_before_and_after_matched": True, "downloaded_body_bytes": self.received,
                "reserved_budget_bytes": self.budget.by_archive.get((self.bucket, self.key), 0),
                "cache_hits": self.cache_hits, "range_receipts": self.receipts,
                "whole_archive_sha256_verified": False, "whole_hdf5_sha256_verified": False,
                "archive_sha256_claimed_by_prior_receipt": self.archive.get("sha256")}


def _safe_member(archive, source):
    members = []
    for member in archive:
        if len(members) >= 4:
            raise ValueError("Unexpected extra tar members")
        if (not member.isfile() or member.name not in ("episode.hdf5", "result.json") or
                any(key in ("path", "linkpath") or "sparse" in key.lower() for key in member.pax_headers) or
                member.sparse is not None or member.offset_data < 0 or
                member.offset_data + member.size > source.size):
            raise ValueError("Unsupported tar member, link, sparse layout or PAX path override")
        members.append(member)
    if sorted(member.name for member in members) != ["episode.hdf5", "result.json"]:
        raise ValueError("Tar must contain exactly one episode.hdf5 and one result.json")
    return next(member for member in members if member.name == "episode.hdf5")


def _dataset(file, name):
    import h5py
    link = file.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        raise ValueError("Only direct hard-linked HDF5 datasets are permitted")
    dataset = file[name]
    if (not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external or
            dataset.dtype.hasobject or dataset.compression not in (None, "gzip")):
        raise ValueError("Unsupported external, virtual or non-numeric HDF5 dataset")
    creation = dataset.id.get_create_plist()
    if any(creation.get_filter(index)[0] != 1 for index in range(creation.get_nfilters())):
        raise ValueError("Unsupported HDF5 filter")
    return dataset


def read_episode(client, bucket, receipt, frames, *, budget):
    import h5py
    import numpy as np
    if (not isinstance(frames, list) or not frames or any(type(frame) is not int for frame in frames) or
            frames != sorted(set(frames))):
        raise ValueError("Frames must be explicit, sorted, unique integers")
    source = FrozenRanges(client, bucket, receipt["archive"], budget=budget)
    try:
        with tarfile.open(fileobj=source, mode="r:") as archive:
            member = _safe_member(archive, source)
            with archive.extractfile(member) as member_file:
                if not isinstance(member_file, tarfile.ExFileObject):
                    raise ValueError("Expected bounded tar ExFileObject")
                with h5py.File(member_file, "r") as file:
                    if (file.attrs.get("kind") != "native_full_episode_collection" or
                            file.attrs.get("schema_version") != 3 or not bool(file.attrs.get("complete", False))):
                        raise ValueError("Unknown or incomplete native HDF5 episode")
                    n = file.attrs.get("control_steps")
                    if not isinstance(n, numbers.Integral) or n != receipt["steps"]:
                        raise ValueError("HDF5 step count differs from receipt")
                    n = int(n)
                    if frames[0] < 0 or frames[-1] >= n:
                        raise ValueError("Requested frame outside exact HDF5 bounds")
                    metadata, result = json.loads(file.attrs["metadata"]), json.loads(file.attrs["result"])
                    for name in ("task", "instance_id", "attempt_id", "run_id", "source_manifest_sha256"):
                        if metadata.get(name) != receipt[name]:
                            raise ValueError("HDF5 metadata identity differs from receipt")
                    if (result != receipt["result"] or file.attrs.get("content_sha256") != receipt["content_sha256"] or
                            file.attrs.get("subtask_sha256") != receipt["subtask_sha256"]):
                        raise ValueError("HDF5 recorded result/hash attributes differ from receipt")
                    datasets = {name: _dataset(file, name) for name in (*FIELDS, "frame_index")}
                    state, action = datasets["observation.state"], datasets["action"]
                    if (state.ndim != 2 or state.shape[0] != n or not 57 <= state.shape[1] <= 512 or
                            state.dtype != np.dtype("float32") or action.shape != (n, 23) or action.dtype != np.dtype("float32")):
                        raise ValueError("Invalid native state/action shape or dtype")
                    for name in CAMERAS:
                        dataset = datasets[name]
                        if (dataset.ndim != 4 or dataset.shape[0] != n or dataset.shape[-1] != 3 or
                                any(not 1 <= size <= 224 for size in dataset.shape[1:3]) or dataset.dtype != np.dtype("uint8")):
                            raise ValueError("Invalid native RGB shape or dtype")
                    for name in FIELDS:
                        if datasets[name].chunks != (1, *datasets[name].shape[1:]):
                            raise ValueError("Native frame fields must use independent per-frame chunks")
                    index = datasets["frame_index"]
                    if index.shape != (n,) or index.dtype.kind not in "iu":
                        raise ValueError("Invalid recorded frame index")
                    if any(int(index[frame]) != frame for frame in frames):
                        raise ValueError("Requested frame does not match recorded frame_index")
                    arrays = {name: np.stack([datasets[name][frame] for frame in frames]) for name in FIELDS}
                    if any(not np.isfinite(value).all() for value in arrays.values()):
                        raise ValueError("Nonfinite selected frame values")
                    arrays["frame_index"] = np.array(frames, dtype=np.int64)
                    layout = {name: {"shape": list(dataset.shape), "dtype": dataset.dtype.str,
                                     "chunks": list(dataset.chunks) if dataset.chunks else None,
                                     "compression": dataset.compression} for name, dataset in datasets.items()}
            provenance = source.finish()
            provenance.update({"tar_member": {"name": member.name, "offset_data": member.offset_data,
                                                "bytes": member.size}, "hdf5_datasets": layout,
                               "frames": frames, "selected_array_sha256": {
                                   name: sha256(value.tobytes()) for name, value in arrays.items()},
                               "observation_alignment": "recorded pre-action observation at zero-based frame_index",
                               "terminal_observation_group_read": False,
                               "last_index_note": "steps-1 precedes the final recorded action; not post-terminal state",
                               "hash_attribute_comparison_only": True})
        return arrays, provenance
    finally:
        source.close()


def load_manifest(path):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != KIND or manifest.get("bucket") != BUCKET or
            manifest.get("per_archive_limit_bytes") != PER_ARCHIVE_LIMIT or manifest.get("total_limit_bytes") != TOTAL_LIMIT):
        raise ValueError("Unexpected manifest kind, bucket or frozen byte budgets")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 2:
        raise ValueError("Manifest must select one or two authorized episodes")
    selected, seen = [], set()
    for entry in entries:
        identity = entry.get("id")
        if identity not in ALLOWED or identity in seen or entry.get("receipt_file") != identity + ".json":
            raise ValueError("Episode selection outside authorized scope")
        seen.add(identity)
        data = (HERE / entry["receipt_file"]).read_bytes()
        if sha256(data) != entry.get("receipt_file_sha256"):
            raise ValueError("Local receipt file SHA256 differs from manifest")
        receipt = json.loads(data)
        instance, success, frames = ALLOWED[identity]
        if (receipt.get("kind") != "native_full_episode_object_checkpoint" or receipt.get("task") != "tidying_bathroom" or
                receipt.get("instance_id") != instance or receipt.get("success") is not success or entry.get("frames") != frames or
                entry.get("archive") != receipt.get("archive") or
                receipt["archive"]["object_key"] != f"{PREFIX}tidying_bathroom/{instance}/attempt-1.tar"):
            raise ValueError("Receipt identity, archive or fixed frame selection differs")
        if any(type(frame) is not int or not 0 <= frame < receipt["steps"] for frame in frames):
            raise ValueError("Manifest frame outside receipt step bounds")
        selected.append((entry, receipt))
    return manifest, selected, sha256(raw)


def make_client():
    """Reuse the configured AOSS identity, with SDK retries disabled for budgets."""
    import boto3
    from botocore.config import Config
    sys.path.insert(0, str(PROJECT / "BEHAVIOR-VLAs/sco"))
    from video_eval_adapter import aoss_client
    original = aoss_client()
    try:
        credentials = original._request_signer._credentials.get_frozen_credentials()
        return boto3.client("s3", endpoint_url=original.meta.endpoint_url, region_name=original.meta.region_name,
                            aws_access_key_id=credentials.access_key, aws_secret_access_key=credentials.secret_key,
                            aws_session_token=credentials.token,
                            config=original.meta.config.merge(Config(retries={"total_max_attempts": 1})))
    finally:
        original.close()


def run(client, manifest_path, output):
    import numpy as np
    from PIL import Image
    manifest, selected, manifest_hash = load_manifest(manifest_path)
    output = Path(output)
    if output.exists():
        raise ValueError("Output path must not exist")
    budget = RangeBudget()
    results = []
    # Complete bounded reads and all validation before writing any frame artifact.
    for entry, receipt in selected:
        arrays, provenance = read_episode(client, manifest["bucket"], receipt, entry["frames"], budget=budget)
        results.append((entry, arrays, provenance))
    output.mkdir(parents=True)
    report = {"kind": KIND, "manifest_sha256": manifest_hash, "per_archive_limit_bytes": PER_ARCHIVE_LIMIT,
              "total_limit_bytes": TOTAL_LIMIT, "reserved_budget_bytes": budget.claimed,
              "whole_archive_sha256_verified": False, "whole_hdf5_sha256_verified": False,
              "selection_note": manifest.get("selection_note"), "episodes": []}
    for entry, arrays, provenance in results:
        target = output / entry["id"]
        target.mkdir()
        npz = target / "frames.npz"
        np.savez_compressed(npz, **arrays)
        artifacts = [{"file": str(npz.relative_to(output)), "sha256": sha256(npz.read_bytes())}]
        for index, frame in enumerate(entry["frames"]):
            for name in CAMERAS:
                png = target / f"frame_{frame:06d}_{name.removeprefix('observation.rgb.')}.png"
                Image.fromarray(arrays[name][index]).save(png)
                artifacts.append({"file": str(png.relative_to(output)), "sha256": sha256(png.read_bytes())})
        report["episodes"].append({"id": entry["id"], "receipt_file_sha256": entry["receipt_file_sha256"],
                                   "frame_labels": entry.get("frame_labels"),
                                   "provenance": provenance, "artifacts": artifacts})
    (output / "range_read_provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        load_manifest(args.manifest)
        if args.output.exists():
            raise ValueError("Output path must not exist")
        client = make_client()
        try:
            report = run(client, args.manifest, args.output)
        finally:
            client.close()
    except Exception as error:
        # SDK exception messages may contain signed request details.
        print(json.dumps({"error_type": type(error).__name__, "status": "bounded_read_rejected"}), file=sys.stderr)
        return 1
    print(json.dumps({"episodes": len(report["episodes"]), "reserved_budget_bytes": report["reserved_budget_bytes"],
                      "output": str(args.output), "whole_archive_sha256_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
