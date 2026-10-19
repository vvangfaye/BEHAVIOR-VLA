#!/usr/bin/env python3
"""Prepare bounded, immutable raw demonstration probes; never run inference.

``plan`` reads only local metadata. ``prepare`` reads only the selected parquet
columns and the RGB ranges sought by PyAV. It never copies the full dataset.
Credentials are loaded only by ``prepare`` and are never written to artifacts.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
from typing import Any

DEFAULT_TASKS = ("polishing_shoes", "tidying_living_room", "unloading_the_car")
DEFAULT_ARCHIVE_SHA256 = "59ca19b648d4d1346e085da70acbed0cf6a0b9dc387e939651bd15027dc4924b"
HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = HERE.parent / "retraining_20261013/preparation/original_stage/original_expert/expert_metadata.tar.gz"
CAMERA_PREFIX = "observation.rgb."
KIND = "raw_boundary_probe_cache"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError("Unsafe relative artifact path")
    return path


def raw_sample_hash(value: Any) -> str:
    """Same structural hash as eval_openloop.input_content_sha256."""
    import numpy as np

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item):
                if not isinstance(key, str):
                    raise ValueError("Sample keys must be strings")
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}\0".encode())
            for child in item:
                visit(child)
        elif isinstance(item, str):
            raw = item.encode()
            digest.update(f"str:{len(raw)}\0".encode())
            digest.update(raw)
        elif item is None:
            digest.update(b"none\0")
        else:
            array = np.asarray(item)
            if array.dtype.hasobject:
                raise ValueError("Object arrays are forbidden")
            digest.update(f"array:{array.dtype.str}:{array.shape}\0".encode())
            digest.update(np.ascontiguousarray(array).tobytes())

    visit(value)
    return digest.hexdigest()


def load_npz(path: Path) -> dict:
    import numpy as np

    with np.load(path, allow_pickle=False) as packed:
        return {key: (packed[key].item() if packed[key].dtype.kind in "US" and packed[key].ndim == 0
                      else packed[key].copy()) for key in packed.files}


def select_manifest(manifest: dict, tasks: list[str] | tuple[str, ...]) -> dict:
    if manifest.get("kind") != "matched_training_boundary_probe" or manifest.get("model_horizon_H") != 32:
        raise ValueError("Expected the frozen 32-step boundary probe manifest")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Explicit, unique task names are required")
    available = {row["task_name"] for row in manifest["sample_keys"]}
    if set(tasks) - available:
        raise ValueError("Task is absent from the frozen probe manifest")
    selected = [dict(row) for row in manifest["sample_keys"] if row["task_name"] in set(tasks)]
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("Repeated sample ID")
    boundary_ids = {row["boundary_id"] for row in selected}
    result = dict(manifest)
    result["sample_keys"] = selected
    result["boundaries"] = [row for row in manifest["boundaries"] if row["boundary_id"] in boundary_ids]
    result["cache_selection"] = {"tasks": list(tasks), "parent_manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
                                 "selection_changed": False, "description": "Exact task subset; no anchors resampled"}
    result["statistics"] = {"selected_samples": len(selected), "selected_episodes": len({r["episode_index"] for r in selected}),
                            "selected_boundaries": len(boundary_ids), "tasks_with_selected_boundaries": len(tasks)}
    return result


def metadata_plan(archive_path: Path, manifest: dict, expected_sha256: str) -> tuple[dict, dict[str, bytes]]:
    import pyarrow.parquet as pq

    if sha_file(archive_path) != expected_sha256:
        raise ValueError("Metadata archive SHA256 differs from the accepted source")
    wanted = {row["episode_index"] for row in manifest["sample_keys"]}
    blobs, episodes = {}, {}
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            if member.name == "source_objects.json" or member.name.startswith("expert_original/meta/"):
                safe_relative(member.name)
                if member.size > 128 * 1024**2:
                    raise ValueError("Oversized metadata member")
                blob = archive.extractfile(member).read()
                blobs[member.name] = blob
                if member.name.startswith("expert_original/meta/episodes/") and member.name.endswith(".parquet"):
                    for row in pq.read_table(io.BytesIO(blob)).to_pylist():
                        if row["episode_index"] in wanted:
                            if row["episode_index"] in episodes:
                                raise ValueError("Repeated episode metadata")
                            episodes[row["episode_index"]] = row
    if set(episodes) != wanted:
        raise ValueError("Requested episodes are absent from the accepted metadata")
    info = json.loads(blobs["expert_original/meta/info.json"])
    source = json.loads(blobs["source_objects.json"])
    inventory = {row["relative_path"]: row for row in source["objects"]}
    tasks = {row["task_index"]: row for row in map(json.loads, blobs["expert_original/meta/tasks.jsonl"].decode().splitlines())}
    cameras = sorted(key for key, value in info["features"].items() if key.startswith(CAMERA_PREFIX) and value["dtype"] == "video")
    if len(cameras) != 3:
        raise ValueError("Expected the three accepted RGB cameras")
    needed, specs = {}, []

    def object_for(relative):
        safe_relative(relative)
        row = inventory[relative]
        if row["key"] != source["prefix"].rstrip("/") + "/" + relative or row["bytes"] <= 0:
            raise ValueError("Invalid frozen source object identity")
        needed[relative] = row
        return relative

    for sample in manifest["sample_keys"]:
        ep = episodes[sample["episode_index"]]
        frame = sample["frame_index"]
        task = tasks[ep["task_index"]]
        if task["task_name"] != sample["task_name"] or ep["task_instance_id"] != sample["task_instance_id"]:
            raise ValueError("Frozen sample task identity differs from original metadata")
        if not 0 <= frame <= ep["length"] - 32:
            raise ValueError("Frozen probe lacks the complete recorded action horizon")
        data = object_for(info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]))
        videos = {}
        for camera in cameras:
            prefix = "videos/" + camera
            relative = object_for(info["video_path"].format(video_key=camera, chunk_index=ep[prefix + "/chunk_index"],
                                                           file_index=ep[prefix + "/file_index"]))
            videos[camera] = {"relative_path": relative, "from_timestamp": ep[prefix + "/from_timestamp"],
                              "to_timestamp": ep[prefix + "/to_timestamp"], "shape": info["features"][camera]["shape"]}
        specs.append({**sample, "task_index": ep["task_index"], "global_frame_index": ep["dataset_from_index"] + frame,
                      "prompt": task["task"], "episode_length": ep["length"], "data_relative_path": data, "videos": videos})
    plan = {"schema_version": 1, "kind": KIND, "metadata_archive_sha256": expected_sha256,
            "source_bucket": source["bucket"], "source_prefix": source["prefix"], "cameras": cameras,
            "samples": specs, "objects": [needed[key] for key in sorted(needed)],
            "selected_episodes": sorted(wanted), "sample_count": len(specs),
            "whole_selected_object_bytes": sum(row["bytes"] for row in needed.values()),
            "inference_executed": False, "new_supervision_collected": False}
    return plan, blobs


def make_plan(args) -> dict:
    parent = json.loads(args.manifest.read_text())
    subset = select_manifest(parent, args.tasks)
    plan, blobs = metadata_plan(args.metadata_archive, subset, args.metadata_sha256)
    plan["transfer_limits"] = {"maximum_bytes": args.max_bytes, "maximum_seconds": args.max_seconds}
    plan["contract_sha256"] = hashlib.sha256(canonical(plan)).hexdigest()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples").mkdir()
    for name, blob in blobs.items():
        if name.startswith("expert_original/meta/"):
            target = output / "dataset" / Path(name).relative_to("expert_original")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
    atomic_json(output / "probe_manifest.json", subset)
    (output / "selected_episodes.txt").write_text("\n".join(map(str, plan["selected_episodes"])) + "\n")
    atomic_json(output / "cache_plan.json", plan)
    hashes = {str(p.relative_to(output)): sha_file(p) for p in output.rglob("*") if p.is_file()}
    manifest = {"schema_version": 1, "kind": KIND, "dataset_root": "dataset", "episodes_file": "selected_episodes.txt",
                "subtask_prompts": "dataset/meta/original_subtask_segments.parquet", "sample_keys_file": "probe_manifest.json",
                "repo_id": parent["repo_id"], "contract_sha256": plan["contract_sha256"], "artifact_hashes": hashes,
                "records": [], "complete": False, "inference_executed": False, "new_supervision_collected": False,
                "transfer_bytes": 0, "transfer_requests": 0}
    atomic_json(output / "cache_manifest.json", manifest)
    return plan


class TransferBudget:
    def __init__(self, maximum_bytes: int, maximum_seconds: int, used_bytes: int = 0):
        if maximum_bytes <= 0 or maximum_seconds <= 0 or not 0 <= used_bytes <= maximum_bytes:
            raise ValueError("Invalid transfer budget")
        self.maximum_bytes, self.used_bytes, self.requests = maximum_bytes, used_bytes, 0
        self.deadline = time.monotonic() + maximum_seconds

    def reserve(self, amount: int):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Probe preparation deadline reached")
        if amount < 0 or self.used_bytes + amount > self.maximum_bytes:
            raise ValueError("Probe transfer byte budget exceeded")
        self.used_bytes += amount
        self.requests += 1


class ObjectReader(io.RawIOBase):
    """Seekable, If-Match-pinned OSS range reader; no whole-object fallback."""
    def __init__(self, bucket, record: dict, budget: TransferBudget):
        self.bucket, self.record, self.budget, self.position = bucket, record, budget, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position + offset if whence == 1 else self.record["bytes"] + offset
        if whence not in (0, 1, 2) or not 0 <= value <= self.record["bytes"]:
            raise ValueError("Invalid source range seek")
        self.position = value
        return value

    def read(self, size=-1):
        length = self.record["bytes"] - self.position
        if size >= 0:
            length = min(length, size)
        if length == 0:
            return b""
        start, end = self.position, self.position + length - 1
        self.budget.reserve(length)
        result = self.bucket.get_object(self.record["key"], byte_range=(start, end), headers={"If-Match": self.record["etag"]})
        try:
            headers = {str(k).lower(): str(v) for k, v in result.headers.items()}
            if result.status != 206 or headers.get("content-range") != f"bytes {start}-{end}/{self.record['bytes']}":
                raise ValueError("Server did not honor the bounded source range")
            if headers.get("etag", "").strip('"').lower() != self.record["etag"].strip('"').lower():
                raise ValueError("Source object ETag changed")
            data = result.read(length + 1)
            if len(data) != length:
                raise ValueError("Incomplete source range")
        finally:
            result.resp.response.close()
        self.position += length
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def open_source_bucket(source_config: Path | None, bucket_name: str):
    """Consume existing OSS config without exposing any secret or signed URL."""
    import oss2

    values = {}
    if source_config is not None:
        if source_config.suffix == ".json":
            values = json.loads(source_config.read_text())
        else:
            parser = configparser.ConfigParser()
            parser.read(source_config)
            for section in parser.sections():
                values.update(parser[section])
    def option(*names, env=None):
        if env and os.environ.get(env):
            return os.environ[env]
        lowered = {key.lower().replace("_", ""): value for key, value in values.items()}
        return next((lowered[name.lower().replace("_", "")] for name in names if name.lower().replace("_", "") in lowered), None)
    endpoint = option("endpoint", env="OSS_ENDPOINT")
    key = option("accessKeyID", "access_key_id", env="OSS_ACCESS_KEY_ID") or os.environ.get("OSS_ACCESS_ID")
    secret = option("accessKeySecret", "access_key_secret", env="OSS_ACCESS_KEY_SECRET") or os.environ.get("OSS_ACCESS_KEY")
    token = option("securityToken", "security_token", env="OSS_SECURITY_TOKEN")
    if not endpoint or not key or not secret:
        raise ValueError("Existing OSS endpoint and read credentials are required")
    if values.get("bucket", bucket_name) != bucket_name:
        raise ValueError("Source bucket configuration differs from the accepted source")
    auth = oss2.StsAuth(key, secret, token) if token else oss2.Auth(key, secret)
    return oss2.Bucket(auth, endpoint, bucket_name, connect_timeout=60)


def read_selected_parquet(bucket, record, budget, episode_ids):
    import pyarrow.parquet as pq

    stream = ObjectReader(bucket, record, budget)
    # Explicit projection excludes unrelated columns; episode predicate is then
    # applied by Arrow without fetching any video or complete dataset object.
    columns = ["action", "observation.state", "episode_index", "frame_index", "index", "timestamp", "task_index"]
    return pq.read_table(stream, columns=columns, filters=[("episode_index", "in", sorted(episode_ids))],
                         pre_buffer=False, use_threads=False).to_pydict()


def decode_rgb(bucket, record, budget, timestamps: list[float], shape: list[int], tolerance=1e-4):
    """Match the accepted reader's preceding-keyframe PyAV/nearest-PTS rule."""
    import av
    import numpy as np

    first, last = min(timestamps), max(timestamps)
    loaded, actual = [], []
    with av.open(ObjectReader(bucket, record, budget), mode="r") as container:
        stream = container.streams.video[0]
        container.seek(int(first * av.time_base), backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current = float(frame.pts * stream.time_base)
            loaded.append(frame.to_ndarray(format="rgb24"))
            actual.append(current)
            if current >= last:
                break
    if not loaded:
        raise ValueError("No original RGB frames decoded")
    distances = np.abs(np.asarray(timestamps, dtype=np.float64)[:, None] - np.asarray(actual)[None, :])
    positions = distances.argmin(axis=1)
    if not np.all(distances[np.arange(len(positions)), positions] < tolerance):
        raise ValueError("Original RGB timestamp falls outside the accepted tolerance")
    images = []
    for position in positions:
        raw = loaded[position]
        if list(raw.shape) != shape or raw.dtype != np.uint8:
            raise ValueError("Decoded RGB shape/dtype differs from accepted metadata")
        images.append(np.ascontiguousarray(raw.transpose(2, 0, 1), dtype=np.float32) / np.float32(255))
    return images, [actual[position] for position in positions]


def make_raw_sample(spec: dict, columns: dict, images: dict) -> dict:
    import numpy as np

    episodes = np.asarray(columns["episode_index"])
    frames = np.asarray(columns["frame_index"])
    selection = np.flatnonzero((episodes == spec["episode_index"]) & (frames >= spec["frame_index"])
                              & (frames < spec["frame_index"] + 32))
    selection = selection[np.argsort(frames[selection])]
    if len(selection) != 32 or not np.array_equal(frames[selection], np.arange(spec["frame_index"], spec["frame_index"] + 32)):
        raise ValueError("Requested source action rows are incomplete or duplicated")
    indices = np.asarray(columns["index"])[selection]
    if not np.array_equal(indices, np.arange(spec["global_frame_index"], spec["global_frame_index"] + 32)):
        raise ValueError("Original global action row identity changed")
    if not np.all(np.asarray(columns["task_index"])[selection] == spec["task_index"]):
        raise ValueError("Original action rows belong to another task")
    first = selection[0]
    action = np.asarray([columns["action"][i] for i in selection], dtype=np.float32)
    state = np.asarray(columns["observation.state"][first], dtype=np.float32)
    if action.shape != (32, 23) or state.shape != (61,) or not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Original action/state shape or finiteness differs")
    return {"action": action, "action_is_pad": np.zeros(32, dtype=np.bool_), "observation.state": state,
            "episode_index": np.asarray(spec["episode_index"], dtype=np.int64),
            "frame_index": np.asarray(spec["frame_index"], dtype=np.int64),
            "index": np.asarray(spec["global_frame_index"], dtype=np.int64),
            "task_index": np.asarray(spec["task_index"], dtype=np.int64),
            "timestamp": np.asarray(columns["timestamp"][first], dtype=np.float64),
            "prompt": spec["prompt"], "subtask": spec["subtask"], **images}


def validate_cache(output: Path, plan: dict, manifest: dict) -> None:
    if manifest.get("contract_sha256") != plan["contract_sha256"] or manifest.get("kind") != KIND:
        raise ValueError("Cache resume contract differs")
    contract = dict(plan)
    contract.pop("contract_sha256")
    if hashlib.sha256(canonical(contract)).hexdigest() != plan["contract_sha256"]:
        raise ValueError("Cache plan content changed")
    for relative, expected in manifest["artifact_hashes"].items():
        if sha_file(output / safe_relative(relative)) != expected:
            raise ValueError("Frozen cache input changed")
    selected = {row["sample_id"]: row for row in plan["samples"]}
    seen = set()
    for record in manifest["records"]:
        sample_id = record["sample_id"]
        if sample_id not in selected or sample_id in seen:
            raise ValueError("Cache record was not selected uniquely")
        seen.add(sample_id)
        if any(record[key] != selected[sample_id][key] for key in ("episode_index", "frame_index", "task_index", "global_frame_index")):
            raise ValueError("Cache record source identity differs")
        path = output / safe_relative(record["npz_path"])
        if sha_file(path) != record["npz_sha256"] or raw_sample_hash(load_npz(path)) != record["raw_sample_sha256"]:
            raise ValueError("Cached raw sample changed")
    if manifest["complete"] != (len(seen) == len(selected)):
        raise ValueError("Cache completion marker differs from record coverage")


def prepare(output: Path, source_config: Path | None) -> dict:
    import numpy as np

    plan = json.loads((output / "cache_plan.json").read_text())
    manifest = json.loads((output / "cache_manifest.json").read_text())
    validate_cache(output, plan, manifest)
    if manifest["complete"]:
        return manifest
    descriptor = os.open(output / ".prepare.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    limits = plan["transfer_limits"]
    budget = TransferBudget(limits["maximum_bytes"], limits["maximum_seconds"], manifest["transfer_bytes"])
    prior_requests = manifest["transfer_requests"]
    try:
        # Empty directories need not survive a metadata-only archive transfer.
        # Recreate this output directory before any remote bytes are requested.
        (output / "samples").mkdir(exist_ok=True)
        bucket = open_source_bucket(source_config, plan["source_bucket"])
        objects = {row["relative_path"]: row for row in plan["objects"]}
        completed = {row["sample_id"] for row in manifest["records"]}
        pending = [row for row in plan["samples"] if row["sample_id"] not in completed]
        groups = {}
        for row in pending:
            groups.setdefault(row["data_relative_path"], []).append(row)
        for relative, specs in groups.items():
            columns = read_selected_parquet(bucket, objects[relative], budget, {row["episode_index"] for row in specs})
            episodes, frames = np.asarray(columns["episode_index"]), np.asarray(columns["frame_index"])
            images, video_proofs = {}, {}
            # Decode the four neighboring observations together, per episode
            # and camera. Do not decode between distant episodes in one file.
            for episode in sorted({row["episode_index"] for row in specs}):
                ep_specs = [row for row in specs if row["episode_index"] == episode]
                for camera in plan["cameras"]:
                    video = ep_specs[0]["videos"][camera]
                    requested = []
                    for row in ep_specs:
                        matches = np.flatnonzero((episodes == episode) & (frames == row["frame_index"]))
                        if len(matches) != 1:
                            raise ValueError("RGB anchor source row is not unique")
                        requested.append(float(columns["timestamp"][matches[0]]) + video["from_timestamp"])
                    decoded, actual = decode_rgb(bucket, objects[video["relative_path"]], budget, requested, video["shape"])
                    for row, image, query, got in zip(ep_specs, decoded, requested, actual, strict=True):
                        images.setdefault(row["sample_id"], {})[camera] = image
                        video_proofs.setdefault(row["sample_id"], {})[camera] = {
                            **objects[video["relative_path"]], "requested_timestamp": query, "decoded_timestamp": got,
                            "from_timestamp": video["from_timestamp"], "tolerance_seconds": 1e-4}
            for spec in specs:
                raw = make_raw_sample(spec, columns, images[spec["sample_id"]])
                path = output / "samples" / (spec["sample_id"] + ".npz")
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as target:
                    np.savez_compressed(target, **raw)
                temporary.replace(path)
                record = {key: spec[key] for key in ("sample_id", "episode_index", "frame_index", "task_index", "global_frame_index")}
                record.update(npz_path=str(path.relative_to(output)), npz_sha256=sha_file(path), raw_sample_sha256=raw_sample_hash(raw),
                              sources={"parquet": objects[relative], "videos": video_proofs[spec["sample_id"]]})
                if raw_sample_hash(load_npz(path)) != record["raw_sample_sha256"]:
                    raise ValueError("Raw sample changed in NPZ round trip")
                manifest["records"].append(record)
                manifest.update(transfer_bytes=budget.used_bytes, transfer_requests=prior_requests + budget.requests,
                                complete=len(manifest["records"]) == plan["sample_count"])
                manifest.pop("last_failure", None)
                atomic_json(output / "cache_manifest.json", manifest)
                print(json.dumps({"prepared_samples": len(manifest["records"]), "expected_samples": plan["sample_count"],
                                  "transfer_bytes": budget.used_bytes, "inference_executed": False}), flush=True)
        validate_cache(output, plan, manifest)
        return manifest
    except BaseException as error:
        # Transport exceptions may contain signed endpoints or credentials.
        # Persist the class and phase only, never exception text or traceback.
        manifest.update(transfer_bytes=budget.used_bytes, transfer_requests=prior_requests + budget.requests,
                        last_failure={"type": type(error).__name__, "phase": "prepare", "inference_executed": False})
        atomic_json(output / "cache_manifest.json", manifest)
        raise
    finally:
        (output / ".prepare.lock").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    plan = subs.add_parser("plan", help="Read local metadata only; no credentials or network")
    plan.add_argument("--manifest", type=Path, default=HERE / "probe_manifest.json")
    plan.add_argument("--metadata-archive", type=Path, default=DEFAULT_ARCHIVE)
    plan.add_argument("--metadata-sha256", default=DEFAULT_ARCHIVE_SHA256)
    plan.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--max-bytes", type=int, default=4 * 1024**3)
    plan.add_argument("--max-seconds", type=int, default=3600)
    run = subs.add_parser("prepare", help="Run or resume bounded selected-sample preparation")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--source-config", type=Path, default=Path.home() / ".ossutilconfig")
    verify = subs.add_parser("verify", help="Verify local cache hashes without credentials/network")
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            if args.max_bytes <= 0 or args.max_seconds <= 0:
                raise ValueError("Positive byte/time limits are required")
            result = make_plan(args)
            print(json.dumps({"plan_only": True, "samples": result["sample_count"], "episodes": len(result["selected_episodes"]),
                              "whole_selected_object_bytes": result["whole_selected_object_bytes"], "inference_executed": False}))
        elif args.command == "prepare":
            result = prepare(args.output, args.source_config)
            print(json.dumps({"complete": result["complete"], "prepared_samples": len(result["records"]), "inference_executed": False}))
        else:
            plan = json.loads((args.output / "cache_plan.json").read_text())
            manifest = json.loads((args.output / "cache_manifest.json").read_text())
            validate_cache(args.output, plan, manifest)
            print(json.dumps({"hashes_verified": True, "complete": manifest["complete"], "prepared_samples": len(manifest["records"])}))
    except BaseException as error:
        print(json.dumps({"error_type": type(error).__name__, "inference_executed": False}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
