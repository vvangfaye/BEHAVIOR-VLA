#!/usr/bin/env python3
"""Run eval_openloop's unchanged metric path on a verified raw-frame cache.

Example:
  python eval_cached_probe.py --cache-manifest cache/cache_manifest.json \
      --pi05-ckpt /path/to/checkpoint --out results/baseline.json

Only selected NPZ samples are read. The cache supplies raw LeRobot-shaped
observations; the normal model input transforms and policy inference still run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


_EVAL_PATH = Path(__file__).resolve().parents[2] / "BEHAVIOR-VLAs" / "scripts" / "eval_openloop.py"
_SPEC = importlib.util.spec_from_file_location("b1k_cached_eval_openloop", _EVAL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
eval_openloop = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = eval_openloop
_SPEC.loader.exec_module(eval_openloop)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VerifiedRawCache:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.expanduser().resolve()
        self.root = self.manifest_path.parent
        raw = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get("schema_version") != 1 or self.manifest.get("kind") != "raw_boundary_probe_cache":
            raise ValueError("Expected schema 1 raw_boundary_probe_cache")
        if self.manifest.get("complete") is not True:
            raise ValueError("Refusing an incomplete raw-frame cache")
        self.paths = {field: self._path(self.manifest.get(field)) for field in
                      ("dataset_root", "episodes_file", "subtask_prompts", "sample_keys_file")}
        if not self.paths["dataset_root"].is_dir():
            raise ValueError("Cache dataset metadata root is missing")
        artifact_hashes = self.manifest.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict):
            raise ValueError("Cache must bind its metadata and sample selection with artifact_hashes")
        checked = set()
        for relative, expected in artifact_hashes.items():
            path = self._path(relative)
            self._check_sha(path, expected)
            checked.add(path)
        required = {self.paths[name] for name in ("episodes_file", "subtask_prompts", "sample_keys_file")}
        required.add(self.paths["dataset_root"] / "meta" / "tasks.jsonl")
        episode_files = list((self.paths["dataset_root"] / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not episode_files:
            raise ValueError("Cache has no episode metadata parquet")
        required.update(path.resolve() for path in episode_files)
        if not required <= checked:
            raise ValueError(f"Unhashed required cache metadata: {sorted(map(str, required - checked))}")

        self.probe = json.loads(self.paths["sample_keys_file"].read_text())
        if self.probe.get("kind") != "matched_training_boundary_probe":
            raise ValueError("Cache requires an explicit matched boundary probe manifest")
        if self.probe.get("repo_id") != self.manifest.get("repo_id"):
            raise ValueError("Cache and probe repo_id disagree")
        probe_rows = self.probe.get("sample_keys")
        records = self.manifest.get("records")
        if not isinstance(probe_rows, list) or not probe_rows or not isinstance(records, list) or not records:
            raise ValueError("Cache/probe must have nonempty sample records")
        self.records = {}
        for record in records:
            key = self._key(record)
            if key in self.records:
                raise ValueError(f"Duplicate cached episode/frame: {key}")
            for field in ("sample_id", "npz_sha256", "raw_sample_sha256"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(f"Cache record is missing {field}")
            self._path(record.get("npz_path"))
            self.records[key] = record
        probe_keys = [self._key(row) for row in probe_rows]
        if len(set(probe_keys)) != len(probe_keys) or set(probe_keys) != self.records.keys():
            raise ValueError("Raw cache records must match the exact probe sample set")
        for row in probe_rows:
            if self.records[self._key(row)]["sample_id"] != row.get("sample_id"):
                raise ValueError("Cache sample_id differs from the frozen probe")
        self.receipts = {}

    def _path(self, value):
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError("Cache paths must be nonempty paths relative to the cache manifest")
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Cache path escapes its bundle")
        return path

    @staticmethod
    def _key(row):
        values = (row.get("episode_index"), row.get("frame_index"))
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Cache/probe episode and frame IDs must be nonnegative integers")
        return values

    @staticmethod
    def _check_sha(path, expected):
        if not isinstance(expected, str) or len(expected) != 64 or file_sha256(path) != expected:
            raise ValueError(f"Cache artifact SHA256 mismatch: {path}")

    def __call__(self, sample):
        key = (sample.episode_index, sample.frame_index)
        if key not in self.records:
            raise ValueError(f"Requested frame was not explicitly cached: {key}")
        record = self.records[key]
        path = self._path(record["npz_path"])
        self._check_sha(path, record["npz_sha256"])
        with np.load(path, allow_pickle=False) as arrays:
            raw = {name: (arrays[name].item() if arrays[name].ndim == 0 and arrays[name].dtype.kind in "US"
                          else arrays[name].copy()) for name in arrays.files}
        digest = eval_openloop.input_content_sha256(raw)
        if digest != record["raw_sample_sha256"]:
            raise ValueError(f"Raw sample content SHA256 mismatch for episode/frame={key}")
        expected = {"episode_index": sample.episode_index, "frame_index": sample.frame_index,
                    "index": sample.global_frame_index, "task_index": sample.task_index}
        for field, value in expected.items():
            array = np.asarray(raw.get(field))
            if array.size != 1 or not np.issubdtype(array.dtype, np.integer) or int(array.item()) != value:
                raise ValueError(f"Raw cache {field} differs from the selected dataset row: {key}")
        for field, value in (("task_index", sample.task_index), ("global_frame_index", sample.global_frame_index)):
            if record.get(field) != value:
                raise ValueError(f"Cache record {field} differs from the selected dataset row: {key}")
        if raw.get("subtask") != sample.subtask:
            raise ValueError(f"Raw cache subtask differs from current annotations: {key}")
        self.receipts[key] = {"sample_id": record["sample_id"], "episode_index": key[0], "frame_index": key[1],
                              "npz_sha256": record["npz_sha256"], "raw_sample_sha256": digest}
        return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    options, remaining = parser.parse_known_args(argv)
    cache = VerifiedRawCache(options.cache_manifest)
    controlled = ("--dataset", "--episodes-file", "--subtask-prompts", "--sample-keys-file", "--repo-id",
                  "--horizon", "--action-noise-seed", "--subtask-conditioning", "--split-label")
    for argument in remaining:
        if argument.split("=", 1)[0] in controlled:
            parser.error(f"{argument.split('=', 1)[0]} is fixed by the verified cache manifest")
    probe = cache.probe
    fixed = ["--dataset", str(cache.paths["dataset_root"]),
             "--episodes-file", str(cache.paths["episodes_file"]),
             "--subtask-prompts", str(cache.paths["subtask_prompts"]),
             "--sample-keys-file", str(cache.paths["sample_keys_file"]),
             "--repo-id", cache.manifest["repo_id"], "--subtask-conditioning", "provided",
             "--split-label", "in_sample", "--horizon", str(probe["model_horizon_H"]),
             "--action-noise-seed", str(probe["pairing_contract"]["action_noise_seed"])]
    build_args = []

    def build_policy(args, episode_ids):
        if args.out.exists():
            raise ValueError(f"Refusing to overwrite an existing result: {args.out}")
        build_args.append(args)
        return eval_openloop.build_pi05(args, episode_ids, sample_provider=cache)

    eval_openloop.main([*remaining, *fixed], build_policy=build_policy)
    if len(build_args) != 1:
        raise RuntimeError("Expected one model construction through the raw-cache provider")
    output = build_args[0].out
    result = json.loads(output.read_text())
    result_keys = {(row["episode_index"], row["frame_index"]) for row in result["sample_keys"]}
    if result_keys != cache.receipts.keys():
        raise RuntimeError("Result sample identities differ from raw-cache read receipts")
    result["raw_cache"] = {"manifest_path": str(cache.manifest_path),
                           "manifest_sha256": cache.manifest_sha256,
                           "source_samples_available": len(cache.records), "samples_read": len(cache.receipts),
                           "unselected_raw_frames_read": False,
                           "read_receipts": list(cache.receipts.values()),
                           "scope": "Offline action error with the normal policy transforms; not closed-loop success."}
    temporary = output.with_suffix(output.suffix + ".cache-audit.tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
