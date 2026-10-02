#!/usr/bin/env python3
"""Nebula entry point and integrity checks for GoalKeeper H20 training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import List, Optional, Tuple

_HASH_RE = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_tree_sha256(root: Path) -> Tuple[str, int, int]:
    """Hash immutable tree contents as sorted ``sha256  relative/path`` records.

    ``_READY`` is excluded because it may contain the resulting tree digest.
    Symlinks and special files are rejected so staging cannot change what a path
    resolves to or silently omit content.
    """
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Feature root is not a directory: {root}")
    records = []  # type: List[Tuple[str, str, int]]
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise ValueError(f"Feature tree contains an unsupported path: {relative!r}")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Feature tree contains a symlink: {relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"Feature tree contains a special file: {relative}")
        if relative != "_READY":
            records.append((relative, file_sha256(path), path.stat().st_size))
    digest = hashlib.sha256()
    for relative, item_hash, _ in records:
        digest.update(f"{item_hash}  {relative}\n".encode("utf-8"))
    return digest.hexdigest(), len(records), sum(size for _, _, size in records)


def _required_regular_file(root: Path, name: str, *, allow_empty: bool = False) -> Path:
    path = root / name
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"Required feature file is missing: {name}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"Required feature path is not a regular file: {name}")
    if not allow_empty and path.stat().st_size == 0:
        raise ValueError(f"Required feature file is empty: {name}")
    return path


def _check_expected_hash(name: str, value: str) -> None:
    if not _HASH_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def validate_feature_root(
    root: Path,
    expected_tree: str,
    expected_manifest: str,
    expected_contract: str,
) -> dict:
    for name, value in (
        ("expected tree hash", expected_tree),
        ("expected manifest hash", expected_manifest),
        ("expected contract hash", expected_contract),
    ):
        _check_expected_hash(name, value)

    root = root.resolve(strict=True)
    manifest = _required_regular_file(root, "manifest.jsonl")
    contract_path = _required_regular_file(root, "dataset_contract.json")
    ready_path = _required_regular_file(root, "_READY", allow_empty=True)

    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset_contract.json must be valid UTF-8 JSON") from error
    if not isinstance(contract, dict):
        raise ValueError("dataset_contract.json must contain a JSON object")
    if contract.get("format") != "goalkeeper-dataset-v1" or contract.get("manifest") != "manifest.jsonl":
        raise ValueError("dataset_contract.json has an unsupported format or manifest")
    files = contract.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("dataset_contract.json has no file inventory")

    rows = 0
    sources = set()
    feature_ids = set()
    action_specs = set()
    record_slots = set()
    required_fields = {"id", "group_id", "source", "record", "bddl", "feature_id", "action_spec"}
    try:
        with manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"manifest line {line_number} is not a JSON object")
                missing = required_fields - set(row)
                if missing:
                    raise ValueError(f"manifest line {line_number} is missing fields: {sorted(missing)}")
                if row["source"] not in {"training_demo", "training_rollout"}:
                    raise ValueError("Formal GoalKeeper features require real training sources")
                index = row.get("record_index")
                if index is not None and (type(index) is not int or index < 0):
                    raise ValueError(f"manifest line {line_number} has an invalid record_index")
                slot = (Path(row["record"]).as_posix(), index)
                if slot in record_slots:
                    raise ValueError(f"manifest line {line_number} duplicates a record archive slot")
                record_slots.add(slot)
                sources.add(row["source"])
                feature_ids.add(row["feature_id"])
                action_specs.add(row["action_spec"])
                for field in ("record", "bddl"):
                    if not isinstance(row[field], str) or not row[field]:
                        raise ValueError(f"manifest line {line_number} has an invalid {field} path")
                    relative = Path(row[field])
                    if relative.is_absolute():
                        raise ValueError(f"manifest line {line_number} has an absolute {field} path")
                    candidate = (root / relative).resolve(strict=True)
                    try:
                        candidate.relative_to(root)
                    except ValueError as error:
                        raise ValueError(f"manifest line {line_number} {field} escapes the feature root") from error
                    if not candidate.is_file():
                        raise ValueError(f"manifest line {line_number} {field} is not a file")
                rows += 1
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest.jsonl must contain valid UTF-8 JSON lines") from error
    if rows == 0:
        raise ValueError("manifest.jsonl is empty")
    if len(feature_ids) != 1 or len(action_specs) != 1:
        raise ValueError("Manifest mixes feature encoders or action specifications")

    actual_manifest = file_sha256(manifest)
    actual_contract = file_sha256(contract_path)
    actual_tree, file_count, byte_count = feature_tree_sha256(root)
    actual = {
        "tree_sha256": actual_tree,
        "manifest_sha256": actual_manifest,
        "contract_sha256": actual_contract,
    }
    expected = {
        "tree_sha256": expected_tree,
        "manifest_sha256": expected_manifest,
        "contract_sha256": expected_contract,
    }
    for key in expected:
        if actual[key] != expected[key]:
            raise ValueError(f"Feature {key} mismatch: {actual[key]} != {expected[key]}")

    if contract.get("manifest_sha256") != actual_manifest:
        raise ValueError("Dataset contract manifest hash does not match manifest.jsonl")
    if contract.get("row_count") != rows:
        raise ValueError("Dataset contract row count does not match manifest.jsonl")
    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Dataset contract contains an unsafe path: {relative!r}")
        if not isinstance(expected_hash, str) or not _HASH_RE.fullmatch(expected_hash):
            raise ValueError(f"Dataset contract contains an invalid hash for {relative!r}")
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Dataset contract path escapes the feature root: {relative}") from error
        if not candidate.is_file() or file_sha256(candidate) != expected_hash:
            raise ValueError(f"Dataset contract file hash mismatch: {relative}")
    if files.get("manifest.jsonl") != actual_manifest:
        raise ValueError("Dataset contract inventory does not bind manifest.jsonl")

    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("_READY must be valid UTF-8 JSON") from error
    expected_ready = {
        "format": "goalkeeper-ready-v1",
        "dataset_contract_sha256": actual_contract,
    }
    if ready != expected_ready:
        raise ValueError("_READY does not bind the exact dataset contract")

    return {
        "format": "goalkeeper-feature-validation-v1",
        "root": str(root),
        **actual,
        "files_excluding_ready": file_count,
        "bytes_excluding_ready": byte_count,
        "manifest_rows": rows,
        "ready_present": True,
    }


def runtime_contract() -> dict:
    if sys.version_info < (3, 11):
        raise RuntimeError(f"GoalKeeper requires Python >=3.11; found {sys.version.split()[0]}")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed") from error
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if match is None or tuple(map(int, match.groups())) < (2, 7):
        raise RuntimeError(f"GoalKeeper requires Torch >=2.7; found {torch.__version__}")
    count = torch.cuda.device_count()
    if count != 8:
        raise RuntimeError(f"GoalKeeper H20 training requires exactly 8 visible GPUs; found {count}")
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    if any("H20" not in name.upper() for name in names):
        raise RuntimeError(f"GoalKeeper H20 training requires eight H20 GPUs; found {names}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and len([part for part in visible.split(",") if part.strip()]) != 8:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly eight devices")
    return {
        "format": "goalkeeper-h20-runtime-v1",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": count,
        "gpu_names": names,
        "global_batch_size": 64,
        "per_rank_batch_size": 8,
        "world_size": 8,
    }


def verify_artifacts(output: Path, expected_manifest: str, expected_contract: str) -> dict:
    import torch

    checkpoint = output / "best.pt"
    calibration = output / "calibration.json"
    history_path = output / "training.json"
    for path in (checkpoint, calibration, history_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required training artifact is missing or empty: {path.name}")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if artifact.get("format") != "goalkeeper-v1":
        raise ValueError("best.pt has an unexpected format")
    if artifact.get("manifest_sha256") != expected_manifest:
        raise ValueError("best.pt was not trained from the expected manifest")
    if artifact.get("dataset_contract_sha256") != expected_contract:
        raise ValueError("best.pt was not trained from the expected dataset contract")
    if artifact.get("synthetic") is not False:
        raise ValueError("Nebula training refuses a synthetic GoalKeeper checkpoint")
    calibration_data = json.loads(calibration.read_text(encoding="utf-8"))
    if calibration_data.get("format") != "goalkeeper-calibration-v1":
        raise ValueError("calibration.json has an unexpected format")
    checkpoint_hash = file_sha256(checkpoint)
    if calibration_data.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("calibration.json does not describe best.pt")
    if calibration_data.get("dataset_contract_sha256") != expected_contract:
        raise ValueError("calibration.json was not produced from the expected dataset contract")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError("training.json contains no completed epochs")
    return {
        "format": "goalkeeper-h20-artifacts-v1",
        "best_pt_sha256": checkpoint_hash,
        "calibration_sha256": file_sha256(calibration),
        "training_sha256": file_sha256(history_path),
        "epochs_completed": len(history),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    feature = commands.add_parser("validate-feature")
    feature.add_argument("root", type=Path)
    feature.add_argument("--expected-tree", required=True)
    feature.add_argument("--expected-manifest", required=True)
    feature.add_argument("--expected-contract", required=True)
    runtime = commands.add_parser("validate-runtime")
    runtime.add_argument("--output", type=Path)
    artifacts = commands.add_parser("verify-artifacts")
    artifacts.add_argument("output", type=Path)
    artifacts.add_argument("--expected-manifest", required=True)
    artifacts.add_argument("--expected-contract", required=True)
    artifacts.add_argument("--result", type=Path)
    marker = commands.add_parser("write-json")
    marker.add_argument("path", type=Path)
    marker.add_argument("payload")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command is None:
        script = Path(__file__).with_name("train_goalkeeper_h20.sh")
        os.execv("/bin/bash", ["bash", str(script)])
    try:
        if args.command == "validate-feature":
            result = validate_feature_root(
                args.root, args.expected_tree, args.expected_manifest, args.expected_contract
            )
        elif args.command == "validate-runtime":
            result = runtime_contract()
            if args.output:
                _write_json(args.output, result)
        elif args.command == "verify-artifacts":
            result = verify_artifacts(args.output, args.expected_manifest, args.expected_contract)
            if args.result:
                _write_json(args.result, result)
        elif args.command == "write-json":
            result = json.loads(args.payload)
            if not isinstance(result, dict):
                raise ValueError("Marker payload must be a JSON object")
            _write_json(args.path, result)
        else:  # pragma: no cover - argparse controls this
            raise AssertionError(args.command)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
