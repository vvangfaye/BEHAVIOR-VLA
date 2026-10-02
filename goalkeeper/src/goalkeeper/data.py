"""Offline records and an immutable, content-addressed dataset contract."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import re
import shutil
import numpy as np
import torch
from .goals import compile_bddl

DATASET_FORMAT = "goalkeeper-dataset-v1"
READY_FORMAT = "goalkeeper-ready-v1"
_REAL_SOURCES = {"training_demo", "training_rollout"}
_ALL_SOURCES = _REAL_SOURCES | {"synthetic_test"}
_RECORD_FIELDS = ("visual", "proprio", "actions", "truth", "visibility", "future_truth")


def _record_member(index: int, field: str) -> str:
    return f"record_{index:07d}_{field}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_split(group_id: str, seed: str = "goalkeeper-v1") -> str:
    bucket = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) % 10
    return "calibration" if bucket == 0 else "dev" if bucket == 1 else "train"


def _safe_resolve(base: Path, value: str, *, must_exist: bool = True) -> Path:
    """Resolve a manifest path without permitting absolute or parent traversal."""
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"Dataset paths must be nonempty relative paths: {value!r}")
    base = base.resolve()
    result = (base / value).resolve(strict=must_exist)
    try:
        result.relative_to(base)
    except ValueError as error:
        raise ValueError(f"Dataset path escapes its root: {value}") from error
    if must_exist and not result.is_file():
        raise ValueError(f"Dataset path is not a file: {value}")
    return result


def _raw_manifest(path: Path) -> list[dict]:
    rows, ids, record_slots = [], set(), set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"id", "group_id", "source", "record", "bddl", "feature_id", "action_spec"}
        if not required <= row.keys():
            raise ValueError(f"Missing manifest fields: {required - row.keys()}")
        if not all(isinstance(row[k], str) and row[k] for k in required):
            raise ValueError("Manifest fields must be nonempty strings")
        if "record_index" in row and (type(row["record_index"]) is not int or row["record_index"] < 0):
            raise ValueError("record_index must be a nonnegative integer")
        if row["id"] in ids:
            raise ValueError("Duplicate record ID")
        ids.add(row["id"])
        if row["source"] not in _ALL_SOURCES:
            raise ValueError("Training inputs must not contain public_test or hidden_test instances")
        record_path = _safe_resolve(path.parent, row["record"])
        _safe_resolve(path.parent, row["bddl"])
        slot = (record_path, row.get("record_index"))
        if slot in record_slots:
            raise ValueError("Duplicate record archive slot")
        record_slots.add(slot)
        rows.append(row)
    if not rows:
        raise ValueError("Empty dataset")
    if len({r["feature_id"] for r in rows}) != 1 or len({r["action_spec"] for r in rows}) != 1:
        raise ValueError("Mixed feature encoders or robot action specifications")
    return rows


def load_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    rows = _raw_manifest(path)
    result = []
    for original in rows:
        row = dict(original)
        for key in ("record", "bddl"):
            row[key] = str(_safe_resolve(path.parent, row[key]))
        row["split"] = group_split(row["group_id"])
        result.append(row)
    return result


def _canonical_json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace immutable dataset file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_record_archive(path: str | Path, records: list[dict], queries: np.ndarray, atom_keys: np.ndarray) -> None:
    if not records:
        raise ValueError("Record archive cannot be empty")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace record archive: {path}")
    payload = {
        "record_count": np.asarray(len(records), dtype=np.int64),
        "queries": np.asarray(queries),
        "atom_keys": np.asarray(atom_keys),
    }
    for index, record in enumerate(records):
        missing = set(_RECORD_FIELDS) - set(record)
        if missing:
            raise ValueError(f"Record {index} is missing fields: {missing}")
        for field in _RECORD_FIELDS:
            payload[_record_member(index, field)] = np.asarray(record[field])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_dataset_contract(manifest: str | Path) -> dict:
    """Build the deterministic contract for a completed real-training manifest."""
    path = Path(manifest).resolve()
    rows = load_manifest(path)
    if any(row["source"] not in _REAL_SOURCES for row in rows):
        raise ValueError("Formal datasets cannot contain synthetic rows")
    split_groups = {split: sorted({r["group_id"] for r in rows if r["split"] == split})
                    for split in ("train", "dev", "calibration")}
    if any(not groups for groups in split_groups.values()):
        raise ValueError("Formal dataset requires train, dev, and calibration group coverage")
    root = path.parent
    raw_rows = _raw_manifest(path)
    files = {path.name: sha256_file(path)}
    asset_paths = {
        Path(raw[key]).as_posix()
        for raw in raw_rows
        for key in ("record", "bddl")
    }
    for relative in sorted(asset_paths):
        files[relative] = sha256_file(_safe_resolve(root, relative))
    schemas = []
    label_counts = {"current_total": 0, "current_positive": 0, "future_total": 0,
                    "future_positive": 0, "visibility_total": 0, "visibility_known": 0,
                    "visibility_positive": 0}
    for row in rows:
        record = load_record(row)
        schemas.append({
            "history": record["visual"].shape[0], "cameras": record["visual"].shape[1],
            "patches": record["visual"].shape[2], "visual_dim": record["visual"].shape[3],
            "proprio_dim": record["proprio"].shape[1], "text_dim": record["queries"].shape[1],
            "horizon": record["actions"].shape[1], "action_dim": record["actions"].shape[2],
        })
        label_counts["current_total"] += record["truth"].numel()
        label_counts["current_positive"] += int(record["truth"].sum())
        label_counts["future_total"] += record["future_truth"].numel()
        label_counts["future_positive"] += int(record["future_truth"].sum())
        known = record["visibility"] >= 0
        label_counts["visibility_total"] += record["visibility"].numel()
        label_counts["visibility_known"] += int(known.sum())
        label_counts["visibility_positive"] += int((record["visibility"] > 0).sum())
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("Formal dataset contains incompatible tensor schemas")
    return {
        "format": DATASET_FORMAT,
        "manifest": path.name,
        "manifest_sha256": files[path.name],
        "files": dict(sorted(files.items())),
        "row_count": len(rows),
        "record_file_count": len({row["record"] for row in rows}),
        "group_count": len({r["group_id"] for r in rows}),
        "splits": {name: {"groups": groups, "rows": sum(r["split"] == name for r in rows)}
                   for name, groups in split_groups.items()},
        "sources": sorted({r["source"] for r in rows}),
        "feature_id": rows[0]["feature_id"],
        "action_spec": rows[0]["action_spec"],
        "tensor_schema": schemas[0],
        "label_counts": label_counts,
    }


def write_dataset_contract(manifest: str | Path) -> tuple[str, str]:
    """Seal a dataset. Existing nonidentical contract/READY files are never overwritten."""
    path = Path(manifest).resolve()
    contract = build_dataset_contract(path)
    contract_path = path.parent / "dataset_contract.json"
    payload = _canonical_json(contract)
    _write_immutable(contract_path, payload)
    contract_hash = hashlib.sha256(payload).hexdigest()
    ready = {"format": READY_FORMAT, "dataset_contract_sha256": contract_hash}
    _write_immutable(path.parent / "_READY", _canonical_json(ready))
    check_dataset(contract_path)
    return str(contract_path), contract_hash


def check_dataset(path: str | Path) -> dict:
    """Verify the READY marker and every path/hash in a formal dataset contract."""
    path = Path(path).resolve()
    contract_path = path / "dataset_contract.json" if path.is_dir() else path
    if contract_path.name != "dataset_contract.json" or not contract_path.is_file():
        raise ValueError("Expected a dataset directory or dataset_contract.json")
    root = contract_path.parent
    payload = contract_path.read_bytes()
    contract = json.loads(payload)
    if contract.get("format") != DATASET_FORMAT:
        raise ValueError("Unknown dataset contract format")
    ready_path = root / "_READY"
    if not ready_path.is_file():
        raise ValueError("Dataset is not sealed: missing _READY")
    ready = json.loads(ready_path.read_text())
    expected_hash = hashlib.sha256(payload).hexdigest()
    if ready != {"format": READY_FORMAT, "dataset_contract_sha256": expected_hash}:
        raise ValueError("Dataset READY marker does not match its contract")
    files = contract.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Dataset contract has no file inventory")
    for relative, expected in files.items():
        actual = sha256_file(_safe_resolve(root, relative))
        if actual != expected:
            raise ValueError(f"Dataset file hash mismatch: {relative}")
    manifest = _safe_resolve(root, contract.get("manifest", ""))
    rebuilt = build_dataset_contract(manifest)
    if rebuilt != contract:
        raise ValueError("Dataset contract metadata does not match its manifest")
    return {**contract, "dataset_contract_sha256": expected_hash,
            "contract_path": str(contract_path), "manifest_path": str(manifest)}


def dataset_identity(manifest: str | Path, contract: str | Path | None = None) -> tuple[str, dict | None]:
    """Return the formal contract hash; retain legacy synthetic smoke-test support."""
    path = Path(manifest).resolve()
    rows = load_manifest(path)
    sources = {row["source"] for row in rows}
    if sources == {"synthetic_test"}:
        if contract is not None:
            raise ValueError("Synthetic smoke data cannot claim a formal dataset contract")
        return sha256_file(path), None
    if not sources <= _REAL_SOURCES:
        raise ValueError("Synthetic and real rows cannot be mixed")
    checked = check_dataset(contract if contract is not None else path.parent)
    if Path(checked["manifest_path"]) != path:
        raise ValueError("Dataset contract points to a different manifest")
    return checked["dataset_contract_sha256"], checked


def merge_manifests(manifests: list[str | Path], output: str | Path, *, seal: bool = True) -> str:
    """Copy manifest assets into one self-contained, relative-path dataset."""
    if not manifests:
        raise ValueError("At least one input manifest is required")
    output = Path(output).resolve()
    manifest_path = output if output.suffix == ".jsonl" else output / "manifest.jsonl"
    root = manifest_path.parent
    if (root / "dataset_contract.json").exists() or (root / "_READY").exists():
        raise FileExistsError("Refusing to modify a sealed dataset")
    root.mkdir(parents=True, exist_ok=True)
    (root / "records").mkdir(exist_ok=True)
    (root / "bddl").mkdir(exist_ok=True)
    merged, ids = [], set()
    asset_cache = {}
    for source_manifest in manifests:
        source_path = Path(source_manifest).resolve()
        raw_rows = _raw_manifest(source_path)
        for raw in raw_rows:
            if raw["source"] not in _REAL_SOURCES:
                raise ValueError("Formal manifest merge rejects synthetic rows")
            if raw["id"] in ids or not re.fullmatch(r"[A-Za-z0-9._-]+", raw["id"]):
                raise ValueError(f"Duplicate or unsafe record ID: {raw['id']}")
            ids.add(raw["id"])
            record_source = _safe_resolve(source_path.parent, raw["record"])
            bddl_source = _safe_resolve(source_path.parent, raw["bddl"])
            if record_source not in asset_cache:
                digest = sha256_file(record_source)
                asset_cache[record_source] = (digest, f"records/{digest}.npz")
            if bddl_source not in asset_cache:
                digest = sha256_file(bddl_source)
                asset_cache[bddl_source] = (digest, f"bddl/{digest}.bddl")
            record_relative = asset_cache[record_source][1]
            bddl_relative = asset_cache[bddl_source][1]
            for source, relative in ((record_source, record_relative), (bddl_source, bddl_relative)):
                destination = root / relative
                source_hash = asset_cache[source][0]
                if destination.exists() and sha256_file(destination) != source_hash:
                    raise FileExistsError(f"Conflicting immutable asset: {relative}")
                if not destination.exists():
                    shutil.copy2(source, destination)
            row = {**{k: raw[k] for k in ("id", "group_id", "source", "feature_id", "action_spec")},
                   "record": record_relative, "bddl": bddl_relative}
            if "record_index" in raw:
                row["record_index"] = raw["record_index"]
            merged.append(row)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted(merged, key=lambda r: r["id"])).encode()
    _write_immutable(manifest_path, payload)
    load_manifest(manifest_path)
    if seal:
        write_dataset_contract(manifest_path)
    return str(manifest_path)


@lru_cache(maxsize=256)
def _load_goal(path: str):
    return compile_bddl(Path(path).read_text())


def load_record(row: dict) -> dict:
    goal = _load_goal(str(Path(row["bddl"]).resolve()))
    if not goal.atoms or not goal.clauses or any(not c for c in goal.clauses):
        raise ValueError("Training records require a nontrivial supported goal")
    with np.load(row["record"], allow_pickle=False) as f:
        index = row.get("record_index")
        if index is None:
            names = {field: field for field in _RECORD_FIELDS}
        else:
            if type(index) is not int or index < 0:
                raise ValueError("record_index must be a nonnegative integer")
            if "record_count" not in f.files or np.asarray(f["record_count"]).shape != ():
                raise ValueError("Indexed record archive is missing scalar record_count")
            if index >= int(f["record_count"]):
                raise ValueError("record_index exceeds record archive size")
            names = {field: _record_member(index, field) for field in _RECORD_FIELDS}
        required = {"queries", "atom_keys", *names.values()}
        if not required <= set(f.files):
            raise ValueError(f"Missing NPZ fields: {required - set(f.files)}")
        arrays = {field: f[name].copy() for field, name in names.items()}
        arrays["queries"] = f["queries"].copy()
        arrays["atom_keys"] = f["atom_keys"].copy()
    keys = arrays.pop("atom_keys").tolist()
    if keys != [a.key for a in goal.atoms]:
        raise ValueError("Labels/features must use exactly the compiler's sorted grounded atoms")
    v, s, q, a = (arrays[k] for k in ["visual", "proprio", "queries", "actions"])
    if v.ndim != 4 or s.ndim != 2 or q.ndim != 2 or a.ndim != 3:
        raise ValueError("Invalid record dimensions")
    t, n, k = len(v), len(goal.atoms), len(a)
    if t < 1 or k < 1 or s.shape[0] != t or q.shape[0] != n:
        raise ValueError("Empty or mismatched feature dimensions")
    for field, shape in [("truth", (t, n)), ("visibility", (t, n)), ("future_truth", (k, n))]:
        if arrays[field].shape != shape:
            raise ValueError(f"Incorrect shape: {field}")
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Nonfinite training record")
    for field in ["truth", "future_truth"]:
        if not np.isin(arrays[field], [0, 1]).all():
            raise ValueError("Goal truth must be binary")
    if not np.isin(arrays["visibility"], [-1, 0, 1]).all():
        raise ValueError("Visibility labels must be -1 (unknown), 0 or 1")
    atom_index = {a: i for i, a in enumerate(goal.atoms)}
    clauses = np.zeros((len(goal.clauses), n), np.float32)
    for j, clause in enumerate(goal.clauses):
        for literal in clause:
            clauses[j, atom_index[literal.atom]] = 1 if literal.positive else -1
    arrays["clauses"] = clauses
    return {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in arrays.items()}


def collate(records: list[dict]) -> dict:
    """Pad only N, K and clause axes; history and action horizon must match."""
    n = max(r["queries"].shape[0] for r in records)
    k = max(r["actions"].shape[0] for r in records)
    c = max(r["clauses"].shape[0] for r in records)
    batched = {}
    for key in records[0]:
        values = []
        for r in records:
            x = r[key]
            shape = list(x.shape)
            if key == "queries": shape[0] = n
            elif key in ["truth", "visibility"]: shape[1] = n
            elif key == "actions": shape[0] = k
            elif key == "future_truth": shape = [k, n]
            elif key == "clauses": shape = [c, n]
            y = torch.full(shape, -1. if key == "visibility" else 0.)
            y[tuple(slice(0, s) for s in x.shape)] = x
            values.append(y)
        batched[key] = torch.stack(values)
    batched["atom_mask"] = torch.tensor([[i < r["queries"].shape[0] for i in range(n)] for r in records])
    batched["candidate_mask"] = torch.tensor([[i < r["actions"].shape[0] for i in range(k)] for r in records])
    return batched
