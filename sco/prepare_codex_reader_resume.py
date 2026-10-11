#!/usr/bin/env python3
"""Prepare an offline reader retry from a complete, stopped staging failure.

Use the existing b1k-eval Python. This does not download, submit or run a job,
decode video, validate raw success, or touch authentication. The caller supplies
the freshly observed terminal job status and its separately saved submission.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import prepare_codex_original_data as staging


def load(path):
    return json.loads(Path(path).read_text())


def _sha(path, deadline):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            if time.monotonic() >= deadline:
                raise TimeoutError("Offline reader-resume preparation exceeded its deadline")
            digest.update(block)
    return digest.hexdigest()


def _relative(value, allowed):
    if not isinstance(value, str) or "\r" in value:
        raise ValueError("Invalid frozen file path")
    path = staging.relative(value)
    if path.parts[0] not in allowed or path.as_posix() != value:
        raise ValueError("Frozen file has a noncanonical or out-of-scope path")
    return path


def _checked_file(root, row, deadline, *, allowed):
    path = root / _relative(row["path"], allowed)
    if (path.resolve() != path or not path.is_file()
            or type(row.get("bytes")) is not int or row["bytes"] < 1
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256")))
            or path.stat().st_size != row["bytes"] or _sha(path, deadline) != row["sha256"]):
        raise ValueError("Frozen file failed size/hash/path verification")
    return path


def _inventory(rows, label):
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Missing {label} inventory")
    names = [row.get("path") for row in rows]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError(f"Invalid or duplicate {label} inventory")
    return {row["path"]: row for row in rows}


def _signed_inventory(rows):
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Missing frozen signed-object identity")
    result = {}
    for row in rows:
        name = row.get("relative_path")
        _relative(name, {"data", "videos"})
        if (name in result or type(row.get("bytes")) is not int or row["bytes"] <= 0
                or not isinstance(row.get("etag"), str) or not row["etag"].strip('"')):
            raise ValueError("Invalid frozen signed-object identity")
        if "sha256" in row and not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])):
            raise ValueError("Invalid frozen signed-object SHA256")
        result[name] = row
    return result


def _separate(roots):
    for i, first in enumerate(roots):
        for second in roots[i+1:]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("Old evidence, original metadata, reuse and next output must be separate")


def prepare_resume(failed_root, payload, *, previous_job, previous_submission,
                   reuse_root, payload_path, max_seconds=3600):
    """Verify every source byte, then prepare a fresh private staging payload."""
    if type(max_seconds) is not int or not 1 <= max_seconds <= 3600:
        raise ValueError("Preparation needs a deadline of at most one hour")
    started = time.monotonic()
    deadline = started + max_seconds
    staging.validate_payload(payload)
    if (previous_job.get("state") not in {"FAILED", "SUCCEEDED", "SUSPENDED"}
            or not isinstance(previous_job.get("name"), str) or not previous_job["name"]
            or not re.fullmatch(r"[0-9a-f]{32}", str(previous_job.get("uid")))
            or any(previous_submission.get(key) != previous_job[key] for key in ("name", "uid"))):
        raise ValueError("Exact prior terminal job name/UID must match the saved submission")
    root = Path(failed_root).resolve(strict=True)
    state_path, old_payload_path = root / "status.json", root / "payload.json"
    status_hash, payload_hash = _sha(state_path, deadline), _sha(old_payload_path, deadline)
    state, old = load(state_path), load(old_payload_path)
    if ((root / "_RUNNING").exists() or not (root / "_FAILED").is_file()
            or (root / "_ORIGINAL_DATA_READY.json").exists()
            or state.get("kind") != staging.KIND or state.get("state") != "failed"
            or state.get("failure_phase") != "real_reader_and_demonstrations"
            or Path(state.get("root", "")).resolve() != root
            or Path(old.get("output_root", "")).resolve() != root):
        raise ValueError("Resume requires a stopped complete-transfer reader failure")
    if (state.get("source") != staging.SOURCE or old.get("source", staging.SOURCE) != staging.SOURCE
            or state.get("selected_episodes") != payload["episodes"]
            or old.get("episodes") != payload["episodes"]
            or old.get("demonstrations") != payload["demonstrations"]):
        raise ValueError("Reader retry changed original episode/task/raw-proof scope")
    for name in ("segments", "retained_episodes"):
        key = name + "_sha256"
        if (not old.get(key) or payload.get(key) != old[key]
                or _sha(payload[name + "_path"], deadline) != old[key]):
            raise ValueError("Original labels or retained episodes changed")
    if not old.get("metadata_source") or not Path(old["metadata_source"]).is_absolute():
        raise ValueError("Original bundled metadata source is missing")
    metadata_root = Path(old["metadata_source"]).resolve(strict=True)
    if payload.get("metadata_source") is not None and Path(payload["metadata_source"]).resolve() != metadata_root:
        raise ValueError("Retry must identify the unchanged original bundled metadata source")
    reuse, private, output = map(lambda path: Path(path).resolve(),
                                 (reuse_root, payload_path, payload["output_root"]))
    for path in (Path(reuse_root), Path(payload_path), Path(payload["output_root"])):
        if path.exists() or path.is_symlink():
            raise ValueError("Reuse directory, private payload and next output must be fresh")
    _separate([root, metadata_root, reuse, output])
    if any(private == folder or folder in private.parents for folder in (root, metadata_root, reuse, output)):
        raise ValueError("Private payload must be outside all dataset directories")

    metadata = _inventory(state.get("metadata_files"), "original metadata")
    if "meta/info.json" not in metadata or sum(row["bytes"] for row in metadata.values()) > 512 * 1024**2:
        raise ValueError("Missing or oversized original metadata")
    actual_metadata = set()
    for path in (metadata_root / "meta").rglob("*"):
        if path.is_symlink():
            raise ValueError("Original bundled metadata must not contain symlinks")
        if path.is_file():
            actual_metadata.add(path.relative_to(metadata_root).as_posix())
    if actual_metadata != set(metadata):
        raise ValueError("Original bundled metadata inventory differs from the old receipt")
    verified_metadata = [(_checked_file(metadata_root, row, deadline, allowed={"meta"}), row)
                         for row in metadata.values()]
    # Recompute selection from the pristine metadata, not the old depth-stripped
    # working copy. This is index verification only, not video/raw-success proof.
    scripts = str(Path(payload["code_root"]) / "scripts")
    sys.path.insert(0, scripts)
    try:
        from stage_subtask_replay import required_files
        required, selected = required_files(metadata_root, payload["episodes"], [])
    finally:
        sys.path.remove(scripts)
    if (required != state.get("required_files") or selected != state.get("selected_metadata")
            or any(type(row["task_instance_id"]) is not int or not 0 <= row["task_instance_id"] <= 300 for row in selected)):
        raise ValueError("Frozen train-only selection differs from original metadata")
    old_signed = _signed_inventory(old.get("signed_files"))
    new_signed = _signed_inventory(payload.get("signed_files"))
    transfers = _inventory(state.get("transfers"), "complete transfer")
    if set(transfers) != set(required) or set(old_signed) != set(required) or set(new_signed) != set(required):
        raise ValueError("Reader retry needs every required transfer, with no missing or extra objects")
    verified_data = []
    for name in required:
        row, before, after = transfers[name], old_signed[name], new_signed[name]
        if (row.get("bytes") != before["bytes"] or row.get("bytes") != after["bytes"]
                or any(not isinstance(item.get("etag"), str) for item in (row, before, after))
                or row["etag"].strip('"') != before["etag"].strip('"')
                or row["etag"].strip('"') != after["etag"].strip('"')
                or any(item.get("sha256", row.get("sha256")) != row.get("sha256") for item in (before, after))):
            raise ValueError("Downloaded object differs from its frozen signed identity")
        verified_data.append((_checked_file(root, row, deadline, allowed={"data", "videos"}), row))
    if (status_hash != _sha(state_path, deadline) or payload_hash != _sha(old_payload_path, deadline)
            or (root / "_RUNNING").exists()):
        raise ValueError("Old staging evidence changed during preparation")
    if sum(row["bytes"] for _, row in verified_data) > payload.get("max_download_bytes", 64 * 1024**3):
        raise ValueError("Verified local objects exceed the next staging byte budget")

    reuse.mkdir(parents=True, mode=0o700)
    copied = 0
    for source, row in verified_metadata:
        destination = reuse / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)  # Metadata is never hard-linked.
        _checked_file(reuse, row, deadline, allowed={"meta"})
    for source, row in verified_data:
        destination = reuse / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination, follow_symlinks=False)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, destination)
            copied += row["bytes"]
        _checked_file(reuse, row, deadline, allowed={"data", "videos"})
    prepared = {**payload, "metadata_source": str(reuse), "local_source_roots": [str(reuse)],
                "signed_files": [{**row, "sha256": transfers[row["relative_path"]]["sha256"]}
                                 for row in payload["signed_files"]]}
    result = dict(kind="codex_original_reader_resume_v1", state="prepared_only",
        previous_job={key: previous_job[key] for key in ("name", "uid", "state")},
        previous_job_matches_submission=True, failed_root=str(root),
        previous_status_sha256=status_hash, previous_payload_sha256=payload_hash,
        original_metadata_source=str(metadata_root), output_root=str(output), reuse_root=str(reuse),
        metadata_files=len(verified_metadata), metadata_bytes=sum(row["bytes"] for _, row in verified_metadata),
        reused_files=len(verified_data), reused_bytes=sum(row["bytes"] for _, row in verified_data),
        remaining_files=0, remaining_download_bytes=0, local_copy_fallback_bytes=copied,
        ready_for_training=False, original_reader_verified=False, raw_success_verified=False,
        elapsed_seconds=round(time.monotonic()-started, 3),
        source_files=[dict(path=row["path"], bytes=row["bytes"], sha256=row["sha256"], etag=row["etag"])
                      for _, row in verified_data])
    with (reuse / "reader_resume_receipt.json").open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    private.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(prepared, stream, indent=2); stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("failed-root", "payload", "previous-job", "previous-submission", "reuse-root", "output-payload"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        result = prepare_resume(args.failed_root, load(args.payload), previous_job=load(args.previous_job),
                                previous_submission=load(args.previous_submission), reuse_root=args.reuse_root,
                                payload_path=args.output_payload, max_seconds=args.max_seconds)
    except Exception as error:
        # Never interpolate payload data, which may include signed bearer URLs.
        print(json.dumps({"state": "preparation_failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
