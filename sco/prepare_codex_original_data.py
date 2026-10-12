#!/usr/bin/env python3
"""Finite CPU staging of explicit original episodes and visual teacher examples.

The payload contains paths and source URIs, never credentials. Only metadata and
the shared data/video files resolved for the selected episodes are transferred.
This driver does not submit jobs or modify any source dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

KIND = "codex_original_pilot_staging_v1"
SOURCE = "oss://traffic-model/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in value or "\n" in value:
        raise ValueError("Dataset file path escapes its root")
    return path


def validate_payload(payload):
    if payload.get("kind") != KIND:
        raise ValueError("Unexpected original-data staging contract")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes or len(episodes) > 6 or len(set(episodes)) != len(episodes) or any(type(ep) is not int or ep < 0 for ep in episodes):
        raise ValueError("Select one to six explicit original episode IDs")
    if payload.get("source", SOURCE).rstrip("/") != SOURCE:
        raise ValueError("Pilot source differs from the verified original demonstration prefix")
    if not 60 <= payload.get("max_seconds", 3600) <= 7200 or not 1 <= payload.get("max_download_bytes", 64 * 1024**3) <= 128 * 1024**3:
        raise ValueError("CPU staging needs bounded time and download bytes")
    demonstrations = payload.get("demonstrations", [])
    if not demonstrations or any(row.get("episode_index") not in episodes or not row.get("raw_path") for row in demonstrations):
        raise ValueError("Explicit matching raw task-success demonstrations are required")
    if len({row["episode_index"] for row in demonstrations}) != len(demonstrations):
        raise ValueError("Repeated visual demonstration source")
    if {row["episode_index"] for row in demonstrations} != set(episodes):
        raise ValueError("Every selected original episode needs raw success and real reader verification")
    if "raw_source_sha256" in payload:
        hashes = payload["raw_source_sha256"]
        if (not isinstance(hashes, dict) or set(hashes) != {row["raw_path"] for row in demonstrations}
                or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in hashes.values())):
            raise ValueError("Frozen raw SHA256 mapping must exactly cover the selected demonstrations")
    required_subtask = payload.get("required_demonstration_subtask")
    if required_subtask is not None and (not isinstance(required_subtask, str) or not required_subtask.strip()):
        raise ValueError("Required demonstration subtask must be nonempty text")
    for key in ("output_root", "code_root", "segments_path", "retained_episodes_path"):
        if not Path(payload[key]).is_absolute():
            raise ValueError("Runtime paths must be absolute")
    signed = payload.get("signed_files", [])
    if len({row["relative_path"] for row in signed}) != len(signed):
        raise ValueError("Repeated signed original file")
    for row in signed:
        relative(row["relative_path"])
        uri = urllib.parse.urlsplit(row["url"])
        if uri.scheme != "https" or not uri.hostname or uri.username or uri.password or type(row.get("bytes")) is not int or row["bytes"] <= 0:
            raise ValueError("Signed original file needs HTTPS and an explicit positive size")


def find_ossutil(payload):
    binaries = payload.get("ossutil_candidates", []) + [shutil.which("ossutil"), "/usr/local/bin/ossutil", "/usr/bin/ossutil"]
    binary = next((str(Path(p)) for p in binaries if p and Path(p).is_file() and os.access(p, os.X_OK)), None)
    if not binary:
        raise RuntimeError("Configured ossutil binary not found")
    candidates = payload.get("ossutil_config_candidates", [])
    if os.environ.get("OSSUTIL_CONFIG_PATH"):
        candidates.append(os.environ["OSSUTIL_CONFIG_PATH"])
    candidates.append(str(Path.home() / ".ossutilconfig"))
    config = next((str(Path(p)) for p in candidates if p and Path(p).is_file()), None)
    # Existing env-based credentials are consumed by ossutil itself. They are
    # never read, copied to files, added to arguments or returned in a report.
    if config is None and not (os.environ.get("OSS_ACCESS_KEY_ID") and os.environ.get("OSS_ACCESS_KEY_SECRET")):
        raise RuntimeError("No existing OSS configuration or authenticated runtime was found")
    return binary, (["--config-file", config] if config else [])


def stat_values(text):
    size = re.search(r"(?im)^\s*(?:content[-_ ]?length|size)\s*:\s*(\d+)\s*$", text)
    etag = re.search(r'(?im)^\s*etag\s*:\s*"?([^\s"]+)"?\s*$', text)
    if not size or not etag or int(size.group(1)) <= 0:
        raise ValueError("OSS stat did not return a nonempty size and ETag")
    return dict(bytes=int(size.group(1)), etag=etag.group(1))


def redact_diagnostic(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = str(value or "")
    value = re.sub(r"(?:https?|oss)://[^\s<>\"']+", "<URL redacted>", value, flags=re.IGNORECASE)
    value = re.sub(r"(?i)(\b(?:[A-Z_]*ACCESS_KEY[A-Z_]*|[A-Z_]*SECRET[A-Z_]*|[A-Z_]*TOKEN[A-Z_]*|PASSWORD|SIGNATURE)\b\s*[:=]\s*)([^\s,;]+)",
                   r"\1<redacted>", value)
    return value[-8192:]


def failure_diagnostic(error):
    """Keep transport identity without dumping requests or nested URL reasons."""
    if isinstance(error, urllib.error.HTTPError):
        return dict(failure_detail=redact_diagnostic(f"HTTP request failed (status {error.code})"),
                    http_status=error.code)
    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        result = dict(failure_detail=redact_diagnostic("URL request failed (" + type(reason).__name__ + ")"),
                      url_reason_type=type(reason).__name__)
        if isinstance(reason, OSError) and type(reason.errno) is int:
            result["url_reason_errno"] = reason.errno
        return result
    return dict(failure_detail=redact_diagnostic(str(error)))


def open_signed_request(request, *, deadline):
    """Retry only transient DNS failures before an HTTP response is opened."""
    for attempt in range(4):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Signed original request exceeded staging deadline")
        try:
            return urllib.request.urlopen(request, timeout=min(60, remaining)), attempt + 1
        except urllib.error.URLError as error:
            reason = error.reason
            if (attempt == 3 or not isinstance(reason, socket.gaierror)
                    or reason.errno != socket.EAI_AGAIN):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Signed original DNS retry exceeded staging deadline") from None
            threading.Event().wait(min(2**attempt, remaining))


def run(payload):
    validate_payload(payload)
    started = time.monotonic()
    deadline = started + payload.get("max_seconds", 3600)
    budget = payload.get("max_download_bytes", 64 * 1024**3)
    reserve = payload.get("reserve_bytes", 8 * 1024**3)
    root = Path(payload["output_root"])
    if root.exists():
        raise ValueError("Original staging needs a fresh output; preserve earlier partial evidence")
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < reserve + 512 * 1024**2:
        raise ValueError("Insufficient disk reserve for original-data staging")
    root.mkdir()
    # Signed URLs are temporary bearer access. Do not duplicate them into logs,
    # manifests or the persistent evidence payload.
    evidence_payload = {**payload, "signed_files": [{k: v for k, v in row.items() if k != "url"}
                                                   for row in payload.get("signed_files", [])]}
    write(root / "payload.json", evidence_payload)
    state = dict(kind=KIND, state="starting", selected_episodes=payload["episodes"], root=str(root),
                 source=SOURCE, transfers=[], demonstrations=[], ready_for_training=False)
    write(root / "status.json", state)
    (root / "_RUNNING").touch()
    cli = None
    signed_files = {row["relative_path"]: row for row in payload.get("signed_files", [])}

    def child_failure(arguments, *, phase, returncode, stdout, stderr, timeout=False):
        # Only decoder subprocesses use this path. OSS CLI streams may contain
        # credential configuration and remain discarded in their entirety.
        evidence = dict(phase=phase, program=Path(arguments[0]).name,
                        script=Path(arguments[1]).name if len(arguments) > 1 else None,
                        returncode=returncode, timed_out=timeout,
                        diagnostic=redact_diagnostic(str(stderr or "") + "\n" + str(stdout or "")))
        target = root / "child_failure.json"
        with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            json.dump(evidence, stream, indent=2); stream.write("\n")

    def command(arguments, *, limit=300, diagnostic_phase=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Original-data staging reached its wall-time limit")
        try:
            result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=min(limit, remaining), check=False)
        except subprocess.TimeoutExpired as error:
            if diagnostic_phase:
                child_failure(arguments, phase=diagnostic_phase, returncode=None,
                              stdout=error.stdout, stderr=error.stderr, timeout=True)
            raise
        if result.returncode:
            # CLI failures can embed signed request details. Neither stream is
            # printed or persisted. The failed phase and exception class suffice.
            if diagnostic_phase:
                child_failure(arguments, phase=diagnostic_phase, returncode=result.returncode,
                              stdout=result.stdout, stderr=result.stderr)
            raise RuntimeError("Bounded original-data subprocess failed")
        return result.stdout

    def oss(*arguments, limit=300):
        nonlocal cli
        if cli is None:
            cli = find_ossutil(payload)
        binary, options = cli
        return command([binary, *arguments, *options], limit=limit)

    def download(relative_path):
        target = root / relative(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        local_source = next((Path(candidate) / relative_path for candidate in payload.get("local_source_roots", [])
                             if (Path(candidate) / relative_path).is_file()), None)
        signed = signed_files.get(relative_path)
        identity = ({key: signed[key] for key in ("bytes", "etag") if key in signed}
                    if signed is not None else None)
        request_attempts = None
        if local_source is not None:
            size = local_source.stat().st_size
            source = str(local_source)
        elif signed is not None:
            size = signed["bytes"]
            source = SOURCE + "/" + relative_path
        else:
            source = SOURCE + "/" + relative_path
            identity = stat_values(oss("stat", source))
            size = identity["bytes"]
        previous = sum(item["bytes"] for item in state["transfers"])
        if size <= 0 or previous + size > budget or shutil.disk_usage(root).free < size + reserve:
            raise ValueError("Selected shared files exceed the finite transfer/disk budget")
        partial = target.with_suffix(target.suffix + ".partial")
        if local_source is not None:
            shutil.copyfile(local_source, partial)
            expected = sha(local_source)
        elif signed is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Signed original transfer exceeded staging deadline")
            headers = {}
            if signed.get("etag"):
                headers["If-Match"] = '"' + signed["etag"].strip('"') + '"'
            request = urllib.request.Request(signed["url"], headers=headers)
            response, request_attempts = open_signed_request(request, deadline=deadline)
            with response, partial.open("xb") as stream:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != size:
                    raise ValueError("Signed original response length differs from frozen source")
                remote_etag = response.headers.get("ETag")
                if signed.get("etag") and (not remote_etag or remote_etag.strip('"') != signed["etag"].strip('"')):
                    raise ValueError("Signed original response ETag differs from frozen source")
                received = 0
                while block := response.read(8 * 1024**2):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Signed original transfer exceeded staging deadline")
                    received += len(block)
                    if received > size:
                        raise ValueError("Signed original transfer exceeded expected file size")
                    stream.write(block)
            expected = signed.get("sha256")
        else:
            oss("cp", source, str(partial), "--parallel", "2", "--retry-times", "2", "--no-progress", limit=900)
            if stat_values(oss("stat", source)) != identity:
                raise ValueError("Original OSS object changed during transfer")
            expected = None
        actual = sha(partial)
        if partial.stat().st_size != size or (expected is not None and actual != expected):
            raise ValueError("Original data transfer length or local-source checksum differs")
        if signed is not None and (size != signed['bytes'] or
                (signed.get('sha256') is not None and actual != signed['sha256'])):
            raise ValueError("Reused original file differs from frozen signed object identity")
        partial.rename(target)
        state["transfers"].append(dict(path=relative_path, bytes=size, sha256=actual,
                                      source=source, etag=None if identity is None else identity.get("etag"),
                                      signed_request_attempts=request_attempts))
        write(root / "status.json", state)

    try:
        state["state"] = "metadata"; write(root / "status.json", state)
        meta_source = payload.get("metadata_source")
        if meta_source and (Path(meta_source) / "meta/info.json").is_file():
            shutil.copytree(Path(meta_source) / "meta", root / "meta", symlinks=False)
        else:
            (root / "meta").mkdir()
            oss("cp", SOURCE + "/meta/", str(root / "meta") + "/", "--recursive", "--job", "2",
                "--parallel", "2", "--retry-times", "2", "--no-progress", limit=600)
        metadata = [p for p in (root / "meta").rglob("*") if p.is_file()]
        if not (root / "meta/info.json").is_file() or sum(p.stat().st_size for p in metadata) > 512 * 1024**2:
            raise ValueError("Original metadata is missing or exceeds its bounded size")
        state["metadata_files"] = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(metadata)]
        sys.path.insert(0, str(Path(payload["code_root"]) / "scripts"))
        from stage_subtask_replay import required_files
        from strip_depth_from_info import main as strip_depth
        paths, selected = required_files(root, payload["episodes"], [])
        if any(not 0 <= row["task_instance_id"] <= 300 for row in selected):
            raise ValueError("Original replay selection includes public test instances")
        retained_path = Path(payload["retained_episodes_path"])
        segment_path = Path(payload["segments_path"])
        for key, path in (("retained_episodes_sha256", retained_path), ("segments_sha256", segment_path)):
            if payload.get(key) and sha(path) != payload[key]:
                raise ValueError("Frozen original training labels or selection changed")
        retained = {int(line) for line in retained_path.read_text().splitlines() if line.strip()}
        if not set(payload["episodes"]) <= retained:
            raise ValueError("Original replay is absent from retained training episodes")
        state["selected_metadata"] = selected
        state["required_files"] = paths
        state["state"] = "selected_files"; write(root / "status.json", state)
        for path in paths:
            download(path)
        strip_depth(str(root))
        labels = root / "labels"; labels.mkdir()
        shutil.copyfile(segment_path, labels / "subtask_segments.parquet")
        shutil.copyfile(retained_path, labels / "retained_episodes.txt")
        (root / "selected_episodes.txt").write_text("".join(f"{ep}\n" for ep in payload["episodes"]))
        state["state"] = "real_reader_and_demonstrations"; write(root / "status.json", state)
        for source in payload["demonstrations"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("Staging deadline reached before visual demonstration readback")
            output = root / "demonstrations" / str(source["episode_index"])
            raw_sha = payload.get("raw_source_sha256", {}).get(source["raw_path"])
            source_identity = ["--expected-raw-sha256", raw_sha] if raw_sha is not None else []
            required_subtask = payload.get("required_demonstration_subtask")
            if required_subtask is not None:
                source_identity += ["--required-subtask", required_subtask]
            command([sys.executable, str(Path(payload["code_root"]) / "scripts/codex_collection_demonstrations.py"),
                     "original", "--dataset-root", str(root), "--segments-path", str(labels / "subtask_segments.parquet"),
                     "--raw-path", source["raw_path"], "--episode-index", str(source["episode_index"]),
                     "--output-dir", str(output), "--retained-episodes-path", str(labels / "retained_episodes.txt"),
                     "--max-examples", "6", *source_identity], limit=900, diagnostic_phase="real_reader_and_demonstrations")
            result = json.loads((output / "manifest.json").read_text())
            state["demonstrations"].append(dict(episode_index=source["episode_index"], task=result["metadata"]["task"],
                manifest=str(output / "manifest.json"), fingerprint=result["fingerprint"], examples=len(result["examples"])))
            write(root / "status.json", state)
        state.update(state="complete", original_reader_verified=True, raw_success_verified=True,
                     elapsed_seconds=round(time.monotonic() - started, 3), bytes_staged=sum(row["bytes"] for row in state["transfers"]))
        write(root / "status.json", state)
        write(root / "_ORIGINAL_DATA_READY.json", state)
        return state
    except BaseException as error:
        state.update(failure_phase=state["state"], state="failed", failure_type=type(error).__name__,
                     elapsed_seconds=round(time.monotonic() - started, 3), **failure_diagnostic(error))
        write(root / "status.json", state)
        (root / "_FAILED").write_text(type(error).__name__ + "\n")
        raise
    finally:
        (root / "_RUNNING").unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(json.loads(args.payload.read_text()))
        print(json.dumps({key: result[key] for key in ("state", "root", "selected_episodes", "bytes_staged", "demonstrations")}, indent=2))
    except Exception as error:
        print("Original data staging failed (" + type(error).__name__ + "); inspect the owned status.json.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
