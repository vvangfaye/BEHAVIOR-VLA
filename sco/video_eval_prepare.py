"""Prepare the immutable 2M checkpoint and source for video evaluation.

The local entry point is idempotent and has exactly one CPU job creation intent.
It never creates GPU jobs. Signed URLs remain in memory / the submitted command;
all saved evidence is credential free. Run with the transfer Python environment.
"""
from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import inspect
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import time
import zlib

REPO = Path(__file__).resolve().parents[1]
BASE = "/mnt/afs/260010165/behavior_1k"
VOLUME = "01a04263-91e5-7603-bc01-c67e503da6b5"
SCO = "/private/tmp/behavior-sco-runtime/bin/sco"


class PreparationStopped(RuntimeError):
    pass


def require_running(root):
    if (Path(root) / "STOP").exists():
        raise PreparationStopped("preparation stopped by STOP marker")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with temp.open("wb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or str(p) != name:
        raise ValueError("unsafe relative path")
    return name


def direct_aoss_client(internal=False):
    import boto3
    from botocore.config import Config
    text = Path("/Users/faye/.config/sco/profiles/default.toml").read_text()
    values = {k: ast.literal_eval(v) for k, v in re.findall(
        r"^(access_key_id|access_key_secret)\s*=\s*([\"'].*[\"'])\s*$", text, re.M)}
    host = "aoss-internal" if internal else "aoss"
    return boto3.client("s3", endpoint_url=f"https://{host}.cn-sh-01b.sensecoreapi-oss.cn",
        region_name="cn-sh-01b", aws_access_key_id=values["access_key_id"],
        aws_secret_access_key=values["access_key_secret"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      proxies={}, connect_timeout=8, read_timeout=60,
                      retries={"total_max_attempts": 2, "mode": "standard"}))


def cli(args, timeout=45):
    env = os.environ.copy()
    env.update(SCO_CONFIG="/private/tmp/behavior-sco-config", SCO_HOME="/private/tmp/behavior-sco-runtime",
               SCO_DATA_HOME="/private/tmp/behavior-sco-data")
    return subprocess.run([SCO] + args, env=env, capture_output=True, text=True, timeout=timeout)


def default_config(root, plan):
    run_id = plan["run_id"]
    return dict(checkpoint_step=2000000, local_checkpoint=str(REPO.parent / "checkpoints/pi05_b1k_full100_joint_2000000"),
        checkpoint=f"{BASE}/checkpoints/{run_id}", code=f"{BASE}/code_snapshots/{run_id}",
        runtime=f"{BASE}/runtime_20261007_1306",
        oldcode=f"{BASE}/code_snapshots/public-test-1688977-aoss-20261010",
        old_source_sha256="7f2a16738961dfb1ca5b95d280ee61450f996757c5514b8c144e86dce3c982d4",
        overlay_files=["sco/video_eval_worker.py"], bucket="260010165",
        prefix=f"behavior1k/video-evaluation/{run_id}/preparation/",
        min_free_after_checkpoint_bytes=64 * 1024**3)


def verify_local_checkpoint(local_root, evidence_root):
    """Return None until the atomic final directory and verified marker exist."""
    local_root, evidence_root = Path(local_root), Path(evidence_root)
    marker = local_root / "TRANSFER_VERIFIED.json"
    if not marker.is_file():
        return None
    raw = marker.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("state") != "verified" or
        manifest.get("checkpoint_directory_step", manifest.get("checkpoint_step")) != 2000000 or
        manifest.get("file_count") != 24 or len(manifest.get("files", [])) != 24):
        raise ValueError("checkpoint verification marker does not identify the 24-file 2M checkpoint")
    rows, names, stats = [], set(), {}
    for row in manifest["files"]:
        name = safe_relative(row["relative_path"])
        if name in names or not (name == "_CHECKPOINT_METADATA" or name.startswith(("params/", "assets/"))):
            raise ValueError("unexpected checkpoint file")
        names.add(name)
        path = local_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint file missing or symlink")
        size = row.get("size", row.get("bytes"))
        stat = path.stat()
        if stat.st_size != size or not re.fullmatch("[0-9a-f]{64}", row["sha256"]):
            raise ValueError("checkpoint size or SHA metadata invalid")
        rows.append(dict(relative_path=name, bytes=size, sha256=row["sha256"]))
        stats[name] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    if sum(r["bytes"] for r in rows) != manifest["total_bytes"]:
        raise ValueError("checkpoint byte count mismatch")
    marker_sha = hashlib.sha256(raw).hexdigest()
    receipt = evidence_root / "local_checkpoint_verified.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get("transfer_manifest_sha256") != marker_sha or old.get("file_stats") != stats:
            raise ValueError("verified checkpoint changed after preparation started")
        return old
    for row in rows:
        path = local_root / row["relative_path"]
        if digest(path) != row["sha256"]:
            raise ValueError("independent local SHA256 verification failed")
        after = path.stat()
        if [after.st_size, after.st_mtime_ns, after.st_ino] != stats[row["relative_path"]]:
            raise ValueError("checkpoint changed during SHA256 verification")
    result = dict(state="verified", checkpoint_step=2000000, files=rows, file_count=24,
                  total_bytes=manifest["total_bytes"], transfer_manifest_sha256=marker_sha,
                  file_stats=stats, independent_full_readback=True, observed_at=now())
    write_json(receipt, result, exclusive=True)
    return result


def upload_checkpoint(root, config, verified, client):
    """Resume objects only by matching immutable size and SHA metadata."""
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError
    root = Path(root)
    require_running(root)
    plan = dict(checkpoint_step=2000000, bucket=config["bucket"], prefix=config["prefix"] + "checkpoint/",
                files=verified["files"], total_bytes=verified["total_bytes"],
                transfer_manifest_sha256=verified["transfer_manifest_sha256"])
    plan_path = root / "checkpoint_upload_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("immutable checkpoint upload plan changed")
    else:
        write_json(plan_path, plan, exclusive=True)
    done = []
    def one(row):
        require_running(root)
        key = plan["prefix"] + row["relative_path"]
        try:
            head = client.head_object(Bucket=plan["bucket"], Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
            head = None
        if head is not None and (head["ContentLength"] != row["bytes"] or
                                  head.get("Metadata", {}).get("sha256") != row["sha256"]):
            raise ValueError("existing AOSS checkpoint object conflicts with frozen upload plan")
        if head is None:
            require_running(root)
            path = Path(config["local_checkpoint"]) / row["relative_path"]
            before = path.stat()
            if [before.st_size, before.st_mtime_ns, before.st_ino] != verified["file_stats"][row["relative_path"]]:
                raise ValueError("checkpoint changed before upload")
            client.upload_file(str(path), plan["bucket"], key, ExtraArgs={"Metadata": {"sha256": row["sha256"]}},
                Config=TransferConfig(multipart_threshold=64 * 1024**2, multipart_chunksize=32 * 1024**2,
                                      max_concurrency=3, use_threads=True))
            after = path.stat()
            if [after.st_size, after.st_mtime_ns, after.st_ino] != verified["file_stats"][row["relative_path"]]:
                raise ValueError("checkpoint changed during upload")
            head = client.head_object(Bucket=plan["bucket"], Key=key)
            if head["ContentLength"] != row["bytes"] or head.get("Metadata", {}).get("sha256") != row["sha256"]:
                raise ValueError("AOSS upload HEAD verification failed")
        return dict(row, key=key, aoss_metadata_verified=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one, row) for row in plan["files"]]):
            done.append(future.result())
            write_json(root / "checkpoint_upload_status.json", dict(state="uploading", files=done,
                uploaded_bytes=sum(r["bytes"] for r in done), observed_at=now()))
    require_running(root)
    client.put_object(Bucket=plan["bucket"], Key=plan["prefix"] + "MANIFEST.json", Body=canonical(plan))
    write_json(root / "checkpoint_upload_status.json", dict(state="uploaded", files=done,
        uploaded_bytes=sum(r["bytes"] for r in done), observed_at=now(), cloud_full_sha_readback="pending"))
    return plan


def quota_free_lower_bound(output):
    """SCO prints rounded SI sizes; round quota down and usage up conservatively."""
    rows = [line for line in output.splitlines() if "/260010165" in line and VOLUME in line]
    if len(rows) != 1:
        raise ValueError("personal AFS quota row missing or ambiguous")
    sizes = re.findall(r"(\d+(?:\.\d+)?)\s*(TB|GB|MB|KB|B)\b", rows[0])
    if len(sizes) != 2:
        raise ValueError("cannot parse personal AFS capacity")
    units = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
    bounds = []
    for value, unit in sizes:
        rounding = 10 ** (-len(value.split(".")[1])) if "." in value else 1
        bounds.append((float(value) * units[unit], rounding * units[unit]))
    return max(0, int(bounds[0][0] - bounds[0][1] - bounds[1][0] - bounds[1][1]))


def freeze_overlay(root, config):
    root = Path(root)
    archive = root / "frozen_source_overlay.tar.gz"
    manifest = root / "frozen_source_overlay.json"
    hashes = {safe_relative(n): digest(REPO / n) for n in config["overlay_files"]}
    def persist_frozen_worker(blob, expected):
        destination = root / "frozen/video_eval_worker.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            worker_bytes = tar.extractfile("sco/video_eval_worker.py").read()
        if hashlib.sha256(worker_bytes).hexdigest() != expected:
            raise ValueError("frozen worker does not match overlay")
        if destination.exists():
            if digest(destination) != expected:
                raise ValueError("frozen worker was changed")
        else:
            with destination.open("xb") as stream:
                stream.write(worker_bytes)
                stream.flush()
                os.fsync(stream.fileno())
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["files_sha256"] != hashes or digest(archive) != saved["sha256"]:
            raise ValueError("source overlay changed after freezing")
        value = archive.read_bytes()
        persist_frozen_worker(value, hashes["sco/video_eval_worker.py"])
        return saved, value
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        for name in sorted(hashes):
            content = (REPO / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = len(content), 0o644, 0
            tar.addfile(member, io.BytesIO(content))
    value = data.getvalue()
    with archive.open("xb") as stream:
        stream.write(value)
    saved = dict(files_sha256=hashes, sha256=hashlib.sha256(value).hexdigest())
    write_json(manifest, saved, exclusive=True)
    persist_frozen_worker(value, hashes["sco/video_eval_worker.py"])
    return saved, value


def cloud_prepare(p):
    """Self-contained standard-library CPU worker, embedded in one SCO command."""
    import base64, hashlib, io, json, os, shutil, tarfile, time, urllib.request
    from pathlib import Path, PurePosixPath
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(block)
        return h.hexdigest()
    def encoded(value):
        return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    checkpoint, code, runtime = Path(p["checkpoint"]), Path(p["code"]), Path(p["runtime"])
    def publish(value):
        if checkpoint.exists():
            (checkpoint / "prepare_status.json").write_bytes(encoded(value))
        print("STATUS_JSON=" + json.dumps(value), flush=True)
        request = urllib.request.Request(p["report_put"], data=encoded(value), method="PUT")
        with opener.open(request, timeout=60) as response:
            response.read()
    try:
        if checkpoint.exists() or code.exists():
            raise RuntimeError("preparation targets already exist; require explicit investigation")
        checkpoint.mkdir(parents=True, exist_ok=False)
        if shutil.disk_usage(checkpoint).free < p["total_bytes"] + p["min_free_after_checkpoint_bytes"]:
            raise RuntimeError("filesystem capacity insufficient")
        rows = []
        for item in p["items"]:
            name = item["relative_path"]
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError("unsafe checkpoint path")
            target = checkpoint / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".download")
            for attempt in range(3):
                try:
                    count = 0
                    with opener.open(item["url"], timeout=60) as source, temp.open("wb") as destination:
                        while True:
                            block = source.read(8 * 1024**2)
                            if not block:
                                break
                            count += len(block)
                            if count > item["bytes"]:
                                raise RuntimeError("download exceeded expected size")
                            destination.write(block)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if count != item["bytes"] or sha(temp) != item["sha256"]:
                        raise RuntimeError("AFS full readback SHA256 mismatch")
                    temp.rename(target)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            rows.append(dict(relative_path=name, bytes=item["bytes"], sha256=item["sha256"], full_local_readback=True))
            publish(dict(state="preparing", checkpoint_step=2000000, stage="checkpoint_download",
                         verified_files=len(rows), total_files=24, run_id=p["run_id"]))
        if len(rows) != 24 or sum(r["bytes"] for r in rows) != p["total_bytes"]:
            raise RuntimeError("checkpoint count mismatch")
        conditioning = json.loads((checkpoint / "assets/subtask_conditioning.json").read_text())
        if conditioning["action_conditioning"] != "task_and_subtask":
            raise RuntimeError("unexpected action conditioning")
        if not (runtime / "_RUNTIME_READY").is_file() or not (runtime / "graphics-runtime/_READY").is_file():
            raise RuntimeError("runtime not ready")
        oldcode = Path(p["oldcode"])
        marker = (oldcode / "_PUBLIC_TEST_SOURCE_READY").read_bytes()
        if hashlib.sha256(marker).hexdigest() != p["old_source_sha256"]:
            raise RuntimeError("frozen base source provenance mismatch")
        old = json.loads(marker)
        base_bytes = (oldcode / "_CODE_SNAPSHOT_READY").read_bytes()
        if hashlib.sha256(base_bytes).hexdigest() != old["base_source_sha256"]:
            raise RuntimeError("base snapshot provenance mismatch")
        base = json.loads(base_bytes)
        base_files = dict(base.get("overlay_files_sha256", {}))
        base_files.update(old["files_sha256"])
        for name, expected in base_files.items():
            if sha(oldcode / name) != expected:
                raise RuntimeError("old frozen source was changed")
        if sha(runtime / "graphics-runtime/_READY") != base["graphics_runtime_manifest_sha256"]:
            raise RuntimeError("graphics provenance changed")
        shutil.copytree(oldcode, code, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "config.sh"))
        overlay = base64.b64decode(p["overlay"])
        if hashlib.sha256(overlay).hexdigest() != p["overlay_sha256"]:
            raise RuntimeError("overlay archive mismatch")
        with tarfile.open(fileobj=io.BytesIO(overlay), mode="r:gz") as tar:
            for member in tar.getmembers():
                rel = PurePosixPath(member.name)
                if not member.isfile() or rel.is_absolute() or ".." in rel.parts or rel.name == "config.sh":
                    raise RuntimeError("unsafe source overlay")
                target = code / member.name
                if target.is_symlink():
                    raise RuntimeError("overlay target is symlink")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
                target.chmod(member.mode & 0o777)
        for name, expected in p["overlay_files_sha256"].items():
            if sha(code / name) != expected:
                raise RuntimeError("frozen source overlay mismatch")
        files, symlinks = {}, {}
        for path in sorted(code.rglob("*")):
            name = str(path.relative_to(code))
            if name == "_PUBLIC_TEST_SOURCE_READY" or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                symlinks[name] = os.readlink(path)
                # External source symlinks would defeat immutable source isolation.
                if not path.resolve().is_relative_to(code.resolve()):
                    raise RuntimeError("source symlink points outside independent snapshot")
            if path.is_file():
                files[name] = sha(path)
        if not all(name in files for name in ("scripts/eval_b1k.sh", "scripts/pi05_eval_queue.py", "sco/video_eval_worker.py")):
            raise RuntimeError("source snapshot missing required evaluator files")
        library = old["supplemental_library"]
        libroot = Path(library["root"])
        if sha(libroot / "_READY") != library["ready_manifest_sha256"]:
            raise RuntimeError("supplemental runtime provenance changed")
        libready = json.loads((libroot / "_READY").read_text())
        for name, expected in libready["libraries_sha256"].items():
            if sha(Path(libready["library_path"]) / name) != expected:
                raise RuntimeError("supplemental runtime library changed")
        provenance = dict(checkpoint_step=2000000, run_id=p["run_id"],
            base_source_sha256=old["base_source_sha256"], previous_source_manifest_sha256=p["old_source_sha256"],
            files_sha256=files, symlinks=symlinks, checkpoint_files=rows,
            supplemental_library=library, overlay_sha256=p["overlay_sha256"],
            code=str(code), runtime=str(runtime), scope=p["scope"], source_policy="independent frozen copy; only declared overlays")
        source_bytes = encoded(provenance)
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        (code / "_PUBLIC_TEST_SOURCE_READY").write_bytes(source_bytes)
        ready = dict(checkpoint_step=2000000, files=24, total_bytes=p["total_bytes"],
                     full_readback_sha256=True, source_manifest_sha256=source_sha, run_id=p["run_id"])
        ready_bytes = encoded(ready)
        (checkpoint / "_READY").write_bytes(ready_bytes)
        report = dict(state="ready", checkpoint_step=2000000, run_id=p["run_id"],
            files=24, checkpoint=str(checkpoint), code=str(code), runtime=str(runtime),
            source_manifest_sha256=source_sha, checkpoint_ready_sha256=hashlib.sha256(ready_bytes).hexdigest(),
            scope=p["scope"], source_manifest=provenance, checkpoint_ready=ready,
            worker_sha256=files["sco/video_eval_worker.py"],
            checkpoint_full_sha256_readback=True, plan_sha256=p["plan_sha256"])
        publish(report)
    except BaseException as exc:
        # Exception text can include signed URLs. Persist only its class.
        try:
            publish(dict(state="failed", checkpoint_step=2000000, run_id=p["run_id"], error_type=type(exc).__name__))
        except Exception:
            pass
        raise RuntimeError("preparation failed; inspect saved status and source without logging secrets") from None


def build_payload(root, plan, config, uploaded, client, internal):
    frozen, overlay = freeze_overlay(root, config)
    items = []
    for row in uploaded["files"]:
        items.append(dict(row, url=internal.generate_presigned_url("get_object", Params={
            "Bucket": uploaded["bucket"], "Key": uploaded["prefix"] + row["relative_path"]}, ExpiresIn=604800)))
    scope = dict(tasks=sorted({e["task"] for e in plan["episodes"]}), instance_ids=list(range(301, 311)),
                 rollout_ids=[0], expected_rollouts=1000, prompt_mode="whole_task", write_video=True,
                 oracle_prompts=False, external_subtask_selector=None, max_steps="official default")
    return {**{k: config[k] for k in ("checkpoint", "code", "runtime", "oldcode", "old_source_sha256", "min_free_after_checkpoint_bytes")},
        "run_id": plan["run_id"], "plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(),
        "scope": scope, "items": items, "total_bytes": uploaded["total_bytes"],
        "overlay": base64.b64encode(overlay).decode(), "overlay_sha256": frozen["sha256"],
        "overlay_files_sha256": frozen["files_sha256"],
        "report_put": client.generate_presigned_url("put_object", Params={"Bucket": config["bucket"],
            "Key": config["prefix"] + "ready.json"}, ExpiresIn=604800)}


def describe_preparation(record):
    # SCO's positional identifier is the saved resource name, not its UID.
    # Validate the immutable UID in the returned resource instead.
    result = cli(["acp", "jobs", "describe", record["name"], "--workspace-name=share-space", "-o", "json"])
    if result.returncode:
        return None
    job = json.loads(result.stdout)
    if (job.get("name") != record["name"] or job.get("ownership", {}).get("user_name") != "260010165" or
        (record.get("uid") and job.get("uid") != record["uid"])):
        raise ValueError("preparation identity mismatch")
    return dict(uid=job["uid"], platform_state=job["state"])


def inspect_preparation(root, config, plan, client):
    from botocore.exceptions import ClientError
    root = Path(root)
    record_path = root / "preparation_submission.json"
    record = json.loads(record_path.read_text())
    observation = describe_preparation(record)
    if observation is None:
        return dict(state="preparation_state_unknown", name=record["name"], no_resubmit=True)
    record.update(observation, observed_at=now())
    write_json(record_path, record)
    try:
        response = client.get_object(Bucket=config["bucket"], Key=config["prefix"] + "ready.json")
        report = json.loads(response["Body"].read(8 * 1024**2))
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        terminal_failure = observation["platform_state"] in ("FAILED", "STOPPED", "DELETED", "CANCELED", "CANCELLED")
        return dict(state="preparation_failed" if terminal_failure else "preparing", **observation)
    write_json(root / "preparation_progress.json", report)
    if report.get("state") != "ready" or observation["platform_state"] != "SUCCEEDED":
        terminal_failure = observation["platform_state"] in ("FAILED", "STOPPED", "DELETED", "CANCELED", "CANCELLED")
        return dict(state="preparation_failed" if report.get("state") == "failed" or terminal_failure else "preparing", **observation)
    if (report.get("checkpoint_step") != 2000000 or report.get("run_id") != plan["run_id"] or
        report.get("plan_sha256") != hashlib.sha256(canonical(plan)).hexdigest() or report.get("files") != 24 or
        not report.get("checkpoint_full_sha256_readback") or
        any(report.get(k) != config[k] for k in ("checkpoint", "code", "runtime")) or
        report.get("source_manifest_sha256") != hashlib.sha256(canonical(report["source_manifest"])).hexdigest() or
        report.get("checkpoint_ready_sha256") != hashlib.sha256(canonical(report["checkpoint_ready"])).hexdigest() or
        report.get("worker_sha256") != report["source_manifest"]["files_sha256"].get("sco/video_eval_worker.py") or
        report.get("worker_sha256") != digest(root / "frozen/video_eval_worker.py")):
        raise ValueError("invalid cloud preparation receipt")
    report.update(preparation_uid=record["uid"], preparation_name=record["name"], verified_at=now())
    write_json(root / "preparation_ready.json", report)
    return report


def step(root, plan_path=None):
    """Perform one preparation phase; GPU admissions belong to another module."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".preparation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(state="another_preparer_active")
        if (root / "STOP").exists():
            return dict(state="stopped", no_submit=True)
        plan = json.loads(Path(plan_path or root / "plan.json").read_text())
        episodes = plan.get("episodes", [])
        if (plan.get("checkpoint_step") != 2000000 or len(episodes) != 1000 or
            len({e["task"] for e in episodes}) != 100 or len({(e["task"], e["instance_id"]) for e in episodes}) != 1000 or
            {e["instance_id"] for e in episodes} != set(range(301, 311)) or
            not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,50}", plan.get("run_id", ""))):
            raise ValueError("preparation requires fixed 100-task, 1000-episode 2M plan")
        config_path = root / "preparation_config.json"
        if not config_path.exists():
            write_json(config_path, default_config(root, plan), exclusive=True)
        config = json.loads(config_path.read_text())
        frozen_config_path = root / "preparation_config_frozen.json"
        frozen = dict(config=config, plan_sha256=hashlib.sha256(canonical(plan)).hexdigest())
        if frozen_config_path.exists():
            if json.loads(frozen_config_path.read_text()) != frozen:
                raise ValueError("preparation configuration or plan changed")
        else:
            write_json(frozen_config_path, frozen, exclusive=True)
        if (root / "preparation_submission.json").exists():
            return inspect_preparation(root, config, plan, direct_aoss_client())
        verified = verify_local_checkpoint(config["local_checkpoint"], root)
        if verified is None:
            return dict(state="download_pending", checkpoint=config["local_checkpoint"])
        client = direct_aoss_client()
        try:
            uploaded = upload_checkpoint(root, config, verified, client)
        except PreparationStopped:
            return dict(state="stopped", no_submit=True)
        quota = cli(["afs", "dir-quota", "list", "--volumeID=" + VOLUME, "--dirPath=/260010165"])
        if quota.returncode:
            return dict(state="afs_quota_unknown", no_submit=True)
        lower_bound = quota_free_lower_bound(quota.stdout)
        required = verified["total_bytes"] + config["min_free_after_checkpoint_bytes"]
        write_json(root / "preparation_quota_before.json", dict(observed_at=now(), quota=quota.stdout,
            free_bytes_lower_bound=lower_bound, required_bytes=required, no_cleanup=True))
        if lower_bound < required:
            return dict(state="insufficient_afs_space", free_bytes_lower_bound=lower_bound, required_bytes=required)
        payload = build_payload(root, plan, config, uploaded, client, direct_aoss_client(internal=True))
        source = inspect.getsource(cloud_prepare)
        write_source = root / "frozen_preparation_worker.py"
        if write_source.exists() and write_source.read_text() != source:
            raise ValueError("preparation worker changed")
        if not write_source.exists():
            with write_source.open("x") as stream:
                stream.write(source)
        program = source + "\ncloud_prepare(" + repr(payload) + ")\n"
        encoded = base64.b64encode(zlib.compress(program.encode())).decode()
        startup = "import base64,zlib;exec(compile(zlib.decompress(base64.b64decode(" + repr(encoded) + ")),\"video_eval_prepare\",\"exec\"))"
        name = plan["run_id"] + "-prepare"
        record = dict(name=name, state="submission_intent", gpu_count=0, cpu_count=8, worker_spec="n6ls.iu.i40.8c32g",
            quota_type="spot", created_at=now(), checkpoint=config["checkpoint"], code=config["code"],
            report_key=config["prefix"] + "ready.json", plan_sha256=payload["plan_sha256"],
            source_sha256=hashlib.sha256(source.encode()).hexdigest(), overlay_sha256=payload["overlay_sha256"])
        if (root / "STOP").exists():
            return dict(state="stopped", no_submit=True)
        write_json(root / "preparation_submission.json", record, exclusive=True)
        args = ["acp", "jobs", "create", "--workspace-name=share-space", "--aec2-name=share-cluster",
            "--name=" + name, "--job-name=" + name, "--training-framework=pytorch",
            "--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630",
            "--worker-spec=n6ls.iu.i40.8c32g", "--worker-nodes=1", "--quota-type=spot",
            "--storage-mount=" + VOLUME + ":/mnt/afs",
            "--command=timeout --signal=TERM --kill-after=15 21600 python3 -c " + shlex.quote(startup)]
        try:
            require_running(root)
            result = cli(args, timeout=90)
            record.update(state="submitted" if result.returncode == 0 else "submission_unknown", exit_code=result.returncode)
        except subprocess.TimeoutExpired:
            record.update(state="submission_unknown", timeout=True)
        except PreparationStopped:
            record.update(state="stopped_before_submission", no_submit=True)
            write_json(root / "preparation_submission.json", record)
            return dict(state="stopped", no_submit=True)
        write_json(root / "preparation_submission.json", record)
        return inspect_preparation(root, config, plan, client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--once", action="store_true", help="perform one phase (also the default)")
    args = parser.parse_args()
    try:
        report = step(args.root, args.plan)
        keys = ("state", "checkpoint_step", "run_id", "name", "uid", "platform_state", "checkpoint", "code",
                "source_manifest_sha256", "checkpoint_ready_sha256", "free_bytes_lower_bound", "required_bytes")
        print(json.dumps({key: report[key] for key in keys if key in report}))
    except Exception as exc:
        # Network exception strings may contain signed URLs. Never print them.
        print(json.dumps(dict(state="preparation_error", error_type=type(exc).__name__)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
