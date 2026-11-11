#!/usr/bin/env python3
"""Preflight the frozen action-calibration screen; submit only with --submit.

Default execution reads local artifacts only. The submit path uploads at most
the approved 22,502-byte patch, reuses the already uploaded parent, and makes a
single SCO copy attempt. Signed URLs remain in memory and the remote command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
PLAN_SHA = "7b5890613d8f6c48f3dd65a8b94d8b7a384b83db2b2ce2707a4c196a98f0cc47"
PATCH_SHA = "216062af8e93c2f4a0ce04c9655244e3098e2309968d4f38b0329755eef583c7"
PARENT_SHA = "394edd270711b32ab63ec3bad2c533759fd417d3ea91ecfcb5c49fa24edbc529"
JOB = "b1k-action-change-20261020-screen01"
BUCKET = "260010165"
PREFIX = f"behavior1k/action-change-audit-20261020/{JOB}/"
PARENT_KEY = "behavior1k/training-boundary-audit-20261019/b1k-boundary-probe-20261019-pilot01/bundle-" + PARENT_SHA + ".tar.gz"
PATCH_KEY = PREFIX + "code_patch-" + PATCH_SHA + ".tar.gz"
TEMPLATE = "b1k-repeat1688-1019-batch-00"
TEMPLATE_UID = "168d5223752b415caf67bd27ed504a83"
WORKSPACE = "share-space"
POOL = "computing-cluster-5090-01g"
WORKER_SPEC = "n12lp.nn.i10a.1"
MOUNT_ID = "01a04263-91e5-7603-bc01-c67e503da6b5"
IMAGE = "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634"
AFS = Path("/mnt/afs/260010165/behavior_1k")
RUNTIME = AFS / "runtime_20261007_1306"
REMOTE_ROOT = AFS / "training_boundary_audit_20261019/calibration_screen01"
CLI = "/home/faye/.local/bin/sco"
CHECKPOINTS = ["baseline1688977", "mixed1000", "mixed5000", "expert1000"]
PATCH_NAMES = {"plan.json", "cloud_probe_worker.py", "project/BEHAVIOR-VLAs/scripts/eval_openloop.py"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_files(path):
    """Read bounded regular members without extracting any archive locally."""
    files = {}
    total = 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = member.name
            relative = PurePosixPath(name)
            if (not name or not relative.parts or str(relative) != name or relative.is_absolute() or ".." in relative.parts
                    or "\\" in name or name in files or not member.isfile() or member.issparse()
                    or set(member.pax_headers) - {"mtime", "atime", "ctime", "path"}
                    or member.pax_headers.get("path", name) != name):
                raise ValueError("Unsafe, duplicate, or non-regular archive member")
            total += member.size
            if member.size < 0 or member.size > 4 * 1024 ** 2 or total > 128 * 1024 ** 2 or len(files) >= 512:
                raise ValueError("Archive exceeds frozen bootstrap safety bounds")
            with archive.extractfile(member) as stream:
                data = stream.read(member.size + 1)
            if len(data) != member.size:
                raise ValueError("Archive member length mismatch")
            files[name] = data
    for name in files:
        if any(str(parent) in files for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise ValueError("Archive contains a file/directory path collision")
    return files


def validate_template(path):
    template = json.loads(path.read_text())
    if (template.get("uid") != TEMPLATE_UID or template.get("name") != TEMPLATE
            or template.get("workspace_name") != WORKSPACE
            or template.get("resource_pool", {}).get("name") != POOL
            or template.get("scheduling", {}).get("quota_type") != "SPOT"):
        raise ValueError("Cached template identity or Spot resource pool differs")
    roles = template.get("roles", [])
    if len(roles) != 1 or roles[0].get("image_path") != IMAGE or roles[0].get("total_replicas") != 1:
        raise ValueError("Cached template image or worker count differs")
    specs = roles[0].get("resource_spec", [])
    if (len(specs) != 1 or specs[0].get("name") != WORKER_SPEC or specs[0].get("replicas") != 1
            or specs[0].get("limits", {}).get("nvidia.com/gpu") != "1"):
        raise ValueError("Cached template is not the approved single 5090 worker")
    mounts = template.get("mount", [])
    if not any(m.get("id") == MOUNT_ID and m.get("mount_path") == "/mnt/afs" for m in mounts):
        raise ValueError("Cached template lacks the approved AFS mount")
    return template


def preflight():
    plan_path = HERE / "execution_plan.json"
    if sha(plan_path) != PLAN_SHA:
        raise ValueError("Frozen execution_plan.json changed")
    plan = json.loads(plan_path.read_text())
    fixed = {"job_name": JOB, "workspace": WORKSPACE, "pool": POOL, "gpu_count": 1,
             "quota_type": "spot", "max_seconds": 3600, "cache_unchanged": True,
             "sample_count_per_checkpoint": 36, "checkpoint_names": CHECKPOINTS,
             "training": False, "simulation": False, "proposed_destination_bucket": BUCKET,
             "proposed_destination_prefix": PREFIX, "parent_bundle_sha256": PARENT_SHA,
             "parent_bundle_bucket": BUCKET, "parent_bundle_key": PARENT_KEY,
             "patch_bytes": 22502, "patch_sha256": PATCH_SHA}
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError("Execution plan exceeds the approved fixed screen scope")
    patch_path, parent_path = HERE / "code_patch.tar.gz", HERE.parent / "pilot_cloud_bundle.tar.gz"
    full_path = HERE / "bundle.tar.gz"
    for path, expected_hash, expected_bytes in ((patch_path, PATCH_SHA, 22502),
                                               (parent_path, PARENT_SHA, None),
                                               (full_path, plan["bundle_sha256"], plan["bundle_bytes"])):
        if sha(path) != expected_hash or (expected_bytes is not None and path.stat().st_size != expected_bytes):
            raise ValueError(f"Frozen archive identity changed: {path.name}")
    parent, patch, full = archive_files(parent_path), archive_files(patch_path), archive_files(full_path)
    if set(patch) != PATCH_NAMES or set(plan["changed_archive_entries"]) != PATCH_NAMES:
        raise ValueError("Patch must contain exactly the three authorized existing files")
    for name in PATCH_NAMES:
        if name not in parent or hashlib.sha256(patch[name]).hexdigest() != plan["changed_archive_entries"][name]:
            raise ValueError("Patch member content differs from the frozen plan")
    if set(parent) != set(full) or {**parent, **patch} != full:
        raise ValueError("Parent plus exact three-file patch does not reproduce the approved bundle contents")
    worker = json.loads(patch["plan.json"])
    if (worker != json.loads((HERE / "worker_plan.json").read_text())
            or worker.get("max_seconds") != 3600 or worker.get("sample_count") != 36
            or [row["name"] for row in worker["checkpoints"]] != CHECKPOINTS
            or worker.get("runtime") != str(RUNTIME) or worker.get("gpu_count") != 1
            or worker.get("quota_type") != "spot" or worker.get("save_raw_actions") is not True
            or worker.get("diagnostic") != "task_independent_action_change_calibration"
            or worker.get("training_executed") is not False or worker.get("simulator_executed") is not False):
        raise ValueError("Patched worker plan differs from the frozen action-only diagnostic")
    old_plan = json.loads(parent["plan.json"])
    excluded = {"max_seconds", "save_raw_actions", "diagnostic", "job_name"}
    if {k: v for k, v in worker.items() if k not in excluded} != {k: v for k, v in old_plan.items() if k not in excluded}:
        raise ValueError("Checkpoint or cache scope changed from the parent")
    needle = b'        with (output / f"{name}.log").open("w") as log:'
    expected_worker = parent["cloud_probe_worker.py"].replace(
        needle, b'        command.append("--save-raw-actions")\n' + needle, 1)
    if expected_worker != patch["cloud_probe_worker.py"]:
        raise ValueError("Worker change exceeds adding --save-raw-actions")
    validate_template(HERE / "template_preflight.json")
    return {"state": "local_preflight_passed", "job_name": JOB, "workspace": WORKSPACE,
            "pool": POOL, "template": TEMPLATE, "template_uid": TEMPLATE_UID,
            "template_preflight_sha256": sha(HERE / "template_preflight.json"),
            "execution_plan_sha256": PLAN_SHA, "patch_sha256": PATCH_SHA, "patch_bytes": 22502,
            "patch_key": PATCH_KEY, "parent_bundle_sha256": PARENT_SHA,
            "parent_bundle_bytes": parent_path.stat().st_size, "parent_bundle_key": PARENT_KEY,
            "patched_bundle_sha256": plan["bundle_sha256"], "archive_entries": len(full),
            "changed_archive_entries": plan["changed_archive_entries"],
            "checkpoint_names": CHECKPOINTS, "samples_per_checkpoint": 36,
            "gpu_count": 1, "quota_type": "spot", "worker_plan_seconds": 3600,
            "outer_term_seconds": 3540, "outer_kill_after_seconds": 60,
            "training": False, "simulation": False, "cfm_diagnostic": False,
            "remote_output_root": str(REMOTE_ROOT), "result_key": PREFIX + "results.tar.gz"}


_BOOTSTRAP = r'''
import hashlib, io, json, os, pathlib, subprocess, tarfile, urllib.request
c = __CONFIG__
def download(url, size, digest):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=120) as response:
        if response.status != 200 or int(response.headers.get('Content-Length', '-1')) != size:
            raise ValueError('Unexpected bootstrap response')
        data = response.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Bootstrap archive identity mismatch')
    return data
def members(data):
    out = {}; total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for member in archive:
            name = member.name; relative = pathlib.PurePosixPath(name)
            if (not name or not relative.parts or str(relative) != name or relative.is_absolute() or '..' in relative.parts
                or '\\' in name or name in out or not member.isfile() or member.issparse()
                or set(member.pax_headers) - {'mtime', 'atime', 'ctime', 'path'}
                or member.pax_headers.get('path', name) != name):
                raise ValueError('Unsafe bootstrap archive member')
            total += member.size
            if member.size < 0 or member.size > 4*1024**2 or total > 128*1024**2 or len(out) >= 512:
                raise ValueError('Bootstrap archive exceeds bounds')
            with archive.extractfile(member) as stream:
                content = stream.read(member.size + 1)
            if len(content) != member.size:
                raise ValueError('Bootstrap member length mismatch')
            out[name] = content
    for name in out:
        if any(str(p) in out for p in pathlib.PurePosixPath(name).parents if str(p) != '.'):
            raise ValueError('Bootstrap path collision')
    return out
def run():
    os.umask(0o077)
    root = pathlib.Path(c['root']); root.mkdir(parents=True, exist_ok=False)
    parent = members(download(c['parent_url'], c['parent_bytes'], c['parent_sha']))
    patch = members(download(c['patch_url'], c['patch_bytes'], c['patch_sha']))
    if set(patch) != set(c['changes']) or not set(patch) <= set(parent):
        raise ValueError('Unexpected patch scope')
    for name, content in patch.items():
        if hashlib.sha256(content).hexdigest() != c['changes'][name]:
            raise ValueError('Unexpected patch content')
    destination = root / 'bundle'; destination.mkdir()
    for name, content in parent.items():
        path = destination / name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(content)
    for name, content in patch.items():
        path = destination / name
        if not path.is_file() or path.is_symlink(): raise ValueError('Patch target is not an existing file')
        with path.open('wb') as stream: stream.write(content)
    plan = json.loads((destination / 'plan.json').read_text())
    if (plan['max_seconds'] != 3600 or plan['sample_count'] != 36 or plan.get('save_raw_actions') is not True
        or [p['name'] for p in plan['checkpoints']] != c['checkpoints']
        or plan['training_executed'] is not False or plan['simulator_executed'] is not False):
        raise ValueError('Patched worker plan exceeds fixed scope')
    os.environ['BOUNDARY_PROBE_RESULT_PUT_URL'] = c['put_url']
    return subprocess.call([c['python'], str(destination / 'cloud_probe_worker.py'),
                            '--bundle', str(destination), '--output', str(root / 'results')])
try:
    code = run()
except Exception as error:
    print(json.dumps({'bootstrap_error_type': type(error).__name__}), flush=True)
    code = 1
raise SystemExit(code)
'''


def bootstrap_command(receipt, parent_url, patch_url, put_url):
    config = {"root": str(REMOTE_ROOT), "python": str(RUNTIME / "envs/b1k-pi05/bin/python"),
              "parent_url": parent_url, "parent_bytes": receipt["parent_bundle_bytes"], "parent_sha": PARENT_SHA,
              "patch_url": patch_url, "patch_bytes": 22502, "patch_sha": PATCH_SHA,
              "put_url": put_url, "changes": receipt["changed_archive_entries"], "checkpoints": CHECKPOINTS}
    code = _BOOTSTRAP.replace("__CONFIG__", repr(config), 1)
    return "timeout --signal=TERM --kill-after=60 3540 " + shlex.quote(config["python"]) + " -u -c " + shlex.quote(code)


def save_state(path, state, *, exclusive=False):
    content = json.dumps(state, indent=2, sort_keys=True) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(content)
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content)
        temporary.replace(path)


def check_head(head, expected_size, expected_sha):
    if head.get("ContentLength") != expected_size:
        raise ValueError("Object size differs from the frozen artifact")
    recorded = head.get("Metadata", {}).get("sha256")
    if recorded is not None and recorded != expected_sha:
        raise ValueError("Object metadata digest differs from the frozen artifact")


def reconcile(state, runner=None):
    """Never persist describe's command field or the copy response containing URLs."""
    runner = subprocess.run if runner is None else runner
    try:
        queried = runner([CLI, "acp", "jobs", "describe", JOB, "--workspace-name=" + WORKSPACE, "-o", "json"],
                         capture_output=True, text=True, timeout=60)
        response = json.loads(queried.stdout) if queried.returncode == 0 else None
        if (not isinstance(response, dict) or response.get("name") != JOB
                or not isinstance(response.get("uid"), str) or not re.fullmatch(r"[a-fA-F0-9]{32}", response["uid"])
                or not isinstance(response.get("state"), str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", response["state"])):
            raise ValueError("Exact-name query did not establish a unique job identity")
        state.update({key: response[key] for key in ("name", "uid", "state")})
    except Exception as error:
        state.update(state="submission_response_requires_exact_name_query", query_error_type=type(error).__name__)
    return state


def submit(receipt):
    intent = HERE / "submission_intent.json"
    state = {**receipt, "state": "preparing_single_submission", "submission_attempted": False,
             "created_at_utc": datetime.now(timezone.utc).isoformat()}
    save_state(intent, state, exclusive=True)
    client = None
    try:
        sys.path.insert(0, str(PROJECT / "BEHAVIOR-VLAs/sco"))
        from video_eval_adapter import aoss_client

        client = aoss_client()
        parent_head = client.head_object(Bucket=BUCKET, Key=PARENT_KEY)
        check_head(parent_head, receipt["parent_bundle_bytes"], PARENT_SHA)
        try:
            patch_head = client.head_object(Bucket=BUCKET, Key=PATCH_KEY)
        except client.exceptions.ClientError as error:
            if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 404:
                raise
            with (HERE / "code_patch.tar.gz").open("rb") as stream:
                client.put_object(Bucket=BUCKET, Key=PATCH_KEY, Body=stream, Metadata={"sha256": PATCH_SHA})
            patch_head = client.head_object(Bucket=BUCKET, Key=PATCH_KEY)
        check_head(patch_head, 22502, PATCH_SHA)
        body = client.get_object(Bucket=BUCKET, Key=PATCH_KEY)["Body"]
        try:
            patch_bytes = body.read(22503)
        finally:
            body.close()
        if len(patch_bytes) != 22502 or hashlib.sha256(patch_bytes).hexdigest() != PATCH_SHA:
            raise ValueError("Uploaded patch failed complete content verification")
        parent_url = client.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": PARENT_KEY}, ExpiresIn=604800)
        patch_url = client.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": PATCH_KEY}, ExpiresIn=604800)
        put_url = client.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": state["result_key"]}, ExpiresIn=604800)
        command = bootstrap_command(receipt, parent_url, patch_url, put_url)
        state.update(submission_attempted=True, startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        save_state(intent, state)
        argv = [CLI, "acp", "jobs", "copy", "--workspace-name=" + WORKSPACE,
                "--copy-job-name=" + TEMPLATE, "--name=" + JOB, "--job-name=" + JOB,
                "--worker-spec=" + WORKER_SPEC, "--worker-nodes=1", "--quota-type=spot", "--retry-times=0",
                "--enable-fault-tolerance=false", "--enable-anomaly-detection=false",
                "--storage-mount=" + MOUNT_ID + ":/mnt/afs", "--container-image-url=" + IMAGE,
                "--env=NVIDIA_DRIVER_CAPABILITIES:all", "--command=" + command]
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            state["cli_returncode"] = completed.returncode
        except Exception as error:
            # Exception strings can contain the signed command; retain only its type.
            state["copy_error_type"] = type(error).__name__
        reconcile(state)
    except Exception as error:
        state.update(state="pre_submission_failed" if not state["submission_attempted"] else
                     "submission_response_requires_exact_name_query", error_type=type(error).__name__)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as error:
                state["client_close_error_type"] = type(error).__name__
        save_state(intent, state)
    return {key: state.get(key) for key in ("job_name", "name", "uid", "state", "cli_returncode", "error_type")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="make the single authorized network submission attempt")
    args = parser.parse_args(argv)
    try:
        receipt = preflight()
        result = submit(receipt) if args.submit else receipt
    except Exception as error:
        # Do not stringify errors from a submission path: argv can contain URLs.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] not in ("failed", "pre_submission_failed", "submission_response_requires_exact_name_query") else 1


if __name__ == "__main__":
    raise SystemExit(main())
