#!/usr/bin/env python3
"""Freeze and submit one finite, four-checkpoint cached-observation diagnostic.

No simulator or training runs. Submission intent is exclusive; a failed or
ambiguous CLI response must be inspected by exact job name, not resubmitted.
The SDK consumes existing credentials and never persists signed URLs here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BUCKET = "260010165"
JOB = "b1k-boundary-probe-20261019-pilot01"
PREFIX = f"behavior1k/training-boundary-audit-20261019/{JOB}/"
AFS = Path("/mnt/afs/260010165/behavior_1k")
CLI = "/home/faye/.local/bin/sco"
TEMPLATE = "b1k-repeat1688-1019-batch-00"
TEMPLATE_UID = "168d5223752b415caf67bd27ed504a83"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def checkpoint_plan():
    root = PROJECT / "analysis/retraining_20261016_quality_taskbalanced_v1"
    local_base = PROJECT / "analysis/organizer_rehearsal_1688977_20261016/runtime/policy-bundle/rootfs"
    base_path = AFS / "checkpoints/pi05_b1k_full100_joint_1688977_aoss_20261010"
    local_checkpoint = local_base / str(base_path).lstrip("/")
    rows = []
    for line in (root / "formal_stage/bundle/inputs/checkpoint.SHA256SUMS").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = local_checkpoint / relative
        if sha(path) != digest:
            raise ValueError("Local baseline differs from accepted checkpoint")
        rows.append({"relative_path": relative, "bytes": path.stat().st_size, "sha256": digest})
    checkpoints = [{"name": "baseline1688977", "path": str(base_path), "files": rows}]
    for name, campaign in [("mixed1000", "paired_fixed20_step1000"),
                           ("mixed5000", "paired_fixed20_step5000"),
                           ("expert1000", "expert_fixed20_step1000")]:
        exported = json.loads((root / campaign / "reports/export.json").read_text())
        if exported["state"] != "accepted" or not exported["full_afs_sha256_readback"]:
            raise ValueError("Fine-tuned weights have no accepted export")
        checkpoints.append({"name": name, "path": exported["checkpoint"],
                            "files": [{key: row[key] for key in ("relative_path", "bytes", "sha256")}
                                      for row in exported["files"]]})
    return checkpoints


def build_bundle(cache: Path):
    from eval_cached_probe import VerifiedRawCache
    verified = VerifiedRawCache(cache / "cache_manifest.json")
    if len(verified.records) != 36:
        raise ValueError("This submission is bounded to the frozen 36-sample pilot")
    for row in verified.probe["sample_keys"]:
        record = verified.records[(row["episode_index"], row["frame_index"])]
        verified(SimpleNamespace(**{**row, **record}))
    plan = {"kind": "finite_cached_boundary_probe", "sample_count": 36,
            "max_seconds": 10800, "gpu_count": 1, "quota_type": "spot",
            "runtime": str(AFS / "runtime_20261007_1306"),
            "norm_asset_id": "b1k_full100_joint_subtask_v1_drop_episode_s1",
            "checkpoints": checkpoint_plan(),
            "cache_manifest_sha256": sha(cache / "cache_manifest.json"),
            "training_executed": False, "simulator_executed": False}
    paths = {}
    for source_root in [PROJECT / "BEHAVIOR-VLAs/openpi/src",
                        PROJECT / "BEHAVIOR-VLAs/openpi/packages/openpi-client/src"]:
        for path in source_root.rglob("*.py"):
            if "__pycache__" not in path.parts and not path.name.endswith("_test.py"):
                paths["project/" + path.relative_to(PROJECT).as_posix()] = path
    for relative in ["BEHAVIOR-VLAs/scripts/eval_openloop.py",
                     "analysis/training_boundary_audit_20261019/eval_cached_probe.py"]:
        paths["project/" + relative] = PROJECT / relative
    for path in cache.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            paths["project/analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1/"
                  + path.relative_to(cache).as_posix()] = path
    paths["cloud_probe_worker.py"] = HERE / "cloud_probe_worker.py"
    output = HERE / "pilot_cloud_bundle.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        data = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("plan.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        for relative, path in sorted(paths.items()):
            archive.add(path, arcname=relative, recursive=False)
    if output.stat().st_size > 120 * 1024**2:
        raise ValueError("Pilot bundle exceeds the bounded bootstrap limit")
    receipt = {"job_name": JOB, "template": TEMPLATE, "template_uid": TEMPLATE_UID,
               "workspace": "share-space", "pool": "computing-cluster-5090-01g",
               "gpu_count": 1, "quota_type": "spot", "max_seconds": 10800,
               "bundle_sha256": sha(output), "bundle_bytes": output.stat().st_size,
               "source_files": {name: sha(path) for name, path in sorted(paths.items())},
               "checkpoint_names": [row["name"] for row in plan["checkpoints"]]}
    save(HERE / "pilot_bundle_receipt.json", receipt)
    return output, receipt


def submit(bundle, receipt):
    sys.path.insert(0, str(PROJECT / "BEHAVIOR-VLAs/sco"))
    from video_eval_adapter import aoss_client
    client = aoss_client()
    intent = HERE / "pilot_submission_intent.json"
    with intent.open("x") as target:
        json.dump({**receipt, "submission_attempted": False}, target, indent=2)
    key = PREFIX + "bundle-" + receipt["bundle_sha256"] + ".tar.gz"
    try:
        head = client.head_object(Bucket=BUCKET, Key=key)
    except client.exceptions.ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
            raise
        with bundle.open("rb") as stream:
            client.put_object(Bucket=BUCKET, Key=key, Body=stream,
                              Metadata={"sha256": receipt["bundle_sha256"]})
        head = client.head_object(Bucket=BUCKET, Key=key)
    if head["ContentLength"] != bundle.stat().st_size or head["Metadata"].get("sha256") != receipt["bundle_sha256"]:
        raise ValueError("Uploaded bundle identity differs")
    body = client.get_object(Bucket=BUCKET, Key=key)["Body"]
    digest = hashlib.sha256()
    try:
        for data in iter(lambda: body.read(1024**2), b""):
            digest.update(data)
    finally:
        body.close()
    if digest.hexdigest() != receipt["bundle_sha256"]:
        raise ValueError("Uploaded bundle readback failed")
    get_url = client.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=604800)
    result_key = PREFIX + "results.tar.gz"
    put_url = client.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": result_key}, ExpiresIn=604800)
    root = AFS / "training_boundary_audit_20261019/pilot01"
    bootstrap = (
        "import hashlib,os,pathlib,subprocess,tarfile,urllib.request;"
        f"r=pathlib.Path({str(root)!r});r.mkdir(parents=True,exist_ok=False);"
        "o=urllib.request.build_opener(urllib.request.ProxyHandler({}));"
        f"b=o.open({get_url!r},timeout=120).read(128*1024**2);"
        f"assert hashlib.sha256(b).hexdigest()=={receipt['bundle_sha256']!r};"
        "p=r/'bundle.tar.gz';p.write_bytes(b);t=tarfile.open(p);"
        "assert all(not pathlib.Path(m.name).is_absolute() and '..' not in pathlib.Path(m.name).parts "
        "and m.isfile() for m in t.getmembers());"
        "d=r/'bundle';d.mkdir();t.extractall(d);t.close();"
        f"os.environ['BOUNDARY_PROBE_RESULT_PUT_URL']={put_url!r};"
        "raise SystemExit(subprocess.call(["
        f"{str(AFS / 'runtime_20261007_1306/envs/b1k-pi05/bin/python')!r},"
        "str(d/'cloud_probe_worker.py'),'--bundle',str(d),'--output',str(r/'results')]))"
    )
    command = "timeout --signal=TERM --kill-after=60 10740 " + shlex.quote(
        str(AFS / "runtime_20261007_1306/envs/b1k-pi05/bin/python")) + " -u -c " + shlex.quote(bootstrap)
    state = json.loads(intent.read_text())
    state.update(submission_attempted=True, startup_sha256=hashlib.sha256(command.encode()).hexdigest(),
                 result_key=result_key, output_root=str(root))
    save(intent, state)
    argv = [CLI, "acp", "jobs", "copy", "--workspace-name=share-space", "--copy-job-name=" + TEMPLATE,
            "--name=" + JOB, "--job-name=" + JOB, "--worker-spec=n12lp.nn.i10a.1", "--worker-nodes=1",
            "--quota-type=spot", "--retry-times=0", "--enable-fault-tolerance=false",
            "--enable-anomaly-detection=false",
            "--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs",
            "--container-image-url=registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/"
            "wangfaye-base-20261010:wangfaye-debug-20261010154634",
            "--env=NVIDIA_DRIVER_CAPABILITIES:all", "--command=" + command]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as error:
        # TimeoutExpired includes argv, which contains temporary signed URLs.
        state.update(state="submission_response_requires_exact_name_query", error_type=type(error).__name__)
        save(intent, state)
        print(json.dumps({key: state.get(key) for key in ("job_name", "state", "error_type")}))
        return
    state["cli_returncode"] = completed.returncode
    if completed.returncode == 0:
        try:
            queried = subprocess.run([CLI, "acp", "jobs", "describe", JOB,
                                      "--workspace-name=share-space", "-o", "json"],
                                     capture_output=True, text=True, timeout=60)
            if queried.returncode:
                raise ValueError("Created job requires exact-name reconciliation")
            response = json.loads(queried.stdout)
            state.update({key: response.get(key) for key in ("name", "uid", "state")})
        except (ValueError, AttributeError, subprocess.TimeoutExpired):
            state["state"] = "submission_response_requires_exact_name_query"
    else:
        state["state"] = "submission_response_requires_exact_name_query"
        # Diagnostics are bounded and scrubbed before reaching the workspace.
        message = re.sub(r"https?://[^\s\"']+", "<redacted URL>", completed.stderr)
        state["cli_error"] = message[:3000]
    save(intent, state)
    print(json.dumps({key: state.get(key) for key in ("job_name", "name", "uid", "state", "cli_returncode")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / "pilot_three_tasks_cache_v1")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    bundle, receipt = build_bundle(args.cache.resolve())
    if args.submit:
        submit(bundle, receipt)
    else:
        print(json.dumps({key: receipt[key] for key in ("job_name", "bundle_sha256", "bundle_bytes", "gpu_count")}))
