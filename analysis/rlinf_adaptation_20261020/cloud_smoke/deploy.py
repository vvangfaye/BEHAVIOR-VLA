"""Prepare immutable artifacts and make one reconciled Spot submission."""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
RUN = HERE.parent
JOB = "b1k-rlinf-1020-smoke01"
ROOT = "/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01"
BUCKET = "260010165"
PREFIX = "behavior1k/rlinf-adapter-20261020/" + JOB + "/"
TEMPLATE = "b1k-repeat1688-1019-batch-00"
TEMPLATE_UID = "168d5223752b415caf67bd27ed504a83"
IMAGE = "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634"
FILES = [
    "common.py",
    "simulator.py",
    "helpers.py",
    "rollout.py",
    "gate.py",
    "ppo.py",
    "driver.py",
]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def prepared():
    previous = json.loads(
        (
            PROJECT
            / "analysis/training_boundary_audit_20261019/snapshot_probe_20261020/plan.json"
        ).read_text()
    )
    plan = {
        "job_name": JOB,
        "upstream_commit": "91a302ec50ddacd1fdabc5859b85e3661d768236",
        "output": ROOT + "/results",
        "converted": ROOT + "/checkpoint",
        "code": previous["code"],
        "runtime": previous["runtime"],
        "original": previous["unused_checkpoint_path"],
        "tokenizer": previous["runtime"]
        + "/openpi-cache/big_vision/paligemma_tokenizer.model",
        "supplemental_library": previous["supplemental_library"]["root"],
        "supplemental_library_sha256": previous["supplemental_library"][
            "ready_manifest_sha256"
        ],
        "original_source_sha256": previous["source_sha256"],
        "original_metadata_sha256": "1a59e30e43f7af9b868e032aaaf3245a8701602207b08ddf1bde8edf7c1fd2dd",
        "task": "turning_on_radio",
        "seed": 0,
        "eval_instances": [301, 302],
        "eval_steps": 3224,
        "train_instance": 1,
        "ppo_steps": 256,
        "gate": {
            "minimum_probes": 8,
            "mean_action_error": 0.005,
            "max_action_error": 0.05,
            "initial_error": 0.0001,
            "score_gap": 0.1,
        },
        "gpu_count": 1,
        "quota_type": "spot",
        "max_job_seconds": 21600,
        "training_condition": "All frozen closed-loop and same-input checks pass.",
        "training_scope": "One on-policy batch, one epoch, original RLinf GAE/PPO functions; single process, no Ray/FSDP.",
        "rules_snapshot": "2026-09-11; online source inaccessible on 2026-10-20, conservative 6 GPU-hour bound.",
        "authorization": "Current user request plus research_autonomy_authorization_20261020.json",
    }
    path = HERE / "plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        write(path, plan)
    archive = HERE / "bundle.tar.gz"
    if not archive.exists():
        with tarfile.open(archive, "w:gz") as tar:
            for name in FILES + ["plan.json"]:
                tar.add(HERE / name, arcname="run/" + name, recursive=False)
            roots = [
                (
                    PROJECT / "BEHAVIOR-VLAs/rlinf_adapter/b1k_rlinf",
                    "adapter/b1k_rlinf",
                ),
                (RUN / "upstream", "upstream"),
                (RUN / "venv/lib/python3.11/site-packages", "site"),
            ]
            for source, prefix in roots:
                for file in sorted(source.rglob("*")):
                    if not file.is_file() or file.is_symlink():
                        continue
                    rel = file.relative_to(source)
                    if (
                        any(
                            x in {"__pycache__", ".git", ".pytest_cache", ".ruff_cache"}
                            for x in rel.parts
                        )
                        or file.suffix == ".pyc"
                    ):
                        continue
                    if prefix == "site" and rel.parts[0] in {
                        "ruff",
                        "pip",
                        "setuptools",
                        "pkg_resources",
                        "_distutils_hack",
                    }:
                        continue
                    tar.add(file, arcname=prefix + "/" + str(rel), recursive=False)
    checkpoint = {}
    for file in sorted((RUN / "checkpoint1688977_openpi").iterdir()):
        if file.is_file():
            checkpoint[file.name] = {
                "bytes": file.stat().st_size,
                "sha256": digest(file),
                "key": PREFIX + "checkpoint/" + file.name,
            }
    receipt = {
        "job_name": JOB,
        "workspace": "share-space",
        "pool": "computing-cluster-5090-01g",
        "worker_spec": "n12lp.nn.i10a.1",
        "gpu_count": 1,
        "max_job_seconds": 21600,
        "bundle": {
            "bytes": archive.stat().st_size,
            "sha256": digest(archive),
            "key": PREFIX + "bundle.tar.gz",
        },
        "checkpoint": checkpoint,
        "plan_sha256": digest(path),
    }
    write(HERE / "artifacts.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submit", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--resume-preparation", action="store_true")
    args = p.parse_args()
    artifacts = prepared()
    if not args.submit:
        print(
            json.dumps(
                {
                    "job_name": JOB,
                    "bundle_bytes": artifacts["bundle"]["bytes"],
                    "checkpoint_bytes": sum(
                        x["bytes"] for x in artifacts["checkpoint"].values()
                    ),
                }
            )
        )
        return
    intent = HERE / "submission_intent.json"
    if args.resume_preparation:
        saved = json.loads(intent.read_text())
        assert saved["submission_attempted"] is False and saved["artifacts"] == artifacts
        saved.setdefault("preparation_events", []).append({
            "reason": "Installed missing local boto3 before first cloud submission",
            "resumed_at": datetime.now(timezone.utc).isoformat(),
        })
        write(intent, saved)
    else:
        with intent.open("x") as f:
            json.dump(
                {
                    "job_name": JOB,
                    "state": "preparing",
                    "submission_attempted": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "artifacts": artifacts,
                },
                f,
                indent=2,
            )
    sys.path.insert(0, str(PROJECT / "BEHAVIOR-VLAs/sco"))
    from video_eval_adapter import aoss_client

    cli = str(Path.home() / ".local/bin/sco")
    response = subprocess.run(
        [
            cli,
            "acp",
            "jobs",
            "describe",
            TEMPLATE,
            "--workspace-name=share-space",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    template = json.loads(response.stdout)
    assert (
        template["uid"] == TEMPLATE_UID and template["ownership"]["user_name"] == BUCKET
    )
    assert template["resource_pool"]["name"] == "computing-cluster-5090-01g"
    assert (
        template["roles"][0]["image_path"] == IMAGE
        and template["roles"][0]["total_replicas"] == 1
    )
    assert template["roles"][0]["resource_spec"][0]["name"] == "n12lp.nn.i10a.1"
    # Never persist template startup URLs.
    write(
        HERE / "template_verified.json",
        {
            k: template[k]
            for k in ["uid", "name", "state", "ownership", "resource_pool", "mount"]
        },
    )
    state = json.loads(intent.read_text())
    client = aoss_client()
    try:
        entries = [(HERE / "bundle.tar.gz", artifacts["bundle"])]
        entries += [
            (RUN / "checkpoint1688977_openpi" / name, item)
            for name, item in artifacts["checkpoint"].items()
        ]
        for file, item in entries:
            print("Uploading " + file.name, flush=True)
            client.upload_file(
                str(file),
                BUCKET,
                item["key"],
                ExtraArgs={"Metadata": {"sha256": item["sha256"]}},
            )
            head = client.head_object(Bucket=BUCKET, Key=item["key"])
            assert (
                head["ContentLength"] == item["bytes"]
                and head["Metadata"]["sha256"] == item["sha256"]
            )

        def get_url(item):
            return {
                **{k: v for k, v in item.items() if k != "key"},
                "url": client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": BUCKET, "Key": item["key"]},
                    ExpiresIn=86400,
                ),
            }

        config = {
            "root": ROOT,
            "bundle": get_url(artifacts["bundle"]),
            "checkpoint": {
                name: get_url(item) for name, item in artifacts["checkpoint"].items()
            },
            "status_put": client.generate_presigned_url(
                "put_object",
                Params={"Bucket": BUCKET, "Key": PREFIX + "status.json"},
                ExpiresIn=86400,
            ),
            "result_put": client.generate_presigned_url(
                "put_object",
                Params={"Bucket": BUCKET, "Key": PREFIX + "result.tar.gz"},
                ExpiresIn=86400,
            ),
        }
        source = (HERE / "bootstrap.py").read_text() + "\nrun(" + repr(config) + ")\n"
        packed = base64.b64encode(zlib.compress(source.encode())).decode()
        wire = (
            "import base64,zlib;exec(zlib.decompress(base64.b64decode("
            + repr(packed)
            + ")))"
        )
        command = (
            "timeout --signal=TERM --kill-after=60s 21540s python3 -u -c "
            + shlex.quote(wire)
        )
        state.update(
            state="submitting",
            submission_attempted=True,
            startup_sha256=hashlib.sha256(command.encode()).hexdigest(),
        )
        write(intent, state)
        argv = [
            cli,
            "acp",
            "jobs",
            "copy",
            "--workspace-name=share-space",
            "--copy-job-name=" + TEMPLATE,
            "--name=" + JOB,
            "--job-name=" + JOB,
            "--worker-spec=n12lp.nn.i10a.1",
            "--worker-nodes=1",
            "--quota-type=spot",
            "--retry-times=0",
            "--enable-fault-tolerance=false",
            "--enable-anomaly-detection=false",
            "--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs",
            "--container-image-url=" + IMAGE,
            "--env=NVIDIA_DRIVER_CAPABILITIES:all",
            "--command=" + command,
        ]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            state["submit_returncode"] = r.returncode
            if r.returncode:
                state["submit_error_type"] = "CLI_NONZERO"
        except subprocess.TimeoutExpired:
            state["submit_error_type"] = "CLI_TIMEOUT_OUTCOME_UNKNOWN"
        # Reconcile exact name instead of resubmitting on uncertain outcomes.
        r = subprocess.run(
            [
                cli,
                "acp",
                "jobs",
                "describe",
                JOB,
                "--workspace-name=share-space",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if r.returncode == 0:
            job = json.loads(r.stdout)
            state.update(uid=job["uid"], state=job["state"])
            write(
                HERE / "job_status.json",
                {
                    k: job.get(k)
                    for k in [
                        "uid",
                        "name",
                        "state",
                        "resource_pool",
                        "create_time",
                        "start_time",
                        "complete_time",
                    ]
                },
            )
        else:
            state["state"] = "requires_exact_name_query"
        write(intent, state)
        print(
            json.dumps(
                {
                    k: state.get(k)
                    for k in ["job_name", "uid", "state", "submit_returncode"]
                }
            )
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
