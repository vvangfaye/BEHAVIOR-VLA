"""One retry after the old worker is terminal; reuse verified AFS weights."""

import base64
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "cloud_smoke"
JOB = "b1k-ppo5-1023-p01"
PREFIX = "behavior1k/rlinf-multiround-20261023/" + JOB + "/"
BASE = "/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01"
REMOTE = "/mnt/afs/260010165/behavior_1k/rlinf_multiround_20261023/pilot01"
CLI = str(Path.home() / ".local/bin/sco")
IMAGE = "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634"
FILES = ["common.py", "simulator.py", "helpers.py", "training.py", "experiment.py", "driver.py", "plan.json"] + [str(path.relative_to(HERE)) for path in sorted((HERE / "b1k_rlinf").glob("*.py"))]



def write(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def describe(name):
    result = subprocess.run(
        [
            CLI,
            "acp",
            "jobs",
            "describe",
            name,
            "--workspace-name=share-space",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    return json.loads(result.stdout)


def main():
    assert len(sys.argv) == 1
    accepted = json.loads((HERE.parent / "cloud_smoke_gpu_20261023/acceptance_report.json").read_text())
    assert accepted["passed"] and all(accepted["checks"].values())
    assert json.loads((HERE / "training_test_result.json").read_text())["passed"]
    assert json.loads((HERE / "local_preflight.json").read_text())["passed"]
    plan = json.loads((HERE / "plan.json").read_text())
    assert plan["gpu_count"] == 1 and plan["quota_type"] == "spot"
    assert plan["rounds"] == 5 and plan["train_steps_per_round"] == 3224
    assert plan["eval_instances"] == [301,302] and plan["eval_seeds"] == [0,1,2]
    assert plan["max_job_seconds"] == 28800
    members = {}
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name in FILES:
            data = (HERE / name).read_bytes()
            member = tarfile.TarInfo("run/" + name)
            member.size = len(data)
            member.mtime = 0
            tar.addfile(member, io.BytesIO(data))
            members[name] = hashlib.sha256(data).hexdigest()
    bundle = archive.getvalue()
    (HERE / "bundle.tar.gz").write_bytes(bundle)
    oldart = json.loads((PREVIOUS / "artifacts.json").read_text())
    reuse = {
        "checkpoint/" + name: entry["sha256"]
        for name, entry in oldart["checkpoint"].items()
    }
    reuse["bundle.tar.gz"] = oldart["bundle"]["sha256"]
    oldmanifest = json.loads((PREVIOUS / "local_preflight.json").read_text())
    reuse.update(
        {
            k: v
            for k, v in oldmanifest["execution_source_sha256"].items()
            if k.startswith("adapter/")
        }
    )
    for rel in [
        "rlinf/algorithms/advantages.py",
        "rlinf/algorithms/losses.py",
        "rlinf/models/embodiment/openpi/tasks/rl.py",
    ]:
        reuse["upstream/" + rel] = hashlib.sha256(
            (HERE.parent / "upstream" / rel).read_bytes()
        ).hexdigest()
    state = {
        "job_name": JOB,
        "previous_job": "b1k-rlinf-1023-r07",
        "previous_uid": "cc2919cc6b5045d3ad07009a842762d7",
        "attempt": 1,
        "state": "preparing",
        "submission_attempted": False,
        "gpu_count": 1,
        "max_seconds": 28800,
        "bundle_bytes": len(bundle),
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "source_sha256": members,
        "changes": [
            "One RTX5090, 8h cap; 5 additional PPO rounds from the verified smoke checkpoint",
            "Full training episodes up to 3224 controls; persistent AdamW state, unchanged reward",
            "Original1688977 and final-round model each evaluated on instances301/302 and seeds0/1/2",
            "Freeze evaluation and final checkpoint selection before training",
        ],
        "frozen_plan": True,
        "reuse_existing_base_weights": True,
        "separate_result_checkpoint_uploads": True,
    }
    with (HERE / "submission_intent.json").open("x") as f:
        json.dump(state, f, indent=2)
    old = describe(state["previous_job"])
    assert old["uid"] == state["previous_uid"] and old["state"] in {"FAILED", "TERMINATED", "CANCELED", "STOPPED"}
    write(
        "previous_terminal.json",
        {k: old.get(k) for k in ["uid", "name", "state", "complete_time"]},
    )
    template = describe("b1k-repeat1688-1019-batch-00")
    assert template["ownership"]["user_name"] == "260010165"
    assert template["resource_pool"]["name"] == "computing-cluster-5090-01g"
    assert template["roles"][0]["image_path"] == IMAGE
    assert template["roles"][0]["resource_spec"][0]["name"] == "n12lp.nn.i10a.1"
    write(
        "template_verified.json",
        {
            k: template.get(k)
            for k in ["uid", "name", "state", "ownership", "resource_pool", "mount"]
        },
    )
    sys.path.insert(0, str(ROOT / "BEHAVIOR-VLAs/sco"))
    from video_eval_adapter import aoss_client

    client = aoss_client()
    try:
        key = PREFIX + "bundle-" + state["bundle_sha256"] + ".tar.gz"
        client.put_object(
            Bucket="260010165",
            Key=key,
            Body=bundle,
            Metadata={"sha256": state["bundle_sha256"]},
        )
        readback = client.get_object(Bucket="260010165", Key=key)["Body"].read(
            len(bundle) + 1
        )
        assert readback == bundle

        def signed(method, key):
            return client.generate_presigned_url(
                method, Params={"Bucket": "260010165", "Key": key}, ExpiresIn=604800
            )

        artifact_puts = {}
        for index in range(1, 6):
            for name in ("trainable.pt", "optimizer.pt"):
                relative = f"checkpoints/round_{index:02d}/{name}"
                key = PREFIX + relative
                artifact_puts[relative] = {"url": signed("put_object", key), "key": key}
        config = {
            "bundle_inline": base64.b64encode(bundle).decode(),
            "root": REMOTE,
            "base": BASE,
            "reuse_sha256": reuse,
            "bundle": {
                "bytes": len(bundle),
                "sha256": state["bundle_sha256"],
            },
            "status_put": signed("put_object", PREFIX + "status.json"),
            "result_put": signed("put_object", PREFIX + "result.tar.gz"),
            "artifact_puts": artifact_puts,
        }
        source = (
            (HERE / "bootstrap.py").read_text()
            + "\nguarded_run("
            + repr(config)
            + ")\n"
        )
        packed = base64.b64encode(zlib.compress(source.encode())).decode()
        command = (
            "timeout --signal=TERM --kill-after=60s 28740s python3 -u -c "
            + shlex.quote(
                "import base64,zlib;exec(zlib.decompress(base64.b64decode("
                + repr(packed)
                + ")))"
            )
        )
        assert len(command) < 64000
        state.update(
            state="submitting",
            submission_attempted=True,
            startup_sha256=hashlib.sha256(command.encode()).hexdigest(),
        )
        write("submission_intent.json", state)
        argv = [
            CLI,
            "acp",
            "jobs",
            "copy",
            "--workspace-name=share-space",
            "--copy-job-name=b1k-repeat1688-1019-batch-00",
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
            result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            state["submit_returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            state["submit_error"] = "timeout_outcome_unknown"
        try:
            actual = describe(JOB)
            assert actual["ownership"]["user_name"] == "260010165"
            assert (
                hashlib.sha256(
                    actual["roles"][0]["startup_script"].encode()
                ).hexdigest()
                == state["startup_sha256"]
            )
            state.update(uid=actual["uid"], state=actual["state"])
        except Exception as error:
            state.update(
                state="requires_exact_name_query", query_error=type(error).__name__
            )
        write("submission_intent.json", state)
        print(
            json.dumps(
                {
                    k: state.get(k)
                    for k in [
                        "job_name",
                        "uid",
                        "state",
                        "submit_returncode",
                        "bundle_bytes",
                    ]
                }
            )
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
