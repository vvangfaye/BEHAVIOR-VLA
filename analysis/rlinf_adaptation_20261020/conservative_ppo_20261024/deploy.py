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
JOB = "b1k-ppoc2-1024-a01"
PREFIX = "behavior1k/rlinf-conservative-20261024/" + JOB + "/"
BASE = "/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01"
REMOTE = "/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/pilot01"
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
    afs = json.loads((HERE / "afs_preflight.json").read_text())
    assert all(len(afs['training_instances'][str(i)]) == 1 for i in [1,2,3,4])
    assert afs['free_bytes'] > 40*1024**3
    plan = json.loads((HERE / "plan.json").read_text())
    assert plan["gpu_count"] == 1
    assert plan["rounds"] == 2 and plan["train_steps_per_round"] == 3224
    assert plan["eval_instances"] == [301,302] and plan["eval_seeds"] == [0,1,2]
    assert plan["max_job_seconds"] == 28800
    from datetime import datetime, timezone
    import os
    sys.path.insert(0, str(ROOT / "BEHAVIOR-VLAs/sco"))
    from public_test_admission import query_standard_capacity, validate_standard_capacity
    capacity = query_standard_capacity(CLI, dict(os.environ))
    quota = "reserved" if capacity["used_gpus"] < capacity["standard_limit"] else "spot"
    plan["quota_type"] = quota
    plan["admission"] = {"observed_at":capacity["observed_at"],"used_reserved_gpus":capacity["used_gpus"],
                         "decision":quota,"policy":"Prefer unused reserved capacity, otherwise spot; never stop other jobs."}
    write("plan.json",plan)
    write("initial_capacity.json",capacity)
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
        "previous_job": "b1k-ppo5-1023-r01",
        "previous_uid": "0a97d630ec844e61b6bec78401f3bf6a",
        "attempt": 1,
        "state": "preparing",
        "submission_attempted": False,
        "gpu_count": 1,
        "quota_type": quota,
        "max_seconds": 28800,
        "bundle_bytes": len(bundle),
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "source_sha256": members,
        "changes": [
            "One RTX5090, 8h cap; prefer free reserved allocation and otherwise spot",
            "Recover only 3 missing old final evaluations, preserving all completed cases",
            "Reset original policy, warm up critic only using existing terminal training trajectories",
            "Two rounds, 4 distinct train instances per round, shuffled accumulation32, actor LR1e-7",
            "Whole-batch KL limit0.01 after each candidate update with exact rollback",
        ],
        "frozen_plan": True,
        "reuse_existing_base_weights": True,
        "separate_result_checkpoint_uploads": True,
    }
    with (HERE / "submission_intent.json").open("x") as f:
        json.dump(state, f, indent=2)
    old = describe(state["previous_job"])
    assert old["uid"] == state["previous_uid"] and old["state"] in {"FAILED", "TERMINATED", "CANCELED", "STOPPED", "SUSPENDED"}
    workers = subprocess.run([CLI,'acp','jobs','get-workers',state['previous_job'],'--workspace-name=share-space'],
                             capture_output=True,text=True,check=True,timeout=45)
    assert state['previous_uid'] in workers.stdout and 'Failed' in workers.stdout
    write('previous_worker_verified.json',{'uid':state['previous_uid'],'phase':'Failed','writer_stopped':True})
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
        for index in range(1, plan['rounds']+1):
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
            "--quota-type=" + quota,
            "--retry-times=0",
            "--enable-fault-tolerance=false",
            "--enable-anomaly-detection=false",
            "--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs",
            "--container-image-url=" + IMAGE,
            "--env=NVIDIA_DRIVER_CAPABILITIES:all",
            "--command=" + command,
        ]
        if quota == 'reserved':validate_standard_capacity(capacity, now=datetime.now(timezone.utc))
        assert (datetime.now(timezone.utc)-datetime.fromisoformat(capacity['observed_at'])).total_seconds()<60
        write("submission_admission.json", capacity)
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            state["submit_returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            state["submit_error"] = "timeout_outcome_unknown"
        try:
            actual = describe(JOB)
            assert actual["ownership"]["user_name"] == "260010165"
            assert actual["scheduling"]["quota_type"] == quota.upper()
            assert actual["resource_pool"]["name"] == "computing-cluster-5090-01g"
            assert len(actual["roles"]) == 1 and actual["roles"][0]["total_replicas"] == 1
            assert actual["roles"][0]["resource_spec"][0]["name"] == "n12lp.nn.i10a.1"
            assert int(actual["roles"][0]["resource_spec"][0]["limits"]["nvidia.com/gpu"]) == 1
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
