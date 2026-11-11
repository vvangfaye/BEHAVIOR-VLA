"""Submit the selected finite trial once, using credentials only on the Mac.

Importing this file and running validate_local() are local, read-only operations.
main() is the only cloud submission entry point.
"""
import configparser
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGE = HERE / "selected_stage"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_json_value(current, recorded):
    return json.dumps(current, sort_keys=True) == json.dumps(recorded, sort_keys=True)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def validate_local():
    if not (STAGE / "selection_proof.json").exists():
        raise ValueError("Selected stage is absent; do not submit the unselected candidate")
    if any((STAGE / name).exists() for name in ("intent.json", "receipt.json", "runtime_receipt.json")):
        raise ValueError("Existing submission evidence: reconcile the exact run, never blindly resubmit")
    proof = read(STAGE / "selection_proof.json")
    if digest(HERE / "selection.json") != proof["selection_sha256"]:
        raise ValueError("Selection changed")
    gate = module("huber_reference_gate", ROOT / "prepare_selected_huber.py")
    expected_run, expected_delta = gate.VARIANTS[HERE.name]
    # Paired comparisons contain tuple task identities in memory; JSON stores
    # them as arrays. Compare the serialized values, not tuple/list identity.
    policy = proof.get("admission_policy", "complete_current_panel")
    if policy == "preregistered_training_ablation" and expected_delta != 50.0:
        raise ValueError("Independent ablation admission is only valid for the selected Huber 50 trial")
    if not same_json_value(gate.validate_admission(policy), proof["evaluation"]):
        raise ValueError("Reference evaluation proof changed")
    bundle = STAGE / "bundle"
    if digest(bundle / "bundle_manifest.json") != proof["selected_manifest_sha256"]:
        raise ValueError("Selected bundle identity changed")
    manifest = read(bundle / "bundle_manifest.json")
    for name, expected in manifest.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or digest(bundle / path) != expected:
            raise ValueError("Selected bundle file changed: " + name)
    job = read(bundle / "nebula/worker_job.json")
    auth = read(bundle / "inputs/formal_authorization.json")
    argv = read(bundle / "inputs/compiled/training_argv.json")
    if (job["run_id"] != expected_run or job["resume"]
            or job["max_steps"] != 1000 or job["stage_end_step"] != 1000
            or argv.get("launch_authorized") is not True
            or auth.get("action_loss") != {"schema_version": 1, "kind": "scaled_huber", "delta": expected_delta}
            or proof["maximum_runtime_seconds"] != 21600 or proof["maximum_gpu_hours"] != 96):
        raise ValueError("Finite Huber experiment contract changed")
    sys.path.insert(0, str(bundle))
    worker = module("selected_huber_worker", bundle / "nebula/train_four_source.py")
    worker.validate(job, auth, {})
    return bundle, job


def load_credentials():
    if platform.system() != "Darwin":
        raise ValueError("AIHub submission must run on the configured Mac")
    values = {}
    for line in (Path.home() / ".ai-hub-cli/config.yaml").read_text().splitlines():
        match = re.fullmatch(r"(namespace|token):\s*(\S+)\s*", line)
        if match:
            values[match[1]] = match[2].strip("\"'")
    if set(values) != {"namespace", "token"}:
        raise ValueError("Unexpected AIHub configuration layout")
    parser = configparser.ConfigParser()
    parser.read(Path.home() / ".ossutilconfig")
    source = parser["Credentials"]
    os.environ.update(AI_HUB_NAMESPACE=values["namespace"], AI_HUB_TOKEN=values["token"],
        OSS_ACCESS_ID=source["accesskeyid"], OSS_ACCESS_KEY=source["accesskeysecret"],
        OSS_ENDPOINT=source["endpoint"], OSS_BUCKET="traffic-model")


def freeze_exact(folder):
    """Archive the already validated manifest without silently regenerating it."""
    bundle = folder / "bundle"
    manifest = read(bundle / "bundle_manifest.json")
    archive = folder / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in sorted([*manifest, "bundle_manifest.json"]):
            path = bundle / name
            if name in manifest and digest(path) != manifest[name]:
                raise ValueError("Bundle changed before packaging: " + name)
            output.add(path, arcname=name, recursive=False)
    return archive


def main():
    # Check the host before reading credentials or creating submission evidence.
    if platform.system() != "Darwin":
        raise ValueError("AIHub submission must run on the configured Mac")
    if not STAGE.exists():
        raise ValueError("Complete reference evaluation and select the trial first")
    with (STAGE / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bundle, job = validate_local()
        load_credentials()
        dispatch = module("selected_huber_dispatch", ROOT / "expert_control/dispatch_support.py")
        from nebula.submit_oss import OSSClient
        client = OSSClient()
        if hashlib.sha256(client.read(job["data_manifest_key"])).hexdigest() != job["data_manifest_sha256"]:
            raise ValueError("Accepted source data manifest changed")
        guard = module("selected_huber_checkpoint_guard", ROOT / "expert_control/submit_two_source_smoke.py")
        dispatch.save(STAGE / "checkpoint_source_preflight.json", guard.verify_checkpoint_source(client, bundle, job))
        inventory = module("selected_huber_inventory_guard", ROOT / "expert_control/verify_source_inventory.py")
        dispatch.save(STAGE / "original_source_preflight.json",
                      inventory.verify_original_objects(client, read(bundle / "inputs/original_source_objects.json")))
        for field in ("environment", "cache"):
            head = client.bucket.head_object(job[field]["key"])
            if head.content_length != job[field]["bytes"] or head.etag != job[field]["etag"]:
                raise ValueError("Frozen runtime object changed: " + field)
        dispatch.HERE = STAGE
        dispatch.freeze = freeze_exact
        delta = read(bundle / "inputs/formal_run_contract.json")["action_loss"]["delta"]
        dispatch.create(client, "expert_huber" + str(int(delta)) + "_formal", STAGE, "nebula/train_four_source.py",
                        bundle / "nebula/training_cluster.json", job["output"], 2, 16, 21600,
                        job["run_id"] + "-" + job["attempt_id"])


if __name__ == "__main__":
    main()
