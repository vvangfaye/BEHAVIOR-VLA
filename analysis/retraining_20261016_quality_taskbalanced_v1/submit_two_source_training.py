"""Submit one authorized formal stage after real CPU/GPU acceptance proofs."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify_smoke(client, prefix):
    success = json.loads(client.read(prefix + "/_SMOKE_SUCCESS.json"))
    if success.get("completed_updates") != 2 or success.get("gpus") != 16 or not success.get("checkpoint_restore"):
        raise ValueError("Actual16GPU smoke proof invalid")
    state = json.loads(client.read(prefix + "/checkpoint_restore_ok.json"))
    if state.get("restored_step") != 2 or state.get("global_devices") != 16:
        raise ValueError("Smoke restored state/topology invalid")
    reports = [json.loads(x) for x in client.read(prefix + "/smoke_metrics.jsonl").decode().splitlines()]
    updates = [x for x in reports if x["kind"] == "optimizer_update"]
    restores = [x for x in reports if x["kind"] == "checkpoint_restore"]
    batches = [x for x in reports if x["kind"] == "real_two_source_batch"]
    if len(updates) != 2 or len(restores) != 1 or not restores[0].get("all_state_leaves_exact") or restores[0].get("optimizer_clock") != 2:
        raise ValueError("Smoke exact two-update/optimizer restore proof missing")
    if len(batches) != 2 or [x.get("step") for x in batches] != [0, 1]:
        raise ValueError("Smoke must contain exactly two real global batches")
    for batch in batches:
        counts = batch.get("source_counts", {})
        if (set(counts) != {"0", "3"} or min(counts.values()) <= 0
                or sum(counts.values()) != 192 or batch.get("actions_shape") != [192, 32, 32]
                or batch.get("full_episode_text_target_tokens") != 0
                or batch.get("supervised_text_target_tokens", 0) <= 0
                or len(batch.get("images", {})) != 3
                or any(shape != [192, 224, 224, 3] for shape in batch["images"].values())):
            raise ValueError("Actual GPU batch/source/text-mask proof invalid")
    fingerprints = {batch["mixture_fingerprint"] for batch in batches}
    if len(fingerprints) != 1 or restores[0].get("next_global_sample") != 384 or restores[0].get("third_update_executed") is not False:
        raise ValueError("Smoke sampler continuity or finite two-update bound invalid")
    # This experiment has a deterministic finite sample stream. Prove the new
    # global sampler was actually used, rather than only checking both sources
    # appeared in a batch that reported the expected manifest fingerprint.
    expected = json.loads(Path(__file__).with_name("metadata_sampling_validation.json").read_text())
    if not expected.get("passed") or fingerprints != {expected["mixture_fingerprint"]}:
        raise ValueError("Smoke differs from the exhaustive finite-sampler audit")
    if len(expected["first_two_batches"]) != 2:
        raise ValueError("Expected exactly two audited source-count records")
    for batch, counts in zip(batches, expected["first_two_batches"]):
        if batch["source_counts"] != {"0": counts["replay"], "3": counts["full_episode"]}:
            raise ValueError("Actual global batch source counts differ from the finite sampler")
    for i, record in enumerate(updates):
        metrics = record["metrics"]
        expected = (4.975124378109452e-9, 9.950248756218905e-9)[i]
        if (record.get("step") != i or not all(math.isfinite(v) for v in metrics.values())
                or metrics.get("lr_clock_matches_step") != 1
                or not math.isclose(metrics["learning_rate"], expected, rel_tol=2e-5)):
            raise ValueError("Applied GPU learning rate or metrics invalid")
    for rank in (0, 1):
        record = json.loads(client.read(prefix + f"/rank-{rank}/_RANK_SUCCESS.json"))
        if record.get("completed_updates") != 2:
            raise ValueError("Both smoke workers must finish")
    return dict(success=success, checkpoint_state=state, reports=reports, source_prefix=prefix,
                verified_mixture_fingerprint=next(iter(fingerprints)))


def verify_original_objects(client, source):
    expected = {x["key"]: x for x in source["objects"] if not x["relative_path"].startswith("meta/")}
    observed = {}
    marker = ""
    while True:
        page = client.bucket.list_objects(prefix=source["prefix"], marker=marker, max_keys=1000)
        for item in page.object_list:
            if item.key in expected:
                observed[item.key] = (item.size, item.etag.strip('"').lower())
        if not page.is_truncated:
            break
        if page.next_marker == marker:
            raise ValueError("Original OSS inventory pagination stalled")
        marker = page.next_marker
    if set(observed) != set(expected):
        raise ValueError("Original training object(s) missing")
    for key, item in expected.items():
        if observed[key] != (item["bytes"], item["etag"].strip('"').lower()):
            raise ValueError("Original training object identity changed: " + item["relative_path"])
    return dict(files=len(expected), bytes=sum(x["bytes"] for x in expected.values()), all_sizes_etags_verified=True)


def main(prepared):
    prepared = Path(prepared)
    bundle = prepared / "bundle"
    sys.path.insert(0, str(bundle))
    jobpath = bundle / "nebula/worker_job.json"
    job = json.loads(jobpath.read_text())
    if (prepared / "intent.json").exists():
        raise ValueError("Existing creation intent; reconcile exact task before any new submission")
    oss = module("formal_oss_client", bundle / "nebula/submit_oss.py")
    client = oss.OSSClient()
    smoke = verify_smoke(client, job["gpu_smoke_prefix"])
    if smoke["verified_mixture_fingerprint"] != job["mixture_fingerprint"]:
        raise ValueError("Accepted GPU smoke differs from this formal mixture")
    (prepared / "gpu_smoke_preflight.json").write_text(json.dumps(smoke, indent=2) + "\n")
    (bundle / "inputs/gpu_smoke_success.json").write_text(json.dumps(smoke["success"], indent=2) + "\n")
    worker = module("formal_worker_validation", bundle / "nebula/train_four_source.py")
    auth = json.loads((bundle / "inputs/formal_authorization.json").read_text())
    worker.validate(job, auth, smoke["success"])
    if hashlib.sha256(client.read(job["data_manifest_key"])).hexdigest() != job["data_manifest_sha256"]:
        raise ValueError("Accepted data manifest changed")
    original = verify_original_objects(client, json.loads((bundle / "inputs/original_source_objects.json").read_text()))
    (prepared / "original_source_preflight.json").write_text(json.dumps(original, indent=2) + "\n")
    # Reuse the reviewed source-weight check without importing any credentials into the bundle.
    helper = module("formal_source_guard", prepared / "submit_two_source_smoke.py")
    checkpoint = helper.verify_checkpoint_source(client, bundle, job)
    (prepared / "checkpoint_source_preflight.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    for field in ("environment", "cache"):
        record = job[field]
        h = client.bucket.head_object(record["key"])
        if h.etag != record["etag"] or h.content_length != record["bytes"]:
            raise ValueError("Frozen packed runtime changed")
    job["storage_endpoint"] = os.environ["OSS_ENDPOINT"]
    job["download_endpoint"] = os.environ["OSS_DOWNLOAD_ENDPOINT"]
    output = client.key(job["output"])
    if job.get("resume"):
        client.unused(output + "/attempts/" + job["attempt_id"])
        parent = output + "/checkpoints/pi05_b1k/" + job["run_id"] + "/" + str(job["resume_from_step"])
        if not client.head(parent + "/_CHECKPOINT_METADATA"):
            raise ValueError("Resume parent checkpoint is not committed")
        if json.loads(client.read(parent + "/assets/formal_run_contract.json")) != json.loads((bundle / "inputs/formal_run_contract.json").read_text()):
            raise ValueError("Resume parent formal training contract changed")
    else:
        client.unused(output)
    jobpath.write_text(json.dumps(job, indent=2) + "\n")
    manifest = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in bundle.rglob("*") if p.is_file() and p.name != "bundle_manifest.json"
                and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    (bundle / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    intent = dict(run_id=job["run_id"], attempt_id=job["attempt_id"], observed_at=time.time(),
        formal_training=True, stage_end_step=job["stage_end_step"], gpus=16, automatic_resubmit=False)
    with (prepared / "intent.json").open("x") as stream:
        json.dump(intent, stream, indent=2)
    client.put_new(output + "/attempts/" + job["attempt_id"] + ".claim.json", prepared / "intent.json")
    secret = [os.environ[k] for k in ("AI_HUB_TOKEN", "OSS_ACCESS_ID", "OSS_ACCESS_KEY")]
    args = ["ai-hub-cli", "train", "mdl", "--token=" + secret[0],
        "--namespace=" + os.environ["AI_HUB_NAMESPACE"], "--name=behavior", "--queue=demand_general",
        "--entry=nebula/train_four_source.py", "--worker_count=2", "--file.cluster_file=nebula/training_cluster.json",
        "--oss_access_id=" + secret[1], "--oss_access_key=" + secret[2], "--oss_bucket=" + client.bucket_name,
        "--oss_endpoint=" + os.environ["OSS_ENDPOINT"],
        "--env=OSS_ACCESS_KEY_ID=" + secret[1] + ",OSS_ACCESS_KEY_SECRET=" + secret[2],
        "--job_name=" + job["run_id"] + "-" + job["attempt_id"], "--algo_name=pytorch2120", "--gpu_card_type=l20z",
        "--oss_appendable=true", "--max_failover_times=0", "--debug_mode", "--ignore=__pycache__,*.pyc,*.pyo"]
    receipt = dict(**intent, state="unknown", output=job["output"],
        bundle_manifest_sha256=hashlib.sha256((bundle / "bundle_manifest.json").read_bytes()).hexdigest())
    try:
        result = subprocess.run(args, cwd=bundle, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        text = result.stdout
        for value in secret:
            text = text.replace(value, "[REDACTED]")
        (prepared / "submission.log").write_text(text)
        receipt.update(state="submitted" if result.returncode == 0 else "failed_or_unknown", exit_code=result.returncode,
            identifiers=sorted(set(re.findall(r"(?i)\b(?:iid|instanceId|instance_id|task_id)\W+([A-Za-z0-9_-]+)", text))))
    except BaseException as exc:
        receipt["error_type"] = type(exc).__name__
    finally:
        (prepared / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main(sys.argv[1])
