"""Authorized 16-GPU four-source fine-tuning, with bounded scientific stages."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("training_worker_helpers", REPO / "nebula/four_source_smoke.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def validate(job, authorization, smoke):
    if (job.get("mode") != "formal_finetune" or job.get("formal_training_authorized") is not True
            or authorization.get("formal_training_allowed") is not True):
        raise ValueError("Explicit formal training authorization required")
    if job["process_count"] != 2 or authorization["workers"] != 2 or authorization["gpus_per_worker"] != 8:
        raise ValueError("Formal job must use exactly sixteen GPUs on two workers")
    if (not 2 < job["stage_end_step"] <= authorization["max_completed_updates"] <= 20000
            or job["max_steps"] != authorization["max_completed_updates"]):
        raise ValueError("Formal update budget differs from authorization")
    direct_target = (
        authorization.get("execution_policy") == "direct_to_authorized_target"
        and authorization.get("direct_target_completed_updates") == job["stage_end_step"]
        and authorization.get("first_formal_stage_completed_updates") == job["stage_end_step"]
    )
    if job["stage_end_step"] > 5000 and not job.get("validation_gate_passed") and not direct_target:
        raise ValueError("Continuation beyond5000 requires scientific validation or explicit direct-target authorization")
    if not job.get("cache_training_original_files") or job.get("cache_smoke_original_files"):
        raise ValueError("Formal training requires all frozen original data locally available")
    if smoke.get("completed_updates") != 2 or not smoke.get("checkpoint_restore") or smoke.get("gpus") != 16:
        raise ValueError("Actual sixteen-GPU smoke and restore proof missing")


def execute():
    for name, digest in helpers.read(REPO / "bundle_manifest.json").items():
        if helpers.sha(REPO / name) != digest:
            raise ValueError("Frozen formal bundle changed: " + name)
    job = helpers.configuration()
    authorization = helpers.read(REPO / "inputs/formal_authorization.json")
    smoke = helpers.read(REPO / "inputs/gpu_smoke_success.json")
    validate(job, authorization, smoke)
    base = Path(job["output"])
    attempt = base / "attempts" / job["attempt_id"]
    attempt.mkdir(parents=True, exist_ok=True)
    rank, coordinator = helpers.register_rank(attempt, 2)
    output = attempt / ("rank-" + str(rank))
    output.mkdir(exist_ok=False)
    scratch = Path("/var/tmp/wangfaye/four-source-training") / job["run_id"] / job["attempt_id"] / ("rank-" + str(rank))
    scratch.mkdir(parents=True, exist_ok=False)
    helpers.write(output / "started.json", dict(run_id=job["run_id"], attempt_id=job["attempt_id"],
        rank=rank, formal_training=True, stage_end_step=job["stage_end_step"], started=time.time()))
    try:
        python, env, checkpoint = helpers.bootstrap_worker(job, rank, 2, base, scratch, output)
        compiled = REPO / "inputs/compiled"
        argv = helpers.read(compiled / "training_argv.json")["argv"]
        replacements = {"--exp-name": job["run_id"], "--num-train-steps": str(job["stage_end_step"]),
            "--save-interval": "1000", "--keep-period": "5000", "--log-interval": "10",
            "--data.subtask-mixture-path": str(compiled / "mixture.json"),
            "--data.subtask-mixture-roots-path": str(scratch / "datasets/roots.json"),
            "--learning-rate-contract-path": str(compiled / "learning_rate_contract.json"),
            "--data.assets.assets-dir": str(checkpoint / "assets"),
            "--weight-loader.params-path": str(checkpoint / "params")}
        for key, value in replacements.items():
            argv[argv.index(key) + 1] = value
        argv += ["--checkpoint-base-dir", str(base / "checkpoints"), "--assets-base-dir", str(scratch / "assets"),
                 "--fsdp-devices", "8", "--num-workers", "8", "--no-wandb-enabled"]
        if job.get("resume"):
            argv += ["--resume"]
        env.update(OPENPI_PROCESS_COUNT="2", OPENPI_PROCESS_ID=str(rank),
            OPENPI_COORDINATOR_ADDRESS=coordinator, OPENPI_LOCAL_DEVICE_IDS="0,1,2,3,4,5,6,7",
            OPENPI_EXPECTED_GLOBAL_DEVICES="16", OPENPI_DISTRIBUTED_TIMEOUT="1200",
            OPENPI_REQUIRE_FINITE_METRICS="1", OPENPI_VERIFY_FINAL_CHECKPOINT="1",
            OPENPI_FORMAL_RUN_CONTRACT=str(REPO / "inputs/formal_run_contract.json"),
            OPENPI_FORMAL_METRICS=str(attempt / "training_metrics.jsonl"),
            OPENPI_CHECKPOINT_LOCAL_READBACK="/var/tmp/wangfaye/four-source-readback/" + job["run_id"] + "/" + job["attempt_id"],
            OPENPI_OSSUTIL_PATH=str(REPO / "nebula/ossutil"),
            OPENPI_OSSUTIL_SHA256=job["ossutil_sha256"], OPENPI_DOWNLOAD_ENDPOINT=job["download_endpoint"],
            OPENPI_SUCCESS_PATH=str(attempt / "checkpoint_restore_ok.json"))
        helpers.write(output / "effective_training.json", dict(argv=argv, global_batch=192, gpus=16,
            processes=2, rank=rank, stage_end_step=job["stage_end_step"], formal_training=True,
            learning_rate_decay_end=20000, fresh_from1688977=not job.get("resume")))
        helpers.log("Starting authorized formal fine-tuning to completed update " + str(job["stage_end_step"]))
        with (output / "trainer.log").open("w") as stream:
            helpers.run([python, REPO / "openpi/scripts/b1k/train_b1k.py", *argv], env=env,
                        stdout=stream, stderr=helpers.subprocess.STDOUT,
                        timeout=max(86400, job["stage_end_step"] * 30 + 3600))
        deadline = time.monotonic() + 300
        while not (attempt / "checkpoint_restore_ok.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Formal checkpoint restore result missing")
            time.sleep(2)
        proof = helpers.read(attempt / "checkpoint_restore_ok.json")
        if (proof.get("restored_step") != job["stage_end_step"] or proof.get("global_devices") != 16
                or proof.get("all_state_leaves_exact") is not True or proof.get("optimizer_clock") != job["stage_end_step"]):
            raise ValueError("Formal final checkpoint failed progress/topology check")
        helpers.write(output / "_RANK_SUCCESS.json", dict(rank=rank, completed_updates=job["stage_end_step"]))
        deadline = time.monotonic() + 300
        while len(list(attempt.glob("rank-*/_RANK_SUCCESS.json"))) != 2:
            if list(attempt.glob("rank-*/failure.json")):
                raise RuntimeError("Peer formal worker failed")
            if time.monotonic() > deadline:
                raise TimeoutError("Peer formal worker completion not proven")
            time.sleep(2)
        if rank == 0:
            helpers.write(attempt / "_TRAINING_STAGE_SUCCESS.json", dict(run_id=job["run_id"],
                completed_updates=job["stage_end_step"], gpus=16, processes=2, formal_training=True,
                checkpoint_restore=True, all_state_leaves_exact=True,
                next_phase=("training_budget_complete" if job["stage_end_step"] == job["max_steps"]
                            else "heldout_evaluation_before_further_updates"), finished=time.time()))
    except BaseException as exc:
        helpers.write(output / "failure.json", dict(error_type=type(exc).__name__, message=str(exc)[-2000:],
            formal_training=True, observed_at=time.time()))
        raise


if __name__ == "__main__":
    execute()
