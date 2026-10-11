#!/usr/bin/env python3
"""Prepare, inspect and submit a frozen two-node H200 subtask training job.

Preparation is local and requires no cloud SDK, credentials or mounted dataset.
Large datasets, the verified checkpoint and the Python environment must already
be available at the explicitly configured worker paths. This is not a transfer
tool, and deliberately does not source the older full100 launcher defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
KIND = "b1k_subtask_mainline_nebula"


class MainlineError(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def check(value, message):
    if not value:
        raise MainlineError(message)


def safe_relative(value):
    path = PurePosixPath(value)
    check(value and not path.is_absolute() and ".." not in path.parts, "Unsafe bundle-relative path")
    return path


def absolute(value, label):
    check(isinstance(value, str) and value.startswith("/") and not any(c in value for c in "\n\r\0"),
          label + " must be an explicit absolute worker path, not an unstaged oss:// or AOSS URI")
    return value


def hash_value(value, label):
    check(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Missing SHA256 evidence: " + label)
    return value


def inventory(root):
    root = Path(root)
    check(not any(p.is_symlink() for p in root.rglob("*")), "Bundle contains a symlink")
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def mixture_module(repo=REPO):
    """Load the shared stdlib-only sampler without importing JAX or Torch."""
    spec = importlib.util.spec_from_file_location("mainline_mixture_contract", Path(repo) / "openpi/src/openpi/training/subtask_mixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_config(config, mixture):
    check(config.get("schema_version") == 1, "Unsupported job config schema")
    check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", config.get("run_id", "")), "Invalid fresh experiment run_id")
    check(type(config.get("resume", False)) is bool, "resume must be a JSON boolean")
    for key in ("runtime_python", "output_root", "openpi_data_home"):
        absolute(config.get(key), key)
    hash_value(config.get("tokenizer_sha256"), "staged PaliGemma tokenizer")
    resources = config.get("resources", {})
    for key in ("queue", "gpu_card_type", "algo_name"):
        check(isinstance(resources.get(key), str) and resources[key].strip(), "Set resources." + key + " explicitly")
    # Queue/card aliases are scheduling metadata, never evidence of actual H200 hardware.
    for key in ("cpu", "memory"):
        check(type(resources.get(key)) is int and resources[key] > 0, "Set per-worker resources." + key)
    check(config.get("process_count", 2) == 2 and config.get("local_devices", 8) == 8
          and config.get("fsdp_devices", 8) == 8, "This launcher requires 2 hosts x 8 H200, FSDP=8")
    for key in ("train_steps", "save_interval", "global_batch_size"):
        check(type(config.get(key)) is int and config[key] > 0, key + " must be a positive integer")
    check(config["global_batch_size"] % 16 == 0, "Global microbatch must be divisible by 16")
    check(config["save_interval"] <= config["train_steps"], "Save interval exceeds the finite step budget")
    check(type(config.get("keep_period", config["save_interval"] * 5)) is int
          and config.get("keep_period", config["save_interval"] * 5) >= config["save_interval"], "keep_period must be at least save_interval")
    check(type(config.get("num_workers", 4)) is int and config.get("num_workers", 4) >= 0, "Invalid loader worker count")
    check(type(config.get("seed", 42)) is int, "seed must be an integer")
    check(type(config.get("rendezvous_timeout", 600)) is int and 1 <= config.get("rendezvous_timeout", 600) <= 3600,
          "rendezvous_timeout must be 1..3600 seconds")
    check(type(config.get("coordinator_port", 12355)) is int and 1024 <= config.get("coordinator_port", 12355) <= 65535,
          "Invalid coordinator port")
    check(config.get("mode", "train") in ("preflight", "train"), "Mode must be preflight or train")
    checkpoint = config.get("checkpoint", {})
    absolute(checkpoint.get("root"), "checkpoint.root")
    check(checkpoint.get("step") == 1375000, "Initialize this mainline from verified checkpoint 1375000")
    hash_value(checkpoint.get("manifest_sha256"), "checkpoint manifest")
    hash_value(checkpoint.get("norm_sha256"), "base normalization")
    norm = str(safe_relative(checkpoint.get("norm_relative_path", "")))
    check(norm.startswith("assets/") and norm.endswith("/norm_stats.json"), "Use the baseline checkpoint's normalization asset")
    learning = config.get("learning_rate", {})
    for key in ("peak", "floor"):
        value = learning.get(key)
        check(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0,
              "Set a finite positive learning_rate." + key)
    check(learning["floor"] <= learning["peak"], "Learning-rate floor exceeds peak")
    check(type(learning.get("warmup_steps")) is int and 0 <= learning["warmup_steps"] < config["train_steps"],
          "Warmup must be shorter than the finite step budget")
    check(type(learning.get("decay_steps")) is int and learning["decay_steps"] > learning["warmup_steps"],
          "Set fixed learning_rate.decay_steps independently from this attempt's train_steps")
    check(mixture.get("schema_version") == 1 and mixture.get("kind") == "b1k_three_source_mixture", "Invalid mixture manifest")
    check(mixture.get("global_batch_size") == config["global_batch_size"], "Manifest/global batch mismatch")
    check(mixture.get("action_horizon") == 32, "Baseline action horizon must stay 32")
    check(mixture.get("seed") == config.get("seed", 42), "Training and mixture seeds must agree")
    phases = mixture.get("phases", [])
    end = 0
    check(bool(phases), "Mixture has no finite phases")
    for phase in phases:
        check(phase.get("start_step") == end and type(phase.get("end_step")) is int and phase["end_step"] > end,
              "Mixture phases must form contiguous finite optimizer-step intervals")
        end = phase["end_step"]
        sources = phase.get("sources", [])
        check(bool(sources) and len({s.get("kind") for s in sources}) == len(sources)
              and all(s.get("kind") in {"recovered", "collected", "replay"} for s in sources), "Missing, unknown or duplicate mixture sources")
        check(all(isinstance(s.get("weight"), (int, float)) and not isinstance(s["weight"], bool)
                  and math.isfinite(s["weight"]) and s["weight"] >= 0 for s in sources), "Invalid source weights")
        check(math.isclose(sum(s["weight"] for s in sources), 1.0, abs_tol=1e-8), "Source weights must sum to one")
        check(all(s.get("indexes") for s in sources if s["weight"] > 0), "A positive source needs frozen indexes")
    check(end == config["train_steps"], "Phase end must equal the explicit train_steps; no inherited epoch budget")
    roots = config.get("data_roots", {})
    check(isinstance(roots, dict) and roots, "Provide data_roots mapping root_id to staged worker paths")
    for root_id, path in roots.items():
        check(isinstance(root_id, str) and root_id, "Invalid dataset root_id")
        absolute(path, "data_roots[" + root_id + "]")


def source_files(repo):
    paths = []
    for area in ("openpi/src", "openpi/packages/openpi-client/src"):
        paths.extend(p for p in (repo / area).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix in (".py", ".json", ".yaml", ".yml", ".typed"))
    paths.extend(repo / name for name in ("openpi/scripts/b1k/train_b1k.py", "nebula/subtask_mainline.py", "nebula/train_subtask_mainline.py"))
    check(all(p.is_file() and not p.is_symlink() for p in paths), "Missing or symlinked mainline training source")
    return sorted(set(paths))


def prepare(config_path, mixture_path, destination, *, repo=REPO):
    config, mixture = read(config_path), read(mixture_path)
    validate_config(config, mixture)
    semantic_plan = mixture_module(repo).MixturePlan.load(mixture_path)
    check(set(config["data_roots"]) == set(semantic_plan.roots), "data_roots must map exactly the mixture's root IDs")
    destination, repo = Path(destination), Path(repo)
    check(not destination.exists(), "Use a fresh prepared directory; prior submission evidence must be preserved")
    original_mixture = json.loads(json.dumps(mixture))
    inputs = {}
    for phase in mixture["phases"]:
        for source in phase["sources"]:
            for index in source["indexes"]:
                expected = hash_value(index.get("sha256"), "valid interval index")
                path = Path(index["path"])
                if not path.is_absolute():
                    path = Path(mixture_path).resolve().parent / path
                check(path.is_file() and sha(path) == expected, "Index missing or changed: " + str(path))
                content = read(path)
                check(content.get("root_id") in config["data_roots"], "Missing worker mapping for root_id " + str(content.get("root_id")))
                check(content.get("episodes") and content.get("intervals"), "Empty interval index")
                check(content.get("provenance", {}).get("source_files"), "Index lacks source metadata/label evidence")
                for evidence in content["provenance"]["source_files"]:
                    hash_value(evidence.get("sha256"), "index source evidence")
                inputs[expected] = path
                index["path"] = "indexes/" + expected + ".json"
    files = source_files(repo)
    source_identity = {p.relative_to(repo).as_posix(): sha(p) for p in files}
    # Physical deployment paths and attempt identities are excluded. Data indexes
    # retain their exact original bytes; their hashes bind the logical contract.
    logical_mixture = json.loads(json.dumps(original_mixture))
    for phase in logical_mixture["phases"]:
        for source in phase["sources"]:
            for index in source["indexes"]:
                index.pop("path", None)
    contract_config = {k: v for k, v in config.items() if k not in
                       {"resume", "mode", "runtime_python", "openpi_data_home", "data_roots", "output_root", "resources", "checkpoint", "rendezvous_timeout", "coordinator_port"}}
    contract_config["checkpoint"] = {k: v for k, v in config["checkpoint"].items() if k != "root"}
    optimizer_config = {k: v for k, v in contract_config.items() if k not in {"train_steps", "save_interval", "keep_period", "num_workers"}}
    optimizer_contract = {"config": optimizer_config, "source_files": source_identity,
                          "conditioning": "task_and_subtask", "action_horizon": 32, "max_token_len": 512,
                          "process_count": 2, "local_devices": 8, "fsdp_devices": 8}
    contract = {"config": contract_config, "mixture": logical_mixture, "source_files": source_identity,
                "conditioning": "task_and_subtask", "action_horizon": 32, "max_token_len": 512,
                "process_count": 2, "local_devices": 8, "fsdp_devices": 8}
    plan = {"schema_version": 1, "kind": KIND, "config": config, "attempt_id": uuid.uuid4().hex,
            "training_contract": contract, "training_contract_sha256": digest(contract),
            "optimizer_contract": optimizer_contract, "optimizer_contract_sha256": digest(optimizer_contract),
            "input_mixture_sha256": sha(mixture_path), "bundled_mixture_sha256": digest(mixture)}
    payload = destination / "bundle"
    payload.mkdir(parents=True)
    for path in files:
        target = payload / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for expected, path in inputs.items():
        target = payload / "inputs/indexes" / (expected + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        check(sha(target) == expected, "Index changed during preparation")
    write(payload / "inputs/mixture.json", mixture)
    write(payload / "inputs/roots.json", config["data_roots"])
    write(payload / "nebula/mainline_job.json", plan)
    write(payload / "nebula/mainline_cluster.json", {"worker": {
        "cpu": config["resources"]["cpu"], "gpu": 800, "memory": config["resources"]["memory"]}})
    manifest = inventory(payload)
    write(payload / "bundle_manifest.json", manifest)
    write(destination / "bundle_manifest.json", manifest)
    write(destination / "review.json", {"kind": KIND, "run_id": config["run_id"], "attempt_id": plan["attempt_id"],
          "training_contract_sha256": plan["training_contract_sha256"], "resources": config["resources"],
          "workers": 2, "gpus_per_worker": 8, "fsdp_devices": 8, "steps": config["train_steps"],
          "global_batch_size": config["global_batch_size"], "resume": config.get("resume", False),
          "large_inputs_staged_by_this_tool": False, "cloud_submitted": False,
          "required_worker_inputs": {"checkpoint": config["checkpoint"]["root"], "runtime_python": config["runtime_python"],
                                     "openpi_data_home": config["openpi_data_home"], "data_roots": config["data_roots"]}})
    return payload


def verify_bundle(payload):
    payload = Path(payload)
    expected = read(payload / "bundle_manifest.json")
    actual = inventory(payload)
    actual.pop("bundle_manifest.json", None)
    check(actual == expected, "Frozen bundle changed; create a new reviewed preparation")
    plan = read(payload / "nebula/mainline_job.json")
    check(plan["kind"] == KIND and digest(plan["training_contract"]) == plan["training_contract_sha256"], "Changed training contract")
    check(digest(plan["optimizer_contract"]) == plan["optimizer_contract_sha256"], "Changed optimizer contract")
    validate_config(plan["config"], read(payload / "inputs/mixture.json"))
    return plan


def submission_command(plan, environ):
    required = ("AI_HUB_TOKEN", "AI_HUB_NAMESPACE", "AIHUB_MODEL_NAME", "OSS_ACCESS_ID", "OSS_ACCESS_KEY", "OSS_BUCKET", "OSS_ENDPOINT")
    check(all(environ.get(k) for k in required), "Actual submission requires explicit AI-Hub/OSS credential environment")
    check(all(not any(c in environ[k] for c in "\n\r\0") for k in required), "Invalid multiline credential/configuration value")
    check(all("," not in environ[k] for k in ("OSS_ACCESS_ID", "OSS_ACCESS_KEY")), "OSS credentials contain unsupported env delimiters")
    config, resources = plan["config"], plan["config"]["resources"]
    return [environ.get("AI_HUB_CLI", "ai-hub-cli"), "train", "mdl",
            "--token=" + environ["AI_HUB_TOKEN"], "--namespace=" + environ["AI_HUB_NAMESPACE"],
            "--name=" + environ["AIHUB_MODEL_NAME"], "--queue=" + resources["queue"],
            "--entry=nebula/train_subtask_mainline.py", "--worker_count=2", "--file.cluster_file=nebula/mainline_cluster.json",
            "--oss_access_id=" + environ["OSS_ACCESS_ID"], "--oss_access_key=" + environ["OSS_ACCESS_KEY"],
            "--oss_bucket=" + environ["OSS_BUCKET"], "--oss_endpoint=" + environ["OSS_ENDPOINT"],
            "--job_name=" + config["run_id"] + "-" + plan["attempt_id"][:8], "--algo_name=" + resources["algo_name"],
            "--gpu_card_type=" + resources["gpu_card_type"], "--oss_appendable=true", "--max_failover_times=0",
            "--debug_mode", "--ignore=__pycache__,*.pyc,*.pyo"]


def submit(prepared, *, dry_run=False, environ=None, runner=subprocess.run):
    prepared = Path(prepared)
    plan = verify_bundle(prepared / "bundle")
    if dry_run:
        return {"cloud_submitted": False, "dry_run": True, "run_id": plan["config"]["run_id"],
                "training_contract_sha256": plan["training_contract_sha256"]}
    environ = dict(os.environ if environ is None else environ)
    command = submission_command(plan, environ)
    intent = prepared / "submission_intent.json"
    write(intent, {"state": "submission_intent", "attempt_id": plan["attempt_id"],
          "training_contract_sha256": plan["training_contract_sha256"], "time": time.time(), "automatic_resubmit": False}, exclusive=True)
    receipt = {"state": "unknown", "run_id": plan["config"]["run_id"], "attempt_id": plan["attempt_id"],
               "training_contract_sha256": plan["training_contract_sha256"], "automatic_resubmit": False}
    try:
        result = runner(command, cwd=prepared / "bundle", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        output = result.stdout or ""
        for key in ("AI_HUB_TOKEN", "OSS_ACCESS_ID", "OSS_ACCESS_KEY"):
            output = output.replace(environ[key], "[REDACTED]")
        (prepared / "submission.log").write_text(output)
        receipt.update(exit_code=result.returncode, state="submitted" if result.returncode == 0 else "failed_or_unknown")
    except Exception as error:
        # Some CLI exceptions include argv, which contains credentials. Preserve
        # an unknown intent and a redacted diagnostic, never a raw traceback.
        message = str(error)
        for key in ("AI_HUB_TOKEN", "OSS_ACCESS_ID", "OSS_ACCESS_KEY"):
            message = message.replace(environ[key], "[REDACTED]")
        receipt.update(error_type=type(error).__name__, error=message)
        raise MainlineError("Submission outcome unknown; reconcile the existing intent: " + message) from None
    finally:
        write(prepared / "submission_receipt.json", receipt, exclusive=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    for name in ("config", "mixture", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    launch = subs.add_parser("submit")
    launch.add_argument("--prepared", type=Path, required=True)
    launch.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"bundle": str(prepare(args.config, args.mixture, args.output)), "cloud_submitted": False}))
    else:
        print(json.dumps(submit(args.prepared, dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
