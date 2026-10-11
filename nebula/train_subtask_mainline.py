#!/usr/bin/env python3
"""Worker entry for one node of the explicit 2 x 8 H200 mainline job."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import ipaddress
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

from subtask_mainline import MainlineError, check, digest, mixture_module, read, safe_relative, sha, verify_bundle, write


def verify_gpus(output):
    rows = [line.split(",") for line in output.strip().splitlines() if line.strip()]
    check(len(rows) == 8, "This worker requires exactly 8 allocated H200 GPUs")
    check(all(len(row) == 2 and re.search(r"\bH200\b", row[0]) and int(row[1].strip()) >= 130000 for row in rows),
          "Actual GPU names/memory must identify H200; L20/L20Z aliases are not hardware evidence")
    return [{"name": row[0].strip(), "memory_mib": int(row[1].strip())} for row in rows]


def verify_tokenizer(config):
    path = Path(config["openpi_data_home"]) / "big_vision/paligemma_tokenizer.model"
    check(path.is_file() and sha(path) == config["tokenizer_sha256"], "PaliGemma tokenizer cache missing or changed; stage it before submission")
    return path


def verify_checkpoint(config):
    identity = config["checkpoint"]
    root = Path(identity["root"])
    manifest = root / "SHA256SUMS.inference"
    check(manifest.is_file() and sha(manifest) == identity["manifest_sha256"], "Baseline checkpoint manifest missing or changed")
    ready = read(root / "_READY")
    check(ready.get("step") == 1375000 and ready.get("manifest_sha256") == identity["manifest_sha256"], "Checkpoint is not verified step 1375000")
    entries = {}
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        check(match is not None, "Malformed checkpoint manifest")
        expected, name = match.groups()
        path = root / safe_relative(name)
        check(name not in entries and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
              "Missing, unsafe or duplicate checkpoint entry")
        check(sha(path) == expected, "Baseline checkpoint content changed: " + name)
        entries[name] = expected
    required = {"params/_METADATA", "_CHECKPOINT_METADATA", "assets/subtask_conditioning.json", identity["norm_relative_path"]}
    check(required <= entries.keys(), "Baseline manifest omits parameters, conditioning or normalization")
    check(not (root / "model.safetensors").exists(), "Mainline expects the verified JAX checkpoint")
    params = {p.relative_to(root).as_posix() for p in (root / "params").rglob("*") if p.is_file()}
    check(params and params <= entries.keys(), "Baseline parameter manifest is incomplete")
    check(read(root / "assets/subtask_conditioning.json").get("action_conditioning") == "task_and_subtask", "Baseline conditioning changed")
    norm = root / identity["norm_relative_path"]
    check(sha(norm) == identity["norm_sha256"], "Base normalization hash changed")
    stats = read(norm)["norm_stats"]
    import math
    for feature in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = stats[feature][field]
            check(len(values) == 23 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values),
                  "Baseline normalization must contain 23 finite transformed dimensions")
        check(all(v >= 0 for v in stats[feature]["std"]) and all(a <= b for a, b in zip(stats[feature]["q01"], stats[feature]["q99"])),
              "Invalid baseline normalization scale/quantiles")
    return norm


def verify_data(payload, config):
    mixture = read(payload / "inputs/mixture.json")
    mixture_module(payload).MixturePlan.load(payload / "inputs/mixture.json")
    identities = {}
    for phase in mixture["phases"]:
        for source in phase["sources"]:
            for entry in source.get("indexes", []):
                expected = entry["sha256"]
                if expected in identities:
                    continue
                path = payload / "inputs" / safe_relative(entry["path"])
                check(sha(path) == expected, "Frozen index changed")
                index = read(path)
                root = Path(config["data_roots"][index["root_id"]])
                check((root / "meta/info.json").is_file(), "Dataset is not staged: " + str(root))
                for source_file in index["provenance"]["source_files"]:
                    # External labels are bound by the frozen interval index;
                    # only root-relative dataset metadata must exist on workers.
                    if source_file.get("role") in {"metadata", "collection_evidence"}:
                        file = root / safe_relative(source_file["path"])
                        check(file.is_file() and file.resolve().is_relative_to(root.resolve())
                              and sha(file) == source_file["sha256"], "Dataset metadata identity changed: " + str(file))
                identities[expected] = index["root_id"]
    return identities


def training_command(payload, plan, *, output=None):
    config = plan["config"]
    output = Path(output or config["output_root"])
    learning = config["learning_rate"]
    command = [config["runtime_python"], str(payload / "openpi/scripts/b1k/train_b1k.py"), "pi05_b1k",
               "--exp-name", config["run_id"], "--assets-base-dir", str(output / "assets"),
               "--checkpoint-base-dir", str(output / "checkpoints"), "--batch-size", str(config["global_batch_size"]),
               "--num-workers", str(config.get("num_workers", 4)), "--num-train-steps", str(config["train_steps"]),
               "--seed", str(config.get("seed", 42)), "--save-interval", str(config["save_interval"]),
               "--keep-period", str(config.get("keep_period", config["save_interval"] * 5)), "--log-interval", "10", "--fsdp-devices", "8",
               "--no-wandb-enabled", "--data.repo-id", "b1k_mainline_" + config["run_id"],
               "--data.subtask-mixture-path", str(payload / "inputs/mixture.json"),
               "--data.subtask-mixture-roots-path", str(payload / "inputs/roots.json"),
               "--data.video-backend", "pyav", "--data.subtask-whole-task-prob", "0", "--data.frame-stride", "1",
               "--model.train-subtask-prediction", "--model.subtask-action-conditioning", "task_and_subtask",
               "--model.action-horizon", "32", "--model.max-token-len", "512", "--model.max-subtask-len", "64",
               "--model.subtask-loss-weight", "1.0", "--model.subtask-loss-chunk-size", "8",
               "--lr-schedule.peak-lr", str(learning["peak"]), "--lr-schedule.decay-lr", str(learning["floor"]),
               "--lr-schedule.warmup-steps", str(learning["warmup_steps"]),
               "--lr-schedule.decay-steps", str(learning["decay_steps"])]
    if config.get("resume", False):
        command.append("--resume")
    else:
        command.extend(["--weight-loader.params-path", str(Path(config["checkpoint"]["root"]) / "params")])
    return command


def training_environment(payload, plan, rank, coordinator, proof):
    # Do not inherit full100 epoch budgets, alternate checkpoints or debug modes.
    env = {k: v for k, v in os.environ.items() if (k == "OPENPI_DATA_HOME" or not k.startswith(("OPENPI_", "TRAIN_", "LR_", "JAX_")))
           and k not in ("PYTHONPATH", "PYTHONHOME", "CUDA_VISIBLE_DEVICES")}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               OPENPI_DATA_HOME=plan["config"]["openpi_data_home"],
               PYTHONPATH=str(payload / "openpi/src") + os.pathsep + str(payload / "openpi/packages/openpi-client/src"),
               OPENPI_PROCESS_COUNT="2", OPENPI_PROCESS_ID=str(rank), OPENPI_LOCAL_DEVICE_IDS="0,1,2,3,4,5,6,7",
               OPENPI_EXPECTED_GLOBAL_DEVICES="16", OPENPI_COORDINATOR_ADDRESS=coordinator,
               OPENPI_DISTRIBUTED_TIMEOUT=str(plan["config"].get("rendezvous_timeout", 600)),
               OPENPI_VERIFY_FINAL_CHECKPOINT="1", OPENPI_REQUIRE_FINITE_METRICS="1",
               OPENPI_SUCCESS_PATH=str(proof), OPENPI_TRAINING_CONTRACT_SHA256=plan["training_contract_sha256"])
    return env


def wait_reports(directory, pattern, count, deadline):
    while True:
        failures = list(directory.glob("*.failed.json"))
        check(not failures, "A peer failed its mainline checks: " + ",".join(p.name for p in failures))
        files = sorted(directory.glob(pattern))
        check(len(files) <= count, "Too many workers registered; do not reuse an attempt")
        if len(files) == count:
            try:
                return [read(p) for p in files]
            except (json.JSONDecodeError, OSError):
                pass  # Another rank has opened its exclusive report but is still writing.
        check(time.time() < deadline, "Bounded rank rendezvous timed out")
        time.sleep(1)


def prepare_output(plan, norm, payload):
    config, output = plan["config"], Path(plan["config"]["output_root"])
    contract_path = output / "optimizer_contract.json"
    ckpts = output / "checkpoints/pi05_b1k" / config["run_id"]
    if config.get("resume", False):
        check(contract_path.is_file() and read(contract_path).get("sha256") == plan["optimizer_contract_sha256"],
              "Resume requires the identical code, initialization and fixed optimizer/LR contract")
        complete = [p for p in ckpts.glob("*") if p.name.isdigit() and (p / "train_state").is_dir()
                    and (p / "params/_METADATA").is_file() and (p / "_CHECKPOINT_METADATA").is_file()]
        check(bool(complete), "Resume requires a complete optimizer checkpoint; initialization is a fresh experiment")
        latest = max(complete, key=lambda p: int(p.name))
        completed_steps = int(latest.name) + 1
        check(completed_steps < config["train_steps"], "Resume has no remaining optimizer updates")
        module = mixture_module(payload)
        module.check_consumed_contract(latest / "assets", module.MixturePlan.load(payload / "inputs/mixture.json"), completed_steps)
    else:
        check(not ckpts.exists(), "Checkpoint directory exists; use explicit resume with the same contract")
        if contract_path.exists():
            check(read(contract_path).get("sha256") == plan["optimizer_contract_sha256"], "Output belongs to a different optimizer contract")
        else:
            write(contract_path, {"sha256": plan["optimizer_contract_sha256"], "contract": plan["optimizer_contract"]}, exclusive=True)
    write(output / "attempts" / plan["attempt_id"] / "training_contract.json",
          {"sha256": plan["training_contract_sha256"], "contract": plan["training_contract"]}, exclusive=True)
    destination = output / "assets/pi05_b1k" / ("b1k_mainline_" + config["run_id"]) / "norm_stats.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        check(sha(destination) == sha(norm), "Existing training normalization differs from the baseline")
    else:
        with destination.open("xb") as stream:
            stream.write(norm.read_bytes())


def reader_preflight(payload, report):
    """Run in the staged environment: decode real samples from every index/root."""
    from types import SimpleNamespace
    import numpy as np
    from openpi.training.lerobot_compat import LeRobotDataset
    from openpi.training.subtask_mixture_data import create_dataset
    data = SimpleNamespace(subtask_prediction=True, prompt_from_task=True, subtask_whole_task_prob=0,
                           frame_stride=1, episodes_index=None, memory_contract=None,
                           subtask_mixture_path=str(payload / "inputs/mixture.json"),
                           subtask_mixture_roots_path=str(payload / "inputs/roots.json"),
                           action_sequence_keys=("action",), data_cls=LeRobotDataset,
                           dataset_kwargs={"video_backend": "pyav", "tolerance_s": 5e-3})
    dataset = create_dataset(data, 32)
    decoded = []
    for index in dataset.mixture_plan.indexes:
        interval = index["intervals"][0]
        sample = dataset.readers[index["root_id"]].read(interval["episode_index"], interval["start"])
        # read() checks finite raw state against the source's 57/61/etc shape,
        # all three RGB frames and the entire 32 x 23 action chunk.
        decoded.append({"root_id": index["root_id"], "source_kind": index["source_kind"],
                        "episode_index": interval["episode_index"], "frame": interval["start"],
                        "state_shape": list(np.asarray(sample["observation.state"]).shape),
                        "action_shape": list(np.asarray(sample["action"]).shape),
                        "rgb_streams": len(dataset.readers[index["root_id"]].video_shapes)})
    write(report, {"complete": True, "actual_training_reader": True, "decoded": decoded,
                   "tokenization_verified": False, "gpu_training_verified": False}, exclusive=True)


def worker(payload):
    payload = Path(payload).resolve()
    plan = verify_bundle(payload)
    config = plan["config"]
    output = Path(config["output_root"])
    rendezvous = output / "rendezvous" / plan["attempt_id"]
    rendezvous.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    identity = host + "-" + str(os.getpid())
    try:
        python = Path(config["runtime_python"])
        check(python.is_file() and os.access(python, os.X_OK), "Prepared Python runtime is missing; stage the environment before submission")
        gpu = verify_gpus(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True))
        verify_tokenizer(config)
        norm = verify_checkpoint(config)
        indexes = verify_data(payload, config)
        addresses = [ipaddress.ip_address(value) for value in subprocess.check_output(["hostname", "-i"], text=True).split()]
        valid = [address for address in addresses if address.version == 4 and not address.is_loopback and not address.is_link_local]
        check(bool(valid), "Worker requires a routable IPv4 host IP")
        ip = str(valid[0])
        write(rendezvous / (identity + ".host.json"), {"identity": identity, "host": host, "ip": ip, "gpu": gpu,
              "contract": plan["training_contract_sha256"], "indexes": indexes,
              "norm_sha256": sha(norm), "bundle_sha256": digest(read(payload / "bundle_manifest.json"))}, exclusive=True)
        deadline = time.time() + config.get("rendezvous_timeout", 600)
        peers = sorted(wait_reports(rendezvous, "*.host.json", 2, deadline), key=lambda r: r["identity"])
        check(len({r["host"] for r in peers}) == len({r["ip"] for r in peers}) == 2, "Expected two distinct worker hosts")
        for field in ("contract", "indexes", "norm_sha256", "bundle_sha256"):
            check(peers[0][field] == peers[1][field], "Ranks disagree on " + field)
        rank = next(i for i, peer in enumerate(peers) if peer["identity"] == identity)
        coordinator = peers[0]["ip"] + ":" + str(config.get("coordinator_port", 12355))
        if rank == 0:
            prepare_output(plan, norm, payload)
            write(rendezvous / "output.ready.json", {"contract": plan["training_contract_sha256"]}, exclusive=True)
        wait_reports(rendezvous, "output.ready.json", 1, deadline)
        proof = rendezvous / "checkpoint_restore_verified.json"
        env = training_environment(payload, plan, rank, coordinator, proof)
        reader_report = rendezvous / f"rank-{rank}.reader.json"
        reader_env = dict(env, JAX_PLATFORMS="cpu")
        for key in list(reader_env):
            if key.startswith("OPENPI_PROCESS") or key in {"OPENPI_COORDINATOR_ADDRESS", "OPENPI_LOCAL_DEVICE_IDS", "OPENPI_EXPECTED_GLOBAL_DEVICES"}:
                reader_env.pop(key)
        subprocess.run([config["runtime_python"], str(payload / "nebula/train_subtask_mainline.py"),
                        "--reader-preflight", str(payload), str(reader_report)], env=reader_env, check=True)
        wait_reports(rendezvous, "rank-*.reader.json", 2, time.time() + config.get("rendezvous_timeout", 600))
        if config.get("mode", "train") == "preflight":
            write(rendezvous / (f"rank-{rank}.preflight.json"), {"complete": True, "rank": rank, "gpus": gpu,
                  "contract": plan["training_contract_sha256"], "gpu_training_verified": False}, exclusive=True)
            return
        log = rendezvous / f"rank-{rank}.train.log"
        with log.open("x") as stream:
            subprocess.run(training_command(payload, plan), env=env, cwd=payload / "openpi", stdout=stream, stderr=subprocess.STDOUT, check=True)
        # Trainer performs collective restore and synchronization before rank 0
        # writes the proof. All ranks verify it before publishing success.
        result = wait_reports(rendezvous, "checkpoint_restore_verified.json", 1, time.time() + 60)[0]
        check(result.get("restored_step") == config["train_steps"] and result.get("processes") == 2
              and result.get("global_devices") == 16 and result.get("action_horizon") == 32
              and result.get("conditioning_mode") == "joint" and result.get("mesh") == {"batch": 2, "fsdp": 8}
              and result.get("contract_sha256") == plan["training_contract_sha256"], "Final checkpoint restore does not match the training contract")
        saved_assets = output / "checkpoints/pi05_b1k" / config["run_id"] / str(config["train_steps"] - 1) / "assets"
        check(read(saved_assets / "subtask_conditioning.json").get("action_conditioning") == "task_and_subtask",
              "Saved checkpoint changed action conditioning")
        check(read(saved_assets / ("b1k_mainline_" + config["run_id"]) / "norm_stats.json") == read(norm),
              "Saved checkpoint changed baseline normalization")
        write(rendezvous / f"rank-{rank}.success.json", {"rank": rank, "contract": plan["training_contract_sha256"], "proof": result}, exclusive=True)
        wait_reports(rendezvous, "rank-*.success.json", 2, time.time() + 60)
        if rank == 0:
            success = {"attempt_id": plan["attempt_id"], "contract": plan["training_contract_sha256"], "proof": result}
            write(output / "attempts" / plan["attempt_id"] / "success.json", success, exclusive=True)
            pointer = output / (".latest_success_" + plan["attempt_id"] + ".json")
            write(pointer, success, exclusive=True)
            pointer.replace(output / "latest_success.json")
    except BaseException as error:
        write(rendezvous / (identity + ".failed.json"), {"error_type": type(error).__name__, "message": str(error)[:1000]})
        raise


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--reader-preflight":
        reader_preflight(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        worker(Path(__file__).resolve().parents[1])
