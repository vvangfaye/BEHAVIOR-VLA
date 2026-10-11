"""Finite CPU preparation of audited successful trajectories for the training pilot.

Run only after collection and its read-only audit have ended. This worker never
authenticates, calls a teacher, allocates GPUs, or edits source trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

AFS = "/mnt/afs/260010165/behavior_1k/codex_pilot_20261011"
RUNTIME = "/mnt/afs/260010165/behavior_1k/runtime_20261007_1306"
CHECKPOINT = "/mnt/afs/260010165/behavior_1k/checkpoints/pi05_b1k_full100_joint_1688977_aoss_20261010"
BUNDLE_SHA = "b2f054f13ccc76f2db46fa990e5d40ef3bfc8ca4cfaaae18baf6b231f42803e0"
CODE9_BUNDLE_SHA = "46bb77c810fde0cbdd5779a7cf49658b2066540edad71238fc86726d885fa512"
CODE9_BUNDLE_NAME = "collection_code_v9_r2.tar.gz"
NATIVE_CHANGED_FILES = {"scripts/collect_b1k_codex.py", "scripts/codex_collection_teacher.py",
                        "scripts/codex_collection_trajectory.py", "scripts/codex_teacher_bridge.py",
                        "scripts/codex_teacher_mailbox.py", "scripts/run_codex_collection.sh",
                        "sco/run_codex_collection.py", "scripts/codex_collection_actions.py"}
ORIGINAL_SHA = "481c24066c6a53fecd1c0d8bce512b2ec76fd106d06a8317c98e3af6a0896d2b"
CHECKPOINT_SHA = "156f0508b558f150402072e393ceca840554d3f345b3414c81c138fe1ed84a20"
TRAINING_DRIVER_SHA = "089aefc0151e2eb051734ba7561b3d5b7054de199b642bc01cbe4853f3619cf1"
TASK_EPISODES = {"turning_on_radio": [28, 68, 102], "putting_shoes_on_rack": [4424, 4487, 4517]}
KIND = "codex_pilot_data_preparation_v1"
EXPECTED_CODE_INVENTORY = None  # Supplied by the builder from the SHA-pinned bundle.


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def validate_payload(payload):
    ensure(payload.get("kind") == KIND, "Unknown data preparation contract")
    run_id = payload.get("run_id")
    ensure(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id), "Invalid fresh run ID")
    audits = payload.get("success_audits")
    ensure(isinstance(audits, list) and 1 <= len(audits) <= 10, "One to ten audited successes are required")
    paths = set()
    for ref in audits:
        ensure(isinstance(ref, dict) and isinstance(ref.get("path"), str), "Audit reference required")
        path = Path(ref["path"])
        ensure(path.is_absolute() and path.name == "report.json" and path.parent.parent == Path(AFS)
               and path.parent.name.startswith("audit-") and path not in paths
               and re.fullmatch(r"[0-9a-f]{64}", str(ref.get("content_sha256"))), "Invalid or repeated audit reference")
        paths.add(path)
    version = payload.get("code_version", "v8")
    ensure(version in {"v8", "v9"}, "Unknown preparation code version")
    ensure(not set(payload) - {"kind", "run_id", "success_audits", "code_version"}, "No task, data or budget overrides are allowed")
    return {**payload, "code_version":version}


def paths_for(run_id, code_version="v8"):
    ensure(code_version in {"v8", "v9"}, "Unknown preparation code version")
    return dict(code=AFS + "/code-" + code_version, runtime=RUNTIME,
                report_root=AFS + "/data-preparation-" + run_id,
                dataset_root=AFS + "/mixed-lerobot-" + run_id,
                run_root=AFS + "/training/" + run_id,
                training_report_root=AFS + "/training-job-" + run_id,
                destination=AFS + "/inference/" + run_id)


def verify_original(helpers, deadline):
    """Recheck the complete v14 source chain, without copying or downloading.

    The pinned READY fixes status/payload hashes, all transfer identities and
    demonstration manifests. Only hash the private staging payload; never load
    its signed URLs into this worker or its reports.
    """
    dataset, reader = Path(AFS) / "original-v3", Path(AFS) / "original-reader-v4"
    expected_episodes = sorted(ep for values in TASK_EPISODES.values() for ep in values)
    checks = {}

    def checked(path, expected, size=None, *, allow_link=False):
        path = Path(path)
        if time.monotonic() >= deadline - 30:
            raise helpers.WorkerDeadline("Original-data verification reached the cleanup reserve")
        ensure(path.is_absolute() and re.fullmatch(r"[0-9a-f]{64}", str(expected))
               and path.is_file() and (allow_link or not path.is_symlink()), "Invalid original source reference")
        ensure(size is None or path.stat().st_size == size, "Original source size changed: " + str(path))
        actual = helpers.sha(path)
        ensure(actual == expected, "Original source SHA256 changed: " + str(path))
        checks[str(path)] = actual
        return path

    def inventory(rows):
        ensure(isinstance(rows, list) and rows, "Original inventory is empty")
        result = {}
        for row in rows:
            ensure(isinstance(row, dict) and isinstance(row.get("path"), str)
                   and row["path"] not in result and type(row.get("bytes")) is int and row["bytes"] > 0
                   and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))), "Invalid original inventory entry")
            result[row["path"]] = row
        return result

    def source_file(row, allowed):
        name = row["path"]; relative = Path(name)
        ensure(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name
               and relative.parts and relative.parts[0] in allowed, "Original source path escapes its allowed root")
        path = dataset / relative
        ensure(not any(parent.is_symlink() for parent in path.parents if parent != dataset and dataset in parent.parents),
               "Original source directory is linked")
        return checked(path, row["sha256"], row["bytes"])

    writer_markers = (dataset / "_RUNNING", reader / "_RUNNING", Path(AFS) / "preparation-v13/_RUNNING",
                      Path(AFS) / "preparation-v14/_RUNNING")
    ensure(not dataset.is_symlink() and not reader.is_symlink()
           and not any(path.exists() for path in writer_markers), "Original source writer is active")
    ready_path = checked(reader / "_ORIGINAL_DATA_READY.json", ORIGINAL_SHA)
    ready = helpers.load(ready_path)
    ensure(ready.get("kind") == "codex_original_pilot_staging_v1" and ready.get("state") == "complete"
           and ready.get("root") == str(reader) and ready.get("dataset_root") == str(dataset)
           and ready.get("selected_episodes") == expected_episodes
           and ready.get("original_reader_verified") is True and ready.get("raw_success_verified") is True,
           "Original reader readiness identity differs")
    receipts = ready.get("source_receipt_sha256", {})
    ensure(set(receipts) == {"status", "payload"}, "Original source receipt chain differs")
    for name, expected in receipts.items():
        checked(dataset / (name + ".json"), expected)
    status = helpers.load(dataset / "status.json")
    ensure(status.get("state") == "failed" and status.get("failure_phase") == "real_reader_and_demonstrations"
           and (dataset / "_FAILED").is_file() and not (dataset / "_ORIGINAL_DATA_READY.json").exists()
           and status.get("root") == str(dataset) and status.get("selected_episodes") == expected_episodes,
           "Original-v3 source is not the pinned stopped v13 receipt")
    transfers = inventory(ready.get("transfers"))
    ensure(len(transfers) == 20 and status.get("transfers") == ready["transfers"]
           and set(status.get("required_files", [])) == set(transfers), "Original complete transfer inventory differs")
    for row in transfers.values():
        source_file(row, {"data", "videos"})
    metadata = inventory(status.get("metadata_files"))
    ensure("meta/info.json" in metadata, "Original metadata inventory is incomplete")
    for name, row in metadata.items():
        if name != "meta/info.json" or not (dataset / "meta/info.json.orig").exists():
            source_file(row, {"meta"}); continue
        original = source_file({**row, "path": "meta/info.json.orig"}, {"meta"})
        value = helpers.load(original); features = value["features"]
        depth = [key for key, feature in features.items() if feature.get("dtype") == "video"
                 and (feature.get("info") or {}).get("is_depth_map")]
        ensure(depth, "Original RGB metadata backup does not contain depth features")
        for key in depth:
            features.pop(key)
        # Match the already verified strip_depth_from_info serialization exactly.
        expected = json.dumps(value, indent=4).encode()
        checked(dataset / name, hashlib.sha256(expected).hexdigest(), len(expected))
    selected = dataset / "selected_episodes.txt"
    selected_bytes = "".join(str(ep) + "\n" for ep in expected_episodes).encode()
    checked(selected, hashlib.sha256(selected_bytes).hexdigest(), len(selected_bytes))
    demos = ready.get("demonstrations", [])
    ensure([row.get("episode_index") for row in demos] == expected_episodes, "Original demonstration selection differs")
    covered, label_hashes = set(), {}
    for row in demos:
        episode = row["episode_index"]
        task = next(task for task, episodes in TASK_EPISODES.items() if episode in episodes)
        manifest_path = reader / "demonstrations" / str(episode) / "manifest.json"
        ensure(row.get("manifest") == str(manifest_path) and row.get("task") == task,
               "Original demonstration manifest path differs")
        manifest = helpers.load(checked(manifest_path, row["manifest_sha256"]))
        ensure(manifest.get("fingerprint") == row.get("fingerprint")
               == helpers.digest({key: value for key, value in manifest.items() if key != "fingerprint"}),
               "Original demonstration fingerprint differs")
        meta = manifest["metadata"]
        ensure(meta.get("source_kind") == "original_train" and meta.get("task") == task
               and meta.get("episode_index") == episode and meta.get("source_dataset") == str(dataset)
               and meta.get("identity_source") == "published_metadata_raw_id_and_sha"
               and meta.get("success_source") == "original_behavior_task_terminated_without_truncation"
               and meta.get("expected_raw_sha256") == meta.get("source_raw_sha256"), "Original demonstration identity differs")
        proof = meta["termination_evidence"]
        ensure(proof.get("final_terminated") is True and proof.get("final_truncated") is False
               and type(proof.get("terminated_count")) is int and proof["terminated_count"] > 0,
               "Original demonstration success evidence differs")
        checked(Path(meta["source_raw"]), meta["source_raw_sha256"], allow_link=True)
        source_rows = meta["source_files"]
        ensure(isinstance(source_rows, list) and source_rows
               and len({item["path"] for item in source_rows}) == len(source_rows), "Original demonstration sources differ")
        for item in source_rows:
            ensure(set(item) == {"path", "sha256"} and item["path"] in transfers
                   and item["sha256"] == transfers[item["path"]]["sha256"], "Demonstration source hash differs")
            covered.add(item["path"])
        for label, name in (("segments_sha256", "subtask_segments.parquet"), ("retained_episodes_sha256", "retained_episodes.txt")):
            expected = meta[label]
            ensure(name not in label_hashes or label_hashes[name] == expected, "Original label identities disagree")
            if name not in label_hashes:
                checked(dataset / "labels" / name, expected)
                label_hashes[name] = expected
        examples = manifest["examples"]
        ensure(1 <= len(examples) <= 6 and len({item["image"] for item in examples}) == len(examples), "Original demonstration images differ")
        for example in examples:
            ensure(Path(example["image"]).name == example["image"], "Original demonstration image path is unsafe")
            checked(manifest_path.parent / example["image"], example["image_sha256"])
    ensure(covered == set(transfers), "Demonstrations do not cover the complete source inventory")
    # Detect changed receipts while the multi-file read was in progress.
    checked(ready_path, ORIGINAL_SHA)
    for name, expected in receipts.items():
        checked(dataset / (name + ".json"), expected)
    ensure(not any(path.exists() for path in writer_markers), "Original writer appeared during verification")
    return dict(ready_sha256=ORIGINAL_SHA, verified_files=len(transfers), verified_metadata_files=len(metadata),
                verified_raw_demonstrations=len(demos), verified_label_sha256=label_hashes,
                verified_source_files=len(checks), source_inventory_sha256=helpers.digest(checks),
                verified_bytes=sum(row["bytes"] for row in transfers.values()), downloaded_bytes=0)


def audit_successes(refs, helpers, inspect, *, code_version="v8"):
    """Check actual audit bytes, semantic identity and current full HDF5 content."""
    selected, identities, content_hashes = [], set(), set()
    for ref in refs:
        path = Path(ref["path"])
        ensure(path.is_file() and not path.is_symlink(), "Audit must be a real file")
        audit = helpers.load(path)
        ensure(helpers.digest(audit) == ref["content_sha256"], "Audit content changed")
        ensure(audit.get("kind") == "codex_collection_readonly_audit_v1"
               and audit.get("state") == "audit_complete" and audit.get("training_eligible") is True
               and audit.get("source_unchanged_during_audit") is True and audit.get("issues") == []
               and audit.get("official_benchmark") is False
               and audit.get("model_calls") == 0 and audit.get("network_calls") == 0
               and audit.get("model_calls_scope") == "audit_process_only"
               and audit.get("audit_root") == str(path.parent), "Audit does not accept this success")
        identity = audit.get("identity", {})
        key = tuple(identity.get(k) for k in ("task", "instance_id", "attempt_id"))
        ensure(key[0] in TASK_EPISODES and type(key[1]) is int and 311 <= key[1] <= 320
               and isinstance(key[2], str) and key not in identities, "Invalid or duplicate collection identity")
        root = Path(audit.get("run_root", ""))
        ensure(root.is_absolute() and root.parent == Path(AFS) / "collection", "Collection root differs")
        trajectory = root / "attempt/trajectory.hdf5"
        ensure(trajectory.is_file() and not trajectory.is_symlink(), "Trajectory must be a real file")
        info = inspect(trajectory, require_success=True)
        mode = info["metadata"].get("collection_mode", "teacher_conditioned")
        ensure(mode in {"teacher_conditioned", "native_with_corrections"}
               and audit.get("collection_mode", "teacher_conditioned") == mode,
               "Audit and actual trajectory collection modes differ")
        if mode == "native_with_corrections":
            ensure(code_version == "v9" and isinstance(CODE9_BUNDLE_SHA, str)
                   and info["metadata"].get("subtask_source") == "mixed_native_and_teacher"
                   and audit.get("collection_code") == AFS + "/code-v9"
                   and audit.get("collection_bundle_sha256") == CODE9_BUNDLE_SHA
                   and isinstance(EXPECTED_CODE_INVENTORY, dict)
                   and audit.get("collection_source_files_sha256") == EXPECTED_CODE_INVENTORY
                   and audit.get("trajectory", {}).get("collection_mode") == mode
                   and audit.get("trajectory", {}).get("subtask_source") == "mixed_native_and_teacher",
                   "Native data requires its exact reviewed code-v9 audit and reader")
        proof = audit.get("trajectory", {})
        ensure(tuple(info["metadata"].get(k) for k in ("task", "instance_id", "attempt_id")) == key
               and info["metadata"].get("checkpoint") == "pi05_b1k_full100_joint_1688977"
               and info["content_sha256"] == proof.get("content_sha256")
               and info["subtask_sha256"] == proof.get("subtask_sha256")
               and info["content_sha256"] not in content_hashes
               and info["steps"] == proof.get("steps")
               and proof.get("valid_complete_trajectory") is True
               and proof.get("environment_goal_success") is True, "Successful trajectory differs from its audit")
        identities.add(key);content_hashes.add(info["content_sha256"])
        selected.append(dict(identity=identity, trajectory=str(trajectory), audit=ref,
                             collection_mode=mode, content_sha256=info["content_sha256"], steps=info["steps"]))
    ensure(all(sum(row["identity"]["task"] == task for row in selected) <= 5 for task in TASK_EPISODES),
           "At most five verified successes per task belong in this pilot")
    return selected


def commands(paths, run_id, selected, python, episodes):
    code = Path(paths["code"])
    export = [python, str(code / "scripts/prepare_codex_training_data.py"), "export", "--trajectories",
              *[row["trajectory"] for row in selected], "--base-dataset", AFS + "/original-v3",
              "--base-repo-id", "behavior-1k/2026-challenge-demos", "--base-segments",
              AFS + "/original-v3/labels/subtask_segments.parquet", "--base-episodes", str(episodes),
              "--output", paths["dataset_root"], "--repo-id", "b1k_codex_" + run_id]
    verify = [python, str(code / "scripts/prepare_codex_training_data.py"), "verify", "--dataset-root",
              paths["dataset_root"], "--report", paths["report_root"] + "/mixed_reader.json"]
    prepare = [python, str(code / "scripts/finetune_codex_collection.py"), "prepare", "--dataset-root",
               paths["dataset_root"], "--checkpoint", CHECKPOINT, "--checkpoint-manifest",
               AFS + "/bundle-v2/checkpoint_upload_plan.json", "--expected-manifest-sha256", CHECKPOINT_SHA,
               "--base-repo-id", "b1k_full100_joint_subtask_v1_drop_episode_s1", "--run-root", paths["run_root"],
               "--run-id", run_id, "--devices", "4", "--batch-size", "4", "--steps", "256"]
    return [("export", export, 4200), ("verify", verify, 1800), ("prepare", prepare, 1800)]


def run(payload, helpers=None):
    sys.dont_write_bytecode = True
    os.umask(0o077)
    if helpers is None:
        import run_codex_pilot_training as helpers
    payload = validate_payload(payload)
    paths = paths_for(payload["run_id"], payload["code_version"])
    helpers.disjoint(list(paths.values()) + [CHECKPOINT, AFS + "/original-v3", AFS + "/original-reader-v4"])
    for key in ("report_root", "dataset_root", "run_root", "training_report_root", "destination"):
        ensure(not Path(paths[key]).exists() and not Path(paths[key]).is_symlink(), "Outputs must all be fresh")
    root = Path(paths["report_root"]);root.mkdir(parents=True, mode=0o700)
    report = dict(kind=KIND, state="preparing", stage="gates", ready_for_training=False,
                  model_calls=0, network_calls=0, official_benchmark=False, run_id=payload["run_id"],
                  code_version=payload["code_version"], phases={})
    started = time.monotonic()
    def publish():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        temporary = root / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(root / "report.json")
    def expired(*_):
        raise helpers.WorkerDeadline("Finite data-preparation deadline reached")
    old_alarm = signal.signal(signal.SIGALRM, expired)
    old_term = signal.signal(signal.SIGTERM, expired)
    timer_started = time.monotonic()
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    budget = min(7800, old_timer[0]) if old_timer[0] > 0 else 7800
    deadline = timer_started + budget
    report["effective_wall_timeout_seconds"] = budget
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
    log_path = None
    try:
        report["original_sources_before"] = verify_original(helpers, deadline)
        code_ready = helpers.file_ref(Path(paths["code"]) / "_CODEX_PILOT_CODE_READY")
        ready = helpers.load(code_ready["path"])
        expected_bundle_sha = BUNDLE_SHA if payload["code_version"] == "v8" else CODE9_BUNDLE_SHA
        ensure(isinstance(EXPECTED_CODE_INVENTORY, dict) and len(EXPECTED_CODE_INVENTORY) == 29
               and isinstance(expected_bundle_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha)
               and ready["bundle_sha256"] == expected_bundle_sha
               and ready.get("source_files_sha256") == EXPECTED_CODE_INVENTORY, "Use the exact reviewed collection bundle inventory")
        report["code"] = helpers.verify_code({"code": paths["code"], "code_ready": code_ready})
        ensure(helpers.sha(Path(AFS) / "bundle-v2/checkpoint_upload_plan.json") == CHECKPOINT_SHA,
               "Baseline checkpoint manifest changed")
        space = shutil.disk_usage(root)
        report["storage"] = dict(free_bytes=space.free, required_free_bytes=192 * 1024**3)
        ensure(space.free >= 192 * 1024**3, "Insufficient space for separate mixed data and smoke/full checkpoints")
        sys.path[:0] = [paths["code"] + "/scripts", paths["code"] + "/sco"]
        from codex_collection_trajectory import inspect_codex_collection_trajectory
        selected = audit_successes(payload["success_audits"], helpers, inspect_codex_collection_trajectory, code_version=payload["code_version"])
        tasks = sorted({row["identity"]["task"] for row in selected})
        episodes = root / "original_episodes.txt"
        with episodes.open("x") as stream:
            stream.write("".join(str(ep) + "\n" for task in tasks for ep in TASK_EPISODES[task]))
        report.update(selected_successes=selected, tasks=tasks, original_episodes=helpers.file_ref(episodes));publish()
        env = helpers.environment(paths)
        env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
        python = paths["runtime"] + "/envs/b1k-eval/bin/python"
        ensure(Path(sys.executable).resolve() == Path(python).resolve(), "Run preparation with the verified b1k-eval Python")
        for stage, command, seconds in commands(paths, payload["run_id"], selected, python, episodes):
            report.update(state="running", stage=stage);publish()
            log_path = root / (stage + ".log")
            helpers.bounded_process(command, env=env, cwd=root, log_path=log_path, seconds=seconds,
                                    deadline=deadline, heartbeat=publish)
            report["phases"][stage] = dict(state="complete", log=helpers.file_ref(log_path));publish()
        training_payload = helpers.validate_payload(dict(kind=helpers.KIND, code=paths["code"], runtime=paths["runtime"],
            report_root=paths["training_report_root"], destination=paths["destination"], code_ready=code_ready,
            training_plan=helpers.file_ref(Path(paths["run_root"]) / "training_plan.json"),
            driver_sha256=TRAINING_DRIVER_SHA))
        plan = helpers.verify_plan(training_payload)
        ensure(set(plan["tasks"]) == set(tasks), "Prepared plan tasks differ from successful collection")
        # Reread complete evidence after export to detect any concurrent writer.
        ensure(audit_successes(payload["success_audits"], helpers, inspect_codex_collection_trajectory, code_version=payload["code_version"]) == selected,
               "Source success evidence changed during preparation")
        report["stage"] = "verify_original_after";publish()
        report["original_sources_after"] = verify_original(helpers, deadline)
        ensure(report["original_sources_before"] == report["original_sources_after"],
               "Original source evidence changed during preparation")
        helpers.write_new(root / "training_payload.json", training_payload)
        report.update(state="complete", stage="complete", ready_for_training=True,
                      training_payload=training_payload, training_payload_ref=helpers.file_ref(root / "training_payload.json"),
                      training_plan=training_payload["training_plan"], code_ready=code_ready,
                      mixed_reader=helpers.file_ref(root / "mixed_reader.json"),
                      source_unchanged_during_preparation=True)
    except BaseException as error:
        report.update(state="failed", ready_for_training=False, error_type=type(error).__name__, error_detail=helpers.redact(error))
        if log_path is not None and log_path.is_file():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 12000))
                report["failure_tail"] = helpers.redact(stream.read().decode("utf-8", errors="replace"))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0);publish()
        signal.signal(signal.SIGALRM, old_alarm);signal.signal(signal.SIGTERM, old_term)
        if old_timer[0] > 0:
            remaining = old_timer[0] - (time.monotonic() - timer_started)
            if remaining > 0:signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])
    return report


def build_source(payload, training_source=None):
    """Embed pinned shared helpers so the immutable code overlay stays unchanged."""
    payload = validate_payload(payload)
    version = payload["code_version"]
    expected_bundle_sha = BUNDLE_SHA if version == "v8" else CODE9_BUNDLE_SHA
    ensure(isinstance(expected_bundle_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha),
           "Requested code version has not been reviewed and frozen")
    training_source = training_source or Path(__file__).with_name("run_codex_pilot_training.py").read_text()
    ensure(hashlib.sha256(training_source.encode()).hexdigest() == TRAINING_DRIVER_SHA, "Training helpers changed")
    bundle_name = "collection_code_v8.tar.gz" if version == "v8" else CODE9_BUNDLE_NAME
    bundle = Path(__file__).resolve().parents[1] / "eval_results/codex_pilot_20261011" / bundle_name
    manifest = json.loads(bundle.with_suffix(".gz.manifest.json").read_text())
    ensure(hashlib.sha256(bundle.read_bytes()).hexdigest() == expected_bundle_sha == manifest["bundle_sha256"]
           and manifest["version"] == version and manifest["code"] == AFS + "/code-" + version
           and len(manifest["source_files_sha256"]) == 29, "Frozen bundle changed")
    if version == "v9":
        previous = bundle.with_name("collection_code_v8.tar.gz")
        prior = json.loads(previous.with_suffix(".gz.manifest.json").read_text())
        ensure(hashlib.sha256(previous.read_bytes()).hexdigest() == BUNDLE_SHA == prior["bundle_sha256"]
               and set(manifest["source_files_sha256"]) == set(prior["source_files_sha256"])
               and {name for name in prior["source_files_sha256"]
                    if prior["source_files_sha256"][name] != manifest["source_files_sha256"][name]} == NATIVE_CHANGED_FILES,
               "Native bundle changes more than the reviewed collector/reader interface")
    own = Path(__file__).read_text()
    source = "from types import SimpleNamespace\n"
    source += "_training={'__name__':'_preparation_training_helpers'}\nexec(compile(" + repr(training_source) + ",'<training_helpers>','exec'),_training)\n"
    source += "_preparation={'__name__':'_data_preparation_worker'}\nexec(compile(" + repr(own) + ",'<data_preparation>','exec'),_preparation)\n"
    source += "_preparation['EXPECTED_CODE_INVENTORY']=" + repr(manifest["source_files_sha256"]) + "\n"
    source += "BOOTSTRAP_REPORT = _preparation['run'](" + repr(payload) + ",helpers=SimpleNamespace(**_training))\n"
    compile(source, "<data_preparation_job>", "exec")
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = build_source(json.loads(args.payload.read_text()))
    with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write(source)
    print(json.dumps(dict(source=str(args.output.resolve()), sha256=hashlib.sha256(source.encode()).hexdigest(),
                          max_seconds=7800, recommended_cloud_max_seconds=8400, actual_job_submitted=False)))


if __name__ == "__main__":
    main()
