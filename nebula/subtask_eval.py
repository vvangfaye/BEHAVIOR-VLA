#!/usr/bin/env python3
"""Prepare and supervise the 100-task exact-state diagnostic using existing runners."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from subtask_sample_panel import load_panel, challenge_tasks


def load_inputs(panel_path, inventory_path):
    panel = load_panel(panel_path)
    inventory = json.loads(Path(inventory_path).read_text())
    if panel["selection"].get("task_scope") != "full100" or set(panel["selection"]["tasks"]) != challenge_tasks():
        raise ValueError("subtasks100 requires the complete 100-task role-bound panel")
    if inventory.get("schema_version") != 1 or inventory.get("panel_fingerprint") != panel["fingerprint"]:
        raise ValueError("Raw inventory does not match the frozen panel")
    expected = {f"task-{e['task_index']:04d}/episode_{e['raw_episode_id']:08d}.hdf5" for e in panel["episodes"]}
    items = inventory.get("episodes")
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError("Raw inventory does not cover every selected episode")
    actual = [item.get("relative_path") for item in items]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Raw inventory has duplicate, unexpected or missing paths")
    for item in items:
        if type(item.get("bytes")) is not int or item["bytes"] <= 0 or item.get("missing"):
            raise ValueError("Raw inventory contains a missing or empty object")
        source = inventory.get("raw_source", "").rstrip("/")
        if not source.startswith("oss://") or item.get("key") != source.split("/", 3)[3] + "/" + item["relative_path"]:
            raise ValueError("Raw inventory object does not belong to its frozen source")
    return panel, inventory


def preflight(panel_path, inventory_path, raw_source, client=None):
    from submit_oss import OSSClient
    panel, inventory = load_inputs(panel_path, inventory_path)
    client = client or OSSClient()
    if client.key(raw_source) != client.key(inventory["raw_source"]):
        raise ValueError("Requested raw source differs from the frozen inventory")
    # HEAD checks freeze both content identity and size without downloading states.
    for item in inventory["episodes"]:
        head = client.bucket.head_object(item["key"])
        if head.content_length != item["bytes"] or head.etag != item["etag"] or head.headers.get("x-oss-hash-crc64ecma") != item["crc64"]:
            raise ValueError(f"Raw object changed since panel preparation: {item['relative_path']}")
    print(json.dumps({"raw_preflight": "passed", "tasks": len(panel["selection"]["tasks"]),
                      "episode_jobs": len(panel["episodes"]), "segment_rollouts": len(panel["samples"]),
                      "excluded_segments": len(panel["exclusions"]),
                      "raw_bytes": sum(item["bytes"] for item in inventory["episodes"]),
                      "panel_fingerprint": panel["fingerprint"]}), flush=True)


def stage_raw(inventory, source, destination):
    source, destination = Path(source), Path(destination) / "2026-challenge-rawdata"
    if destination.exists():
        raise ValueError("Raw staging requires a new destination")
    needed = sum(item["bytes"] for item in inventory["episodes"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < needed + 5 * 1024**3:
        raise ValueError("Insufficient local disk for selected raw states plus 5 GiB reserve")
    destination.mkdir()

    def copy(item):
        original = source / item["relative_path"]
        output = destination / item["relative_path"]
        if original.stat().st_size != item["bytes"]:
            raise ValueError(f"Raw source size changed: {item['relative_path']}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, output)
        if output.stat().st_size != item["bytes"]:
            raise ValueError(f"Incomplete raw copy: {item['relative_path']}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(copy, inventory["episodes"]), 1):
            if index % 25 == 0 or index == len(inventory["episodes"]):
                print(f"Staged raw states: {index}/{len(inventory['episodes'])}", flush=True)
    # Existing diagnostic preflight validates each HDF5's schema, frame ranges
    # and topology transitions, and records a full SHA256 before any rollout.
    return destination.parent


def sync_results(local, persistent, copied=None):
    """Incrementally publish stable files without depending on system rsync.

    The queue publishes validated results in its canonical results directory.
    Attempts, live SQLite files, temporary files and completion markers stay
    local. Each destination is replaced only after a complete, stable copy.
    """
    local, persistent = Path(local), Path(persistent)
    if not local.exists():
        return
    copied = copied if copied is not None else {}
    excluded = {"_SUCCESS", "_FAILED", "_RUNNING", ".supervisor.lock"}
    for directory, folders, files in os.walk(local):
        folders[:] = [name for name in folders if name != "attempts"]
        for name in folders:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Result synchronization rejects directory symlinks")
        for name in files:
            if name in excluded or name.startswith("queue.sqlite3") or name.endswith(".tmp"):
                continue
            source = Path(directory) / name
            if source.is_symlink():
                raise ValueError("Result synchronization rejects file symlinks")
            try:
                before = source.stat()
            except FileNotFoundError:
                continue  # A transient file disappeared between walk and stat.
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            relative = source.relative_to(local)
            destination = persistent / relative
            if copied.get(relative) == identity and destination.is_file() and destination.stat().st_size == before.st_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copyfile(source, temporary)
                try:
                    after = source.stat()
                except FileNotFoundError:
                    continue
                current = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                if current != identity:
                    continue  # Growing logs or replaced JSON will be retried.
                if temporary.stat().st_size != before.st_size:
                    raise OSError(f"Incomplete result copy: {relative}")
                os.replace(temporary, destination)
                copied[relative] = identity
            finally:
                temporary.unlink(missing_ok=True)


def supervise_results(command, local_root, persistent, *, poll_interval=30, env=None):
    """Preserve the original failure when termination or final sync is needed."""
    child = subprocess.Popen(command, cwd=REPO_ROOT, **({"env": env} if env is not None else {}))
    copied = {}
    try:
        while child.poll() is None:
            sync_results(local_root, persistent, copied)
            time.sleep(poll_interval)
        if child.returncode:
            raise RuntimeError(f"Subtask diagnostic failed with exit code {child.returncode}")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        try:
            sync_results(local_root, persistent, copied)
        except Exception as error:
            print(f"WARNING: final result synchronization failed: {error}", file=sys.stderr, flush=True)
        raise
    sync_results(local_root, persistent, copied)


def write_reports(summary, output):
    """Export task and prompt counts, including unsupported-only prompts."""
    output = Path(output)
    columns = ["annotated_segments", "scheduled_segments", "valid_results", "metric_eligible",
               "setup_excluded", "source_excluded", "unsupported", "successes", "completion_rate"]
    groups = [("task_success.csv", ["task"], [
        {"task": task, **counts} for task, counts in sorted(summary["per_task"].items())]),
        ("subtask_success.csv", ["task", "subtask"], [
            {"task": task, "subtask": prompt, **counts}
            for task, prompts in sorted(summary["per_prompt_label"].items())
            for prompt, counts in sorted(prompts.items())])]
    for filename, keys, rows in groups:
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys + columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def run_validation(args, panel, inventory):
    from eval_b1k_subtask_probe import KIND, load_plan, validate_results
    plan, samples = load_plan(args.validation_plan, panel)
    local, persistent = Path(args.run_root), Path(args.persistent_output)
    if local.exists() or args.workers != 1 or args.sample_max_steps != plan["max_steps"]:
        raise ValueError("Validation requires a fresh run root, one GPU and its frozen step limit")
    local.mkdir(parents=True)
    persistent.mkdir(parents=True, exist_ok=True)
    paths = {f"task-{s['task_index']:04d}/episode_{s['raw_episode_id']:08d}.hdf5" for s in samples}
    selected = {**inventory, "episodes": [item for item in inventory["episodes"] if item["relative_path"] in paths]}
    raw_root = stage_raw(selected, args.raw_source, args.raw_root)
    ready = json.loads((Path(args.checkpoint_source)/"_READY").read_text())
    if plan['kind'] == 'subtask_native_snapshot_model_validation':
        from validate_b1k_native_snapshot_model import validate_input_artifacts, INPUT_DIRECTORY
        validate_input_artifacts(Path(args.validation_plan).parent/INPUT_DIRECTORY,panel,plan)
    if plan['kind'] == 'subtask_native_snapshot_model_validation' or plan.get('require_recorded_native_state'):
        if ready.get('step')!=plan['checkpoint_step'] or ready.get('manifest_sha256')!=plan['checkpoint_manifest_sha256']:
            raise ValueError('Native model validation checkpoint differs from its frozen plan')
    contract = {"run_id": args.run_id, "profile": plan["kind"], "official_benchmark": False,
                "checkpoint_step": ready.get("step"), "checkpoint_source": args.checkpoint_source,
                "parent_panel_fingerprint": panel["fingerprint"], "sample_ids": plan["sample_ids"],
                "expected_samples": len(samples), "expected_episode_jobs": len(paths),
                "workers": 1, "max_steps": plan["max_steps"]}
    (local/"validation_contract.json").write_text(json.dumps(contract, indent=2)+"\n")
    shutil.copyfile(args.panel, local/"parent_panel.json")
    shutil.copyfile(args.validation_plan, local/"validation_plan.json")
    environment = os.environ.copy()
    environment.update(MODEL="pi05", MODE="train", PROMPT_MODE="whole_task", NUM_ROLLOUTS="1",
        PI05_CKPT=args.checkpoint, PI05_REPO_ID=args.repo_id, PI05_ENV=args.pi05_env, EVAL_ENV=args.eval_env,
        OMNIGIBSON_DATA_PATH=args.asset_path, OUTPUT_ROOT=str(local), SERVER_GPU="0", EVAL_GPU="0",
        PORT=str(args.base_port), EVAL_RUNTIME_ROOT=str(local.parent/(local.name+"-sim-runtime")),
        PI05_MEM_FRACTION="0.35", SUBTASK_PREDICTION="1", PROVIDED_SUBTASK_AS_PROMPT="0", MAX_TOKEN_LEN="512",
        WRITE_VIDEO="1", PI05_SUBTASK_SAMPLE_PANEL=args.panel, PI05_SUBTASK_RAW_DATA_ROOT=str(raw_root),
        PI05_SUBTASK_PROBE_PLAN=args.validation_plan, SUBTASK_SAMPLE_MAX_STEPS=str(plan["max_steps"]))
    for key in ("PI05_EVAL_QUEUE_DB", "EXTERNAL_SUBTASK", "EXTERNAL_SUBTASK_SELECTOR", "SUBTASK_INTERVAL",
                "SUBTASK_VOTE_WINDOW", "SUBTASK_VOTE_THRESHOLD", "INSTANCE_INDICES", "MAX_STEPS", "GOALKEEPER_CONFIG_DIR"):
        environment.pop(key, None)
    tasks = sorted({s["task"] for s in samples})
    # Reuse policy startup, identity checks, graphics environment and cleanup.
    command = ["/bin/bash", str(REPO_ROOT/"scripts/eval_b1k.sh"), *tasks]
    # supervise_results owns cleanup and stable incremental publication.
    supervise_results(command, local, persistent, env=environment)
    summary = validate_results(local, panel, plan, samples)
    (persistent/"validation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (persistent/"_SUCCESS").write_text(json.dumps({"kind": plan["kind"], "complete": True})+"\n")



def run_collection(args, panel, inventory):
    from subtask_improvement import KIND, load_plan
    from collect_b1k_subtasks import validate_collection
    plan, sample_by_id = load_plan(args.collection_plan, panel)
    local, persistent = Path(args.run_root), Path(args.persistent_output)
    if local.exists() or args.workers != 1 or args.sample_max_steps != plan['max_steps']:
        raise ValueError('Collection requires a fresh root, one GPU and its frozen control limit')
    ready = json.loads((Path(args.checkpoint_source)/'_READY').read_text())
    identity = plan.get('checkpoint_identity') or {}
    if identity.get('checkpoint_step') != ready.get('step') or identity.get('checkpoint_manifest_sha256') != ready.get('manifest_sha256'):
        raise ValueError('Collection checkpoint differs from the verified baseline checkpoint')
    local.mkdir(parents=True)
    persistent.mkdir(parents=True, exist_ok=True)
    samples = [sample_by_id[k] for k in sorted({t['sample_id'] for t in plan['trials']})]
    paths = {f"task-{s['task_index']:04d}/episode_{s['raw_episode_id']:08d}.hdf5" for s in samples}
    selected = {**inventory, 'episodes': [item for item in inventory['episodes'] if item['relative_path'] in paths]}
    raw_root = stage_raw(selected, args.raw_source, args.raw_root)
    contract = dict(run_id=args.run_id, kind=KIND, checkpoint_identity=identity,
                    official_benchmark=False, parent_panel_fingerprint=panel['fingerprint'],
                    plan_fingerprint=plan['fingerprint'], expected_trials=plan['expected_trials'],
                    expected_starts=len(samples), expected_episode_jobs=len(paths), workers=1,
                    max_steps=plan['max_steps'], maximum_control_steps=plan['maximum_control_steps'])
    if plan.get('experiment_purpose'):
        contract.update(experiment_purpose=plan['experiment_purpose'],
                        difficulty_selection_verified=False, require_recorded_native_state=True)
    (local/'collection_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    shutil.copyfile(args.panel, local/'parent_panel.json')
    shutil.copyfile(args.collection_plan, local/'collection_plan.json')
    environment = os.environ.copy()
    environment.update(MODEL='pi05', MODE='train', PROMPT_MODE='whole_task', NUM_ROLLOUTS='1',
        PI05_CKPT=args.checkpoint, PI05_REPO_ID=args.repo_id, PI05_ENV=args.pi05_env, EVAL_ENV=args.eval_env,
        OMNIGIBSON_DATA_PATH=args.asset_path, OUTPUT_ROOT=str(local), SERVER_GPU='0', EVAL_GPU='0',
        PORT=str(args.base_port), EVAL_RUNTIME_ROOT=str(local.parent/(local.name+'-sim-runtime')),
        PI05_MEM_FRACTION='0.35', SUBTASK_PREDICTION='1', PROVIDED_SUBTASK_AS_PROMPT='0', MAX_TOKEN_LEN='512',
        WRITE_VIDEO='1', PI05_SUBTASK_SAMPLE_PANEL=args.panel, PI05_SUBTASK_RAW_DATA_ROOT=str(raw_root),
        PI05_SUBTASK_COLLECTION_PLAN=args.collection_plan, SUBTASK_SAMPLE_MAX_STEPS=str(plan['max_steps']))
    for key in ('PI05_EVAL_QUEUE_DB', 'PI05_SUBTASK_PROBE_PLAN', 'EXTERNAL_SUBTASK', 'EXTERNAL_SUBTASK_SELECTOR',
                'SUBTASK_INTERVAL', 'SUBTASK_VOTE_WINDOW', 'SUBTASK_VOTE_THRESHOLD', 'INSTANCE_INDICES',
                'MAX_STEPS', 'GOALKEEPER_CONFIG_DIR'):
        environment.pop(key, None)
    command = ['/bin/bash', str(REPO_ROOT/'scripts/eval_b1k.sh'), *sorted({s['task'] for s in samples})]
    supervise_results(command, local, persistent, env=environment)
    summary = validate_collection(local, panel, plan)
    (persistent/'collection_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    # Completion of attempts does not imply there are exportable successes.
    (persistent/'_SUCCESS').write_text(json.dumps(dict(kind=KIND, complete=True,
        ready_for_export=summary['ready_for_export'], plan_fingerprint=plan['fingerprint']))+'\n')


def run(args):
    panel, inventory = load_inputs(args.panel, args.inventory)
    if getattr(args, 'collection_plan', None):
        if getattr(args, 'validation_plan', None):raise ValueError('Collection and validation are mutually exclusive')
        return run_collection(args, panel, inventory)
    if getattr(args, "validation_plan", None):
        return run_validation(args, panel, inventory)
    local_root, persistent = Path(args.run_root), Path(args.persistent_output)
    if local_root.exists():
        raise ValueError("Diagnostic requires a new local run root")
    persistent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.panel, persistent / "panel.json")
    shutil.copyfile(args.inventory, persistent / "raw_inventory.json")
    if args.workers not in (4, 8):
        raise ValueError("Full evaluation requires four or eight workers")
    raw_root = stage_raw(inventory, args.raw_source, args.raw_root)
    command = [sys.executable, str(REPO_ROOT / "scripts/eval_b1k_subtask_dynamic.py"), "start",
               "--run-root", str(local_root), "--run-id", args.run_id,
               "--checkpoint", args.checkpoint, "--repo-id", args.repo_id,
               "--panel", args.panel, "--raw-data-root", str(raw_root),
               "--gpus", *map(str, range(args.workers)), "--base-port", str(args.base_port),
               "--sample-max-steps", str(args.sample_max_steps), "--subtask-prediction",
               "--eval-env", args.eval_env, "--pi05-env", args.pi05_env,
               "--omnigibson-data-path", args.asset_path]
    source_record = {"run_id": args.run_id, "profile": "subtasks100", "checkpoint_source": args.checkpoint_source,
                     "panel_fingerprint": panel["fingerprint"],
                     "tasks": panel["selection"]["tasks"], "expected_episode_jobs": len(panel["episodes"]),
                     "expected_segment_rollouts": len(panel["samples"]), "expected_exclusions": len(panel["exclusions"]),
                     "workers": args.workers, "sample_max_steps": args.sample_max_steps,
                     "diagnostic": True, "official_benchmark": False}
    # Step identity is obtained from the verified staged checkpoint, not inferred
    # from a run name or hardcoded model setting.
    ready = json.loads((Path(args.checkpoint_source) / "_READY").read_text())
    source_record["checkpoint_step"] = ready.get("step")
    source_record["checkpoint_manifest_sha256"] = ready.get("manifest_sha256")
    from subtask_paired_retest import evaluator_digest
    source_record["evaluator_code_sha256"] = evaluator_digest()
    conditioning_path = Path(args.checkpoint_source)/"assets/subtask_conditioning.json"
    source_record["conditioning"] = (json.loads(conditioning_path.read_text()).get("action_conditioning")
                                      if conditioning_path.is_file() else "task_and_subtask")
    source_record["paired_retest_seed"] = os.environ.get("PI05_PAIRED_RETEST_SEED", "")
    if source_record["paired_retest_seed"]:
        import hashlib
        runtime_files = {"policy_environment": Path(os.environ["PI05_ENV_ARCHIVE"]),
                         "evaluation_environment": Path(os.environ["EVAL_ENV_ARCHIVE"]),
                         "assets_manifest": Path(os.environ["ASSET_ROOT"])/"SHA256SUMS",
                         "norm_stats": Path(args.checkpoint_source)/"assets"/args.repo_id/"norm_stats.json",
                         "runtime_requirements": REPO_ROOT/"nebula/requirements-eval-runtime.txt"}
        identity = {}
        for key, path in runtime_files.items():
            with path.open("rb") as stream:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                    hasher.update(chunk)
                identity[key] = hasher.hexdigest()
        source_record["runtime_identity"] = identity
    (persistent / "submission_contract.json").write_text(json.dumps(source_record, indent=2) + "\n")
    supervise_results(command, local_root, persistent)
    summary = json.loads((local_root / "summary.json").read_text())
    if not (local_root / "_SUCCESS").is_file() or summary.get("complete") is not True:
        raise RuntimeError("Diagnostic lacks strict successful completion")
    write_reports(summary, persistent)
    shutil.copyfile(local_root / "_SUCCESS", persistent / "_SUCCESS")
    print(f"Subtask diagnostic complete: {persistent}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--panel", required=True)
    check.add_argument("--inventory", required=True)
    check.add_argument("--raw-source", required=True)
    execute = commands.add_parser("run")
    for name in ("panel", "inventory", "raw-source", "raw-root", "run-root", "run-id", "persistent-output",
                 "checkpoint", "checkpoint-source", "repo-id", "eval-env", "pi05-env", "asset-path"):
        execute.add_argument("--" + name, required=True)
    execute.add_argument("--workers", type=int, choices=(1, 4, 8), required=True)
    execute.add_argument("--validation-plan")
    execute.add_argument("--collection-plan")
    execute.add_argument("--base-port", type=int, required=True)
    execute.add_argument("--sample-max-steps", type=int, default=2048)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.panel, args.inventory, args.raw_source)
    else:
        run(args)


if __name__ == "__main__":
    main()
