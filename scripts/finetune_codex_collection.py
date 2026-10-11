#!/usr/bin/env python3
"""Finite joint fine-tuning of verified mixed Codex data from checkpoint 1688977.

Preparation and a two-step model/data smoke must precede the finite full phase.
This helper never submits cloud jobs and never resumes or overwrites another run.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

from finetune_subtask_collection import read, sha, tree, verify_checkpoint as _verify_checkpoint
from finetune_subtask_collection import environment as _environment
from prepare_codex_training_data import verify_mixed, EXPORT_KIND
from subtask_improvement import digest

KIND = "codex_teacher_collection_finetuning"
TRAINED_CHECKPOINT_KIND = "codex_pilot_trained_checkpoint_v1"
REPO = Path(__file__).resolve().parents[1]
BASE_STEP = 1688977
DEFAULT_COLLECTION_CHECKPOINT = "pi05_b1k_full100_joint_1688977"


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), size=path.stat().st_size)


def verify_checkpoint(root, identity, repo_id):
    if identity.get("manifest_format") == "aoss_upload_plan_v1":
        root = Path(root).resolve()
        manifest = Path(identity["checkpoint_manifest_path"])
        ready, source = read(root / "_READY"), read(manifest)
        if sha(manifest) != identity["checkpoint_manifest_sha256"] or ready.get("checkpoint_step") != identity["checkpoint_step"] or source.get("checkpoint_step") != identity["checkpoint_step"]:
            raise ValueError("AOSS checkpoint readiness or manifest differs from frozen identity")
        entries = {}
        for row in source["files"]:
            name, expected = row["relative_path"], row["sha256"]
            relative = PurePosixPath(name)
            path = root / name
            if relative.is_absolute() or ".." in relative.parts or name in entries or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Unsafe or repeated AOSS checkpoint manifest path")
            if path.resolve() != root / name or not path.is_file() or path.stat().st_size != row["size"] or sha(path) != expected:
                raise ValueError("AOSS checkpoint content changed: " + name)
            entries[name] = expected
        norm = f"assets/{repo_id}/norm_stats.json"
        required = {"params/_METADATA", "_CHECKPOINT_METADATA", "assets/subtask_conditioning.json", norm}
        params = {p.relative_to(root).as_posix() for p in (root / "params").rglob("*") if p.is_file()}
        if not required <= entries.keys() or not params or not params <= entries.keys() or (root / "model.safetensors").exists():
            raise ValueError("AOSS manifest does not completely verify JAX inference parameters")
        if read(root / "assets/subtask_conditioning.json").get("action_conditioning") != "task_and_subtask":
            raise ValueError("AOSS initialization conditioning differs from joint training")
        stats = read(root / norm)["norm_stats"]
        for feature in ("state", "actions"):
            values = stats[feature]
            for key in ("mean", "std", "q01", "q99"):
                values_key = values[key]
                if len(values_key) != 23 or any(isinstance(v, bool) or not math.isfinite(v) for v in values_key):
                    raise ValueError("Baseline normalization needs 23 finite transformed dimensions")
            if any(v < 0 for v in values["std"]) or any(a > b for a, b in zip(values["q01"], values["q99"])):
                raise ValueError("Baseline normalization has invalid scales or quantiles")
        return entries, norm
    entries, norm = _verify_checkpoint(root, identity, repo_id)
    if "assets/subtask_conditioning.json" not in entries:
        raise ValueError("Checkpoint manifest must verify joint conditioning metadata")
    return entries, norm


def audit_export(root, *, collection_checkpoint_id=DEFAULT_COLLECTION_CHECKPOINT):
    root = Path(root)
    report = read(root / "export_report.json")
    if report.get("kind") != EXPORT_KIND or report.get("conditioning") != "task_and_subtask":
        raise ValueError("Training requires the isolated Codex mixed-data contract")
    verification = verify_mixed(root)
    if verification.get("complete") is not True or verification.get("real_lerobot_reader") is not True or verification.get("real_lerobot_writer") is not True or verification.get("action_horizon") != 32:
        raise ValueError("Training requires actual LeRobot video/source/action-window readback")
    provenance = read(root / "meta/collection_provenance.json")
    for row in provenance:
        if row["source_kind"] != "original" and row["source_metadata"].get("checkpoint") != collection_checkpoint_id:
            raise ValueError("Collected trajectories came from another baseline checkpoint")
    if verification["export_fingerprint"] != report["fingerprint"] or verification["frames"] != report["frames"]:
        raise ValueError("Verification differs from the mixed export")
    return report, provenance, verification


def prepare(dataset_root, checkpoint, base_repo_id, run_root, *, run_id, devices=4,
            batch_size=4, steps=256, collection_checkpoint_id=DEFAULT_COLLECTION_CHECKPOINT,
            expected_manifest_sha256=None, checkpoint_manifest=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("Training run ID must be a safe fresh name")
    if type(devices) is not int or devices not in (1, 2, 4, 8) or type(batch_size) is not int or not devices <= batch_size <= 64 or batch_size % devices or type(steps) is not int or not 128 <= steps <= 5000:
        raise ValueError("Pilot budget must be 128..5000 steps and a divisible batch <=64")
    run_root = Path(run_root).resolve()
    if run_root.exists():
        raise ValueError("Fine-tuning requires a fresh experiment root")
    for source in (dataset_root, checkpoint):
        source = Path(source).resolve()
        if source == run_root or source in run_root.parents or run_root in source.parents:
            raise ValueError("Training output must be separate from immutable inputs")
    ready = read(Path(checkpoint) / "_READY")
    if ready.get("checkpoint_step" if checkpoint_manifest else "step") != BASE_STEP:
        raise ValueError("Codex experiment must initialize from checkpoint 1688977")
    manifest_sha = sha(checkpoint_manifest) if checkpoint_manifest else ready.get("manifest_sha256")
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError("Baseline manifest differs from the frozen pilot identity")
    identity = dict(checkpoint_step=BASE_STEP, checkpoint_manifest_sha256=manifest_sha)
    if checkpoint_manifest:
        identity.update(manifest_format="aoss_upload_plan_v1", checkpoint_manifest_path=str(Path(checkpoint_manifest).resolve()))
    entries, norm = verify_checkpoint(checkpoint, identity, base_repo_id)
    report, provenance, verification = audit_export(dataset_root, collection_checkpoint_id=collection_checkpoint_id)
    if report["frames"] < batch_size:
        raise ValueError("Mixed data does not contain a full batch")
    payload = dict(schema_version=1, kind=KIND, training_run_id=run_id, conditioning="task_and_subtask",
        dataset_fingerprint=report["fingerprint"], init_checkpoint_identity=identity,
        export_provenance_sha256=digest(provenance), dataset_root=str(Path(dataset_root).resolve()),
        checkpoint=str(Path(checkpoint).resolve()), base_repo_id=base_repo_id,
        repo_id="b1k_codex_" + run_id, run_root=str(run_root), dataset_files=tree(dataset_root),
        checkpoint_files=entries, base_norm_path=norm, base_norm_sha256=entries[norm],
        export_verification=verification, devices=devices, batch_size=batch_size, steps=steps,
        smoke_steps=2, training_action_horizon=32, execution_horizon=16,
        optimizer_resume=False, video_backend="pyav", seed=42,
        collection_checkpoint_id=collection_checkpoint_id, collection_instances=list(range(311, 321)),
        evaluation_instances=list(range(301, 311)), evaluation_agent_enabled=False,
        memory_plan=dict(parameter_dtype="float32", adam_moments=2, ema_decay=0.99,
                         measured_gpu_peak_bytes=None, requires_same_device_count_smoke=True,
                         default_device_reason="1688977 has49.97GiB persistent unsharded train state before gradients and activations"),
        tasks=report["tasks"], new_frame_fraction=report["new_frame_fraction"],
        learning_rate=dict(peak=1e-5, floor=1e-6, warmup=min(100, steps // 4), decay=steps))
    plan = {**payload, "fingerprint": digest(payload)}
    run_root.mkdir(parents=True)
    (run_root / "training_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def environment(plan, phase):
    env = _environment(plan, phase)
    env["CODEX_TRAINING_RUNTIME"] = "1"
    # Cluster environments retain editable-install pointers to runtime/code.
    # Use this experiment's verified source even when reusing that environment.
    env["PYTHONPATH"] = os.pathsep.join(str(REPO / path) for path in
        ("openpi/src", "openpi/packages/openpi-client/src"))
    # Four-device FSDP keeps persistent state small enough for a useful smoke.
    # The remaining activations/workspace still require actual GPU measurement.
    env["NUM_WORKERS"] = "0"
    env["SUBTASK_LOSS_CHUNK_SIZE"] = "4"
    return env


def run(plan, phase):
    if plan.get('fingerprint') != digest({k: v for k, v in plan.items() if k != 'fingerprint'}):
        raise ValueError('Training plan changed')
    if tree(plan['dataset_root']) != plan['dataset_files']:
        raise ValueError('Exported training files changed after preflight')
    verify_checkpoint(plan['checkpoint'], plan['init_checkpoint_identity'], plan['base_repo_id'])
    root = Path(plan['run_root']) / phase
    if root.exists():
        raise ValueError('Training phase already exists; inspect its evidence before any retry')
    if phase == 'full':
        smoke = read(Path(plan['run_root']) / 'smoke/phase_receipt.json')
        if smoke.get('complete') is not True or smoke.get('training_plan_fingerprint') != plan['fingerprint']:
            raise ValueError('Full fine-tuning requires successful training on this data and model')
    env = environment(plan, phase)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    if len(rows) < plan['devices']:
        raise ValueError('Allocated GPU count does not match fine-tuning plan')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if processes:
        raise ValueError('Training GPUs have existing compute processes; do not reuse unrelated allocations')
    root.mkdir()
    norm = Path(env['ASSETS_BASE_DIR']) / 'pi05_b1k' / plan['repo_id'] / 'norm_stats.json'
    norm.parent.mkdir(parents=True)
    shutil.copyfile(Path(plan['checkpoint']) / plan['base_norm_path'], norm)
    if sha(norm) != plan['base_norm_sha256']:
        raise ValueError('Fresh training assets changed the base normalization')
    (root / '_RUNNING').touch()
    try:
        with (root / 'train.log').open('x') as log:
            for mode in ('check', phase):
                subprocess.run(['/bin/bash', str(REPO / 'scripts/train_pi05_subtask_b1k.sh'), mode], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        proof = read(env['OPENPI_SUCCESS_PATH'])
        expected_steps = int(env['TRAIN_STEPS'])
        if proof.get('restored_step') != expected_steps or proof.get('action_horizon') != 32 or proof.get('conditioning_mode') != 'joint' or proof.get('global_devices') != plan['devices'] or proof.get('processes') != 1:
            raise ValueError('Actual final checkpoint restore differs from bounded joint training')
        checkpoint = Path(env['CHECKPOINT_BASE_DIR']) / 'pi05_b1k' / env['EXP_NAME'] / str(expected_steps - 1)
        if not (checkpoint / 'params/_METADATA').is_file() or read(checkpoint / 'assets/subtask_conditioning.json').get('action_conditioning') != 'task_and_subtask':
            raise ValueError('Saved model lacks joint inference parameters or matching conditioning')
        if read(checkpoint / 'assets' / plan['repo_id'] / 'norm_stats.json') != read(norm):
            raise ValueError('Saved checkpoint changed the base normalization values')
        receipt = dict(complete=True, kind=KIND, phase=phase, training_plan_fingerprint=plan['fingerprint'],
            training_run_id=plan['training_run_id'], conditioning='task_and_subtask',
            dataset_fingerprint=plan['dataset_fingerprint'],
            export_provenance_sha256=plan['export_provenance_sha256'],
            init_checkpoint_manifest_sha256=plan['init_checkpoint_identity']['checkpoint_manifest_sha256'],
            saved_checkpoint=str(checkpoint), checkpoint_restore_evidence=proof,
            training_log_sha256=sha(root / 'train.log'), base_norm_sha256=plan['base_norm_sha256'],
            output_checkpoint_manifest_sha256=None, inference_staging_complete=False)
        (root / 'phase_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt
    except BaseException as error:
        (root / '_FAILED').write_text(type(error).__name__ + '\n')
        raise
    finally:
        (root / '_RUNNING').unlink(missing_ok=True)


def stage_final(plan, destination):
    """Stage actual completed inference parameters; only then link both models."""
    if plan.get('fingerprint') != digest({k: v for k, v in plan.items() if k != 'fingerprint'}):
        raise ValueError('Training plan changed')
    if tree(plan['dataset_root']) != plan['dataset_files']:
        raise ValueError('Exported training files changed after training')
    verify_checkpoint(plan['checkpoint'], plan['init_checkpoint_identity'], plan['base_repo_id'])
    report, provenance, verification = audit_export(plan['dataset_root'],
        collection_checkpoint_id=plan['collection_checkpoint_id'])
    if (verification != plan['export_verification'] or report['fingerprint'] != plan['dataset_fingerprint']
            or digest(provenance) != plan['export_provenance_sha256']):
        raise ValueError('Real mixed-data readback differs from the frozen training plan')
    if (Path(plan['run_root']) / 'training_receipt.json').exists():
        raise ValueError('Training receipt already exists; inspect instead of repeating staging')
    root = Path(plan['run_root']) / 'full'
    receipt = read(root / 'phase_receipt.json')
    proof = read(root / 'checkpoint_restore_verified.json')
    if ((root / '_RUNNING').exists() or (root / '_FAILED').exists() or receipt.get('complete') is not True
            or receipt.get('kind') != KIND or receipt.get('phase') != 'full'
            or receipt.get('training_plan_fingerprint') != plan['fingerprint']):
        raise ValueError('Inference staging requires completed training on this frozen plan')
    if receipt.get('checkpoint_restore_evidence') != proof or proof.get('restored_step') != plan['steps'] or proof.get('action_horizon') != 32 or proof.get('conditioning_mode') != 'joint' or proof.get('global_devices') != plan['devices'] or proof.get('processes') != 1 or receipt.get('training_log_sha256') != sha(root / 'train.log'):
        raise ValueError('Actual training completion evidence changed or is mismatched')
    expected = root / 'checkpoints/pi05_b1k' / (plan['training_run_id'] + '_full') / str(plan['steps'] - 1)
    source = Path(receipt['saved_checkpoint'])
    if source.resolve() != expected.resolve():
        raise ValueError('Saved inference parameters belong to another experiment')
    if not (source / 'params/_METADATA').is_file() or read(source / 'assets/subtask_conditioning.json').get('action_conditioning') != 'task_and_subtask':
        raise ValueError('Saved joint checkpoint is missing required inference artifacts')
    norm_source = Path(plan['checkpoint']) / plan['base_norm_path']
    if sha(norm_source) != plan['base_norm_sha256'] or read(source / 'assets' / plan['repo_id'] / 'norm_stats.json') != read(norm_source):
        raise ValueError('Trained checkpoint normalization differs from the paired base model')
    destination = Path(destination).resolve()
    if destination.exists() or source.resolve() == destination or source.resolve() in destination.parents or destination in source.resolve().parents:
        raise ValueError('Inference staging requires a fresh separate directory')
    for path in (plan['dataset_root'], plan['checkpoint']):
        path = Path(path).resolve()
        if path == destination or path in destination.parents or destination in path.parents:
            raise ValueError('Inference output must be separate from immutable inputs')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Checkpoint staging rejects symlinks')
    source_params = tree(source / 'params')
    if source_params == tree(Path(plan['checkpoint']) / 'params'):
        raise ValueError('Trained parameter files are identical to the base checkpoint')
    required = sum(p.stat().st_size for area in ('params', 'assets') for p in (source / area).rglob('*') if p.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + 1024**3:
        raise ValueError('Insufficient disk to stage verified inference parameters plus reserve')
    destination.mkdir()
    for name in ('params', 'assets'):
        shutil.copytree(source / name, destination / name)
    shutil.copyfile(source / '_CHECKPOINT_METADATA', destination / '_CHECKPOINT_METADATA')
    # Preserve identical numeric statistics and exact runtime hash for retesting.
    shutil.copyfile(norm_source, destination / 'assets' / plan['repo_id'] / 'norm_stats.json')
    if (tree(destination / 'params') != source_params
            or sha(destination / 'assets' / plan['repo_id'] / 'norm_stats.json') != plan['base_norm_sha256']):
        raise ValueError('Staged inference readback differs from trained parameters or base normalization')
    entries = tree(destination)
    manifest = ''.join(value + '  ' + name + '\n' for name, value in sorted(entries.items()))
    (destination / 'SHA256SUMS.inference').write_text(manifest)
    manifest_sha = sha(destination / 'SHA256SUMS.inference')
    if manifest_sha == plan['init_checkpoint_identity']['checkpoint_manifest_sha256']:
        raise ValueError('Trained inference artifact is identical to the base manifest')
    (destination / '_READY').write_text(json.dumps(dict(step=plan['steps'] - 1,
        restored_train_state_step=plan['steps'], manifest_sha256=manifest_sha,
        init_checkpoint_step=1688977, training_run_id=plan['training_run_id'])) + '\n')
    checkpoint_id = 'pi05_b1k_codex_' + plan['training_run_id'] + '_' + str(plan['steps'])
    result = {**receipt, 'kind': TRAINED_CHECKPOINT_KIND, 'checkpoint_id': checkpoint_id,
              'parent_checkpoint_step': BASE_STEP, 'optimizer_steps': proof['restored_step'],
              'ready_for_evaluation': True, 'real_lerobot_readback_verified': True,
              'base_normalization_preserved': True, 'base_norm_sha256': plan['base_norm_sha256'],
              'repo_id': plan['repo_id'], 'checkpoint_manifest': file_ref(destination / 'SHA256SUMS.inference'),
              'training_plan': file_ref(Path(plan['run_root']) / 'training_plan.json'),
              'phase_receipt': file_ref(root / 'phase_receipt.json'),
              'checkpoint_restore_receipt': file_ref(root / 'checkpoint_restore_verified.json'),
              'export_verification': verification,
              'output_checkpoint_manifest_sha256': manifest_sha,
              'inference_staging_complete': True, 'staged_checkpoint': str(destination)}
    with (Path(plan['run_root']) / 'training_receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare")
    for name in ("dataset-root", "checkpoint", "run-root"):
        freeze.add_argument("--" + name, required=True, type=Path)
    freeze.add_argument("--base-repo-id", required=True)
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--devices", type=int, default=4)
    freeze.add_argument("--batch-size", type=int, default=4)
    freeze.add_argument("--steps", type=int, default=256)
    freeze.add_argument("--collection-checkpoint-id", default=DEFAULT_COLLECTION_CHECKPOINT)
    freeze.add_argument("--expected-manifest-sha256")
    freeze.add_argument("--checkpoint-manifest", type=Path,
                        help="Original AOSS upload plan for the deployed1688977 checkpoint; verifies every listed file")
    execute = commands.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--phase", choices=("smoke", "full"), required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--plan", type=Path, required=True)
    stage.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.dataset_root, args.checkpoint, args.base_repo_id, args.run_root,
            run_id=args.run_id, devices=args.devices, batch_size=args.batch_size, steps=args.steps,
            collection_checkpoint_id=args.collection_checkpoint_id,
            expected_manifest_sha256=args.expected_manifest_sha256, checkpoint_manifest=args.checkpoint_manifest)
    elif args.command == "run":
        result = run(read(args.plan), args.phase)
    else:
        result = stage_final(read(args.plan), args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
