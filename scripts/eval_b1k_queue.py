#!/usr/bin/env python3
"""Queue-backed wrapper around the baseline OmniGibson evaluator."""

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from pi05_eval_queue import EvaluationQueue

FULL_EPISODE_SCRATCH_BASES = (Path("/var/tmp/behavior-full-episode"), Path("/tmp/behavior-full-episode"))


def _nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must be nonempty")
    return value


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _load_base_module() -> ModuleType:
    return importlib.import_module("omnigibson.eval.eval")


def parse_args(
    argv: list[str] | None = None, *, base_module: ModuleType | None = None
) -> argparse.Namespace:
    """Parse queue flags here and delegate all ordinary flags to OmniGibson."""
    argv = list(sys.argv[1:] if argv is None else argv)
    queue_parser = argparse.ArgumentParser(add_help=False)
    queue_parser.add_argument("--queue-db", type=_nonempty)
    queue_parser.add_argument("--queue-worker", type=_nonempty)
    queue_parser.add_argument("--queue-run-id", type=_nonempty)
    queue_parser.add_argument("--queue-generation", type=_nonnegative_int)
    queue_parser.add_argument("--queue-attempt-root", type=_nonempty)
    queue_parser.add_argument("--full-episode-collection-manifest", type=Path,
                              help="Opt-in native reset-to-terminal trajectory capture manifest.")
    queue_args, base_argv = queue_parser.parse_known_args(argv)

    base = base_module or _load_base_module()
    args = base.parse_args(base_argv)
    queue_values = (
        queue_args.queue_db,
        queue_args.queue_worker,
        queue_args.queue_run_id,
        queue_args.queue_generation,
        queue_args.queue_attempt_root,
    )
    if not all(value is not None for value in queue_values):
        queue_parser.error(
            "--queue-db, --queue-worker, --queue-run-id, --queue-generation, and "
            "--queue-attempt-root are all required by the queue evaluator"
        )
    if isinstance(args.num_rollouts, bool) or args.num_rollouts <= 0:
        queue_parser.error("--num-rollouts must be positive")

    for name, value in vars(queue_args).items():
        setattr(args, name, value)
    return args


def _build_evaluator_cfg(base: ModuleType, args: argparse.Namespace, seed: int):
    robot_config = None
    if args.robot_config is not None:
        robot_config_path = Path(args.robot_config).expanduser()
        robot_config = base.OmegaConf.load(str(robot_config_path))
        base.logger.info(f"Loaded robot config from {robot_config_path}")

    if args.policy == "websocket":
        model_cfg = {
            "_target_": "omnigibson.eval.policies.WebsocketPolicy",
            "host": args.host,
            "port": args.port,
            "expected_server_token": args.expected_server_token,
            "expected_subtask_vote_action_horizon": args.subtask_vote_action_horizon,
            "expected_subtask_vote_interval": args.subtask_vote_interval,
            "expected_subtask_vote_window": args.subtask_vote_window,
            "expected_subtask_vote_threshold": args.subtask_vote_threshold,
        }
    else:
        model_cfg = {
            "_target_": "omnigibson.eval.policies.LocalPolicy",
            "action_dim": None,
        }

    return base.OmegaConf.create(
        {
            "env_wrapper": {"_target_": args.env_wrapper},
            "policy_name": args.policy,
            "model": model_cfg,
            "headless": args.headless,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": args.write_video,
            "mode": args.mode,
            "prompt_mode": args.prompt_mode,
            "external_subtask": args.external_subtask,
            "external_subtask_selector": args.external_subtask_selector,
            "selector_debounce_refreshes": args.selector_debounce_refreshes,
            "subtask_vote_action_horizon": args.subtask_vote_action_horizon,
            "subtask_vote_interval": args.subtask_vote_interval,
            "subtask_vote_window": args.subtask_vote_window,
            "subtask_vote_threshold": args.subtask_vote_threshold,
            "seed": seed,
            "task": {"name": args.task_name},
            "robot": robot_config,
        }
    )


def _validate_queue_metadata(queue: EvaluationQueue, args: argparse.Namespace) -> None:
    metadata = queue.metadata(readonly=True)
    expected = {
        "run_id": args.queue_run_id,
        "mode": args.mode,
        "num_rollouts": args.num_rollouts,
        "require_videos": args.write_video,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"queue metadata does not match evaluator arguments: {mismatches}"
        )


def run_task_instance(
    base: ModuleType,
    evaluator: object,
    args: argparse.Namespace,
    instance_id: int,
    output_dir: str | Path,
) -> list[dict]:
    output_dir = Path(output_dir).expanduser()
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    if args.write_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    try:
        evaluator.reset(prepare_for_task_instance=True)
        evaluator.load_task_instance(int(instance_id))
    except Exception:
        base.logger.exception(f"Failed to load task instance {instance_id}.")
        raise

    results = []
    for rollout_id in range(args.num_rollouts):
        video_path = video_dir / f"{args.task_name}_{instance_id}_{rollout_id}.mp4"
        trajectory = None
        try:
            evaluator.reset()
            manifest_path = getattr(args, "full_episode_collection_manifest", None)
            if manifest_path is not None:
                from full_episode_trajectory import (
                    NativeFullEpisodeTrajectory, capture_observation, load_collection_manifest,
                )
                manifest, manifest_sha = load_collection_manifest(manifest_path)
                if manifest["run_id"] != args.queue_run_id:
                    raise ValueError("Collection run_id differs from the queue run")
                if (args.mode != "public_test" or args.prompt_mode != "whole_task"
                        or args.external_subtask is not None or args.external_subtask_selector is not None):
                    raise ValueError("Native episode capture requires unmodified public-test whole-task policy")
                attempt_root = Path(args.queue_attempt_root).expanduser().resolve()
                attempt_relative = output_dir.resolve().relative_to(attempt_root)
                attempt_id = attempt_relative.as_posix() + f"/rollout-{rollout_id}"
                if manifest.get("trajectory_root"):
                    trajectory_path = (Path(manifest["trajectory_root"]).expanduser().resolve()
                                       / args.task_name / str(instance_id) / attempt_relative
                                       / f"rollout-{rollout_id}.hdf5")
                else:
                    trajectory_path = output_dir / "trajectories" / f"{args.task_name}_{instance_id}_{rollout_id}.hdf5"
                identity = dict(task=args.task_name, instance_id=int(instance_id), rollout_id=rollout_id,
                                run_id=args.queue_run_id, attempt_id=attempt_id,
                                checkpoint=manifest["checkpoint"], fps=int(evaluator.env.env_config["rendering_frequency"]),
                                task_prompt=manifest["task_prompts"][args.task_name],
                                source_manifest_sha256=manifest["source_manifest_sha256"],
                                collection_manifest_sha256=manifest_sha)
                trajectory = NativeFullEpisodeTrajectory(trajectory_path, identity)
            if args.write_video:
                evaluator.start_recording(str(video_path), rate=args.video_fps)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                if trajectory is not None:
                    preceding_observation = capture_observation(evaluator.obs)
                terminated, truncated = evaluator.step()
                if trajectory is not None:
                    trajectory.record(preceding_observation, evaluator.robot_action,
                                      terminated, truncated, evaluator.observed_subtasks[-1],
                                      evaluator.observed_subtask_sources[-1])
                steps += 1

            base.verify_external_subtask_rollout(
                evaluator,
                external_subtask=args.external_subtask,
                external_subtask_selector=args.external_subtask_selector,
            )
            base.verify_subtask_voting_rollout(
                evaluator,
                enabled=args.subtask_vote_window is not None,
                steps=steps,
            )
            success = bool(evaluator.env.task.success)
            metrics = {}
            for metric in evaluator.metrics:
                metrics.update(metric.aggregate(evaluator.env))
            result = {
                "task": args.task_name,
                "instance_id": int(instance_id),
                "rollout_id": rollout_id,
                "steps": steps,
                "success": success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "conditioning": evaluator.conditioning_metadata,
                "prompt_events": evaluator.prompt_events,
                "subtask_events": evaluator.subtask_events,
                "subtask_vote_events": evaluator.subtask_vote_events,
                "subtask_vote_summary": evaluator.subtask_vote_summary,
                "selector_events": evaluator.selector_events,
                "selector_summary": evaluator.selector_summary,
                "final_goal_state": evaluator.final_goal_state,
                **metrics,
            }
            if trajectory is not None:
                from full_episode_trajectory import (
                    KIND, SCHEMA_VERSION, file_sha256, inspect_full_episode_trajectory,
                )
                if load_collection_manifest(manifest_path)[1] != manifest_sha:
                    raise ValueError("Collection manifest changed during the episode")
                result = trajectory.finish(result, capture_observation(evaluator.obs))
                inspected = inspect_full_episode_trajectory(trajectory.path, require_success=False)
                reference = dict(path=str(trajectory.path.resolve()), bytes=trajectory.path.stat().st_size,
                                 sha256=file_sha256(trajectory.path), kind=KIND, schema_version=SCHEMA_VERSION,
                                 steps=inspected["steps"], content_sha256=inspected["content_sha256"],
                                 subtask_sha256=inspected["subtask_sha256"],
                                 terminal_observation_sha256=inspected["terminal_observation_sha256"],
                                 collection_manifest_sha256=manifest_sha)
                if manifest.get("object_storage"):
                    from full_episode_checkpoint import checkpoint_episode
                    result["full_episode_object_checkpoint"] = checkpoint_episode(
                        trajectory.path, result, manifest, attempt_id=identity["attempt_id"])
                result["full_episode_trajectory"] = reference
            out_path = json_dir / f"{args.task_name}_{instance_id}_{rollout_id}.json"
            with out_path.open("w") as stream:
                json.dump(result, stream, indent=2, default=float)
            q_score = metrics.get("q_score", {}).get("final")
            video_msg = f" | video -> {video_path}" if args.write_video else ""
            base.logger.info(
                f"Result: instance={instance_id} rollout={rollout_id} steps={steps} "
                f"success={success} q_score={q_score} -> {out_path}{video_msg}"
            )
            results.append(result)
        except Exception:
            base.logger.exception(
                f"Instance {instance_id} rollout {rollout_id} failed."
            )
            raise
        finally:
            if trajectory is not None:
                trajectory.close()
            if args.write_video:
                evaluator.stop_recording()
    return results


def release_uploaded_trajectories(lease, args):
    """Remove only verified scratch copies after their canonical JSON is durable.

    A failure here retains all affected files. It must never turn a successfully
    uploaded episode into a request to run the policy again.
    """
    manifest_path = getattr(args, "full_episode_collection_manifest", None)
    if manifest_path is None:
        return 0
    from full_episode_trajectory import file_sha256, inspect_full_episode_trajectory, load_collection_manifest
    manifest, manifest_sha = load_collection_manifest(manifest_path)
    if manifest.get("release_uploaded_scratch") is not True:
        return 0
    if not manifest.get("object_storage") or not manifest.get("trajectory_root"):
        raise ValueError("Scratch release requires completed object checkpoints")
    scratch = Path(manifest["trajectory_root"]).expanduser().resolve()
    attempts = Path(args.queue_attempt_root).expanduser().resolve()
    if (not any(scratch.is_relative_to(base.resolve()) and scratch != base.resolve()
                for base in FULL_EPISODE_SCRATCH_BASES)
            or scratch == attempts or scratch.is_relative_to(attempts) or attempts.is_relative_to(scratch)
            or any(scratch.is_relative_to(Path(base)) for base in ("/mnt/afs", "/afs"))):
        raise ValueError("Refusing to release any non-scratch or queue-persistent path")
    validated = []
    for rollout_id in range(args.num_rollouts):
        canonical = Path(args.output_dir).expanduser().resolve() / "json" / f"{lease.task}_{lease.instance_id}_{rollout_id}.json"
        result = json.loads(canonical.read_text())
        reference, receipt = result["full_episode_trajectory"], result["full_episode_object_checkpoint"]
        path = Path(reference["path"])
        resolved = path.resolve()
        if path.is_symlink() or resolved == scratch or not resolved.is_relative_to(scratch):
            raise ValueError("Uploaded raw trajectory is outside its private scratch directory")
        key = f"{lease.task}:{lease.instance_id}:{rollout_id}"
        archive = receipt["archive"]
        expected_archive = manifest["object_storage"][key]["archive"]
        expected_attempt = (Path(lease.attempt_dir).resolve().relative_to(attempts).as_posix()
                            + f"/rollout-{rollout_id}")
        expected_result = {name: value for name, value in result.items()
                           if name not in ("full_episode_trajectory", "full_episode_object_checkpoint")}
        if (receipt.get("kind") != "native_full_episode_object_checkpoint"
                or receipt.get("run_id") != manifest["run_id"]
                or receipt.get("source_manifest_sha256") != manifest["source_manifest_sha256"]
                or receipt.get("attempt_id") != expected_attempt
                or (receipt.get("task"), receipt.get("instance_id"), receipt.get("rollout_id"))
                   != (lease.task, lease.instance_id, rollout_id)
                or receipt.get("result") != expected_result
                or archive.get("object_key") != expected_archive["key"]
                or type(archive.get("bytes")) is not int or archive["bytes"] <= 0
                or type(archive.get("parts")) is not int or archive["parts"] <= 0
                or not isinstance(archive.get("sha256"), str) or len(archive["sha256"]) != 64
                or not archive.get("etag") or archive.get("part_content_md5_verified") is not True
                or reference.get("collection_manifest_sha256") != manifest_sha
                or any(receipt.get(receipt_key) != reference.get(reference_key)
                       for receipt_key, reference_key in (("trajectory_sha256", "sha256"),
                                                         ("content_sha256", "content_sha256"),
                                                         ("subtask_sha256", "subtask_sha256")))):
            raise ValueError("Object receipt does not close the canonical episode")
        if (not resolved.is_file() or resolved.stat().st_size != reference["bytes"]
                or file_sha256(resolved) != reference["sha256"]):
            raise ValueError("Scratch raw trajectory bytes changed after the verified object upload")
        info = inspect_full_episode_trajectory(resolved, require_success=False)
        if (info["result"] != expected_result or info["metadata"]["attempt_id"] != receipt["attempt_id"]
                or info["metadata"].get("collection_manifest_sha256") != manifest_sha):
            raise ValueError("Scratch trajectory outcome or attempt differs from its accepted object receipt")
        validated.append(resolved)
    # Validate all rollouts before deleting any member of this accepted lease.
    for path in validated:
        path.unlink()
    return len(validated)


def run(
    args: argparse.Namespace,
    *,
    base_module: ModuleType | None = None,
    instance_runner: Callable | None = None,
) -> list[dict]:
    base = base_module or _load_base_module()
    instance_runner = instance_runner or run_task_instance
    base.gm.HEADLESS = args.headless
    seed = base.seed_everything(base.DEFAULT_EVAL_SEED)
    base.logger.info(f"Seeded Python, NumPy, and Torch with seed={seed}")
    if args.prompt_mode == "oracle_unsatisfied_goal":
        base.logger.warning(
            "Using privileged simulator BDDL goal state for policy prompts; "
            "these results are not directly comparable to whole-task conditioning."
        )
    if args.external_subtask_selector is not None:
        base.logger.warning(
            "Using privileged simulator state for subtask selection; "
            "this diagnostic is not an official benchmark result."
        )

    queue = EvaluationQueue(args.queue_db)
    _validate_queue_metadata(queue, args)
    worker = (
        f"{args.queue_run_id}:{args.queue_worker}:generation-{args.queue_generation}"
    )
    cfg = _build_evaluator_cfg(base, args, seed)
    results = []
    with base.Evaluator(cfg) as evaluator:
        while True:
            lease = queue.claim(args.task_name, worker, args.queue_attempt_root)
            if lease is None:
                break
            provenance = {
                "run_id": args.queue_run_id,
                "generation": args.queue_generation,
            }
            try:
                instance_results = instance_runner(
                    base, evaluator, args, lease.instance_id, lease.attempt_dir
                )
                queue.publish(
                    lease,
                    args.output_dir,
                    num_rollouts=args.num_rollouts,
                    require_videos=args.write_video,
                    provenance=provenance,
                )
                try:
                    release_uploaded_trajectories(lease, args)
                except Exception:
                    base.logger.warning("Retaining uploaded scratch: post-publication release validation failed.")
                results.extend(instance_results)
            except Exception as exc:
                try:
                    if queue.reconcile(
                        lease,
                        args.output_dir,
                        num_rollouts=args.num_rollouts,
                        require_videos=args.write_video,
                        provenance=provenance,
                    ):
                        try:
                            release_uploaded_trajectories(lease, args)
                        except Exception:
                            base.logger.warning("Retaining uploaded scratch: reconciled release validation failed.")
                        base.logger.warning(
                            f"Reconciled queue job {lease.task}/{lease.instance_id} after "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                except Exception:
                    base.logger.exception(
                        f"Failed to reconcile queue lease {lease.lease_token}."
                    )
                try:
                    state = queue.fail(lease, f"{type(exc).__name__}: {exc}")
                    base.logger.error(
                        f"Queue job {lease.task}/{lease.instance_id} attempt {lease.attempt} failed; state={state}."
                    )
                except Exception:
                    base.logger.exception(
                        f"Failed to record queue failure for lease {lease.lease_token}."
                    )
                raise

    count = len(results)
    successes = sum(result["success"] for result in results)
    mean_q = (
        sum(result.get("q_score", {}).get("final", 0.0) for result in results) / count
        if count
        else 0.0
    )
    base.logger.info(
        f"Eval summary: {successes}/{count} success | mean q_score={mean_q:.3f} | task={args.task_name}"
    )
    return results


def main(argv: list[str] | None = None) -> None:
    base = _load_base_module()
    run(parse_args(argv, base_module=base), base_module=base)


if __name__ == "__main__":
    main()
