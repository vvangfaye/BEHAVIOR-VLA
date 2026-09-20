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
        try:
            evaluator.reset()
            if args.write_video:
                evaluator.start_recording(str(video_path), rate=args.video_fps)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                terminated, truncated = evaluator.step()
                steps += 1

            base.verify_external_subtask_rollout(
                evaluator,
                external_subtask=args.external_subtask,
                external_subtask_selector=args.external_subtask_selector,
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
                "selector_events": evaluator.selector_events,
                "selector_summary": evaluator.selector_summary,
                "final_goal_state": evaluator.final_goal_state,
                **metrics,
            }
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
            if args.write_video:
                evaluator.stop_recording()
    return results


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
