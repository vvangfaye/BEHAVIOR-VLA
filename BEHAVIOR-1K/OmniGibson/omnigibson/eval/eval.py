"""Websocket evaluation runner for the BEHAVIOR-1K challenge.

Drives the OmniGibson ``Evaluator`` against a policy served over a websocket
(e.g. the openpi or GR00T ``scripts/b1k/serve_b1k.py`` server). For each test instance of a
task it runs a rollout and writes a per-rollout result JSON compatible with
``omnigibson/eval/utils/score_utils.py`` (``q_score``, ``time``,
``agent_distance`` / ``normalized_agent_distance``).

Example:
    python -m omnigibson.eval.eval \
        --task-name turning_on_radio \
        --robot-config omnigibson/eval/r1pro.yaml \
        --mode public_test \
        --host 127.0.0.1 --port 8000 \
        --instance-indices 0 --max-steps 500 \
        --output-dir outputs/b1k_eval --write-video
"""

import argparse
import json
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

from omnigibson.eval.evaluator import (
    Evaluator,
    normalize_subtask_text,
    resolve_instance_ids,
    validate_subtask_voting_config,
)
from omnigibson.eval.subtask_selectors import DEBOUNCE_REFRESHES, STATE_MACHINE_V1, SUPPORTED_TASKS
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def _non_empty_string(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must be a non-empty string")
    return value


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _validate_conditioning_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.selector_debounce_refreshes is not None and args.external_subtask_selector != STATE_MACHINE_V1:
        parser.error("--selector-debounce-refreshes requires --external-subtask-selector state_machine_v1")
    conditioning_arg = (
        "--external-subtask-selector" if args.external_subtask_selector is not None else "--external-subtask"
    )
    if args.external_subtask is None and args.external_subtask_selector is None:
        return
    if args.policy != "websocket":
        parser.error(f"{conditioning_arg} requires --policy websocket")
    if args.prompt_mode != "whole_task":
        parser.error(f"{conditioning_arg} requires --prompt-mode whole_task and cannot be used in oracle mode")
    if args.external_subtask_selector is not None and args.task_name not in SUPPORTED_TASKS:
        parser.error(
            f"--external-subtask-selector {args.external_subtask_selector} only supports tasks "
            f"{sorted(SUPPORTED_TASKS)}, got {args.task_name!r}"
        )


def verify_external_subtask_rollout(
    evaluator: Evaluator,
    *,
    external_subtask: str | None,
    external_subtask_selector: str | None,
) -> None:
    """Fail closed if the server did not use the requested external conditioning."""
    if external_subtask is None and external_subtask_selector is None:
        return

    observed_sources = evaluator.observed_subtask_sources
    observed_subtasks = evaluator.observed_subtasks
    if not observed_sources or any(source != "provided" for source in observed_sources):
        raise RuntimeError(
            "External subtask conditioning requires server source 'provided' on every request, "
            f"got {observed_sources!r}."
        )

    normalized_observed = [normalize_subtask_text(subtask) for subtask in observed_subtasks]
    if external_subtask_selector is None:
        expected_subtasks = [normalize_subtask_text(external_subtask)] * len(observed_subtasks)
        if not normalized_observed or normalized_observed != expected_subtasks:
            raise RuntimeError(
                "External subtask conditioning requires the server to report the requested text on every action, "
                f"expected {expected_subtasks!r}, got {normalized_observed!r}."
            )
        return

    selector_summary = evaluator.selector_summary
    selector_events = evaluator.selector_events
    if not selector_summary or selector_summary.get("valid") is not True:
        raise RuntimeError(f"External subtask selector did not produce a valid summary: {selector_summary!r}")
    if not selector_events:
        raise RuntimeError("External subtask selector did not produce any events")
    if selector_summary.get("validated_action_requests") != len(normalized_observed):
        raise RuntimeError(
            "External subtask selector did not validate every action request: "
            f"validated={selector_summary.get('validated_action_requests')!r}, "
            f"observed={len(normalized_observed)}."
        )
    if not normalized_observed or any(subtask is None for subtask in normalized_observed):
        raise RuntimeError("External subtask selector reported an empty subtask for an action request")


def verify_subtask_voting_rollout(evaluator: Evaluator, *, enabled: bool, steps: int) -> None:
    if not enabled:
        return
    events = evaluator.subtask_vote_events
    summary = evaluator.subtask_vote_summary
    if not events or events[0].get("reason") != "initial" or events[0].get("accepted") is not True:
        raise RuntimeError("Temporal subtask voting did not record an accepted initial vote")
    if not summary or summary.get("total_votes") != len(events):
        raise RuntimeError("Temporal subtask voting summary does not match recorded vote events")
    expected_votes = (steps - 1) // (
        evaluator.subtask_vote_action_horizon * evaluator.subtask_vote_interval
    ) + 1
    if len(events) != expected_votes:
        raise RuntimeError(
            f"Temporal subtask voting recorded {len(events)} votes, expected {expected_votes} for {steps} steps"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="BEHAVIOR task name, e.g. turning_on_radio.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy websocket server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy websocket server port.")
    parser.add_argument(
        "--expected-server-token",
        type=_non_empty_string,
        default=None,
        help="Reject a websocket server that was not launched for this evaluator.",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help=(
            "Optional path to YAML/JSON file containing one complete robot config dictionary with canonical "
            "'model' and 'name' fields. Add eval.camera_sensor_names to configure eval camera roles."
        ),
    )
    parser.add_argument(
        "--instance-indices",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Instance indices for the selected mode. For train these are direct train instance IDs; "
            "for public_test / hidden_test these index into that 20-instance split."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("train", "public_test", "hidden_test"),
        default="public_test",
        help="Instance split to evaluate. Default: public_test.",
    )
    parser.add_argument("--num-rollouts", type=int, default=1, help="Rollouts per instance.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode timeout in steps. Default (None) = 1.5x mean human-demo length.",
    )
    parser.add_argument(
        "--env-wrapper",
        default="omnigibson.eval.wrappers.DefaultWrapper",
        help="Target path of the EnvironmentWrapper to apply.",
    )
    parser.add_argument(
        "--policy",
        choices=("websocket", "local"),
        default="websocket",
        help="Policy backend to use. local emits zero actions and is intended for eval smoke tests.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("whole_task", "oracle_unsatisfied_goal"),
        default="whole_task",
        help="Policy prompt source. oracle_unsatisfied_goal uses privileged simulator goal state.",
    )
    external_subtask_group = parser.add_mutually_exclusive_group()
    external_subtask_group.add_argument(
        "--external-subtask",
        type=_non_empty_string,
        default=None,
        help="Fixed subtask text sent with every policy request (websocket + whole_task only).",
    )
    external_subtask_group.add_argument(
        "--external-subtask-selector",
        choices=(STATE_MACHINE_V1,),
        default=None,
        help="Privileged live-state subtask selector (websocket + whole_task on supported tasks only).",
    )
    parser.add_argument(
        "--selector-debounce-refreshes",
        type=_positive_int,
        default=None,
        help=f"Consecutive selector completion observations required before transition (default: {DEBOUNCE_REFRESHES}).",
    )
    parser.add_argument("--subtask-vote-action-horizon", type=_positive_int, default=None)
    parser.add_argument("--subtask-vote-interval", type=_positive_int, default=None)
    parser.add_argument("--subtask-vote-window", type=_positive_int, default=None)
    parser.add_argument("--subtask-vote-threshold", type=_positive_int, default=None)
    parser.add_argument("--output-dir", default="/tmp/b1k_eval", help="Where to write result JSONs.")
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save an MP4 rollout video (head + wrist cameras) per rollout under <output-dir>/videos.",
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Frame rate for saved rollout videos.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OmniGibson headless (default: True).",
    )
    args = parser.parse_args(argv)
    _validate_conditioning_args(parser, args)
    try:
        validate_subtask_voting_config(
            action_horizon=args.subtask_vote_action_horizon,
            interval=args.subtask_vote_interval,
            window=args.subtask_vote_window,
            threshold=args.subtask_vote_threshold,
            external_subtask=args.external_subtask,
            external_subtask_selector=args.external_subtask_selector,
            policy_name=args.policy,
            prompt_mode=args.prompt_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()

    gm.HEADLESS = args.headless

    seed = seed_everything(DEFAULT_EVAL_SEED)
    logger.info(f"Seeded Python, NumPy, and Torch with seed={seed}")
    if args.prompt_mode == "oracle_unsatisfied_goal":
        logger.warning(
            "Using privileged simulator BDDL goal state for policy prompts; "
            "these results are not directly comparable to whole-task conditioning."
        )
    if args.external_subtask_selector is not None:
        logger.warning(
            "Using privileged simulator state for subtask selection; "
            "this diagnostic is not an official benchmark result."
        )

    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, mode=args.mode)
    logger.info(f"Resolved {args.mode} instance ids for {args.task_name}: {instance_ids}")

    robot_config = None
    if args.robot_config is not None:
        robot_config_path = Path(args.robot_config).expanduser()
        robot_config = OmegaConf.load(str(robot_config_path))
        logger.info(f"Loaded robot config from {robot_config_path}")

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
        model_cfg = {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}

    cfg = OmegaConf.create(
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

    json_dir = os.path.join(os.path.expanduser(args.output_dir), "json")
    os.makedirs(json_dir, exist_ok=True)
    video_dir = os.path.join(os.path.expanduser(args.output_dir), "videos")
    if args.write_video:
        os.makedirs(video_dir, exist_ok=True)

    results = []
    with Evaluator(cfg) as evaluator:
        for instance_id in instance_ids:
            try:
                evaluator.reset(prepare_for_task_instance=True)
                evaluator.load_task_instance(int(instance_id))
            except Exception:
                logger.exception(f"Failed to load task instance {instance_id}.")
                raise
            for rollout_id in range(args.num_rollouts):
                video_path = os.path.join(video_dir, f"{args.task_name}_{instance_id}_{rollout_id}.mp4")
                try:
                    evaluator.reset()
                    if args.write_video:
                        evaluator.start_recording(video_path, rate=args.video_fps)
                    terminated = truncated = False
                    steps = 0
                    while not (terminated or truncated):
                        terminated, truncated = evaluator.step()
                        steps += 1

                    verify_external_subtask_rollout(
                        evaluator,
                        external_subtask=args.external_subtask,
                        external_subtask_selector=args.external_subtask_selector,
                    )
                    verify_subtask_voting_rollout(
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
                    out_path = os.path.join(json_dir, f"{args.task_name}_{instance_id}_{rollout_id}.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2, default=float)
                    q_score = metrics.get("q_score", {}).get("final")
                    video_msg = f" | video -> {video_path}" if args.write_video else ""
                    logger.info(
                        f"Result: instance={instance_id} rollout={rollout_id} steps={steps} "
                        f"success={success} q_score={q_score} -> {out_path}{video_msg}"
                    )
                    results.append(result)
                except Exception:
                    logger.exception(f"Instance {instance_id} rollout {rollout_id} failed.")
                    raise
                finally:
                    if args.write_video:
                        evaluator.stop_recording()

    n = len(results)
    n_success = sum(r["success"] for r in results)
    mean_q = (sum(r.get("q_score", {}).get("final", 0.0) for r in results) / n) if n else 0.0
    logger.info(f"Eval summary: {n_success}/{n} success | mean q_score={mean_q:.3f} | task={args.task_name}")


if __name__ == "__main__":
    main()
