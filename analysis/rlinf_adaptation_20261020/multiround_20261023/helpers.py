from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType


class InvalidSampleState(RuntimeError):
    pass


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
