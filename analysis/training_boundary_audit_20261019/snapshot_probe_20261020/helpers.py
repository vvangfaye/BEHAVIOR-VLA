from __future__ import annotations
from pathlib import Path
import copy

class InvalidSampleState(RuntimeError): pass


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

def _dump_full_simulator_state(scene: object, simulator: object) -> Any:
    registry = scene.object_registry
    previous_filter = registry._dump_filter
    registry.set_dump_filter(dump_filter=lambda obj: True)
    try:
        # Raw demo rows use the recording's field order, while a full runtime
        # dump contains all current states. Flattening that dump and reading it
        # with the demo schema shifts object UUID offsets. Keep named fields.
        state = simulator.dump_state(serialized=False)
        if not isinstance(state, dict):
            raise InvalidSampleState("Full simulator snapshot must be a named state dictionary")
        return copy.deepcopy(state)
    finally:
        registry.set_dump_filter(dump_filter=previous_filter)

def _validate_full_snapshot_objects(simulator: object, state: dict) -> None:
    """Refuse missing/extra objects that dictionary loading could silently skip."""
    if set(state) != set(range(len(simulator.scenes))):
        raise InvalidSampleState("Full snapshot scene indices do not match the simulator")
    for index, scene in enumerate(simulator.scenes):
        try:
            expected = state[index]["registry"]["object_registry"]
        except (KeyError, TypeError) as error:
            raise InvalidSampleState("Full snapshot lacks its named object registry") from error
        if not isinstance(expected, dict):
            raise InvalidSampleState("Full snapshot object registry must be a dictionary")
        actual = {obj.name for obj in scene.object_registry.objects}
        if set(expected) != actual:
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            raise InvalidSampleState(
                f"Full snapshot object topology differs in scene {index}: missing={missing}, extra={extra}"
            )
