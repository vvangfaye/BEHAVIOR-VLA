import dataclasses
import logging
import pathlib
import socket

import tyro

from openpi.configs.tasks import TASK_REGISTRY
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_b1k_server
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    robot: str
    task: str
    # Specifies how to load the policy.
    policy: Checkpoint

    # LeRobot repo id whose checkpoint assets contain the norm stats.
    repo_id: str | None = None
    control_mode: str = "receding_horizon"
    # Number of actions to execute before replanning.
    action_horizon: int = 16
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Use the joint subtask-trained checkpoint and generate text before actions.
    subtask_prediction: bool = False
    # Use a selector-provided subtask as the only language prompt for an action-only checkpoint.
    provided_subtask_as_prompt: bool = False
    max_token_len: int = 512
    max_subtask_len: int = 64
    # Refresh every N model action-chunk requests (not simulator frames).
    subtask_interval: int | None = None
    # None reads checkpoint assets; old checkpoints default to task_and_subtask.
    subtask_action_conditioning: str | None = None
    subtask_temperature: float = 0.0


def main(args: Args) -> None:
    # Load task from registry
    task_bucket, task_name = args.task.split("/")
    task_prompt = TASK_REGISTRY[task_bucket][task_name]
    # log the prompt used
    logging.info(f"Using robot: {args.robot}, prompt: {task_prompt}")

    # Load training config and override request-specific fields.
    config = _config.get_config(args.policy.config)
    if args.subtask_prediction and args.provided_subtask_as_prompt:
        raise ValueError("Subtask prediction and provided-subtask prompt modes are mutually exclusive.")
    if args.subtask_prediction:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                train_subtask_prediction=True,
                max_token_len=args.max_token_len,
                max_subtask_len=args.max_subtask_len,
                subtask_loss_chunk_size=min(config.model.subtask_loss_chunk_size, args.max_subtask_len),
            ),
        )
    elif args.provided_subtask_as_prompt:
        if not getattr(config.model, "pi05", False):
            raise ValueError("Provided-subtask prompt mode requires a π0.5 model.")
        if pathlib.Path(args.policy.dir, "model.safetensors").exists():
            raise ValueError("Provided-subtask prompt mode currently supports JAX checkpoints only.")
        if args.max_token_len != 200:
            raise ValueError("Provided-subtask prompt mode requires max_token_len=200.")
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                train_subtask_prediction=False,
                max_token_len=args.max_token_len,
            ),
        )
    norm_stats_repo_id = args.repo_id or args.task
    logging.info("Using norm stats for repo: %s", norm_stats_repo_id)
    config = dataclasses.replace(
        config, data=dataclasses.replace(config.data, repo_id=norm_stats_repo_id, robot_config_name=args.robot)
    )

    repack_transforms = (
        transforms.Group(inputs=[transforms.UseSubtaskAsPrompt()]) if args.provided_subtask_as_prompt else None
    )
    policy = _policy_config.create_trained_policy(
        config,
        args.policy.dir,
        repack_transforms=repack_transforms,
        default_prompt=task_prompt,
        subtask_interval=args.subtask_interval,
        subtask_temperature=args.subtask_temperature,
        subtask_action_conditioning=args.subtask_action_conditioning,
    )
    policy_metadata = dict(policy.metadata)
    if args.provided_subtask_as_prompt:
        policy_metadata["conditioning_mode"] = "provided_subtask_as_prompt"
    logging.info(
        "Loaded policy class=%s conditioning=%s max_token_len=%d",
        type(policy).__name__,
        policy_metadata.get("conditioning_mode", "default"),
        config.model.max_token_len,
    )

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    policy = B1KPolicyWrapper(
        policy=policy,
        robot=args.robot,
        text_prompt=task_prompt,
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        max_len=config.model.action_horizon,
        provided_subtask_as_prompt=args.provided_subtask_as_prompt,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_b1k_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
