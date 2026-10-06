import copy
import cv2
import json
import logging
import os
import sys
import traceback
from signal import SIGINT, signal
from typing import Any, List, Tuple

import numpy as np
import torch as th
from av.container import Container
from av.stream import Stream
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

import omnigibson as og
import omnigibson.utils.transform_utils as T
from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.utils.og_teleop_utils import load_available_tasks
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.eval.batched_evaluator import BatchedEvaluator, evaluate_instances_batched
from omnigibson.eval.single_env import SingleEnvironmentAdapter
from omnigibson.eval.subtask_selectors import (
    DEBOUNCE_REFRESHES,
    LEGACY_SEGMENTS_SHA256,
    SELECTOR_LABELS_SHA256,
    STALL_STEPS,
    STATE_MACHINE_V1,
    SUPPORTED_TASKS,
    TRAINING_ANNOTATION_SHA256,
    create_subtask_selector,
)
from omnigibson.eval.utils.eval_utils import (
    EVAL_TIMEOUT_MULTIPLIER,
    NUM_HIDDEN_TEST_INSTANCES,
    NUM_PUBLIC_TEST_INSTANCES,
    TASK_NAMES_TO_INDICES,
    TEST_INSTANCE_IDS,
    flatten_obs_dict,
    generate_basic_environment_config,
    get_robot_camera_names,
)
from omnigibson.eval.utils.obs_utils import create_video_writer, write_video
from omnigibson.eval.utils.score_utils import load_human_stats
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import Robot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.eval.utils.light_utils import LightToggleSynchronizer, set_light_control_toggles
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import create_module_logger


LIGHT_EVAL_TASKS = {"turning_out_all_lights_before_sleep"}
EVAL_BASE_LINK_MASS = 250.0
EVAL_HEAD_HORIZONTAL_APERTURE = 40.0
DEFAULT_ROBOT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "r1pro.yaml")
EVAL_MODES = ("train", "public_test", "hidden_test")
PROMPT_MODES = ("whole_task", "oracle_unsatisfied_goal")

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def _to_plain_dict(cfg: Any) -> dict | None:
    if cfg is None:
        return None
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def normalize_subtask_text(value: object) -> str | None:
    return " ".join(value.split()) if isinstance(value, str) and value.strip() else None


def validate_subtask_conditioning_config(
    *,
    external_subtask: object,
    external_subtask_selector: object,
    policy_name: object,
    prompt_mode: str,
    task_name: object,
) -> tuple[str | None, str | None]:
    """Validate config-level external conditioning without constructing an environment."""
    if external_subtask is not None and external_subtask_selector is not None:
        raise ValueError("External fixed subtask and external subtask selector modes are mutually exclusive")

    normalized_subtask = normalize_subtask_text(external_subtask)
    if external_subtask is not None and normalized_subtask is None:
        raise ValueError("External subtask must be a non-empty string")

    if external_subtask_selector is not None and external_subtask_selector != STATE_MACHINE_V1:
        raise ValueError(
            f"Unsupported external subtask selector {external_subtask_selector!r}; expected {STATE_MACHINE_V1!r}"
        )

    if external_subtask is not None or external_subtask_selector is not None:
        if policy_name != "websocket":
            raise ValueError("External subtask conditioning requires the websocket policy")
        if prompt_mode != "whole_task":
            raise ValueError("External subtask conditioning requires whole_task prompt mode")

    if external_subtask_selector is not None and task_name not in SUPPORTED_TASKS:
        raise ValueError(f"{STATE_MACHINE_V1} only supports tasks {sorted(SUPPORTED_TASKS)}, got {task_name!r}")
    return normalized_subtask, external_subtask_selector


def validate_subtask_voting_config(
    *,
    action_horizon: object,
    interval: object,
    window: object,
    threshold: object,
    external_subtask: object,
    external_subtask_selector: object,
    policy_name: object,
    prompt_mode: str,
) -> tuple[int | None, int | None, int | None, int | None]:
    values = (action_horizon, interval, window, threshold)
    if all(value is None for value in values):
        return None, None, None, None
    if any(value is None for value in values):
        raise ValueError("Subtask voting action horizon, interval, window, and threshold must be set together")
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Subtask voting action horizon, interval, window, and threshold must be positive integers")
    if window < 2 or threshold < 2 or threshold > window:
        raise ValueError("Subtask voting requires window >= 2 and 2 <= threshold <= window")
    if threshold * 2 <= window:
        raise ValueError("Subtask voting threshold must be a strict majority of the window")
    if external_subtask is not None or external_subtask_selector is not None:
        raise ValueError("Subtask voting cannot be combined with external subtask conditioning")
    if policy_name != "websocket":
        raise ValueError("Subtask voting requires the websocket policy")
    if prompt_mode != "whole_task":
        raise ValueError("Subtask voting requires whole_task prompt mode")
    return action_horizon, interval, window, threshold


def build_conditioning_metadata(
    prompt_mode: str,
    external_subtask: str | None,
    external_subtask_selector: str | None,
    selector_debounce_refreshes: int = DEBOUNCE_REFRESHES,
    subtask_vote_interval: int | None = None,
    subtask_vote_window: int | None = None,
    subtask_vote_threshold: int | None = None,
    subtask_vote_action_horizon: int | None = None,
) -> dict:
    if external_subtask is not None:
        return {
            "mode": "external_fixed",
            "subtask_mode": "external_fixed",
            "prompt_mode": prompt_mode,
            "subtask": external_subtask,
            "oracle": False,
            "source": "provided",
            "subtask_source": "provided",
            "privileged": False,
            "comparable_to_whole_task": False,
        }
    if external_subtask_selector is not None:
        return {
            "mode": "external_state_selector",
            "subtask_mode": "external_state_selector",
            "prompt_mode": prompt_mode,
            "selector": external_subtask_selector,
            "label_format": "legacy",
            "selector_labels_sha256": SELECTOR_LABELS_SHA256,
            "legacy_segments_sha256": LEGACY_SEGMENTS_SHA256,
            "training_annotation_sha256": TRAINING_ANNOTATION_SHA256,
            "selector_debounce_refreshes": selector_debounce_refreshes,
            "selector_stall_steps": STALL_STEPS,
            "source": "provided",
            "subtask_source": "provided",
            "oracle": True,
            "privileged": True,
            "official_benchmark": False,
            "comparable_to_whole_task": False,
        }
    if prompt_mode == "oracle_unsatisfied_goal":
        return {
            "mode": prompt_mode,
            "oracle": True,
            "source": "bddl_top_level_goal_conditions",
            "privileged": True,
            "comparable_to_whole_task": False,
        }
    metadata = {
        "mode": prompt_mode,
        "oracle": False,
        "source": "policy_server_whole_task_prompt",
        "privileged": False,
        "comparable_to_whole_task": True,
    }
    if subtask_vote_window is not None:
        metadata["subtask_voting"] = {
            "mode": "temporal_majority",
            "action_horizon": subtask_vote_action_horizon,
            "interval": subtask_vote_interval,
            "window": subtask_vote_window,
            "threshold": subtask_vote_threshold,
            "oracle": False,
            "privileged": False,
            "official_benchmark": False,
            "inference_ablation": True,
        }
    return metadata


def inject_external_subtask(
    obs: dict,
    *,
    external_subtask: str | None,
    selector_text: str | None,
) -> dict:
    """Inject the current external conditioning text into a preprocessed observation."""
    subtask = external_subtask if external_subtask is not None else selector_text
    if subtask is not None:
        obs["subtask"] = subtask
    return obs


def validate_selector_policy_response(requested_text: str, policy_subtask: object, policy_source: object) -> str:
    """Validate selector conditioning before its returned action reaches the environment."""
    if policy_source != "provided":
        raise RuntimeError(
            f"External subtask selector requires policy.current_subtask_source == 'provided', got {policy_source!r}"
        )
    normalized = normalize_subtask_text(policy_subtask)
    if normalized != requested_text:
        raise RuntimeError(
            "External subtask selector response did not match the requested selector text: "
            f"expected {requested_text!r}, got {normalized!r}"
        )
    return normalized


def collect_final_goal_state(task: Any) -> dict:
    """Return JSON-serializable top-level and grounded BDDL goal status."""
    success, top_level_status = task.compiled_task.check_goal(task._evaluate_predicate)

    grounded_status = []
    for option_index, option in enumerate(task.ground_goal_state_options):
        values = [bool(predicate.evaluate(task._evaluate_predicate)) for predicate in option]
        grounded_status.append(
            {
                "option_index": int(option_index),
                "success": all(values),
                "satisfied": [index for index, value in enumerate(values) if value],
                "unsatisfied": [index for index, value in enumerate(values) if not value],
            }
        )

    return {
        "success": bool(success),
        "top_level": {
            "success": bool(success),
            "satisfied": [int(index) for index in top_level_status["satisfied"]],
            "unsatisfied": [int(index) for index in top_level_status["unsatisfied"]],
        },
        "grounded": grounded_status,
    }


def resolve_instance_ids(task_name: str, instance_indices: list[int], mode: str = "public_test") -> list[int]:
    assert mode in EVAL_MODES, f"Mode must be one of {EVAL_MODES}, got {mode}"
    if mode == "train":
        return [int(instance_id) for instance_id in instance_indices]

    test_instances = (
        TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES]
        if mode == "public_test"
        else TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:]
    )
    num_split_instances = NUM_PUBLIC_TEST_INSTANCES if mode == "public_test" else NUM_HIDDEN_TEST_INSTANCES
    assert set(instance_indices).issubset(set(range(num_split_instances))), (
        f"Instance indices must be in range({num_split_instances}) for mode {mode}"
    )
    return [int(test_instances[i]) for i in instance_indices]


def _wrap_video_text(text: str, max_width: int, font: int, font_scale: float, thickness: int) -> list[str]:
    def width(value: str) -> int:
        return cv2.getTextSize(value, font, font_scale, thickness)[0][0]

    def truncate(value: str) -> str:
        if width(value) <= max_width:
            return value
        suffix = "..."
        low, high = 0, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            if width(value[:middle].rstrip() + suffix) <= max_width:
                low = middle
            else:
                high = middle - 1
        return value[:low].rstrip() + suffix

    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and width(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return [truncate(line) for line in lines[:2]]


class Evaluator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.prompt_mode = self.cfg.get("prompt_mode", "whole_task")
        if self.prompt_mode not in PROMPT_MODES:
            raise ValueError(f"Prompt mode must be one of {PROMPT_MODES}, got {self.prompt_mode}")
        self.external_subtask, self.external_subtask_selector = validate_subtask_conditioning_config(
            external_subtask=self.cfg.get("external_subtask"),
            external_subtask_selector=self.cfg.get("external_subtask_selector"),
            policy_name=self.cfg.get("policy_name"),
            prompt_mode=self.prompt_mode,
            task_name=self.cfg.task.name,
        )
        configured_debounce = self.cfg.get("selector_debounce_refreshes")
        if configured_debounce is not None and self.external_subtask_selector is None:
            raise ValueError("selector_debounce_refreshes requires an external subtask selector")
        if configured_debounce is not None and (
            isinstance(configured_debounce, bool)
            or not isinstance(configured_debounce, int)
            or configured_debounce < 1
        ):
            raise ValueError("selector_debounce_refreshes must be a positive integer")
        self.selector_debounce_refreshes = (
            DEBOUNCE_REFRESHES if configured_debounce is None else configured_debounce
        )
        (
            self.subtask_vote_action_horizon,
            self.subtask_vote_interval,
            self.subtask_vote_window,
            self.subtask_vote_threshold,
        ) = validate_subtask_voting_config(
            action_horizon=self.cfg.get("subtask_vote_action_horizon"),
            interval=self.cfg.get("subtask_vote_interval"),
            window=self.cfg.get("subtask_vote_window"),
            threshold=self.cfg.get("subtask_vote_threshold"),
            external_subtask=self.external_subtask,
            external_subtask_selector=self.external_subtask_selector,
            policy_name=self.cfg.get("policy_name"),
            prompt_mode=self.prompt_mode,
        )

        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()
        self.robot_name = None
        self.robot_eval_config = {}
        self.robot_camera_names = {}
        self._rollout_steps = 0
        self._oracle_goal_index = None
        self._policy_prompt = None
        self._prompt_events = []
        self._current_subtask = None
        self._current_subtask_source = None
        self._subtask_events = []
        self._subtask_vote_events = []
        self._last_subtask_vote_id = None
        self._observed_subtasks = []
        self._observed_subtask_sources = []
        self._selector_validated_action_requests = 0
        self._caption_subtask = None
        self._caption_lines = []
        self._subtask_selector = None
        self._task_instance_loaded = False

        self.num_envs = 1
        if int(self.cfg.get("num_envs", 1)) != 1:
            raise ValueError("Evaluator diagnostics require num_envs=1; use BatchedEvaluator for batches")
        self.env = SingleEnvironmentAdapter(self.load_env(env_wrapper=self.cfg.env_wrapper))
        self.robot = self.load_robot()
        self.robot_camera_names = get_robot_camera_names(self.robot.name, self.robot_eval_config)
        self._validate_robot_eval_config()
        self._apply_robot_eval_settings()
        self.policy = self.load_policy()
        self.metrics = self.load_metrics()
        self.obs = None
        self.light_synchronizer = None

        # Initialize the physics views so the first reset() (which eval.py issues before the first
        # load_task_instance) can read/restore robot joint state.
        og.sim.update_handles()

        self.env._current_episode = 0
        self._video_writer = None
        self._video_path = None
        self._video_rate = 30

    @property
    def should_sync_lights(self) -> bool:
        return self.env.task.activity_name in LIGHT_EVAL_TASKS

    @property
    def conditioning_metadata(self) -> dict:
        return build_conditioning_metadata(
            self.prompt_mode,
            self.external_subtask,
            self.external_subtask_selector,
            self.selector_debounce_refreshes,
            self.subtask_vote_interval,
            self.subtask_vote_window,
            self.subtask_vote_threshold,
            self.subtask_vote_action_horizon,
        )

    @property
    def prompt_events(self) -> list[dict]:
        return [dict(event) for event in self._prompt_events]

    @property
    def subtask_events(self) -> list[dict]:
        return [dict(event) for event in self._subtask_events]

    @property
    def subtask_vote_events(self) -> list[dict]:
        return copy.deepcopy(self._subtask_vote_events)

    @property
    def subtask_vote_summary(self) -> dict | None:
        if self.subtask_vote_window is None:
            return None
        reasons = {}
        candidates = {}
        for event in self._subtask_vote_events:
            reason = event["reason"]
            candidate = event["candidate"]
            reasons[reason] = reasons.get(reason, 0) + 1
            candidates[candidate] = candidates.get(candidate, 0) + 1
        return {
            "mode": "temporal_majority",
            "action_horizon": self.subtask_vote_action_horizon,
            "interval": self.subtask_vote_interval,
            "window": self.subtask_vote_window,
            "threshold": self.subtask_vote_threshold,
            "total_votes": len(self._subtask_vote_events),
            "accepted_switches": reasons.get("majority_switch", 0),
            "rejected_votes": sum(
                count for reason, count in reasons.items() if reason not in {"initial", "majority_switch"}
            ),
            "reasons": dict(sorted(reasons.items())),
            "candidates": dict(sorted(candidates.items())),
        }

    @property
    def selector_events(self) -> list[dict]:
        return self._subtask_selector.events if self._subtask_selector is not None else []

    @property
    def selector_summary(self) -> dict | None:
        if self._subtask_selector is None:
            return None
        summary = copy.deepcopy(self._subtask_selector.summary)
        summary["validated_action_requests"] = self._selector_validated_action_requests
        return summary

    @property
    def final_goal_state(self) -> dict:
        return collect_final_goal_state(self.env.task)

    @property
    def observed_subtasks(self) -> list[str | None]:
        return list(self._observed_subtasks)

    @property
    def observed_subtask_sources(self) -> list[str | None]:
        return list(self._observed_subtask_sources)

    def _reset_prompt_state(self) -> None:
        self._rollout_steps = 0
        self._oracle_goal_index = None
        self._policy_prompt = None
        self._prompt_events = []
        self._current_subtask = None
        self._current_subtask_source = None
        self._subtask_events = []
        self._subtask_vote_events = []
        self._last_subtask_vote_id = None
        self._observed_subtasks = []
        self._observed_subtask_sources = []
        self._selector_validated_action_requests = 0
        self._caption_subtask = None
        self._caption_lines = []

    def _record_subtask(self, step: int, subtask: str | None, source: str | None = None) -> str | None:
        normalized = normalize_subtask_text(subtask)
        normalized_source = source.strip() if isinstance(source, str) and source.strip() else None
        if normalized != self._current_subtask or normalized_source != self._current_subtask_source:
            self._current_subtask = normalized
            self._current_subtask_source = normalized_source
            if normalized is not None or normalized_source is not None or self._subtask_events:
                self._subtask_events.append({"step": int(step), "subtask": normalized, "source": normalized_source})
        return normalized

    def _record_subtask_vote(self, step: int, vote: object, policy_subtask: object) -> None:
        if vote is None:
            return
        if self.subtask_vote_window is None:
            raise RuntimeError("Policy returned a subtask vote when temporal voting is disabled")
        if not isinstance(vote, dict):
            raise RuntimeError(f"Policy returned an invalid subtask vote: {vote!r}")
        vote_id = vote.get("vote_id")
        if isinstance(vote_id, bool) or not isinstance(vote_id, int) or vote_id < 0:
            raise RuntimeError(f"Policy returned an invalid subtask vote id: {vote_id!r}")
        if vote_id == self._last_subtask_vote_id:
            previous = dict(self._subtask_vote_events[-1])
            previous.pop("step")
            if vote != previous:
                raise RuntimeError(f"Policy reused subtask vote id {vote_id} with different content")
            return
        expected_vote_id = 0 if self._last_subtask_vote_id is None else self._last_subtask_vote_id + 1
        if vote_id != expected_vote_id:
            raise RuntimeError(f"Policy subtask vote id is {vote_id}, expected {expected_vote_id}")
        expected_contract = {
            "action_horizon": self.subtask_vote_action_horizon,
            "interval": self.subtask_vote_interval,
            "window": self.subtask_vote_window,
            "threshold": self.subtask_vote_threshold,
        }
        mismatches = {
            name: (vote.get(name), expected)
            for name, expected in expected_contract.items()
            if vote.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(f"Policy subtask vote contract does not match evaluator: {mismatches}")
        request_index = vote.get("request_index")
        if isinstance(request_index, bool) or not isinstance(request_index, int) or request_index < 0:
            raise RuntimeError(f"Policy returned an invalid subtask vote request index: {request_index!r}")
        expected_request_index = vote_id * self.subtask_vote_interval
        if request_index != expected_request_index:
            raise RuntimeError(
                f"Policy subtask vote request index is {request_index}, expected {expected_request_index}"
            )
        expected_step = request_index * self.subtask_vote_action_horizon
        if step != expected_step:
            raise RuntimeError(f"Policy subtask vote arrived at step {step}, expected {expected_step}")
        committed_after = normalize_subtask_text(vote.get("committed_after"))
        if committed_after != normalize_subtask_text(policy_subtask):
            raise RuntimeError(
                "Policy subtask vote committed text does not match the action conditioning: "
                f"vote={committed_after!r}, policy={normalize_subtask_text(policy_subtask)!r}"
            )
        event = copy.deepcopy(vote)
        event["step"] = int(step)
        self._subtask_vote_events.append(event)
        self._last_subtask_vote_id = vote_id

    def _ensure_subtask_selector(self) -> None:
        if self.external_subtask_selector is None or self._subtask_selector is not None:
            return
        if not self._task_instance_loaded:
            return
        self._subtask_selector = create_subtask_selector(
            self.env,
            mode=self.external_subtask_selector,
            robot=self.robot,
            debounce_refreshes=self.selector_debounce_refreshes,
        )

    def _refresh_subtask_selector(self, step: int, *, reset_policy_on_transition: bool) -> dict | None:
        if self._subtask_selector is None:
            return None
        previous_phase = self._subtask_selector.snapshot.get("phase")
        previous_event_count = len(self._subtask_selector.events)
        snapshot = self._subtask_selector.refresh(step)
        selector_text = normalize_subtask_text(self._subtask_selector.current_text)
        if selector_text is None:
            raise RuntimeError(f"External subtask selector produced invalid text at step {step}: {snapshot!r}")
        transitioned = len(self._subtask_selector.events) > previous_event_count
        if reset_policy_on_transition and previous_phase is not None and transitioned:
            logger.info(
                f"External subtask selector transition at step {step}: {previous_phase!r} -> "
                f"{snapshot.get('phase')!r}; resetting policy action queue."
            )
            self.policy.reset()
        return snapshot

    def _current_selector_text(self) -> str | None:
        if self._subtask_selector is None:
            return None
        return normalize_subtask_text(self._subtask_selector.current_text)

    def _refresh_oracle_prompt(self) -> None:
        if self.prompt_mode != "oracle_unsatisfied_goal":
            return

        _, goal_status = self.env.task.compiled_task.check_goal(self.env.task._evaluate_predicate)
        unsatisfied = sorted(int(index) for index in goal_status["unsatisfied"])
        previous_index = self._oracle_goal_index
        if previous_index in unsatisfied:
            selected_index = previous_index
        else:
            selected_index = unsatisfied[0] if unsatisfied else None

        prompt = None
        if selected_index is not None:
            prompts = self.env.task.activity_natural_language_goal_conditions
            if selected_index >= len(prompts):
                raise IndexError(f"Goal condition index {selected_index} has no natural-language prompt")
            prompt = prompts[selected_index]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Goal condition {selected_index} has an invalid natural-language prompt: {prompt!r}")
            prompt = prompt.strip()

        if self._prompt_events and selected_index == previous_index and prompt == self._policy_prompt:
            return

        if not self._prompt_events:
            reason = "initial_goal"
        elif selected_index is None:
            reason = "no_unsatisfied_goal"
        elif previous_index is None:
            reason = "goal_became_unsatisfied"
        else:
            reason = "previous_goal_satisfied"

        self._oracle_goal_index = selected_index
        self._policy_prompt = prompt
        event = {
            "step": int(self._rollout_steps),
            "goal_index": selected_index,
            "prompt": prompt,
            "source": (
                "bddl_top_level_goal_conditions" if selected_index is not None else "policy_server_whole_task_fallback"
            ),
            "reason": reason,
        }
        self._prompt_events.append(event)
        logger.info(
            f"Oracle prompt transition: step={event['step']} goal_index={selected_index} "
            f"reason={reason} prompt={prompt!r}"
        )

    def _reset_light_synchronizer(self) -> None:
        if self.should_sync_lights:
            self.light_synchronizer = LightToggleSynchronizer(self.env.scene)
            self.light_synchronizer.reset_from_current_state()
        else:
            self.light_synchronizer = None

    def _sync_lights_and_get_obs(self, obs: dict | None = None) -> dict:
        if not self.should_sync_lights:
            return obs

        if self.light_synchronizer is None:
            self._reset_light_synchronizer()
        else:
            self.light_synchronizer.sync_from_current_state()
        for _ in range(3):
            og.sim.render()
        obs, _ = self.env.get_obs()
        return obs

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        return BatchedEvaluator.load_env(self, env_wrapper)

    def configure_robot_cameras(self, robot_cfg: dict, env_wrapper: DictConfig) -> None:
        """Apply the upstream wrapper camera contract to an exact-state recorded robot."""
        spec = get_class(env_wrapper["_target_"]).camera_spec()
        robot_cfg["obs_modalities"] = ["proprio", *spec["modalities"]]
        sensor_config = robot_cfg.setdefault("sensor_config", {})
        for camera_id, sensor_name in self.robot_eval_config.get("camera_sensor_names", {}).items():
            if camera_id not in spec["resolution"]:
                continue
            link_key = sensor_name.split(":", 1)[1]
            height, width = spec["resolution"][camera_id]
            sensor_kwargs = sensor_config.setdefault(link_key, {}).setdefault("sensor_kwargs", {})
            sensor_kwargs.update(image_height=height, image_width=width)

    def load_robot(self) -> Robot:
        if self.robot_name is not None:
            return self.env.scene.object_registry("name", self.robot_name)
        return self.env.robots[0]

    def _validate_robot_eval_config(self) -> None:
        missing_camera_sensors = []
        for camera_id, camera_name in self.robot_camera_names.items():
            sensor_name = camera_name.split("::", 1)[1]
            if sensor_name not in self.robot.sensors:
                missing_camera_sensors.append(f"{camera_id}: {sensor_name}")
        if missing_camera_sensors:
            raise ValueError(
                f"Configured eval.camera_sensor_names entries were not found in robot.sensors: {missing_camera_sensors}"
            )

        if self.cfg.get("write_video", False):
            required_camera_ids = {"left_wrist", "right_wrist", "head"}
            missing_camera_ids = sorted(required_camera_ids - set(self.robot_camera_names))
            if missing_camera_ids:
                raise ValueError(
                    "--write-video requires eval.camera_sensor_names roles "
                    f"{sorted(required_camera_ids)}; missing {missing_camera_ids}"
                )

    def _build_robot_config(self, task_name: str, task_cfg: dict) -> dict:
        robot_cfg = _to_plain_dict(OmegaConf.select(self.cfg, "robot"))
        if robot_cfg is None:
            robot_cfg = _to_plain_dict(OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH))
        else:
            robot_cfg = dict(robot_cfg)
        assert "model" in robot_cfg, "Robot config must include canonical 'model'"
        assert "type" not in robot_cfg, "Robot config must use canonical 'model', not 'type'"
        robot_cfg["model"] = robot_cfg["model"].lower()
        assert "name" in robot_cfg, "Robot config must include 'name'"
        self.robot_eval_config = _to_plain_dict(robot_cfg.pop("eval", None)) or {}
        robot_cfg["position"] = task_cfg["robot_start_position"]
        robot_cfg["orientation"] = task_cfg["robot_start_orientation"]

        return robot_cfg

    def _apply_robot_eval_settings(self) -> None:
        if self.robot.model in ("r1", "r1pro"):
            og.sim.stop()
            self.robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
            og.sim.play()

        head_camera_name = self.robot_camera_names.get("head")
        if head_camera_name is not None:
            head_sensor_name = head_camera_name.split("::")[1]
            if head_sensor_name in self.robot.sensors:
                self.robot.sensors[head_sensor_name].horizontal_aperture = EVAL_HEAD_HORIZONTAL_APERTURE

    def load_policy(self) -> Any:
        policy = instantiate(self.cfg.model)
        if hasattr(policy, "set_batch_size"):
            policy.set_batch_size(0)
        if hasattr(policy, "set_action_dim"):
            policy.set_action_dim(self.robot.action_dim)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> List[MetricBase]:
        return [
            AgentMetric(self.human_stats, env_accessor=self.env),
            TaskMetric(self.human_stats, env_accessor=self.env),
        ]

    def step(self) -> Tuple[bool, bool]:
        action_step = self._rollout_steps
        requested_selector_text = self._current_selector_text()
        if self.external_subtask_selector is not None:
            if requested_selector_text is None or self._subtask_selector is None:
                raise RuntimeError("External subtask selector is not initialized for this rollout")
            if normalize_subtask_text(self.obs.get("subtask")) != requested_selector_text:
                raise RuntimeError(
                    "Current observation does not contain the live selector text: "
                    f"expected {requested_selector_text!r}, got {self.obs.get('subtask')!r}"
                )

        self.robot_action = self.policy.forward(obs=self.obs)
        policy_subtask = getattr(self.policy, "current_subtask", None)
        policy_subtask_source = getattr(self.policy, "current_subtask_source", None)
        self._record_subtask_vote(
            action_step,
            getattr(self.policy, "current_subtask_vote", None),
            policy_subtask,
        )
        if requested_selector_text is not None:
            validate_selector_policy_response(requested_selector_text, policy_subtask, policy_subtask_source)
            self._selector_validated_action_requests += 1

        action_subtask = self._record_subtask(action_step, policy_subtask, policy_subtask_source)
        self._observed_subtasks.append(action_subtask)
        self._observed_subtask_sources.append(self._current_subtask_source)
        obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
        self._rollout_steps += 1
        obs = self._sync_lights_and_get_obs(obs)
        self._refresh_subtask_selector(self._rollout_steps, reset_policy_on_transition=True)
        self._refresh_oracle_prompt()
        self.obs = self._preprocess_obs(obs)

        if self._video_path is not None:
            self._write_video(action_subtask)

        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        for metric in self.metrics:
            metric.step(self.env, self.robot_action, obs, 0.0, terminated, truncated, info)
        return terminated, truncated

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            container, stream = self._video_writer
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._video_writer = video_writer

    def load_task_instance(self, instance_id: int) -> None:
        self._task_instance_loaded = False
        self._subtask_selector = None
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        mode = self.cfg.get("mode", "public_test")
        tro_file_path = get_task_instance_path(
            scene_model,
            f"{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state",
            mode=mode,
        )
        if tro_file_path is None:
            raise FileNotFoundError(
                f"Could not find 2026 {mode} task instance {instance_id} for "
                f"{self.env.task.activity_name} in scene {scene_model}."
            )

        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = {key.lower(): value for key, value in tro_state.items()}
                if "robot" in presampled_robot_poses:
                    available_poses = presampled_robot_poses["robot"]
                elif self.robot.model in presampled_robot_poses:
                    logger.info("No generic presampled robot pose found, using robot-specific pose.")
                    available_poses = presampled_robot_poses[self.robot.model]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose found for {self.robot.model}!")
                self.robot.set_position_orientation(available_poses[0]["position"], available_poses[0]["orientation"])
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        if self.should_sync_lights:
            set_light_control_toggles(self.env.task.object_scope.values(), True)

        # Keep all task-relevant entities (including the robot/agent) still while the scene settles,
        # so the snapshotted per-instance initial state is stable. The robot must already be in a clean
        # configuration when this is called -- eval.py resets before load_task_instance for this reason.
        og.sim.update_handles()
        for _ in range(25):
            og.sim.step_physics()
            for inst, entity in self.env.task.object_scope.items():
                if not is_system_bddl_inst(inst) and entity is not None:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()
        self._reset_light_synchronizer()
        self._task_instance_loaded = True

    def _preprocess_obs(self, obs: dict) -> dict:
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in self.robot_camera_names.values():
            sensor_name = camera_name.split("::")[1]
            if sensor_name not in self.robot.sensors:
                continue
            camera = self.robot.sensors[sensor_name]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        if cam_rel_poses:
            obs[f"{self.robot.name}::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        if self._policy_prompt is not None:
            obs["prompt"] = self._policy_prompt
        return inject_external_subtask(
            obs,
            external_subtask=self.external_subtask,
            selector_text=self._current_selector_text(),
        )

    def _write_video(self, subtask: str | None = None) -> None:
        required_camera_ids = ("left_wrist", "right_wrist", "head")
        if not all(camera_id in self.robot_camera_names for camera_id in required_camera_ids):
            return
        if self.robot_camera_names["head"] + "::rgb" not in self.obs:
            return
        left_wrist_rgb = cv2.resize(
            self.obs[self.robot_camera_names["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[self.robot_camera_names["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[self.robot_camera_names["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        frame = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb])[..., :3].copy()
        if subtask:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            margin = 10
            if subtask != self._caption_subtask:
                self._caption_subtask = subtask
                self._caption_lines = _wrap_video_text(
                    f"Subtask: {subtask}", frame.shape[1] - 2 * margin, font, font_scale, thickness
                )
            text_height = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1]
            line_height = text_height + 7
            panel_height = 12 + line_height * len(self._caption_lines)
            panel_top = frame.shape[0] - panel_height
            panel = frame[panel_top:].copy()
            panel.fill(0)
            cv2.addWeighted(panel, 0.65, frame[panel_top:], 0.35, 0, frame[panel_top:])
            for index, line in enumerate(self._caption_lines):
                origin = (margin, panel_top + 8 + text_height + index * line_height)
                cv2.putText(frame, line, origin, font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        frame = np.expand_dims(frame, 0)
        if self._video_writer is None:
            # Writer is created lazily so its resolution matches the composite (H, W) frame above.
            self.video_writer = create_video_writer(
                self._video_path, resolution=frame.shape[1:3], rate=self._video_rate
            )
        write_video(frame, video_writer=self.video_writer, batch_size=1, mode="rgb")

    def start_recording(self, fpath: str, rate: int = 30) -> None:
        # Finalize any in-progress recording, then arm a new one (writer is created on the first frame).
        self.video_writer = None
        self._video_path = fpath
        self._video_rate = rate

    def stop_recording(self) -> None:
        self.video_writer = None
        self._video_path = None

    def reset(self, *, prepare_for_task_instance: bool = False) -> None:
        self._reset_prompt_state()
        if prepare_for_task_instance:
            self._task_instance_loaded = False
            self._subtask_selector = None
        obs = self.env.reset()[0]
        self._reset_light_synchronizer()
        obs = self._sync_lights_and_get_obs(obs)
        self._ensure_subtask_selector()
        if self._subtask_selector is not None:
            self._subtask_selector.reset()
            self._refresh_subtask_selector(0, reset_policy_on_transition=False)
        self._refresh_oracle_prompt()
        self.obs = self._preprocess_obs(obs)
        for metric in self.metrics:
            metric.reset(self.env)
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.video_writer = None
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


__all__ = [
    "Evaluator",
    "build_conditioning_metadata",
    "collect_final_goal_state",
    "inject_external_subtask",
    "normalize_subtask_text",
    "resolve_instance_ids",
    "validate_selector_policy_response",
    "validate_subtask_conditioning_config",
    "validate_subtask_voting_config",
]
