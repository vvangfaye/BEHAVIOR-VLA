#!/usr/bin/env python3
"""Queue-backed exact-state evaluation of fixed training subtasks."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import subtask_sample_panel as panel_module
from eval_b1k_queue import _build_evaluator_cfg, _load_base_module
from pi05_eval_queue import EvaluationQueue, atomic_write_json
from subtask_sample_panel import file_sha256, load_panel, raw_hdf5_path

MAX_SAMPLE_STEPS = 2048
DEBOUNCE_STEPS = 3
QUEUE_PROFILE = "training_subtask_closed_loop_episode_segments"
EVALUATION_KIND = "training_subtask_closed_loop"


class InvalidSampleState(RuntimeError):
    pass


class PredicateObservabilityError(InvalidSampleState):
    pass


def _transition_map(transitions: dict[str, Any]) -> dict[int, dict[str, Any]]:
    normalized = {}
    for value, transition in transitions.items():
        try:
            step = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid transition step {value!r}") from error
        if step < 0:
            raise ValueError(f"Invalid negative transition step {step}")
        if step in normalized:
            raise ValueError(f"Duplicate transition step {step}")
        if not isinstance(transition, dict):
            raise TypeError(f"Transition at step {step} must be an object")
        normalized[step] = transition
    return normalized


def transition_steps_before(transitions: dict[str, Any], frame: int) -> list[int]:
    return sorted(step for step in _transition_map(transitions) if step < frame)


def sparse_materialization_plan(
    target_frames: list[int], transitions: dict[str, Any]
) -> list[tuple[str, int]]:
    """Return the exact load/capture/transition order for sparse recorded states."""
    if not target_frames:
        raise ValueError("At least one target frame is required")
    targets = set()
    for frame in target_frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError(f"Invalid target frame {frame!r}")
        targets.add(frame)
    transition_by_step = _transition_map(transitions)
    last_frame = max(targets)
    plan = []
    for frame in range(last_frame + 1):
        plan.append(("load", frame))
        if frame in targets:
            plan.append(("capture", frame))
        if frame < last_frame and frame in transition_by_step:
            plan.append(("transition", frame))
    return plan


def compatible_phases(task: str, predicate: dict[str, Any]) -> set[str]:
    key = (
        predicate["kind"],
        predicate.get("target"),
        predicate.get("reference"),
        predicate.get("value"),
    )
    if task == "carrying_in_groceries":
        mapping = {
            ("near", "sack", None, None): {
                "move_to_sack_in_car",
                "move_to_sack_for_pour",
            },
            ("grasped", "sack", None, None): {
                "move_to_sack_in_car",
                "pick_sack_from_car",
                "move_to_sack_for_pour",
                "pick_sack_for_pour",
            },
            ("open", "car", None, False): {"close_car"},
            ("near", "passage_door", None, None): {"move_to_passage"},
            ("open", "passage_door", None, True): {"move_to_passage", "open_passage"},
            ("near", "breakfast_table", None, None): {"move_sack_to_table"},
            ("on_top", "sack", "breakfast_table", None): {"stage_sack", "restage_sack"},
            ("grasped", "tomato", None, None): {"pick_tomato"},
            ("near", "refrigerator", None, None): {
                "move_tomato_to_refrigerator",
                "move_milk_to_refrigerator",
                "move_to_refrigerator_to_close",
            },
            ("open", "refrigerator", None, True): {
                "move_tomato_to_refrigerator",
                "open_refrigerator_for_tomato",
                "move_milk_to_refrigerator",
                "open_refrigerator_for_milk",
            },
            ("inside", "tomato", "refrigerator", None): {
                "open_refrigerator_for_tomato",
                "store_tomato",
            },
            ("tilted", "sack", None, None): {"tip_sack"},
            ("not_inside", "milk", "sack", None): {
                "pick_sack_for_pour",
                "tip_sack",
                "pour_sack",
            },
            ("grasped", "milk", None, None): {"pick_milk"},
            ("inside", "milk", "refrigerator", None): {"store_milk"},
            ("open", "refrigerator", None, False): {
                "move_to_refrigerator_to_close",
                "close_refrigerator",
            },
        }
    elif task == "make_rose_centerpieces":
        mapping = {
            ("near", "active_rose", None, None): {"move_to_rose"},
            ("grasped", "active_rose", None, None): {"move_to_rose", "pick_rose"},
            ("near", "vase", None, None): {"move_rose_to_vase", "move_to_vase"},
            ("rose_resolved_increment", None, None, 1): {
                "move_rose_to_vase",
                "place_rose_in_vase",
            },
            ("grasped", "vase", None, None): {"move_to_vase", "pick_vase"},
            ("near", "coffee_table", None, None): {"move_vase_to_coffee_table"},
            ("on_top", "vase", "coffee_table", None): {
                "move_vase_to_coffee_table",
                "place_vase_on_coffee_table",
            },
        }
    else:
        raise ValueError(f"Unsupported task: {task}")
    phases = mapping.get(key)
    if phases is None:
        raise ValueError(f"Unsupported predicate for {task}: {predicate}")
    return phases


def _symbol_ids() -> dict[str, str]:
    from omnigibson.eval import subtask_selectors as selectors

    return {
        "sack": selectors.SACK,
        "tomato": selectors.TOMATO,
        "milk": selectors.MILK,
        "car": selectors.CAR,
        "refrigerator": selectors.REFRIGERATOR,
        "breakfast_table": selectors.BREAKFAST_TABLE,
        "passage_door": selectors.PASSAGE_DOOR,
        "vase": selectors.VASE,
        "coffee_table": selectors.COFFEE_TABLE,
    }


class CompletionMonitor:
    def __init__(
        self,
        evaluator: object,
        predicate: dict[str, Any],
        sample: dict[str, Any] | None = None,
        *,
        selector: object | None = None,
        near_distance: float | None = None,
        tilt_threshold: float | None = None,
    ) -> None:
        selector_factory = None
        if selector is None or near_distance is None or tilt_threshold is None:
            from omnigibson.eval.subtask_selectors import (
                NEAR_ENTER_DISTANCE_M,
                SACK_TILT_THRESHOLD_RADIANS,
                create_subtask_selector,
            )

            selector_factory = create_subtask_selector
            near_distance = (
                NEAR_ENTER_DISTANCE_M if near_distance is None else near_distance
            )
            tilt_threshold = (
                SACK_TILT_THRESHOLD_RADIANS
                if tilt_threshold is None
                else tilt_threshold
            )

        self.near_distance = float(near_distance)
        self.tilt_threshold = float(tilt_threshold)
        self.predicate = copy.deepcopy(predicate)
        self.sample = copy.deepcopy(sample) if sample is not None else None
        self.selector = selector
        self.initial_snapshot: dict[str, Any] = {}
        self.final_snapshot: dict[str, Any] = {}
        self.initial_recoveries = 0
        self.active_rose: str | None = None
        self.dwell = 0
        self.metric_eligible = False
        self.setup_exclusion_reason: str | None = None
        self.setup_evidence: dict[str, Any] = {}
        try:
            if self.selector is None:
                self.selector = selector_factory(
                    evaluator.env, robot=evaluator.robot
                )
            self.initial_snapshot = self.selector.refresh(0)
            self.final_snapshot = self.initial_snapshot
            self.initial_recoveries = int(
                self._selector_summary().get("recoveries", 0)
            )
            self.active_rose = self._active_rose()
            initial_signals = self.selector.adapter.observe()
            initial_value = self._predicate_value(initial_signals)
            initial_phase = self.initial_snapshot.get("phase")
            try:
                expected_phases = sorted(
                    compatible_phases(self.selector.adapter.task_name, self.predicate)
                )
            except (KeyError, TypeError, ValueError):
                expected_phases = []
            self.setup_evidence = {
                "initial_predicate_value": initial_value,
                "initial_phase": initial_phase,
                "compatible_phases": expected_phases,
                "phase_compatible": initial_phase in expected_phases,
            }
            if initial_value:
                self.setup_exclusion_reason = "predicate_already_true"
            else:
                self.metric_eligible = True
        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            PredicateObservabilityError,
        ) as error:
            self.setup_exclusion_reason = "predicate_unobservable"
            self.setup_evidence = {
                "error_type": type(error).__name__,
                "error": str(error),
            }

    @staticmethod
    def _normalized_object_id(value: object) -> str:
        text = str(value).strip().lower().replace(".n.01", "")
        return "".join(character for character in text if character.isalnum())

    def _active_rose(self) -> str | None:
        if self.predicate.get("target") != "active_rose" and self.predicate.get(
            "kind"
        ) != "rose_resolved_increment":
            return None
        adapter = self.selector.adapter
        roses = tuple(adapter.rose_ids)
        target_ids = dict(getattr(adapter, "target_ids", {}))
        resolved_names = dict(getattr(adapter, "resolved_object_names", {}))
        annotated_ids = []
        if self.sample is not None:
            manipulating = list(self.sample.get("manipulating_object_ids") or [])
            objects = list(self.sample.get("object_ids") or [])
            annotated_ids = list(dict.fromkeys([*manipulating, *objects]))
        if annotated_ids:
            matches = set()
            normalized_annotations = {
                self._normalized_object_id(value) for value in annotated_ids
            }
            for rose in roses:
                aliases = {
                    rose,
                    target_ids.get(rose),
                    resolved_names.get(rose),
                }
                if any(
                    alias is not None
                    and self._normalized_object_id(alias) in normalized_annotations
                    for alias in aliases
                ):
                    matches.add(rose)
            if len(matches) == 1:
                return matches.pop()
            specific_roses = {
                value
                for value in normalized_annotations
                if value.startswith("rose") and value != "rose"
            }
            if matches or specific_roses:
                raise PredicateObservabilityError(
                    "Could not uniquely resolve annotated rose from "
                    f"{annotated_ids!r}; matches={sorted(matches)!r}"
                )

        locked_external = self.initial_snapshot.get("locked_object")
        reverse = {value: key for key, value in target_ids.items()}
        active = reverse.get(locked_external, locked_external)
        if active not in roses:
            raise PredicateObservabilityError(
                f"Could not resolve active rose from {locked_external!r}"
            )
        return active

    def _resolve(self, name: str | None) -> str | None:
        if name is None:
            return None
        if name == "active_rose":
            if self.active_rose is None:
                raise PredicateObservabilityError("Active rose is unresolved")
            return self.active_rose
        try:
            return _symbol_ids()[name]
        except KeyError as error:
            raise PredicateObservabilityError(
                f"Unknown predicate target {name!r}"
            ) from error

    @staticmethod
    def _mapping_value(signals: object, name: str, key: object) -> Any:
        mapping = getattr(signals, name, None)
        if mapping is None or key not in mapping:
            raise PredicateObservabilityError(
                f"Predicate signal {name}[{key!r}] is unavailable"
            )
        return mapping[key]

    @staticmethod
    def _grasped(signals: object) -> object:
        grasped = getattr(signals, "grasped", None)
        if grasped is None or isinstance(grasped, (str, bytes)):
            raise PredicateObservabilityError("Predicate signal grasped is unavailable")
        return grasped

    @staticmethod
    def _boolean_signal(value: object, description: str) -> bool:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and not math.isfinite(numeric):
            raise PredicateObservabilityError(
                f"Predicate signal {description} is non-finite"
            )
        try:
            return bool(value)
        except (TypeError, ValueError) as error:
            raise PredicateObservabilityError(
                f"Predicate signal {description} is not boolean"
            ) from error

    def _predicate_value(self, signals: object) -> bool:
        kind = self.predicate.get("kind")
        target = self._resolve(self.predicate.get("target"))
        reference = self._resolve(self.predicate.get("reference"))
        if kind == "near":
            value = float(self._mapping_value(signals, "distances", target))
            if not math.isfinite(value):
                raise PredicateObservabilityError(
                    f"Predicate distance for {target!r} is non-finite"
                )
            return value <= self.near_distance
        if kind == "grasped":
            return target in self._grasped(signals)
        if kind == "open":
            value = self._boolean_signal(
                self._mapping_value(signals, "open", target), f"open[{target!r}]"
            )
            return value is bool(self.predicate["value"])
        if kind in ("inside", "not_inside"):
            inside = self._boolean_signal(
                self._mapping_value(signals, "inside", (target, reference)),
                f"inside[{(target, reference)!r}]",
            )
            return not inside if kind == "not_inside" else (
                inside and target not in self._grasped(signals)
            )
        if kind == "on_top":
            on_top = self._boolean_signal(
                self._mapping_value(signals, "on_top", (target, reference)),
                f"on_top[{(target, reference)!r}]",
            )
            return on_top and target not in self._grasped(signals)
        if kind == "tilted":
            value = float(self._mapping_value(signals, "tilts", target))
            if not math.isfinite(value):
                raise PredicateObservabilityError(
                    f"Predicate tilt for {target!r} is non-finite"
                )
            return value >= self.tilt_threshold
        if kind == "rose_resolved_increment":
            if int(self.predicate.get("value", 0)) != 1:
                raise PredicateObservabilityError(
                    "rose_resolved_increment only supports value 1"
                )
            vase = _symbol_ids()["vase"]
            inside = self._boolean_signal(
                self._mapping_value(
                    signals, "inside", (self.active_rose, vase)
                ),
                f"inside[{(self.active_rose, vase)!r}]",
            )
            return inside and self.active_rose not in self._grasped(signals)
        raise PredicateObservabilityError(f"Unsupported predicate kind: {kind!r}")

    def refresh(self, step: int) -> bool:
        if not self.metric_eligible:
            return False
        self.final_snapshot = self.selector.refresh(step)
        if self._predicate_value(self.selector.adapter.observe()):
            self.dwell += 1
        else:
            self.dwell = 0
        return self.dwell >= DEBOUNCE_STEPS

    def _selector_summary(self) -> dict[str, Any]:
        summary = getattr(self.selector, "summary", {})
        return summary if isinstance(summary, dict) else {}

    def result(self) -> dict[str, Any]:
        summary = self._selector_summary()
        return {
            "predicate": copy.deepcopy(self.predicate),
            "active_rose": self.active_rose,
            "debounce_steps": DEBOUNCE_STEPS,
            "dwell": self.dwell,
            "metric_eligible": self.metric_eligible,
            "setup_exclusion_reason": self.setup_exclusion_reason,
            "setup_evidence": copy.deepcopy(self.setup_evidence),
            "initial_snapshot": self.initial_snapshot,
            "final_snapshot": self.final_snapshot,
            "diagnostic_recoveries": max(
                0, int(summary.get("recoveries", 0)) - self.initial_recoveries
            ),
            "selector_summary": summary,
        }


def _unwrap_environment(env: object) -> object:
    seen: set[int] = set()
    current = env
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        nested = current.env
        if nested is current:
            break
        current = nested
    return current


def _apply_transition(scene: object, transition: dict[str, Any]) -> None:
    import omnigibson as og
    import torch as th
    from omnigibson.envs.data_wrapper import (
        _is_system_particle_template_info,
        _is_system_particle_template_name,
    )
    from omnigibson.utils.python_utils import create_object_from_init_info

    systems = transition.get("systems", {})
    objects = transition.get("objects", {})
    added_systems = set(systems.get("add", []))
    removed_systems = set(systems.get("remove", []))
    for name in systems.get("add", []):
        scene.get_system(name, force_init=True)
    for name in systems.get("remove", []):
        scene.clear_system(name)
    for name in objects.get("remove", []):
        if _is_system_particle_template_name(name, removed_systems):
            continue
        obj = scene.object_registry("name", name)
        if obj is None:
            raise InvalidSampleState(f"Transition removes missing object {name!r}")
        scene.remove_object(obj)
    for index, info in enumerate(objects.get("add", [])):
        if _is_system_particle_template_info(info, added_systems):
            continue
        obj = create_object_from_init_info(info)
        scene.add_object(obj)
        obj.set_position(th.ones(3) * (100.0 + 5 * index))
    og.sim.step()


def complete_recorded_scene_file(
    recorded_config: dict[str, Any],
    recorded_scene_file: dict[str, Any],
    data_root: str | Path,
) -> tuple[dict[str, Any], Path, int]:
    scene_config = recorded_config.get("scene", {})
    scene_model = scene_config.get("scene_model")
    scene_instance = scene_config.get("scene_instance")
    if not isinstance(scene_model, str) or not isinstance(scene_instance, str):
        raise InvalidSampleState("Recorded config has no scene model or instance")
    full_scene_path = (
        Path(data_root)
        / "2026-challenge-task-instances"
        / "scenes"
        / scene_model
        / "json"
        / f"{scene_instance}.json"
    )
    if not full_scene_path.is_file():
        raise FileNotFoundError(full_scene_path)
    full_scene_file = json.loads(full_scene_path.read_text())
    completed = copy.deepcopy(recorded_scene_file)
    recorded_objects = completed["objects_info"]["init_info"]
    recorded_states = completed["state"]["registry"]["object_registry"]
    full_objects = full_scene_file["objects_info"]["init_info"]
    full_states = full_scene_file["state"]["registry"]["object_registry"]
    for robot in recorded_config.get("robots", []):
        robot_state = recorded_states.get(robot.get("name"))
        if not isinstance(robot_state, dict) or "controllers" not in robot_state:
            continue
        if "controller_groups" in robot_state:
            raise InvalidSampleState("Recorded robot state has conflicting controller schemas")
        controller_groups = robot_state.pop("controllers")
        for controller_state in controller_groups.values():
            controller_state["goal_set"] = controller_state.pop("goal_is_valid")
            controller_state["goals"] = controller_state.pop("goal")
        robot_state["controller_groups"] = controller_groups
    missing_states = set(recorded_objects) - set(recorded_states)
    unavailable = missing_states - set(full_states)
    if unavailable:
        raise InvalidSampleState(
            f"Full scene cannot backfill object states: {sorted(unavailable)}"
        )
    for name in sorted(missing_states):
        recorded_info = recorded_objects[name]
        full_info = full_objects.get(name)
        if full_info is None:
            raise InvalidSampleState(f"Full scene has no object definition for {name!r}")
        for key in ("class_module", "class_name"):
            if recorded_info.get(key) != full_info.get(key):
                raise InvalidSampleState(
                    f"Recorded and full scene disagree on {name!r} {key}"
                )
        for key in ("name", "category", "model"):
            if recorded_info.get("args", {}).get(key) != full_info.get("args", {}).get(
                key
            ):
                raise InvalidSampleState(
                    f"Recorded and full scene disagree on {name!r} args.{key}"
                )
        recorded_states[name] = copy.deepcopy(full_states[name])
    return completed, full_scene_path, len(missing_states)


def alias_robot_observation_namespace(
    obs: dict[str, Any], source_name: str, target_name: str
) -> dict[str, Any]:
    if source_name == target_name:
        return obs
    source_prefix = f"{source_name}::"
    target_prefix = f"{target_name}::"
    aliases = {}
    for key, value in obs.items():
        if not key.startswith(source_prefix):
            continue
        suffix = key[len(source_prefix) :]
        sensor_prefix = f"{source_name}:"
        if suffix.startswith(sensor_prefix):
            suffix = f"{target_name}:{suffix[len(sensor_prefix) :]}"
        aliases[f"{target_prefix}{suffix}"] = value
    return {**obs, **aliases}


def _configure_recorded_robot(
    recorded_robot: dict[str, Any], evaluator_robot: dict[str, Any], eval_config: dict
) -> tuple[dict[str, Any], dict[str, Any]]:
    robot = copy.deepcopy(recorded_robot)
    for key in (
        "obs_modalities",
        "proprio_obs",
        "sensor_config",
        "include_sensor_names",
        "exclude_sensor_names",
    ):
        if key in evaluator_robot:
            robot[key] = copy.deepcopy(evaluator_robot[key])
    recorded_name = robot.get("name")
    evaluator_name = evaluator_robot.get("name")
    configured_eval = copy.deepcopy(eval_config)
    camera_names = configured_eval.get("camera_sensor_names", {})
    if isinstance(recorded_name, str) and isinstance(evaluator_name, str):
        configured_eval["camera_sensor_names"] = {
            role: (
                f"{recorded_name}:{sensor_name.split(':', 1)[1]}"
                if sensor_name.startswith(f"{evaluator_name}:")
                else sensor_name
            )
            for role, sensor_name in camera_names.items()
        }
    return robot, configured_eval


_EPISODE_IDENTITY_FIELDS = (
    "task",
    "task_index",
    "episode_index",
    "raw_episode_id",
    "task_instance_id",
    "episode_length",
)


def load_episode_cache(path: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Read and validate one raw episode once for all of its target starts."""
    if not samples:
        raise ValueError("Episode must contain at least one sample")
    import h5py
    import numpy as np

    path = Path(path).expanduser().resolve()
    first = samples[0]
    for sample in samples[1:]:
        if any(sample.get(name) != first.get(name) for name in _EPISODE_IDENTITY_FIELDS):
            raise ValueError("Episode samples disagree on raw episode identity")
    if not path.is_file():
        raise FileNotFoundError(path)

    digest = file_sha256(path)
    with h5py.File(path, "r", locking=False) as file:
        data = file["data"]
        details, state, state_size, transition_frames = (
            panel_module._raw_episode_structure(
                file, path, int(first["episode_length"])
            )
        )
        sample_evidence = {}
        for sample in samples:
            sample_evidence[sample["sample_id"]] = {
                **details,
                **panel_module._raw_frame_evidence(
                    sample, state, state_size, transition_frames, np
                ),
                "sha256": digest,
            }
        max_target_start = max(int(sample["start"]) for sample in samples)
        group = data[details["demo_group"]]
        transitions = json.loads(group.attrs["transitions"])
        transition_by_step = _transition_map(transitions)
        cache = {
            "path": str(path),
            "sha256": digest,
            "details": details,
            "recorded_config": json.loads(data.attrs["config"]),
            "scene_file": json.loads(data.attrs["scene_file"]),
            "init_metadata": {
                name: np.asarray(group["init_metadata"][name][...]).copy()
                for name in group["init_metadata"]
            },
            "states": np.asarray(group["state"][: max_target_start + 1]).copy(),
            "state_sizes": np.asarray(
                group["state_size"][: max_target_start + 1]
            ).copy(),
            "transitions": transitions,
            "transition_by_step": transition_by_step,
            "sample_evidence": sample_evidence,
            "max_target_start": max_target_start,
        }
    return cache


def _validate_recorded_episode(
    evaluator: object, episode_cache: dict[str, Any], sample: dict[str, Any]
) -> None:
    recorded_config = episode_cache["recorded_config"]
    recorded_robots = recorded_config.get("robots", [])
    if recorded_robots:
        recorded_model = str(
            recorded_robots[0].get("model", recorded_robots[0].get("type", ""))
        ).lower()
        evaluator_model = str(getattr(evaluator.robot, "model", "")).lower()
        if recorded_model and evaluator_model and recorded_model != evaluator_model:
            raise InvalidSampleState(
                f"Recorded robot model {recorded_model!r} does not match evaluator {evaluator_model!r}"
            )
    recorded_task = recorded_config.get("task", {})
    recorded_activity = recorded_task.get("activity_name") or recorded_task.get(
        "activity"
    )
    if recorded_activity and recorded_activity != sample["task"]:
        raise InvalidSampleState(
            f"Recorded task {recorded_activity!r} does not match {sample['task']!r}"
        )
    recorded_scene = recorded_config.get("scene", {}).get("scene_model")
    evaluator_scene = getattr(evaluator.env.task, "scene_name", None)
    if recorded_scene and evaluator_scene and recorded_scene != evaluator_scene:
        raise InvalidSampleState(
            f"Recorded scene {recorded_scene!r} does not match evaluator {evaluator_scene!r}"
        )


def _restore_episode_baseline(
    evaluator: object, episode_cache: dict[str, Any]
) -> None:
    import omnigibson as og
    import torch as th
    from omnigibson.envs.data_wrapper import (
        _align_scene_object_states_with_recorded_schema,
    )
    from omnigibson.macros import gm

    completed_scene_file = getattr(
        evaluator, "_exact_state_scene_file", episode_cache["scene_file"]
    )
    init_metadata = {
        name: th.as_tensor(values)
        for name, values in episode_cache["init_metadata"].items()
    }
    transition_rules_enabled = gm.ENABLE_TRANSITION_RULES
    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = True
    try:
        evaluator.env.scene.restore(completed_scene_file, update_initial_file=True)
        og.sim.stop()
        for values in init_metadata.values():
            if len(values) != evaluator.env.scene.n_objects:
                raise InvalidSampleState(
                    "Recorded init_metadata does not match restored scene object count"
                )
        for index, obj in enumerate(evaluator.env.scene.objects):
            for attribute, values in init_metadata.items():
                value = values[index]
                setattr(obj, attribute, value.item() if value.ndim == 0 else value)
        if evaluator.env.scene.transition_rule_api is None:
            from omnigibson.transition_rules import TransitionRuleAPI

            evaluator.env.scene._transition_rule_api = TransitionRuleAPI(
                scene=evaluator.env.scene
            )
        og.sim.play()
        evaluator.env.reset()
        with gm.unlocked():
            gm.ENABLE_TRANSITION_RULES = False
        _align_scene_object_states_with_recorded_schema(
            scene=evaluator.env.scene,
            recorded_scene_file=episode_cache["scene_file"],
        )
        og.sim.update_handles()
    finally:
        with gm.unlocked():
            gm.ENABLE_TRANSITION_RULES = transition_rules_enabled


def _dump_full_simulator_state(scene: object, simulator: object) -> Any:
    registry = scene.object_registry
    previous_filter = registry._dump_filter
    registry.set_dump_filter(dump_filter=lambda obj: True)
    try:
        state = simulator.dump_state(serialized=True)
        return state.clone() if hasattr(state, "clone") else copy.deepcopy(state)
    finally:
        registry.set_dump_filter(dump_filter=previous_filter)


def materialize_episode_start_states(
    evaluator: object,
    episode_cache: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Replay sparse states once and cache a full state at every requested start."""
    import omnigibson as og
    import torch as th
    from omnigibson.macros import gm

    _validate_recorded_episode(evaluator, episode_cache, samples[0])
    _restore_episode_baseline(evaluator, episode_cache)
    states = th.as_tensor(episode_cache["states"])
    state_sizes = episode_cache["state_sizes"]
    transition_by_step = episode_cache["transition_by_step"]
    transition_steps = sorted(transition_by_step)
    snapshots_by_frame = {}
    transition_rules_enabled = gm.ENABLE_TRANSITION_RULES
    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = False
    try:
        for action, frame in sparse_materialization_plan(
            [int(sample["start"]) for sample in samples],
            episode_cache["transitions"],
        ):
            if action == "load":
                og.sim.load_state(
                    states[frame, : int(state_sizes[frame])], serialized=True
                )
            elif action == "capture":
                og.sim.update_handles()
                snapshots_by_frame[frame] = _dump_full_simulator_state(
                    evaluator.env.scene, og.sim
                )
            else:
                _apply_transition(evaluator.env.scene, transition_by_step[frame])
    finally:
        with gm.unlocked():
            gm.ENABLE_TRANSITION_RULES = transition_rules_enabled

    materialized = {}
    for sample in samples:
        frame = int(sample["start"])
        prefix_steps = [step for step in transition_steps if step < frame]
        evidence = copy.deepcopy(episode_cache["sample_evidence"][sample["sample_id"]])
        evidence.update(
            {
                "full_scene_path": getattr(
                    evaluator, "_exact_state_full_scene_path", None
                ),
                "scene_backfilled_states": getattr(
                    evaluator, "_exact_state_backfilled_states", 0
                ),
                "state_prefix_samples": frame + 1,
                "topology_transition_steps": prefix_steps,
                "full_snapshot": True,
            }
        )
        materialized[sample["sample_id"]] = {
            "state": snapshots_by_frame[frame],
            "transition_steps": prefix_steps,
            "restoration": evidence,
        }
    return materialized


def _reset_evaluator_rollout(evaluator: object) -> None:
    base_env = _unwrap_environment(evaluator.env)
    if hasattr(base_env, "_current_step"):
        base_env._current_step = 0
    if hasattr(base_env, "_current_episode"):
        base_env._current_episode = 0
    evaluator._reset_prompt_state()
    evaluator._task_instance_loaded = True
    evaluator._subtask_selector = None
    evaluator._reset_light_synchronizer()
    obs, _ = evaluator.env.get_obs()
    obs = evaluator._sync_lights_and_get_obs(obs)
    evaluator.obs = evaluator._preprocess_obs(obs)
    for metric in evaluator.metrics:
        metric.reset(evaluator.env)
    evaluator.policy.reset()
    evaluator.n_success_trials = 0
    evaluator.n_trials = 0


def restore_materialized_sample_state(
    evaluator: object,
    episode_cache: dict[str, Any],
    materialized: dict[str, Any],
) -> dict[str, Any]:
    import omnigibson as og
    from omnigibson.macros import gm

    _restore_episode_baseline(evaluator, episode_cache)
    transition_rules_enabled = gm.ENABLE_TRANSITION_RULES
    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = False
    try:
        for step in materialized["transition_steps"]:
            _apply_transition(
                evaluator.env.scene, episode_cache["transition_by_step"][step]
            )
        og.sim.load_state(materialized["state"], serialized=True)
        og.sim.update_handles()
        for _ in range(3):
            og.sim.render()
    finally:
        with gm.unlocked():
            gm.ENABLE_TRANSITION_RULES = transition_rules_enabled
    _reset_evaluator_rollout(evaluator)
    return copy.deepcopy(materialized["restoration"])


def create_sample_evaluator(
    base: ModuleType,
    cfg: object,
    episode_cache: dict[str, Any],
    sample: dict[str, Any],
    sample_max_steps: int,
) -> object:
    import omnigibson as og
    from hydra.utils import instantiate
    from omnigibson.eval import evaluator as evaluator_module
    from omnigibson.macros import gm

    recorded_config = episode_cache["recorded_config"]
    scene_file = episode_cache["scene_file"]
    completed_scene_file, full_scene_path, backfilled_states = (
        complete_recorded_scene_file(recorded_config, scene_file, gm.DATA_PATH)
    )

    class ExactStateEvaluator(base.Evaluator):
        def _preprocess_obs(self, obs):
            processed = super()._preprocess_obs(obs)
            return alias_robot_observation_namespace(
                processed, self.robot.name, self._policy_robot_name
            )

        def load_env(self, env_wrapper):
            for rule in evaluator_module.DISABLED_TRANSITION_RULES:
                rule.ENABLED = False
            task_name = self.cfg.task.name
            if task_name != sample["task"]:
                raise InvalidSampleState(
                    f"Evaluator task {task_name!r} does not match sample {sample['task']!r}"
                )
            self.human_stats = evaluator_module.load_human_stats(task_name)
            config = copy.deepcopy(recorded_config)
            config["scene"]["scene_file"] = copy.deepcopy(completed_scene_file)
            config["objects"] = []
            config["task"]["include_obs"] = False
            if config["task"].get("type") == "BehaviorTask":
                config["task"]["online_object_sampling"] = False
                config["task"]["use_presampled_robot_pose"] = False
            config["task"].setdefault("termination_config", {})["max_steps"] = (
                sample_max_steps
            )
            task_cfg = evaluator_module.load_available_tasks()[task_name][0]
            evaluator_robot = self._build_robot_config(
                task_name=task_name, task_cfg=task_cfg
            )
            recorded_robots = config.get("robots", [])
            if len(recorded_robots) != 1:
                raise InvalidSampleState(
                    f"Expected one recorded robot, found {len(recorded_robots)}"
                )
            recorded_robot, self.robot_eval_config = _configure_recorded_robot(
                recorded_robots[0], evaluator_robot, self.robot_eval_config
            )
            self.robot_name = recorded_robot.get("name")
            self._policy_robot_name = evaluator_robot["name"]
            config["robots"] = [recorded_robot]
            env = og.Environment(configs=config)
            env._eval_robot_config = self.robot_eval_config
            self._exact_state_scene_file = completed_scene_file
            self._exact_state_full_scene_path = str(full_scene_path)
            self._exact_state_backfilled_states = backfilled_states
            return instantiate(env_wrapper, env=env)

    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = True
    return ExactStateEvaluator(cfg)


def _nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must be nonempty")
    return value


def parse_args(
    argv: list[str] | None = None, *, base_module: ModuleType | None = None
) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue-db", type=_nonempty, required=True)
    parser.add_argument("--queue-worker", type=_nonempty, required=True)
    parser.add_argument("--queue-run-id", type=_nonempty, required=True)
    parser.add_argument("--queue-generation", type=int, required=True)
    parser.add_argument("--queue-attempt-root", type=_nonempty, required=True)
    parser.add_argument("--subtask-panel", type=Path, required=True)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--sample-max-steps", type=int, default=MAX_SAMPLE_STEPS)
    panel_args, base_argv = parser.parse_known_args(argv)
    base = base_module or _load_base_module()
    args = base.parse_args(base_argv)
    if panel_args.queue_generation < 0:
        parser.error("--queue-generation must be non-negative")
    if panel_args.sample_max_steps <= 0:
        parser.error("--sample-max-steps must be positive")
    if args.num_rollouts != 1:
        parser.error("subtask sample evaluation requires --num-rollouts 1")
    if not args.write_video:
        parser.error("subtask sample evaluation requires --write-video")
    if args.external_subtask is not None or args.external_subtask_selector is not None:
        parser.error(
            "subtask sample evaluation supplies its fixed prompts from --subtask-panel"
        )
    for name, value in vars(panel_args).items():
        setattr(args, name, value)
    return args


def _validate_queue(
    queue: EvaluationQueue, args: argparse.Namespace, panel: dict[str, Any]
) -> None:
    expected = {
        "run_id": args.queue_run_id,
        "mode": args.mode,
        "require_videos": args.write_video,
        "panel_fingerprint": panel["fingerprint"],
        "profile": QUEUE_PROFILE,
        "evaluation_kind": EVALUATION_KIND,
        "expected_episode_jobs": len(panel["episodes"]),
        "expected_segment_rollouts": len(panel["samples"]),
    }
    metadata = queue.metadata(readonly=True)
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"queue metadata does not match sample evaluator arguments: {mismatches}"
        )


def resolve_episode_samples(
    panel: dict[str, Any], task: str, instance_id: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes = [
        episode
        for episode in panel["episodes"]
        if episode["task"] == task
        and int(episode["task_instance_id"]) == int(instance_id)
    ]
    if len(episodes) != 1:
        raise ValueError(
            f"Queue job has no unique panel episode: {task}/{instance_id}"
        )
    episode = episodes[0]
    samples = sorted(
        (
            sample
            for sample in panel["samples"]
            if sample["task"] == task
            and int(sample["task_instance_id"]) == int(instance_id)
            and int(sample["episode_index"]) == int(episode["episode_index"])
        ),
        key=lambda sample: int(sample["rollout_id"]),
    )
    expected_sample_ids = list(episode["sample_ids"])
    if [sample["sample_id"] for sample in samples] != expected_sample_ids:
        raise ValueError(
            f"Panel episode sample membership is inconsistent for {task}/{instance_id}"
        )
    if [int(sample["rollout_id"]) for sample in samples] != list(
        range(len(samples))
    ):
        raise ValueError(
            f"Panel episode rollout IDs are not dense for {task}/{instance_id}"
        )
    return episode, samples


def episode_rollout_specs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rollout_id": int(sample["rollout_id"]),
            "sample_id": sample["sample_id"],
            "video": "required_if_eligible",
        }
        for sample in samples
    ]


def _artifact_stem(sample: dict[str, Any]) -> str:
    return (
        f"{sample['task']}_{sample['task_instance_id']}_"
        f"{int(sample['rollout_id'])}"
    )


def run_sample(
    base: ModuleType,
    evaluator: object,
    args: argparse.Namespace,
    sample: dict[str, Any],
    episode_cache: dict[str, Any],
    materialized: dict[str, Any],
    output_dir: Path,
    panel_fingerprint: str,
) -> dict[str, Any]:
    evaluator.external_subtask = sample["subtask"]
    restoration = restore_materialized_sample_state(
        evaluator, episode_cache, materialized
    )
    with base.gm.unlocked():
        base.gm.ENABLE_TRANSITION_RULES = True
    monitor = CompletionMonitor(evaluator, sample["predicate"], sample)

    json_dir = output_dir / "json"
    video_dir = output_dir / "videos"
    json_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(sample)
    video_path = video_dir / f"{stem}.mp4"

    success = False
    terminated = False
    truncated = False
    failure_reason = monitor.setup_exclusion_reason
    steps = 0
    recording_started = False
    try:
        if monitor.metric_eligible:
            if args.write_video:
                video_dir.mkdir(parents=True, exist_ok=True)
                evaluator.start_recording(str(video_path), rate=args.video_fps)
                recording_started = True
            while steps < args.sample_max_steps:
                evaluator.external_subtask = sample["subtask"]
                terminated, truncated = evaluator.step()
                steps += 1
                success = monitor.refresh(steps)
                if success:
                    break
                if terminated or truncated:
                    failure_reason = (
                        "environment_terminated"
                        if terminated
                        else "environment_truncated"
                    )
                    break
            if not success and failure_reason is None:
                failure_reason = "sample_timeout"
            base.verify_external_subtask_rollout(
                evaluator,
                external_subtask=sample["subtask"],
                external_subtask_selector=None,
            )

        metrics = {}
        if monitor.metric_eligible:
            for metric in evaluator.metrics:
                metrics.update(metric.aggregate(evaluator.env))
        result = {
            "task": sample["task"],
            "instance_id": int(sample["task_instance_id"]),
            "rollout_id": int(sample["rollout_id"]),
            "sample_id": sample["sample_id"],
            "episode_index": int(sample["episode_index"]),
            "raw_episode_id": int(sample["raw_episode_id"]),
            "source_segment_index": int(sample["source_segment_index"]),
            "source_interval_index": int(sample["source_interval_index"]),
            "segment_start": int(sample["start"]),
            "segment_end": int(sample["end"]),
            "segment_frames": int(sample["segment_frames"]),
            "subtask": sample["subtask"],
            "predicate": sample["predicate"],
            "metric_eligible": monitor.metric_eligible,
            "result_state": (
                "evaluated" if monitor.metric_eligible else "setup_excluded"
            ),
            "setup_exclusion_reason": monitor.setup_exclusion_reason,
            "setup_evidence": copy.deepcopy(monitor.setup_evidence),
            "steps": steps,
            "success": success,
            "failure_reason": failure_reason,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "task_success": bool(evaluator.env.task.success),
            "conditioning": {
                **evaluator.conditioning_metadata,
                "source": "provided",
                "subtask_source": "provided",
                "training_set_diagnostic": True,
                "official_benchmark": False,
            },
            "subtask_events": evaluator.subtask_events,
            "observed_subtasks": evaluator.observed_subtasks,
            "observed_subtask_sources": evaluator.observed_subtask_sources,
            "monitor": monitor.result(),
            "restoration": restoration,
            "panel_fingerprint": panel_fingerprint,
            **metrics,
        }
        result = json.loads(json.dumps(result, default=float))
        atomic_write_json(json_dir / f"{stem}.json", result)
        return result
    finally:
        if recording_started:
            evaluator.stop_recording()


def run(
    args: argparse.Namespace, *, base_module: ModuleType | None = None
) -> list[dict[str, Any]]:
    base = base_module or _load_base_module()
    panel = load_panel(args.subtask_panel)
    if not any(sample["task"] == args.task_name for sample in panel["samples"]):
        raise ValueError(f"Panel contains no samples for task {args.task_name}")

    queue = EvaluationQueue(args.queue_db)
    _validate_queue(queue, args, panel)
    worker = (
        f"{args.queue_run_id}:{args.queue_worker}:generation-{args.queue_generation}"
    )
    lease = queue.claim(args.task_name, worker, args.queue_attempt_root)
    if lease is None:
        return []
    try:
        episode, samples = resolve_episode_samples(
            panel, lease.task, lease.instance_id
        )
    except Exception as error:
        queue.fail(lease, f"{type(error).__name__}: {error}")
        raise
    first_sample = samples[0]
    rollout_specs = episode_rollout_specs(samples)
    provenance = {
        "run_id": args.queue_run_id,
        "generation": args.queue_generation,
        "panel_fingerprint": panel["fingerprint"],
        "episode_index": int(episode["episode_index"]),
        "raw_episode_id": int(episode["raw_episode_id"]),
        "sample_ids": [sample["sample_id"] for sample in samples],
    }
    results: list[dict[str, Any]] = []
    try:
        raw_path = raw_hdf5_path(args.raw_data_root, first_sample)
        episode_cache = load_episode_cache(raw_path, samples)
        base.gm.HEADLESS = args.headless
        seed = base.seed_everything(base.DEFAULT_EVAL_SEED)
        evaluator_args = copy.copy(args)
        evaluator_args.external_subtask = first_sample["subtask"]
        evaluator_args.max_steps = args.sample_max_steps
        cfg = _build_evaluator_cfg(base, evaluator_args, seed)
        with create_sample_evaluator(
            base,
            cfg,
            episode_cache,
            first_sample,
            args.sample_max_steps,
        ) as evaluator:
            materialized = materialize_episode_start_states(
                evaluator, episode_cache, samples
            )
            for sample in samples:
                results.append(
                    run_sample(
                        base,
                        evaluator,
                        args,
                        sample,
                        episode_cache,
                        materialized[sample["sample_id"]],
                        lease.attempt_dir,
                        panel["fingerprint"],
                    )
                )
            queue.publish(
                lease,
                args.output_dir,
                rollout_specs=rollout_specs,
                provenance=provenance,
            )
        return results
    except Exception as error:
        try:
            if queue.reconcile(
                lease,
                args.output_dir,
                rollout_specs=rollout_specs,
                provenance=provenance,
            ):
                return results
        except Exception:
            base.logger.exception(
                f"Failed to reconcile episode lease {lease.lease_token}"
            )
        queue.fail(lease, f"{type(error).__name__}: {error}")
        raise


def main(argv: list[str] | None = None) -> None:
    base = _load_base_module()
    run(parse_args(argv, base_module=base), base_module=base)


if __name__ == "__main__":
    main()
