"""Privileged state-based subtask selection for the two supported evaluation tasks.

The state-machine core in this module only depends on the Python standard library.
OmniGibson object-state classes are imported lazily by the runtime adapter.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


STATE_MACHINE_V1 = "state_machine_v1"
SUPPORTED_TASKS = frozenset({"carrying_in_groceries", "make_rose_centerpieces"})
DEBOUNCE_REFRESHES = 3
NEAR_ENTER_DISTANCE_M = 1.0
NEAR_EXIT_DISTANCE_M = 1.4
STALL_STEPS = 2048
SACK_TILT_THRESHOLD_RADIANS = math.radians(45.0)
SACK_UNTILT_THRESHOLD_RADIANS = math.radians(30.0)

CAR = "car.n.01_1"
SACK = "sack.n.01_1"
TOMATO = "beefsteak_tomato.n.01_1"
MILK = "carton__of__milk.n.01_1"
REFRIGERATOR = "electric_refrigerator.n.01_1"
BREAKFAST_TABLE = "breakfast_table"
PASSAGE_DOOR = "garage_passage_door"

VASE = "vase.n.01_1"
CABINET = "cabinet.n.01_1"
COFFEE_TABLE = "coffee_table.n.01_1"
DEFAULT_ROSE_IDS = tuple(f"rose.n.01_{index}" for index in range(1, 4))

GROCERY_PHASE_TEXT = {
    "move_to_sack_in_car": "move to paper bag",
    "pick_sack_from_car": "pick up from paper bag to car ssxsje",
    "close_car": "close lid car ssxsje",
    "move_to_passage": "move to door bexenl",
    "open_passage": "open door door bexenl",
    "move_sack_to_table": "move to breakfast table xftrki",
    "stage_sack": "place on paper bag to breakfast table xftrki",
    "pick_tomato": "pick up from beefsteak tomato to paper bag",
    "move_tomato_to_refrigerator": "move to fridge petcxr",
    "open_refrigerator_for_tomato": "open door fridge petcxr",
    "store_tomato": "place in beefsteak tomato to fridge petcxr",
    "move_to_sack_for_pour": "move to paper bag",
    "pick_sack_for_pour": "pick up from paper bag to breakfast table xftrki",
    "tip_sack": "tip over paper bag",
    "pour_sack": "pour paper bag to carton of milk",
    "restage_sack": "place on paper bag to breakfast table xftrki",
    "pick_milk": "pick up from carton of milk to breakfast table xftrki",
    "move_milk_to_refrigerator": "move to fridge petcxr",
    "open_refrigerator_for_milk": "open door fridge petcxr",
    "store_milk": "place in carton of milk to fridge petcxr",
    "move_to_refrigerator_to_close": "move to fridge petcxr",
    "close_refrigerator": "close door fridge petcxr",
    "complete": "close door fridge petcxr",
}

ROSE_PHASE_TEXT = {
    "move_to_rose": "move to rose",
    "pick_rose": "pick up from rose to bottom cabinet",
    "move_rose_to_vase": "move to vase",
    "place_rose_in_vase": "place in rose to vase",
    "move_to_vase": "move to vase",
    "pick_vase": "pick up from vase to bottom cabinet",
    "move_vase_to_coffee_table": "move to coffee table",
    "place_vase_on_coffee_table": "place on vase to coffee table",
    "complete": "place on vase to coffee table",
}

LEGACY_LABELS_BY_TASK = {
    "carrying_in_groceries": frozenset(GROCERY_PHASE_TEXT.values()),
    "make_rose_centerpieces": frozenset(ROSE_PHASE_TEXT.values()),
}
LEGACY_SEGMENTS_SHA256 = "999c58103ad91046d3a1215e202ab2022c645b55524261b3eeed6d66e7f1322d"
TRAINING_ANNOTATION_SHA256 = "870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d"
_SELECTOR_LABELS_TEXT = "".join(
    f"{task_name}\t{label}\n"
    for task_name in sorted(LEGACY_LABELS_BY_TASK)
    for label in sorted(LEGACY_LABELS_BY_TASK[task_name])
)
SELECTOR_LABELS_SHA256 = hashlib.sha256(_SELECTOR_LABELS_TEXT.encode()).hexdigest()

_GROCERY_ORDER = tuple(GROCERY_PHASE_TEXT)
_GROCERY_RANK = {phase: rank for rank, phase in enumerate(_GROCERY_ORDER)}
_ROSE_ORDER = ("move_to_rose", "pick_rose", "move_rose_to_vase", "place_rose_in_vase")
_ROSE_RANK = {phase: rank for rank, phase in enumerate(_ROSE_ORDER)}
_VASE_ORDER = ("move_to_vase", "pick_vase", "move_vase_to_coffee_table", "place_vase_on_coffee_table", "complete")
_VASE_RANK = {phase: rank for rank, phase in enumerate(_VASE_ORDER)}


class SubtaskSelectorError(ValueError):
    """Base error for selector configuration and runtime-state failures."""


class UnsupportedTaskError(SubtaskSelectorError):
    """Raised when state_machine_v1 is requested for any other task."""


class ObjectBindingError(SubtaskSelectorError):
    """Raised when required task or scene objects cannot be bound uniquely."""


class StateObservationError(SubtaskSelectorError):
    """Raised when an OmniGibson state required by the selector cannot be read."""


@dataclass(frozen=True)
class StateSignals:
    """Simulator-independent state consumed by :class:`StateMachineV1Core`."""

    inside: Mapping[tuple[str, str], bool] = field(default_factory=dict)
    on_top: Mapping[tuple[str, str], bool] = field(default_factory=dict)
    open: Mapping[str, bool] = field(default_factory=dict)
    distances: Mapping[str, float] = field(default_factory=dict)
    tilts: Mapping[str, float] = field(default_factory=dict)
    grasped: frozenset[str] = field(default_factory=frozenset)


class StateMachineV1Core:
    """Deterministic, simulator-independent finite-state-machine selector."""

    def __init__(
        self,
        task_name: str,
        *,
        rose_ids: Sequence[str] | None = None,
        resolved_object_names: Mapping[str, str] | None = None,
        target_ids: Mapping[str, str] | None = None,
        debounce_refreshes: int = DEBOUNCE_REFRESHES,
        stall_steps: int = STALL_STEPS,
    ) -> None:
        if task_name not in SUPPORTED_TASKS:
            raise UnsupportedTaskError(f"{STATE_MACHINE_V1} only supports {sorted(SUPPORTED_TASKS)}, got {task_name!r}")
        if isinstance(debounce_refreshes, bool) or not isinstance(debounce_refreshes, int) or debounce_refreshes < 1:
            raise ValueError("debounce_refreshes must be a positive integer")
        if isinstance(stall_steps, bool) or not isinstance(stall_steps, int) or stall_steps < 1:
            raise ValueError("stall_steps must be a positive integer")

        self.task_name = task_name
        self.rose_ids = tuple(sorted(rose_ids or DEFAULT_ROSE_IDS))
        if task_name == "make_rose_centerpieces" and len(self.rose_ids) != 3:
            raise ObjectBindingError(f"Expected exactly 3 runtime rose objects, found {len(self.rose_ids)}")
        self.resolved_object_names = dict(resolved_object_names or {})
        self.target_ids = dict(target_ids or {})
        self.debounce_refreshes = debounce_refreshes
        self.stall_steps = stall_steps
        self.valid = True
        self.reset()

    def reset(self) -> None:
        """Clear all rollout-local state. Call ``refresh`` on the reset observation next."""
        self._phase: str | None = None
        self._last_step: int | None = None
        self._phase_enter_step: int | None = None
        self._phase_dwell = 0
        self._completion_dwell = 0
        self._recovery_count = 0
        self._near: dict[str, bool] = {}
        self._sack_tipped = False
        self._passage_completed = False
        self._locked_rose: str | None = None
        self._completed_roses: set[str] = set()
        self._phase_stalled = False
        self._transitions: list[dict[str, Any]] = []
        self._stalls: list[dict[str, Any]] = []
        self._phase_completion_counts: Counter[str] = Counter()
        self._snapshot: dict[str, Any] = {
            "step": None,
            "valid": True,
            "mode": STATE_MACHINE_V1,
            "task_name": self.task_name,
            "phase": None,
            "text": None,
            "target_ids": [],
            "predicates": {},
            "dwell_count": 0,
            "completion_dwell": 0,
            "recovery_count": 0,
            "stalled": False,
            "locked_object": None,
        }

    @property
    def current_text(self) -> str | None:
        return self._snapshot["text"]

    @property
    def events(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._transitions)

    @property
    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._snapshot)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "mode": STATE_MACHINE_V1,
            "task_name": self.task_name,
            "label_format": "legacy",
            "selector_labels_sha256": SELECTOR_LABELS_SHA256,
            "legacy_segments_sha256": LEGACY_SEGMENTS_SHA256,
            "training_annotation_sha256": TRAINING_ANNOTATION_SHA256,
            "debounce_refreshes": self.debounce_refreshes,
            "stall_steps": self.stall_steps,
            "transitions": self.events,
            "stalls": copy.deepcopy(self._stalls),
            "recoveries": self._recovery_count,
            "phase_completion_counts": dict(sorted(self._phase_completion_counts.items())),
            "resolved_object_names": dict(sorted(self.resolved_object_names.items())),
            "final_snapshot": self.snapshot,
        }

    def refresh(self, signals: StateSignals, step: int) -> dict[str, Any]:
        """Consume one state observation and return the current structured snapshot."""
        step = int(step)
        if step < 0:
            raise ValueError("step must be non-negative")
        if self._last_step is not None and step < self._last_step:
            raise ValueError(f"step must be monotonic (previous={self._last_step}, current={step})")
        self._last_step = step
        self._refresh_near_latches(signals)
        self._refresh_tilt_latch(signals)

        if self._phase is None:
            if self.task_name == "carrying_in_groceries":
                self._passage_completed = self._grocery_progress_implies_passage(signals) or self._is_open(
                    signals, PASSAGE_DOOR
                )
                phase = self._desired_grocery_phase(signals)
            else:
                self._completed_roses = {rose for rose in self.rose_ids if self._rose_resolved(signals, rose)}
                self._locked_rose = self._choose_unresolved_rose(signals)
                phase = self._desired_rose_or_vase_phase(signals)
            self._transition(phase, step, "initial_state", signals)
            self._phase_dwell = 1
        else:
            self._phase_dwell += 1
            if self.task_name == "carrying_in_groceries":
                self._refresh_groceries(signals, step)
            else:
                self._refresh_roses(signals, step)

        self._maybe_record_stall(signals, step)
        self._snapshot = self._build_snapshot(signals, step)
        return self.snapshot

    def _refresh_near_latches(self, signals: StateSignals) -> None:
        for target, raw_distance in signals.distances.items():
            distance = float(raw_distance)
            if not math.isfinite(distance) or distance < 0:
                raise StateObservationError(f"Invalid planar AABB distance for {target}: {raw_distance!r}")
            previous = self._near.get(target, False)
            if distance <= NEAR_ENTER_DISTANCE_M:
                self._near[target] = True
            elif distance >= NEAR_EXIT_DISTANCE_M:
                self._near[target] = False
            else:
                self._near[target] = previous

    def _refresh_tilt_latch(self, signals: StateSignals) -> None:
        if self.task_name != "carrying_in_groceries":
            return
        tilt = self._tilt(signals, SACK)
        if not math.isfinite(tilt) or tilt < 0:
            raise StateObservationError(f"Invalid sack tilt: {tilt!r}")
        if tilt >= SACK_TILT_THRESHOLD_RADIANS:
            self._sack_tipped = True
        elif tilt <= SACK_UNTILT_THRESHOLD_RADIANS:
            self._sack_tipped = False

    def _refresh_groceries(self, signals: StateSignals, step: int) -> None:
        desired = self._desired_grocery_phase(signals)
        current_rank = _GROCERY_RANK[self._phase]
        desired_rank = _GROCERY_RANK[desired]
        # If the milk exits while the tipping label is active, retain the distinct
        # annotated pour phase rather than skipping directly to restaging.
        if self._phase == "tip_sack" and desired_rank > _GROCERY_RANK["pour_sack"]:
            desired = "pour_sack"
            desired_rank = _GROCERY_RANK[desired]
        if desired_rank < current_rank:
            self._transition(desired, step, "state_regression_or_drop", signals, recovery=True)
            return
        if desired_rank == current_rank:
            self._completion_dwell = 0
            return

        self._completion_dwell += 1
        if self._completion_dwell < self.debounce_refreshes:
            return

        previous = self._phase
        self._phase_completion_counts[previous] += 1
        if current_rank <= _GROCERY_RANK["open_passage"] < desired_rank:
            self._passage_completed = True
        self._transition(desired, step, "debounced_state_completion", signals)

    def _refresh_roses(self, signals: StateSignals, step: int) -> None:
        currently_resolved = {rose for rose in self.rose_ids if self._rose_resolved(signals, rose)}
        regressed = sorted(self._completed_roses - currently_resolved)
        if regressed:
            self._locked_rose = self._choose_unresolved_rose(signals, preferred=regressed)
            self._transition(
                self._rose_action_phase(signals, self._locked_rose),
                step,
                "rose_relation_regressed",
                signals,
                recovery=True,
            )
            return
        self._completed_roses.update(currently_resolved)

        held_unresolved = sorted(
            rose for rose in self.rose_ids if rose not in currently_resolved and self._is_grasped(signals, rose)
        )
        if held_unresolved and self._locked_rose not in held_unresolved:
            self._locked_rose = held_unresolved[0]
            self._transition(
                self._rose_action_phase(signals, self._locked_rose),
                step,
                "adopted_held_rose",
                signals,
                recovery=True,
            )
            return

        if self._phase in _ROSE_RANK:
            assert self._locked_rose is not None
            if self._rose_resolved(signals, self._locked_rose):
                self._completion_dwell += 1
                if self._completion_dwell < self.debounce_refreshes:
                    return
                completed_rose = self._locked_rose
                event_metadata = self._phase_metadata(self._phase, signals)
                self._completed_roses.add(completed_rose)
                self._phase_completion_counts[self._phase] += 1
                self._locked_rose = self._choose_unresolved_rose(signals)
                next_phase = self._desired_rose_or_vase_phase(signals)
                self._transition(
                    next_phase,
                    step,
                    "debounced_rose_placed",
                    signals,
                    event_metadata=event_metadata,
                )
                return

            desired = self._rose_action_phase(signals, self._locked_rose)
            current_rank = _ROSE_RANK[self._phase]
            desired_rank = _ROSE_RANK[desired]
            if desired_rank < current_rank:
                self._transition(desired, step, "rose_dropped_or_moved_away", signals, recovery=True)
            elif desired_rank > current_rank:
                self._completion_dwell += 1
                if self._completion_dwell >= self.debounce_refreshes:
                    self._phase_completion_counts[self._phase] += 1
                    self._transition(desired, step, "debounced_state_completion", signals)
            else:
                self._completion_dwell = 0
            return

        unresolved = self._unresolved_roses(signals)
        if unresolved:
            self._locked_rose = self._choose_unresolved_rose(signals)
            self._transition(
                self._rose_action_phase(signals, self._locked_rose),
                step,
                "rose_relation_regressed",
                signals,
                recovery=True,
            )
            return

        desired = self._desired_vase_phase(signals)
        current_rank = _VASE_RANK[self._phase]
        desired_rank = _VASE_RANK[desired]
        if desired_rank < current_rank:
            self._transition(desired, step, "vase_dropped_or_relation_regressed", signals, recovery=True)
        elif desired_rank > current_rank:
            self._completion_dwell += 1
            if self._completion_dwell >= self.debounce_refreshes:
                self._phase_completion_counts[self._phase] += 1
                self._transition(desired, step, "debounced_state_completion", signals)
        else:
            self._completion_dwell = 0

    def _desired_grocery_phase(self, signals: StateSignals) -> str:
        if self._inside(signals, SACK, CAR):
            return "pick_sack_from_car" if self._is_near(SACK) else "move_to_sack_in_car"
        if self._is_open(signals, CAR):
            return "close_car"
        if not self._passage_completed and self._inside(signals, TOMATO, SACK) and not self._is_grasped(signals, SACK):
            return "pick_sack_from_car" if self._is_near(SACK) else "move_to_sack_in_car"
        if not self._passage_completed and not self._is_open(signals, PASSAGE_DOOR):
            return "open_passage" if self._is_near(PASSAGE_DOOR) else "move_to_passage"

        tomato_stored = self._inside(signals, TOMATO, REFRIGERATOR) and not self._is_grasped(signals, TOMATO)
        sack_staged = self._on_top(signals, SACK, BREAKFAST_TABLE) and not self._is_grasped(signals, SACK)
        if not tomato_stored:
            if self._inside(signals, TOMATO, SACK):
                if not sack_staged:
                    if self._is_grasped(signals, SACK):
                        return "stage_sack" if self._is_near(BREAKFAST_TABLE) else "move_sack_to_table"
                    return "pick_sack_from_car" if self._is_near(SACK) else "move_to_sack_in_car"
                return "pick_tomato"
            if not self._is_grasped(signals, TOMATO):
                return "pick_tomato"
            if not self._is_near(REFRIGERATOR):
                return "move_tomato_to_refrigerator"
            if not self._is_open(signals, REFRIGERATOR):
                return "open_refrigerator_for_tomato"
            return "store_tomato"

        if self._inside(signals, MILK, SACK):
            if not self._is_grasped(signals, SACK):
                return "pick_sack_for_pour" if self._is_near(SACK) else "move_to_sack_for_pour"
            return "pour_sack" if self._sack_tipped else "tip_sack"

        if not sack_staged:
            if not self._is_grasped(signals, SACK):
                return "pick_sack_for_pour" if self._is_near(SACK) else "move_to_sack_for_pour"
            return "restage_sack"

        milk_stored = self._inside(signals, MILK, REFRIGERATOR) and not self._is_grasped(signals, MILK)
        if not milk_stored:
            if not self._is_grasped(signals, MILK):
                return "pick_milk"
            if not self._is_near(REFRIGERATOR):
                return "move_milk_to_refrigerator"
            if not self._is_open(signals, REFRIGERATOR):
                return "open_refrigerator_for_milk"
            return "store_milk"

        if self._is_open(signals, REFRIGERATOR):
            return "close_refrigerator" if self._is_near(REFRIGERATOR) else "move_to_refrigerator_to_close"
        return "complete"

    def _grocery_progress_implies_passage(self, signals: StateSignals) -> bool:
        return (
            self._on_top(signals, SACK, BREAKFAST_TABLE)
            or not self._inside(signals, TOMATO, SACK)
            or not self._inside(signals, MILK, SACK)
            or self._inside(signals, TOMATO, REFRIGERATOR)
            or self._inside(signals, MILK, REFRIGERATOR)
        )

    def _desired_rose_or_vase_phase(self, signals: StateSignals) -> str:
        if self._locked_rose is not None:
            return self._rose_action_phase(signals, self._locked_rose)
        return self._desired_vase_phase(signals)

    def _desired_vase_phase(self, signals: StateSignals) -> str:
        if self._on_top(signals, VASE, COFFEE_TABLE) and not self._is_grasped(signals, VASE):
            return "complete"
        if self._is_grasped(signals, VASE):
            return "place_vase_on_coffee_table" if self._is_near(COFFEE_TABLE) else "move_vase_to_coffee_table"
        return "pick_vase" if self._is_near(VASE) else "move_to_vase"

    def _rose_action_phase(self, signals: StateSignals, rose: str) -> str:
        if self._is_grasped(signals, rose):
            return "place_rose_in_vase" if self._is_near(VASE) else "move_rose_to_vase"
        return "pick_rose" if self._is_near(rose) else "move_to_rose"

    def _unresolved_roses(self, signals: StateSignals) -> list[str]:
        return [rose for rose in self.rose_ids if not self._rose_resolved(signals, rose)]

    def _choose_unresolved_rose(self, signals: StateSignals, preferred: Sequence[str] | None = None) -> str | None:
        unresolved = self._unresolved_roses(signals)
        if not unresolved:
            return None
        held = [rose for rose in unresolved if self._is_grasped(signals, rose)]
        if held:
            return sorted(held)[0]
        preferred_set = set(preferred or ())
        candidates = [rose for rose in unresolved if rose in preferred_set] or unresolved
        return sorted(candidates)[0]

    def _rose_resolved(self, signals: StateSignals, rose: str) -> bool:
        return self._inside(signals, rose, VASE) and not self._is_grasped(signals, rose)

    def _transition(
        self,
        phase: str,
        step: int,
        reason: str,
        signals: StateSignals,
        *,
        recovery: bool = False,
        event_metadata: tuple[dict[str, Any], list[str]] | None = None,
    ) -> None:
        previous_phase = self._phase
        previous_text = self._phase_text(previous_phase)
        metadata_phase = previous_phase if previous_phase is not None else phase
        predicates, target_ids = event_metadata or self._phase_metadata(metadata_phase, signals)
        dwell_count = self._phase_dwell
        completion_dwell = self._completion_dwell
        if recovery:
            self._recovery_count += 1
        self._phase = phase
        self._phase_enter_step = step
        self._phase_dwell = 0
        self._completion_dwell = 0
        self._phase_stalled = False
        self._transitions.append(
            {
                "step": step,
                "previous_phase": previous_phase,
                "current_phase": phase,
                "previous_text": previous_text,
                "current_text": self._phase_text(phase),
                "reason": reason,
                "predicates": predicates,
                "target_ids": target_ids,
                "dwell_count": dwell_count,
                "completion_dwell": completion_dwell,
                "recovery_count": self._recovery_count,
            }
        )

    def _maybe_record_stall(self, signals: StateSignals, step: int) -> None:
        if self._phase_stalled or self._phase_enter_step is None or step - self._phase_enter_step < self.stall_steps:
            return
        predicates, target_ids = self._phase_metadata(self._phase, signals)
        self._phase_stalled = True
        self._stalls.append(
            {
                "step": step,
                "phase": self._phase,
                "text": self._phase_text(self._phase),
                "reason": "phase_stalled",
                "predicates": predicates,
                "target_ids": target_ids,
                "dwell_count": self._phase_dwell,
                "recovery_count": self._recovery_count,
                "steps_in_phase": step - self._phase_enter_step,
            }
        )

    def _build_snapshot(self, signals: StateSignals, step: int) -> dict[str, Any]:
        predicates, target_ids = self._phase_metadata(self._phase, signals)
        return {
            "step": step,
            "valid": self.valid,
            "mode": STATE_MACHINE_V1,
            "task_name": self.task_name,
            "phase": self._phase,
            "text": self._phase_text(self._phase),
            "target_ids": target_ids,
            "predicates": predicates,
            "dwell_count": self._phase_dwell,
            "completion_dwell": self._completion_dwell,
            "recovery_count": self._recovery_count,
            "stalled": self._phase_stalled,
            "locked_object": self._external_id(self._locked_rose) if self._locked_rose is not None else None,
        }

    def _phase_metadata(self, phase: str | None, signals: StateSignals) -> tuple[dict[str, Any], list[str]]:
        if phase is None:
            return {}, []
        if self.task_name == "make_rose_centerpieces":
            return self._rose_phase_metadata(phase, signals)
        return self._grocery_phase_metadata(phase, signals)

    def _grocery_phase_metadata(self, phase: str, signals: StateSignals) -> tuple[dict[str, Any], list[str]]:
        targets_by_phase = {
            "move_to_sack_in_car": (SACK,),
            "pick_sack_from_car": (SACK, CAR),
            "close_car": (CAR,),
            "move_to_passage": (PASSAGE_DOOR,),
            "open_passage": (PASSAGE_DOOR,),
            "move_sack_to_table": (SACK, BREAKFAST_TABLE),
            "stage_sack": (SACK, BREAKFAST_TABLE),
            "pick_tomato": (TOMATO, SACK),
            "move_tomato_to_refrigerator": (TOMATO, REFRIGERATOR),
            "open_refrigerator_for_tomato": (TOMATO, REFRIGERATOR),
            "store_tomato": (TOMATO, REFRIGERATOR),
            "move_to_sack_for_pour": (SACK,),
            "pick_sack_for_pour": (SACK, BREAKFAST_TABLE),
            "tip_sack": (SACK, MILK),
            "pour_sack": (SACK, MILK),
            "restage_sack": (SACK, BREAKFAST_TABLE),
            "pick_milk": (MILK, BREAKFAST_TABLE),
            "move_milk_to_refrigerator": (MILK, REFRIGERATOR),
            "open_refrigerator_for_milk": (MILK, REFRIGERATOR),
            "store_milk": (MILK, REFRIGERATOR),
            "move_to_refrigerator_to_close": (REFRIGERATOR,),
            "close_refrigerator": (REFRIGERATOR,),
            "complete": (TOMATO, MILK, REFRIGERATOR),
        }
        targets = targets_by_phase[phase]
        predicates: dict[str, Any] = {}
        for target in targets:
            if target in signals.distances:
                predicates[f"distance:{target}"] = float(signals.distances[target])
                predicates[f"near:{target}"] = self._is_near(target)
            if target in signals.grasped:
                predicates[f"grasped:{target}"] = True
        for key in ((SACK, CAR), (TOMATO, SACK), (MILK, SACK), (TOMATO, REFRIGERATOR), (MILK, REFRIGERATOR)):
            if key[0] in targets or key[1] in targets:
                predicates[f"inside:{key[0]}:{key[1]}"] = self._inside(signals, *key)
        for key in ((SACK, BREAKFAST_TABLE), (MILK, BREAKFAST_TABLE)):
            if key[0] in targets or key[1] in targets:
                predicates[f"on_top:{key[0]}:{key[1]}"] = self._on_top(signals, *key)
        for target in (CAR, PASSAGE_DOOR, REFRIGERATOR):
            if target in targets:
                predicates[f"open:{target}"] = self._is_open(signals, target)
        if SACK in targets:
            predicates[f"tilt:{SACK}"] = self._tilt(signals, SACK)
            predicates[f"tipped:{SACK}"] = self._sack_tipped
            predicates[f"grasped:{SACK}"] = self._is_grasped(signals, SACK)
        if TOMATO in targets:
            predicates[f"grasped:{TOMATO}"] = self._is_grasped(signals, TOMATO)
        if MILK in targets:
            predicates[f"grasped:{MILK}"] = self._is_grasped(signals, MILK)
        return predicates, [self._external_id(target) for target in targets]

    def _rose_phase_metadata(self, phase: str, signals: StateSignals) -> tuple[dict[str, Any], list[str]]:
        rose = self._locked_rose
        if phase in _ROSE_RANK and rose is not None:
            targets = (rose, VASE)
            predicates = {
                f"inside:{rose}:{VASE}": self._inside(signals, rose, VASE),
                f"grasped:{rose}": self._is_grasped(signals, rose),
            }
            for target in targets:
                if target in signals.distances:
                    predicates[f"distance:{target}"] = float(signals.distances[target])
                    predicates[f"near:{target}"] = self._is_near(target)
        else:
            targets = (VASE, COFFEE_TABLE)
            predicates = {
                f"on_top:{VASE}:{COFFEE_TABLE}": self._on_top(signals, VASE, COFFEE_TABLE),
                f"grasped:{VASE}": self._is_grasped(signals, VASE),
                "all_roses_resolved": not self._unresolved_roses(signals),
            }
            for target in targets:
                if target in signals.distances:
                    predicates[f"distance:{target}"] = float(signals.distances[target])
                    predicates[f"near:{target}"] = self._is_near(target)
        return predicates, [self._external_id(target) for target in targets]

    def _phase_text(self, phase: str | None) -> str | None:
        if phase is None:
            return None
        mapping = GROCERY_PHASE_TEXT if self.task_name == "carrying_in_groceries" else ROSE_PHASE_TEXT
        return mapping[phase]

    def _external_id(self, target: str) -> str:
        return self.target_ids.get(target, target)

    def _is_near(self, target: str) -> bool:
        return self._near.get(target, False)

    @staticmethod
    def _inside(signals: StateSignals, obj: str, container: str) -> bool:
        return bool(signals.inside.get((obj, container), False))

    @staticmethod
    def _on_top(signals: StateSignals, obj: str, surface: str) -> bool:
        return bool(signals.on_top.get((obj, surface), False))

    @staticmethod
    def _is_open(signals: StateSignals, obj: str) -> bool:
        return bool(signals.open.get(obj, False))

    @staticmethod
    def _is_grasped(signals: StateSignals, obj: str) -> bool:
        return obj in signals.grasped

    @staticmethod
    def _tilt(signals: StateSignals, obj: str) -> float:
        return float(signals.tilts.get(obj, 0.0))


class OmniGibsonStateAdapter:
    """Bind live OmniGibson entities and convert their state to pure ``StateSignals``."""

    def __init__(self, env: Any, robot: Any | None = None) -> None:
        task = getattr(env, "task", None)
        scene = getattr(env, "scene", None)
        if task is None or scene is None:
            raise ObjectBindingError("Expected an initialized environment exposing task and scene")
        task_name = getattr(task, "activity_name", None)
        if task_name not in SUPPORTED_TASKS:
            raise UnsupportedTaskError(f"{STATE_MACHINE_V1} only supports {sorted(SUPPORTED_TASKS)}, got {task_name!r}")
        scope = getattr(task, "object_scope", None)
        if not isinstance(scope, Mapping):
            raise ObjectBindingError("Behavior task object_scope is not initialized")

        self.task_name = task_name
        self.robot = robot if robot is not None else self._resolve_robot(env)
        self.objects: dict[str, Any] = {}
        self.target_ids: dict[str, str] = {}

        if task_name == "carrying_in_groceries":
            for bddl_id in (CAR, SACK, TOMATO, MILK, REFRIGERATOR):
                self._bind_scored(scope, bddl_id)
            self._bind_fixture(
                BREAKFAST_TABLE,
                _resolve_unique_scene_fixture(
                    scene,
                    category="breakfast_table",
                    required_room_types={"kitchen"},
                    description="kitchen breakfast table",
                ),
            )
            self._bind_fixture(
                PASSAGE_DOOR,
                _resolve_unique_scene_fixture(
                    scene,
                    category="door",
                    required_room_types={"garage", "corridor"},
                    description="garage-passage door",
                ),
            )
            self.rose_ids: tuple[str, ...] = ()
        else:
            for bddl_id in (VASE, CABINET, COFFEE_TABLE):
                self._bind_scored(scope, bddl_id)
            rose_ids = tuple(sorted(key for key in scope if re.fullmatch(r"rose\.n\.01_\d+", key)))
            if len(rose_ids) != 3:
                raise ObjectBindingError(f"Expected exactly 3 runtime rose.n.01_* objects, found {list(rose_ids)}")
            for bddl_id in rose_ids:
                self._bind_scored(scope, bddl_id)
            self.rose_ids = rose_ids

        self.resolved_object_names = {key: str(getattr(obj, "name", "<unnamed>")) for key, obj in self.objects.items()}
        self.resolved_object_names["robot"] = str(getattr(self.robot, "name", "<unnamed>"))
        self.target_ids["robot"] = self.resolved_object_names["robot"]

    def observe(self) -> StateSignals:
        """Query the current live simulator state. Runtime-specific imports stay local."""
        try:
            from omnigibson.controllers import IsGraspingState
            from omnigibson.object_states import Inside, OnTop, Open
        except ImportError as error:
            raise StateObservationError("OmniGibson runtime state classes are unavailable") from error

        inside: dict[tuple[str, str], bool] = {}
        on_top: dict[tuple[str, str], bool] = {}
        open_states: dict[str, bool] = {}
        tilts: dict[str, float] = {}
        grasped: set[str] = set()

        if self.task_name == "carrying_in_groceries":
            inside_pairs = (
                (SACK, CAR),
                (TOMATO, SACK),
                (MILK, SACK),
                (TOMATO, REFRIGERATOR),
                (MILK, REFRIGERATOR),
            )
            on_top_pairs = ((SACK, BREAKFAST_TABLE), (MILK, BREAKFAST_TABLE))
            open_ids = (CAR, PASSAGE_DOOR, REFRIGERATOR)
            grasp_ids = (SACK, TOMATO, MILK)
            distance_ids = (SACK, PASSAGE_DOOR, BREAKFAST_TABLE, REFRIGERATOR, MILK)
            _, orientation = self.objects[SACK].get_position_orientation()
            tilts[SACK] = _tilt_from_quaternion_xyzw(orientation)
        else:
            inside_pairs = tuple((rose, VASE) for rose in self.rose_ids)
            on_top_pairs = tuple((rose, CABINET) for rose in self.rose_ids) + ((VASE, COFFEE_TABLE),)
            open_ids = ()
            grasp_ids = self.rose_ids + (VASE,)
            distance_ids = self.rose_ids + (VASE, COFFEE_TABLE)

        for obj_id, container_id in inside_pairs:
            inside[(obj_id, container_id)] = self._read_relative_state(
                self.objects[obj_id], Inside, self.objects[container_id], "Inside"
            )
        for obj_id, surface_id in on_top_pairs:
            on_top[(obj_id, surface_id)] = self._read_relative_state(
                self.objects[obj_id], OnTop, self.objects[surface_id], "OnTop"
            )
        for obj_id in open_ids:
            open_states[obj_id] = self._read_absolute_state(self.objects[obj_id], Open, "Open")
        for obj_id in grasp_ids:
            if self._robot_is_grasping(self.objects[obj_id], IsGraspingState):
                grasped.add(obj_id)

        robot_position = _as_float_vector(self.robot.get_position_orientation()[0], expected=3)
        distances = {
            obj_id: _planar_point_to_aabb_distance(robot_position, _read_aabb(self.objects[obj_id]))
            for obj_id in distance_ids
        }
        return StateSignals(
            inside=inside,
            on_top=on_top,
            open=open_states,
            distances=distances,
            tilts=tilts,
            grasped=frozenset(grasped),
        )

    def _bind_scored(self, scope: Mapping[str, Any], bddl_id: str) -> None:
        if bddl_id not in scope or scope[bddl_id] is None:
            raise ObjectBindingError(f"Required BDDL object {bddl_id!r} is missing from task.object_scope")
        self.objects[bddl_id] = scope[bddl_id]
        self.target_ids[bddl_id] = bddl_id

    def _bind_fixture(self, selector_id: str, obj: Any) -> None:
        self.objects[selector_id] = obj
        name = getattr(obj, "name", None)
        if not isinstance(name, str) or not name:
            raise ObjectBindingError(f"Resolved fixture {selector_id!r} has no stable scene object name")
        self.target_ids[selector_id] = f"scene:{name}"

    @staticmethod
    def _resolve_robot(env: Any) -> Any:
        robots = list(getattr(env, "robots", ()) or ())
        if len(robots) != 1:
            raise ObjectBindingError(f"Expected exactly one evaluation robot, found {len(robots)}")
        return robots[0]

    @staticmethod
    def _read_relative_state(obj: Any, state_type: type, other: Any, state_name: str) -> bool:
        state = getattr(obj, "states", {}).get(state_type)
        if state is None:
            raise StateObservationError(f"Object {getattr(obj, 'name', obj)!r} has no {state_name} state")
        return bool(state.get_value(other))

    @staticmethod
    def _read_absolute_state(obj: Any, state_type: type, state_name: str) -> bool:
        state = getattr(obj, "states", {}).get(state_type)
        if state is None:
            raise StateObservationError(f"Object {getattr(obj, 'name', obj)!r} has no {state_name} state")
        return bool(state.get_value())

    def _robot_is_grasping(self, candidate: Any, is_grasping_state: type) -> bool:
        arms = tuple(getattr(self.robot, "arm_names", ()) or ("default",))
        for arm in arms:
            grasp_state = self.robot.is_grasping(arm=arm, candidate_obj=candidate)
            if grasp_state == is_grasping_state.TRUE:
                return True
        return False


class StateMachineV1SubtaskSelector:
    """Evaluator-facing selector combining live-state adaptation with the pure FSM."""

    mode = STATE_MACHINE_V1

    def __init__(
        self,
        env: Any,
        robot: Any | None = None,
        *,
        debounce_refreshes: int = DEBOUNCE_REFRESHES,
        stall_steps: int = STALL_STEPS,
    ) -> None:
        self.adapter = OmniGibsonStateAdapter(env, robot=robot)
        self.core = StateMachineV1Core(
            self.adapter.task_name,
            rose_ids=self.adapter.rose_ids,
            resolved_object_names=self.adapter.resolved_object_names,
            target_ids=self.adapter.target_ids,
            debounce_refreshes=debounce_refreshes,
            stall_steps=stall_steps,
        )
        self._next_step = 0

    def reset(self) -> None:
        self.core.reset()
        self._next_step = 0

    def refresh(self, step: int | None = None) -> dict[str, Any]:
        if step is None:
            step = self._next_step
        step = int(step)
        snapshot = self.core.refresh(self.adapter.observe(), step)
        self._next_step = max(self._next_step, step + 1)
        return snapshot

    @property
    def current_text(self) -> str | None:
        return self.core.current_text

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.core.events

    @property
    def snapshot(self) -> dict[str, Any]:
        return self.core.snapshot

    @property
    def summary(self) -> dict[str, Any]:
        return self.core.summary


def create_subtask_selector(
    env: Any,
    mode: str = STATE_MACHINE_V1,
    robot: Any | None = None,
    *,
    debounce_refreshes: int = DEBOUNCE_REFRESHES,
    stall_steps: int = STALL_STEPS,
) -> StateMachineV1SubtaskSelector:
    """Create the requested privileged selector after the environment and task exist."""
    if mode != STATE_MACHINE_V1:
        raise ValueError(f"Unsupported subtask selector mode {mode!r}; expected {STATE_MACHINE_V1!r}")
    return StateMachineV1SubtaskSelector(
        env,
        robot=robot,
        debounce_refreshes=debounce_refreshes,
        stall_steps=stall_steps,
    )


def _resolve_unique_scene_fixture(
    scene: Any,
    *,
    category: str,
    required_room_types: set[str],
    description: str,
) -> Any:
    registry = getattr(scene, "object_registry", None)
    if registry is None or not callable(registry):
        raise ObjectBindingError("Scene does not expose the OmniGibson object_registry API")
    candidates = sorted(
        registry("category", category, default_val=set()) or (),
        key=lambda obj: str(getattr(obj, "name", "<unnamed>")),
    )
    matching = []
    missing_room_metadata = []
    for obj in candidates:
        rooms = getattr(obj, "in_rooms", None)
        if rooms is None:
            missing_room_metadata.append(str(getattr(obj, "name", "<unnamed>")))
            continue
        room_types = {_room_type(room) for room in rooms}
        if required_room_types.issubset(room_types):
            matching.append(obj)
    if len(matching) != 1:
        names = [str(getattr(obj, "name", "<unnamed>")) for obj in matching]
        detail = f"; candidates missing in_rooms metadata: {missing_room_metadata}" if missing_room_metadata else ""
        raise ObjectBindingError(
            f"Expected exactly one {description} (category={category!r}, rooms={sorted(required_room_types)}), "
            f"found {names}{detail}"
        )
    return matching[0]


def _room_type(room_instance: Any) -> str:
    room = str(room_instance)
    return re.sub(r"_\d+$", "", room)


def _as_float_vector(value: Any, *, expected: int) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    result = tuple(float(component) for component in value)
    if len(result) != expected or not all(math.isfinite(component) for component in result):
        raise StateObservationError(f"Expected a finite {expected}-vector, got {result!r}")
    return result


def _read_aabb(obj: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        lower, upper = obj.aabb
    except (AttributeError, TypeError, ValueError) as error:
        raise StateObservationError(f"Cannot read AABB for {getattr(obj, 'name', obj)!r}") from error
    lower_vector = _as_float_vector(lower, expected=3)
    upper_vector = _as_float_vector(upper, expected=3)
    if any(low > high for low, high in zip(lower_vector, upper_vector)):
        raise StateObservationError(f"Invalid AABB for {getattr(obj, 'name', obj)!r}: {(lower_vector, upper_vector)!r}")
    return lower_vector, upper_vector


def _planar_point_to_aabb_distance(point: Sequence[float], aabb: tuple[Sequence[float], Sequence[float]]) -> float:
    """Return XY distance from a point (the robot base) to the closest point in a target AABB."""
    lower, upper = aabb
    squared_distance = 0.0
    for axis in (0, 1):
        coordinate = float(point[axis])
        delta = max(float(lower[axis]) - coordinate, 0.0, coordinate - float(upper[axis]))
        squared_distance += delta * delta
    return math.sqrt(squared_distance)


def _tilt_from_quaternion_xyzw(quaternion: Any) -> float:
    """Return the unsigned angle between an object's local +Z and world +Z axes."""
    x, y, z, w = _as_float_vector(quaternion, expected=4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise StateObservationError("Cannot compute tilt from a zero quaternion")
    x /= norm
    y /= norm
    local_z_world_z = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.acos(local_z_world_z)


__all__ = [
    "DEBOUNCE_REFRESHES",
    "LEGACY_LABELS_BY_TASK",
    "LEGACY_SEGMENTS_SHA256",
    "NEAR_ENTER_DISTANCE_M",
    "SELECTOR_LABELS_SHA256",
    "NEAR_EXIT_DISTANCE_M",
    "ObjectBindingError",
    "OmniGibsonStateAdapter",
    "STALL_STEPS",
    "STATE_MACHINE_V1",
    "StateMachineV1Core",
    "StateMachineV1SubtaskSelector",
    "StateObservationError",
    "StateSignals",
    "SubtaskSelectorError",
    "SUPPORTED_TASKS",
    "TRAINING_ANNOTATION_SHA256",
    "UnsupportedTaskError",
    "create_subtask_selector",
]
