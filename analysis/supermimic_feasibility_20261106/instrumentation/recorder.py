"""Bounded state reads and timing; no imports or startup of OmniGibson.

Only caller-selected entities are read. Required dynamics fail closed; optional
joints, grasp and explicit contact providers report availability. Private grasp
internals, arbitrary environment dictionaries and scene-wide traversal are never
used. Hooks are disabled by default and installed only by a future launcher.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Callable, Mapping


def _plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Enum):
        return {"name": value.name, "value": value.value}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("State contains a nonfinite numeric value")
        return value
    raise TypeError(f"Unsupported state value type: {type(value).__name__}")


def _vector(value, length=None):
    value = _plain(value)
    if not isinstance(value, list) or any(type(item) not in (int, float) for item in value):
        raise ValueError("Expected a one-dimensional numeric state vector")
    if length is not None and len(value) != length:
        raise ValueError(f"Expected state vector length {length}, got {len(value)}")
    return value


def _required(entity, method, *args, **kwargs):
    function = getattr(entity, method, None)
    if not callable(function):
        raise ValueError(f"Required dynamics method missing: {method}")
    return function(*args, **kwargs)


def _optional(call):
    try:
        return {"available": True, "value": call()}
    except Exception as error:
        # A genuine read failure remains visible; it is never replaced by zeros.
        return {"available": False, "reason": type(error).__name__ + ": " + str(error)[:160]}


def _entity(entity, *, require_joints=False):
    pose = _required(entity, "get_position_orientation")
    if not isinstance(pose, (tuple, list)) or len(pose) != 2:
        raise ValueError("Expected (position, xyzw quaternion) pose")
    state = {
        "position_world": _vector(pose[0], 3),
        "quaternion_xyzw_world": _vector(pose[1], 4),
        "linear_velocity_world": _vector(_required(entity, "get_linear_velocity"), 3),
        "angular_velocity_world": _vector(_required(entity, "get_angular_velocity"), 3),
    }

    def joints():
        # OG EntityPrim exposes joint methods even for rigid entities; these
        # methods assert n_joints > 0. Do not convert that case into zero joints.
        if getattr(entity, "n_joints", None) == 0:
            raise ValueError("Entity reports no articulated joints")
        q = _vector(_required(entity, "get_joint_positions"))
        dq = _vector(_required(entity, "get_joint_velocities"))
        if len(q) != len(dq) or not q:
            raise ValueError("Joint position and velocity dimensions must match and be nonempty")
        return {"positions": q, "velocities": dq}

    state["joints"] = {"available": True, "value": joints()} if require_joints else _optional(joints)
    return state


def snapshot_entities(
    robot,
    objects: Mapping[str, Any],
    *,
    arms=(),
    contact_reader: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
):
    """Read only the robot and explicit name -> object allowlist.

    Pose and linear/angular velocity are required for all selected entities;
    robot joint state is required. Object joints are optional. Public robot
    ``is_grasping`` results preserve their enum (including UNKNOWN), not bool.
    No contact API is presumed: an optional provider must return only allowlist
    object names, e.g. pairwise contact booleans against this robot.
    """
    if not objects or any(not isinstance(name, str) or not name for name in objects):
        raise ValueError("An explicit nonempty task-object name allowlist is required")
    if len(set(arms)) != len(arms) or any(not isinstance(arm, str) for arm in arms):
        raise ValueError("arms must be an explicit sequence of unique names")
    state = {
        "robot": _entity(robot, require_joints=True),
        "objects": {name: _entity(entity) for name, entity in objects.items()},
    }
    if arms:
        state["grasp"] = {
            arm: _optional(lambda arm=arm: {
                name: _plain(_required(robot, "is_grasping", arm=arm, candidate_obj=entity))
                for name, entity in objects.items()
            }) for arm in arms
        }
    else:
        state["grasp"] = {"available": False, "reason": "No explicit arm names configured"}
    if contact_reader is None:
        state["contacts"] = {"available": False, "reason": "No verified bounded contact reader configured"}
    else:
        def contacts():
            result = contact_reader(robot, objects)
            if not isinstance(result, Mapping) or set(result) != set(objects):
                raise ValueError("Contact provider must report exactly the task-object allowlist")
            return {name: _plain(value) for name, value in result.items()}
        state["contacts"] = _optional(contacts)
    return state


class TimingRecorder:
    """Opt-in monotonic spans; nested spans overlap and must not be summed."""

    def __init__(self, *, enabled=False, sink=None, clock=time.monotonic):
        self.enabled, self.sink, self.clock = enabled, sink, clock
        self.spans = []

    @contextmanager
    def measure(self, stage):
        if not self.enabled:
            yield
            return
        start = self.clock()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            end = self.clock()
            span = {"stage": str(stage), "start_monotonic": start,
                    "end_monotonic": end, "wall_seconds": end-start, "status": status}
            self.spans.append(span)
            if self.sink:
                self.sink(span)


@dataclass(frozen=True)
class ControlOutcome:
    terminated: bool = False
    truncated: bool = False
    reward: float | None = None


class StepRecorder:
    """Execute at most one chunk and record its actual control-step interval.

    The caller provides a verified monotone control counter, and a one-control
    ``step_control(action)`` callback. Every callback must increment that counter
    by exactly one. Physics dt and control dt are explicit, measured runtime
    configuration values; no claim of observing hidden physics substeps is made.

    An enabled recorder requires an observation sink. It persists the terminal
    next observation too, instead of assuming a later action will expose it.
    The returned object always includes the actual raw next observation.
    """

    def __init__(self, *, enabled=False, snapshot=None, read_control_count=None,
                 physics_dt=None, control_dt=None, observation_sink=None,
                 record_sink=None, clock=time.monotonic):
        self.enabled = enabled
        self.snapshot = snapshot
        self.read_control_count = read_control_count
        self.physics_dt, self.control_dt = physics_dt, control_dt
        self.observation_sink, self.record_sink = observation_sink, record_sink
        self.clock = clock
        if enabled:
            if not all(callable(item) for item in (snapshot, read_control_count, observation_sink)):
                raise ValueError("Enabled recording requires snapshot, control counter, and observation sink")
            if any(type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0
                   for dt in (physics_dt, control_dt)):
                raise ValueError("Measured positive physics_dt and control_dt are required")

    def record_chunk(self, actions, *, step_control, read_observation):
        start = self.clock() if self.enabled else None
        spans = []

        def timed(stage, operation):
            if not self.enabled:
                return operation()
            begin = self.clock()
            value = operation()
            end = self.clock()
            spans.append({"stage": stage, "start_monotonic": begin,
                          "end_monotonic": end, "wall_seconds": end-begin})
            return value

        before = timed("state_before", self.snapshot) if self.enabled else None
        count_before = self.read_control_count() if self.enabled else None
        requested = len(actions)
        if requested < 1:
            raise ValueError("An action chunk must contain at least one control action")
        outcomes, executed_actions = [], []
        for action in actions:
            previous = self.read_control_count() if self.enabled else None
            outcome = timed("sim_control", lambda action=action: step_control(action))
            if not isinstance(outcome, ControlOutcome):
                raise TypeError("step_control must return ControlOutcome")
            if self.enabled and self.read_control_count() - previous != 1:
                raise ValueError("Control counter did not increase by exactly one per executed action")
            outcomes.append(outcome)
            if self.enabled:
                executed_actions.append(_vector(action))
            if outcome.terminated or outcome.truncated:
                break
        # Intentionally called even when the final action terminates the task.
        next_observation = timed("observation_read", read_observation)
        record = None
        if self.enabled:
            after = timed("state_after", self.snapshot)
            final = outcomes[-1]
            reference = timed("observation_io", lambda: self.observation_sink(
                next_observation, final.terminated or final.truncated))
            if not isinstance(reference, str) or not reference:
                raise ValueError("observation_sink must return a nonempty artifact reference")
            count_after = self.read_control_count()
            if count_after - count_before != len(outcomes):
                raise ValueError("Chunk control count differs from its executed action count")
            end = self.clock()
            record = {
                "start_monotonic": start, "end_monotonic": end, "wall_seconds": end-start,
                "control_step_start": count_before, "control_step_end": count_after,
                "requested_control_steps": requested, "executed_control_steps": len(outcomes),
                "physics_dt": self.physics_dt, "control_dt": self.control_dt,
                "inferred_simulated_seconds": len(outcomes)*self.control_dt,
                "inferred_physics_substeps": len(outcomes)*self.control_dt/self.physics_dt,
                "physics_substep_count_is_measured": False,
                "before": before, "after": after, "execution_actions": executed_actions,
                "terminated": final.terminated, "truncated": final.truncated,
                "rewards": [outcome.reward for outcome in outcomes],
                "next_observation": {"captured": True, "reference": reference,
                                     "terminal": final.terminated or final.truncated},
                "timing_spans": spans,
            }
            if self.record_sink:
                self.record_sink(record)
        return {"next_observation": next_observation, "outcome": outcomes[-1], "record": record}
