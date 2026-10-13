"""Role-bound completion predicates for the 100-task subtask diagnostic.

Only directly observable predicates are supported. Unknown skills and ambiguous
object bindings are excluded with evidence; they are never scored as successes.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

CONTRACT = "annotation_roles_v1"
DEBOUNCE_STEPS = 3


def infer_role_predicate(row):
    description = " ".join(str(row.get("skill_description", "")).lower().split())
    targets = list(dict.fromkeys(row.get("manipulating_object_ids") or (row.get("object_ids") or [])[:1]))
    references = list(dict.fromkeys(row.get("reference_object_ids") or []))
    if not targets or not all(isinstance(x, str) and x for x in targets + references):
        return None
    predicate = {"kind": "role_bound", "contract": CONTRACT, "targets": targets}
    if description == "move to":
        predicate.update(operation="near", distance_m=1.0)
    elif description in ("pick up from", "hold"):
        predicate.update(operation="grasped")
    elif description == "release":
        predicate.update(operation="released")
    elif description in ("place on", "place in", "place under", "place on next to", "place in next to"):
        required = 2 if description.endswith(" next to") else 1
        if len(references) < required:
            return None
        predicate.update(operation="placed", state={"on": "OnTop", "in": "Inside", "under": "Under"}[description.split()[1]], references=references[:required])
        if required == 2:
            predicate["secondary_state"] = "NextTo"
    elif description in ("open door", "open drawer", "open lid", "close door", "close drawer", "close lid"):
        # Annotation may name a handle/lid as manipulated and its owning
        # container as reference. Open is attached to that container.
        predicate.update(operation="state", state="Open", value=description.startswith("open "))
        predicate["targets"] = references[:1] or targets
    elif description in ("turn on switch", "turn off switch"):
        predicate.update(operation="state", state="ToggledOn", value=description == "turn on switch")
        predicate["targets"] = references[:1] or targets
    elif description == "tip over":
        predicate.update(operation="tilted", angle_radians=math.pi / 4)
    else:
        return None
    return predicate


def _normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def resolve_object(env, identifier):
    """Bind exact annotation IDs; a unique normalized alias is permitted."""
    scope = env.task.object_scope
    objects = list(env.scene.objects)
    candidates = []
    for key, obj in scope.items():
        if obj is not None:
            candidates.append((key, obj))
    candidates.extend((getattr(obj, "name", ""), obj) for obj in objects)
    for normalize in (lambda x: x, _normalized):
        matches = {id(obj): obj for alias, obj in candidates
                   if normalize(alias) == normalize(identifier) or normalize(getattr(obj, "name", "")) == normalize(identifier)}
        if len(matches) == 1:
            return next(iter(matches.values()))
        if len(matches) > 1:
            raise ValueError(f"Ambiguous annotation object {identifier!r}")
    raise ValueError(f"Annotation object {identifier!r} is absent from the restored scene")


class RoleStateObserver:
    def __init__(self, evaluator, predicate, *, resolved_roles=None):
        self.robot = evaluator.robot
        self.predicate = predicate
        roles = predicate if resolved_roles is None else resolved_roles
        self.targets = [resolve_object(evaluator.env, x) for x in roles["targets"]]
        self.references = [resolve_object(evaluator.env, x) for x in roles.get("references", [])]

    @staticmethod
    def _state(obj, name, reference=None):
        from omnigibson import object_states
        state_class = getattr(object_states, name)
        if state_class not in obj.states:
            raise ValueError(f"{obj.name!r} does not expose {name}")
        state = obj.states[state_class]
        # Contact-derived tensors can change after restoring a stationary object
        # even when the kinematic cache considers its pose unchanged.
        state.clear_cache()
        value = state.get_value() if reference is None else state.get_value(reference)
        return bool(value)

    def _grasped(self, obj):
        from omnigibson.controllers import IsGraspingState
        if getattr(self.robot, "grasping_mode", None) == "physical":
            raise ValueError("Physical grasping does not identify the candidate object")
        states = [self.robot.is_grasping(arm=arm, candidate_obj=obj) for arm in self.robot.arm_names]
        if not states or any(state == IsGraspingState.UNKNOWN for state in states):
            raise ValueError("Candidate-object grasping state is unobservable")
        return any(state == IsGraspingState.TRUE for state in states)

    def observe(self):
        operation = self.predicate["operation"]
        values, evidence = [], []
        for obj in self.targets:
            row = {"object": obj.name}
            if operation == "near":
                point = self.robot.get_position_orientation()[0]
                lower, upper = obj.aabb
                distance = math.sqrt(sum(max(float(lower[i]) - float(point[i]), 0.0, float(point[i]) - float(upper[i])) ** 2 for i in (0, 1)))
                if not math.isfinite(distance):
                    raise ValueError("Non-finite robot-object distance")
                row["distance_m"] = distance
                value = distance <= self.predicate["distance_m"]
            elif operation in ("grasped", "released"):
                grasped = self._grasped(obj)
                row["grasped"] = grasped
                value = grasped if operation == "grasped" else not grasped
            elif operation == "state":
                state = self._state(obj, self.predicate["state"])
                row[self.predicate["state"]] = state
                value = state is self.predicate["value"]
            elif operation == "placed":
                relation = self._state(obj, self.predicate["state"], self.references[0])
                grasped = self._grasped(obj)
                row.update(relation=relation, reference=self.references[0].name, grasped=grasped)
                value = relation and not grasped
                if "secondary_state" in self.predicate:
                    secondary = self._state(obj, self.predicate["secondary_state"], self.references[1])
                    row.update(secondary_relation=secondary, secondary_reference=self.references[1].name)
                    value = value and secondary
            elif operation == "tilted":
                quaternion = self.robot_orientation(obj)
                x, y, z, w = quaternion
                norm = math.sqrt(x*x + y*y + z*z + w*w)
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError("Invalid object orientation")
                angle = math.acos(max(-1.0, min(1.0, 1 - 2 * (x*x + y*y) / (norm*norm))))
                row["tilt_radians"] = angle
                value = angle >= self.predicate["angle_radians"]
            else:
                raise ValueError(f"Unknown completion operation {operation!r}")
            values.append(bool(value))
            evidence.append(row)
        return all(values), {"operation": operation, "targets": evidence}

    @staticmethod
    def robot_orientation(obj):
        return [float(v) for v in obj.get_position_orientation()[1]]


class RoleCompletionMonitor:
    def __init__(self, evaluator, predicate, sample=None, *, observer=None, source_context=None):
        self.predicate = deepcopy(predicate)
        self.dwell = 0
        self.satisfied = False
        self.metric_eligible = False
        self.setup_exclusion_reason = None
        self.setup_evidence = {}
        self.initial_snapshot = {}
        self.final_snapshot = {}
        try:
            binding = None
            if source_context is not None:
                if observer is not None:
                    raise ValueError("Source role binding requires the actual scene observer")
                from recorded_native_state import source_role_bindings
                binding = source_role_bindings(*source_context, predicate)
                from recorded_fixture_state import validate_runtime_fixture_roles
                binding = validate_runtime_fixture_roles(binding, evaluator)
                self.observer = RoleStateObserver(evaluator, predicate,
                                                 resolved_roles=binding["resolved_roles"])
            else:
                self.observer = observer or RoleStateObserver(evaluator, predicate)
            initial, snapshot = self.observer.observe()
            self.initial_snapshot = deepcopy(snapshot)
            self.final_snapshot = deepcopy(snapshot)
            self.setup_evidence = {"initial_predicate_value": initial, "binding_contract": CONTRACT}
            if binding is not None:
                self.setup_evidence["source_role_binding"] = deepcopy(binding)
            if initial:
                self.setup_exclusion_reason = "predicate_already_true"
            else:
                self.metric_eligible = True
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
            self.setup_exclusion_reason = "predicate_unobservable"
            self.setup_evidence = {"error_type": type(error).__name__, "error": str(error)}

    def refresh(self, step):
        if not self.metric_eligible:
            return False
        self.satisfied, self.final_snapshot = self.observer.observe()
        self.final_snapshot["step"] = step
        self.dwell = self.dwell + 1 if self.satisfied else 0
        return self.dwell >= DEBOUNCE_STEPS

    def result(self):
        return {"predicate": deepcopy(self.predicate), "debounce_steps": DEBOUNCE_STEPS,
                "dwell": self.dwell, "satisfied": self.satisfied,
                "metric_eligible": self.metric_eligible,
                "setup_exclusion_reason": self.setup_exclusion_reason,
                "setup_evidence": deepcopy(self.setup_evidence),
                "initial_snapshot": self.initial_snapshot, "final_snapshot": self.final_snapshot,
                "diagnostic_recoveries": 0, "selector_summary": {}}
