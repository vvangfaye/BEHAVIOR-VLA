"""Bounded Cartesian teacher corrections for the unchanged R1Pro 23-D action space.

The teacher never supplies joint positions. This adapter uses the live Jacobian
and the SAME J-PARSE IK function as OmniGibson's InverseKinematicsController,
then sends the resulting absolute joint targets through evaluator.step(). It
does not teleport a robot, change controllers, or declare a task successful.
"""
from __future__ import annotations

import math
import numpy as np

MAX_TRANSLATION_M = 0.02
MAX_ROTATION_RAD = 0.10
MAX_JOINT_DELTA_RAD = 0.15
MAX_CORRECTION_STEPS = 5
EXTENDED_BURST_PROTOCOL = "small_translation_contact_interrupt_v1"
MAX_EXTENDED_STEPS = 20
MAX_EXTENDED_TRANSLATION_M = 0.005
MAX_BURST_TRANSLATION_M = MAX_TRANSLATION_M * MAX_CORRECTION_STEPS
MAX_BURST_JOINT_TRAVEL_RAD = MAX_JOINT_DELTA_RAD * MAX_CORRECTION_STEPS
FRAME = "robot_base_footprint"


class CorrectionError(ValueError):
    """A correction is not supported or cannot be mapped safely to live joints."""


class RecoverableCorrectionError(CorrectionError):
    """Finite IK output violated a motion bound; no action was executed."""
    def __init__(self, message, *, reason_code, diagnostics):
        if reason_code not in {"ik_joint_bounds", "ik_residual"}:
            raise CorrectionError("Unknown recoverable correction rejection")
        if (diagnostics.get("arm") not in {"left", "right"}
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for key, value in diagnostics.items() if key != "arm")):
            raise CorrectionError("Correction rejection diagnostics must be finite numeric values")
        super().__init__(message)
        self.reason_code = reason_code
        self.diagnostics = dict(diagnostics)


def correction_schema(*, execution_protocol=None):
    if execution_protocol not in (None, EXTENDED_BURST_PROTOCOL):
        raise CorrectionError("Unknown correction execution protocol")
    vector = lambda bound: {"type": "array", "minItems": 3, "maxItems": 3,
                            "description": f"The vector's Euclidean norm must be <= {bound}, not just each component.",
                            "items": {"type": "number", "minimum": -bound, "maximum": bound}}
    return {"type": "object", "additionalProperties": False,
            "required": ["arm", "delta_position_m", "delta_rotation_axis_angle_rad", "gripper", "steps"],
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "delta_position_m": vector(MAX_TRANSLATION_M),
                "delta_rotation_axis_angle_rad": vector(MAX_ROTATION_RAD),
                "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
                "steps": {"type": "integer", "minimum": 1,
                          "maximum": MAX_EXTENDED_STEPS if execution_protocol else MAX_CORRECTION_STEPS}}}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite(value, shape, name):
    value = _array(value)
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise CorrectionError(f"{name} must have finite numeric shape {shape}")
    return value.astype(np.float64, copy=True)


def _indices(value, name):
    value = _array(value)
    if value.ndim != 1 or value.dtype.kind not in "iu" or np.any(value < 0):
        raise CorrectionError(f"{name} must be nonnegative integer indices")
    return value.astype(int)


def validate_correction(value, *, execution_protocol=None):
    schema = correction_schema(execution_protocol=execution_protocol)
    if not isinstance(value, dict) or set(value) != set(correction_schema()["required"]):
        raise CorrectionError("Correction must use the exact Cartesian correction contract")
    if value["arm"] not in ("left", "right") or value["gripper"] not in ("keep", "open", "close"):
        raise CorrectionError("Unknown correction arm or gripper command")
    if type(value["steps"]) is not int or not 1 <= value["steps"] <= schema["properties"]["steps"]["maximum"]:
        raise CorrectionError("Correction control-step count exceeds the execution protocol")
    result = dict(value)
    for key, bound in (("delta_position_m", MAX_TRANSLATION_M),
                       ("delta_rotation_axis_angle_rad", MAX_ROTATION_RAD)):
        vector = _finite(value[key], (3,), key)
        if np.linalg.norm(vector) > bound + 1e-9:
            raise CorrectionError(f"{key} vector norm exceeds {bound} per control step")
        result[key] = vector.tolist()
    if (not any(result["delta_position_m"]) and not any(result["delta_rotation_axis_angle_rad"])
            and result["gripper"] == "keep"):
        raise CorrectionError("A correction must request motion or a gripper change")
    if execution_protocol and result["steps"] > MAX_CORRECTION_STEPS:
        distance = float(np.linalg.norm(result["delta_position_m"]))
        if any(result["delta_rotation_axis_angle_rad"]):
            raise CorrectionError("Rotation bursts remain limited to five control steps")
        if distance and (result["gripper"] != "open" or distance > MAX_EXTENDED_TRANSLATION_M + 1e-9):
            raise CorrectionError("Long translation requires an explicitly open gripper and at most 5mm per step")
        if distance * result["steps"] > MAX_BURST_TRANSLATION_M + 1e-9:
            raise CorrectionError("Requested translation exceeds the original total burst envelope")
    return result


def _rotation_from_quaternion(quaternion):
    x, y, z, w = _finite(quaternion, (4,), "EEF quaternion xyzw")
    if not math.isclose(float(np.linalg.norm([x, y, z, w])), 1.0, abs_tol=1e-3):
        raise CorrectionError("EEF quaternion must be normalized")
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _rotation_from_axis_angle(vector):
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew @ skew)


def _orientation_error(target, current):
    # Same spatial/base-frame error used by OmniGibson transform_utils_np.
    return sum(np.cross(current[:, i], target[:, i]) for i in range(3)) / 2


class _OmniGibsonBackend:
    """Lazy imports keep schema validation / CPU tests independent of Isaac Sim."""
    def __init__(self):
        import torch
        from omnigibson.controllers.controller_view import ControllerView
        from omnigibson.controllers.ik_controller import _compute_ik_qpos_batch_numpy
        from omnigibson.utils.usd_utils import ControllableObjectViewAPI
        self.torch, self.controllers = torch, ControllerView
        self.views, self.solve = ControllableObjectViewAPI, _compute_ik_qpos_batch_numpy

    def pose(self, robot, arm):
        return self.views.get_link_relative_position_orientation(robot.articulation_root_path,
                                                                  robot.eef_link_names[arm])

    def jacobian(self, robot, arm, joint_indices):
        path = robot.articulation_root_path
        rows = _array(self.views.get_member_view_indices(path, [path]))
        if rows.shape != (1,):
            raise CorrectionError("Cannot uniquely resolve robot articulation view row")
        jacobians = _array(self.views.get_all_relative_jacobians(path))
        q = _array(self.views.get_all_joint_positions(path))
        # PhysX includes the root link row AND six root columns for floating
        # articulations. Fixed-base tensors omit both. Subtracting one from
        # every body index selects the preceding link on a floating robot.
        offset = jacobians.shape[-1] - q.shape[-1]
        if offset not in (0, 6):
            raise CorrectionError("Unknown Jacobian floating-base column layout")
        body = self.views.get_link_index(path, robot.eef_link_names[arm])
        row = body if offset == 6 else body - 1
        if not 0 <= row < jacobians.shape[1]:
            raise CorrectionError("EEF link has no valid Jacobian row")
        return jacobians[int(rows[0]), row][:, joint_indices + offset]

    def noop(self, robot, name):
        return self.controllers.compute_no_op_action(*robot.controllers[name])

    def action_tensor(self, action):
        return self.torch.as_tensor(action, dtype=self.torch.float32)


class BoundedCartesianCorrection:
    """Validate once, begin a bounded burst, recompute IK from each live state.

    Integration: include snapshot() in the teacher request; on action='correct'
    call begin(decision['correction'], last_action). Before each evaluator.step(),
    compute one next_action() from the unchanged live state and let the policy
    wrapper consume that prepared action. Record evaluator.robot_action normally. Request
    a new teacher observation at the end of the burst. Never enqueue returned
    actions across robot steps: each Jacobian must correspond to the current state.
    """
    def __init__(self, robot, *, backend=None, execution_protocol=None):
        correction_schema(execution_protocol=execution_protocol)
        self.execution_protocol = execution_protocol
        self.robot = robot
        self.backend = backend
        self.remaining_steps = 0
        self.last_diagnostics = None
        self._validate_layout()
        if self.backend is None:
            self.backend = _OmniGibsonBackend()

    def _validate_layout(self):
        robot = self.robot
        if robot.model != "r1pro" or robot.action_dim != 23 or getattr(robot, "_action_normalize", True):
            raise CorrectionError("Corrections require unnormalized R1Pro 23-D evaluation actions")
        self.action_indices = {name: _indices(idx, name) for name, idx in robot.controller_action_idx.items()}
        dims = {"base": 3, "trunk": 4, "arm_left": 7, "arm_right": 7,
                "gripper_left": 1, "gripper_right": 1}
        if set(self.action_indices) - set(dims) - {"camera"}:
            raise CorrectionError("Unknown robot controller in correction action layout")
        for name, dimension in dims.items():
            if name not in self.action_indices or self.action_indices[name].shape != (dimension,):
                raise CorrectionError(f"Unexpected action dimensions for {name}")
        if "camera" in self.action_indices and self.action_indices["camera"].size:
            raise CorrectionError("Camera must have a zero-dimensional NullJointController")
        all_indices = np.concatenate(list(self.action_indices.values()))
        if sorted(all_indices.tolist()) != list(range(23)):
            raise CorrectionError("Controller action indices must partition the 23-D action")
        config = robot._controller_config
        if "camera" in self.action_indices and config["camera"]["name"] != "NullJointController":
            raise CorrectionError("Camera must use NullJointController")
        self.joint_indices = {}
        for name in ("trunk", "arm_left", "arm_right"):
            cfg = config[name]
            if (cfg["name"] != "JointController" or cfg["motor_type"] != "position"
                    or cfg.get("use_delta_commands", True) or cfg.get("use_impedances", True)
                    or cfg.get("command_input_limits") is not None or cfg.get("command_output_limits") is not None
                    or cfg.get("subsume_controllers")):
                raise CorrectionError(f"{name} must accept unscaled absolute joint positions")
            self.joint_indices[name] = _indices(robot.controller_joint_idx[name], f"{name} joints")
            if self.joint_indices[name].shape != (dims[name],):
                raise CorrectionError(f"Unexpected joint dimensions for {name}")
        for arm in ("left", "right"):
            if not np.array_equal(self.joint_indices[f"arm_{arm}"], _array(robot.arm_control_idx[arm])):
                raise CorrectionError("Arm controller DOF order differs from robot arm order")
            cfg = config[f"gripper_{arm}"]
            if (cfg["name"] != "MultiFingerGripperController" or cfg["mode"] != "smooth"
                    or cfg["motor_type"] != "position" or cfg["command_input_limits"] != "default"
                    or cfg["command_output_limits"] != "default"):
                raise CorrectionError("Grippers require the evaluation smooth position controller")
        if config["base"]["name"] != "HolonomicBaseJointController" or config["base"]["motor_type"] != "velocity":
            raise CorrectionError("Base must use the evaluation holonomic velocity controller")
        for key in ("command_input_limits", "command_output_limits"):
            bounds = _finite(config["base"][key], (2, 3), f"base {key}")
            if not np.allclose(bounds[0], -bounds[1]) or np.any(bounds[1] <= 0):
                raise CorrectionError("Base command zero must map to zero velocity")

    def snapshot(self):
        arms = {}
        q = _array(self.robot.get_joint_positions())
        for arm in ("left", "right"):
            position, quaternion = self.backend.pose(self.robot, arm)
            quaternion = _finite(quaternion, (4,), "EEF quaternion")
            _rotation_from_quaternion(quaternion)
            arms[arm] = {
                "eef_link": self.robot.eef_link_names[arm],
                "eef_position_m": _finite(position, (3,), "EEF position").tolist(),
                "eef_quaternion_xyzw": quaternion.tolist(),
                "joint_position_rad": _finite(q[self.joint_indices[f"arm_{arm}"]], (7,), "arm joints").tolist(),
                "finger_joint_position_m": _finite(q[_array(self.robot.gripper_control_idx[arm]).astype(int)],
                                                    (2,), "gripper joints").tolist()}
        snapshot = {"contract": "r1pro_cartesian_correction_v1", "frame": FRAME,
                "frame_link": self.robot.base_footprint_link_name,
                "frame_definition": "OmniGibson robot base footprint frame, +x forward, +y left, +z up",
                "orientation_convention": "xyzw; axis-angle increments left-multiply current rotation in this frame",
                "arms": arms,
                "limits": {"translation_norm_m_per_step": MAX_TRANSLATION_M,
                           "rotation_norm_rad_per_step": MAX_ROTATION_RAD,
                           "joint_delta_rad_per_step": MAX_JOINT_DELTA_RAD,
                           "max_steps": MAX_CORRECTION_STEPS},
                "correction_behavior": "Deltas repeat once per control step; base, trunk, and other arm hold. "
                                       "Gripper keep preserves its last executed command. No collision-free guarantee."}
        if self.execution_protocol:
            snapshot["contract"] = "r1pro_cartesian_correction_v2"
            snapshot["execution_protocol"] = self.execution_protocol
            snapshot["limits"].update(max_steps=MAX_EXTENDED_STEPS,
                long_translation_norm_m_per_step=MAX_EXTENDED_TRANSLATION_M,
                translation_norm_m_per_burst=MAX_BURST_TRANSLATION_M,
                rotation_max_steps=MAX_CORRECTION_STEPS,
                joint_command_travel_rad_per_joint_per_burst=MAX_BURST_JOINT_TRAVEL_RAD)
            snapshot["correction_behavior"] += (
                " More than five steps requires pure small translation with an explicitly open gripper, "
                "or zero-motion gripper settling. Open-gripper translation returns early on any finger "
                "environment contact; this does not identify the touched object or prove grasping.")
        return snapshot

    def begin(self, correction, last_action=None):
        if self.remaining_steps:
            raise CorrectionError("Finish the pending correction before starting another")
        correction = validate_correction(correction, execution_protocol=self.execution_protocol)
        self._validate_layout()
        hold = np.zeros(23, dtype=np.float64)
        for name, indices in self.action_indices.items():
            if len(indices):
                hold[indices] = _finite(self.backend.noop(self.robot, name), indices.shape, f"{name} hold action")
        hold[self.action_indices["base"]] = 0
        if last_action is not None:
            last_action = _finite(last_action, (23,), "last executed action")
            for arm in ("left", "right"):
                idx = self.action_indices[f"gripper_{arm}"]
                # Evaluator keeps the raw policy output. The validated smooth
                # gripper controller clips its default input to [-1, 1] before
                # mapping to joint targets (BaseController._preprocess_command).
                # Preserve that effective command, including the other hand.
                hold[idx] = np.clip(last_action[idx], -1.0, 1.0)
        if correction["gripper"] != "keep":
            hold[self.action_indices[f"gripper_{correction['arm']}"]] = 1 if correction["gripper"] == "open" else -1
        self._hold, self._correction = hold, correction
        self._joint_travel = np.zeros(7, dtype=np.float64)
        self.remaining_steps = correction["steps"]

    def interrupt_on_contact(self, feedback):
        """Return to the teacher after an executed open approach touches anything."""
        if not (self.execution_protocol and self.remaining_steps and self._correction
                and self._correction["gripper"] == "open"
                and any(self._correction["delta_position_m"])):
            return False
        if feedback.get("contract") != "robot_gripper_execution_v1":
            raise CorrectionError("Contact interruption requires actual robot tactile feedback")
        flags = feedback["arms"][self._correction["arm"]]["finger_environment_contact"]
        if len(flags) != 2 or any(type(flag) is not bool for flag in flags):
            raise CorrectionError("Contact flags must be two robot tactile booleans")
        if any(flags):
            self.cancel()
            return True
        return False

    def cancel(self):
        """Discard a rejected burst without changing the robot or last executed diagnostics."""
        self.remaining_steps = 0
        self._hold = self._correction = None

    def next_action(self):
        if self.remaining_steps < 1:
            raise CorrectionError("No pending correction")
        arm = self._correction["arm"]
        indices = self.joint_indices[f"arm_{arm}"]
        q = _finite(_array(self.robot.get_joint_positions())[indices], (7,), "current arm joints")
        dpos = np.asarray(self._correction["delta_position_m"])
        drot = np.asarray(self._correction["delta_rotation_axis_angle_rad"])
        target = q.copy()
        residual_pos = residual_ori = 0.0
        if np.any(dpos) or np.any(drot):
            position, quaternion = self.backend.pose(self.robot, arm)
            position = _finite(position, (3,), "EEF position")
            rotation = _rotation_from_quaternion(quaternion)
            goal_rotation = _rotation_from_axis_angle(drot) @ rotation
            jacobian = _finite(self.backend.jacobian(self.robot, arm, indices), (6, 7), "EEF Jacobian")
            lo, hi = self.robot.control_limits["position"]
            lower = _finite(_array(lo)[indices], (7,), "joint lower limits")
            upper = _finite(_array(hi)[indices], (7,), "joint upper limits")
            if np.any(lower >= upper) or np.any(q < lower-1e-5) or np.any(q > upper+1e-5):
                raise CorrectionError("Current joints or joint limits are invalid")
            target = self.backend.solve(q=q[None].astype(np.float32), j_eef=jacobian[None].astype(np.float32),
                ee_pos=position[None].astype(np.float32), ee_mat=rotation[None].astype(np.float32),
                goal_pos=(position+dpos)[None].astype(np.float32), goal_ori_mat=goal_rotation[None].astype(np.float32),
                q_lower_limit=lower[None].astype(np.float32), q_upper_limit=upper[None].astype(np.float32))
            target = _finite(target, (1, 7), "IK joint solution")[0]
            delta = target-q
            diagnostics = {"arm": arm, "max_joint_delta_rad": float(np.max(np.abs(delta))),
                           "joint_delta_limit_rad": MAX_JOINT_DELTA_RAD,
                           "joint_lower_violation_rad": max(0.0, float(np.max(lower-target))),
                           "joint_upper_violation_rad": max(0.0, float(np.max(target-upper)))}
            if np.any(np.abs(delta) > MAX_JOINT_DELTA_RAD+1e-6) or np.any(target < lower) or np.any(target > upper):
                raise RecoverableCorrectionError("IK solution exceeds joint position/step bounds",
                    reason_code="ik_joint_bounds", diagnostics=diagnostics)
            desired = np.r_[dpos, _orientation_error(goal_rotation, rotation)]
            residual = jacobian @ delta-desired
            residual_pos, residual_ori = float(np.linalg.norm(residual[:3])), float(np.linalg.norm(residual[3:]))
            position_limit = max(0.002, float(np.linalg.norm(dpos))*0.25)
            orientation_limit = max(0.01, float(np.linalg.norm(drot))*0.25)
            diagnostics.update(linearized_position_residual_m=residual_pos,
                               linearized_position_limit_m=position_limit,
                               linearized_orientation_residual_rad=residual_ori,
                               linearized_orientation_limit_rad=orientation_limit)
            if not math.isfinite(residual_pos) or not math.isfinite(residual_ori):
                raise CorrectionError("IK residual must be finite")
            if residual_pos > position_limit or residual_ori > orientation_limit:
                raise RecoverableCorrectionError("IK linearized residual too large; correction was not executed",
                    reason_code="ik_residual", diagnostics=diagnostics)
        if self.execution_protocol:
            travel = self._joint_travel + np.abs(target-q)
            if np.any(travel > MAX_BURST_JOINT_TRAVEL_RAD + 1e-9):
                raise RecoverableCorrectionError("Original cumulative joint command envelope exceeded",
                    reason_code="ik_joint_bounds", diagnostics={"arm": arm,
                        "max_cumulative_joint_command_travel_rad": float(np.max(travel)),
                        "cumulative_joint_command_travel_limit_rad": MAX_BURST_JOINT_TRAVEL_RAD})
            self._joint_travel = travel
        action = self._hold.copy()
        action[self.action_indices[f"arm_{arm}"]] = target
        self.last_diagnostics = {"arm": arm, "max_joint_delta_rad": float(np.max(np.abs(target-q))),
                                 "linearized_position_residual_m": residual_pos,
                                 "linearized_orientation_residual_rad": residual_ori}
        action = self.backend.action_tensor(_finite(action, (23,), "correction action").astype(np.float32))
        self.remaining_steps -= 1
        return action
