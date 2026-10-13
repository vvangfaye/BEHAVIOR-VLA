"""CPU contract / numerical adapter tests; no simulator data is produced."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from codex_collection_actions import (
    BoundedCartesianCorrection, CorrectionError, RecoverableCorrectionError, _OmniGibsonBackend,
    _orientation_error, _rotation_from_axis_angle, validate_correction, EXTENDED_BURST_PROTOCOL,
)


def correction(**values):
    return {"arm": "left", "delta_position_m": [0.01, 0, 0],
            "delta_rotation_axis_angle_rad": [0, 0, 0], "gripper": "keep", "steps": 2, **values}


class Robot:
    model, action_dim, _action_normalize = "r1pro", 23, False
    articulation_root_path = "/fake/robot"
    base_footprint_link_name = "base_footprint"
    eef_link_names = {"left": "left_eef", "right": "right_eef"}

    def __init__(self):
        self.q = np.zeros(28)
        self.q[6:10] = [0.1, -0.1, -0.2, 0.0]
        self.q[17:24] = 0.15
        self.q[24:] = 0.025
        self.controller_action_idx = {"base": np.arange(3), "trunk": np.arange(3, 7),
            "arm_left": np.arange(7, 14), "arm_right": np.arange(15, 22),
            "gripper_left": np.array([14]), "gripper_right": np.array([22]), "camera": np.array([], dtype=int)}
        self.arm_control_idx = {"left": np.arange(10, 17), "right": np.arange(17, 24)}
        self.gripper_control_idx = {"left": np.arange(24, 26), "right": np.arange(26, 28)}
        self.controller_joint_idx = {"trunk": np.arange(6, 10),
            "arm_left": self.arm_control_idx["left"], "arm_right": self.arm_control_idx["right"]}
        self._controller_config = {
            name: {"name": "JointController", "motor_type": "position", "use_delta_commands": False,
                   "use_impedances": False, "command_input_limits": None, "command_output_limits": None}
            for name in ("trunk", "arm_left", "arm_right")}
        self._controller_config.update({f"gripper_{arm}": {"name": "MultiFingerGripperController",
            "motor_type": "position", "mode": "smooth", "command_input_limits": "default",
            "command_output_limits": "default"} for arm in ("left", "right")})
        self._controller_config["base"] = {"name": "HolonomicBaseJointController", "motor_type": "velocity",
            "command_input_limits": [[-1, -1, -1], [1, 1, 1]],
            "command_output_limits": [[-.75, -.75, -1], [.75, .75, 1]]}
        self._controller_config["camera"] = {"name": "NullJointController"}
        self.control_limits = {"position": (-np.ones(28)*3, np.ones(28)*3)}

    def get_joint_positions(self):
        return self.q.copy()


def upstream_numpy_solver():
    """Load only upstream numerical function bodies, avoiding Isaac Sim imports.

    This exercises the repository's actual J-PARSE implementation, not a fake
    solver or copied alternative that could drift from production behavior.
    """
    path = Path(__file__).resolve().parents[1] / "BEHAVIOR-1K/OmniGibson/omnigibson/controllers/ik_controller.py"
    tree = ast.parse(path.read_text())
    names = {"_jparse_compute_numpy", "_compute_ik_qpos_batch_numpy"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(definitions) == 2
    nt = SimpleNamespace(orientation_error=lambda goal, current: np.stack([
        _orientation_error(target, start) for target, start in zip(goal, current)]))
    namespace = {"np": np, "NT": nt}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_compute_ik_qpos_batch_numpy"]


class Backend:
    def __init__(self):
        self.solve = upstream_numpy_solver()
        self.jac = np.c_[np.eye(6), np.zeros(6)]
        self.pose_calls = []
        self.quaternion = np.array([0, 0, 0, 1.])

    def pose(self, robot, arm):
        self.pose_calls.append(arm)
        return robot.q[robot.arm_control_idx[arm]][:3], self.quaternion

    def jacobian(self, robot, arm, indices):
        return self.jac

    def noop(self, robot, name):
        if name in robot.controller_joint_idx:
            return robot.q[robot.controller_joint_idx[name]]
        return np.zeros(len(robot.controller_action_idx[name]))

    def action_tensor(self, action):
        return action.copy()


def adapter():
    robot, backend = Robot(), Backend()
    return robot, backend, BoundedCartesianCorrection(robot, backend=backend)


@pytest.mark.parametrize('values', [
    {'steps':21}, {'delta_position_m':[.006,0,0]}, {'gripper':'keep'}, {'gripper':'close'},
    {'delta_rotation_axis_angle_rad':[.01,0,0]},
])
def test_extended_bursts_reject_unsafe_long_motion(values):
    proposal=correction(steps=20,gripper='open',delta_position_m=[.005,0,0])
    proposal.update(values)
    with pytest.raises(CorrectionError):
        validate_correction(proposal,execution_protocol=EXTENDED_BURST_PROTOCOL)


def test_twenty_small_steps_preserve_other_controls_and_original_total_translation():
    robot, backend = Robot(), Backend()
    control = BoundedCartesianCorrection(robot,backend=backend,execution_protocol=EXTENDED_BURST_PROTOCOL)
    control.begin(correction(steps=20,gripper='open',delta_position_m=[.005,0,0]))
    for _ in range(20):
        action=control.next_action()
        np.testing.assert_allclose(action[15:22],.15)
        np.testing.assert_allclose(action[:3],0)
        assert action[14]==1
        robot.q[10:17]=action[7:14]
    assert robot.q[10]==pytest.approx(.1,abs=1e-6)
    assert control.remaining_steps==0
    state=control.snapshot()
    assert state['execution_protocol']==EXTENDED_BURST_PROTOCOL
    assert state['limits']['translation_norm_m_per_burst']==.1


def test_extended_burst_rejects_cumulative_joint_travel_before_emitting_next_action():
    robot, backend = Robot(), Backend();backend.jac=np.c_[np.eye(6)*.05,np.zeros(6)]
    control=BoundedCartesianCorrection(robot,backend=backend,execution_protocol=EXTENDED_BURST_PROTOCOL)
    control.begin(correction(steps=20,gripper='open',delta_position_m=[.005,0,0]))
    for _ in range(7):robot.q[10:17]=control.next_action()[7:14]
    before=robot.q.copy()
    with pytest.raises(RecoverableCorrectionError) as caught:control.next_action()
    assert caught.value.reason_code=='ik_joint_bounds'
    assert caught.value.diagnostics['max_cumulative_joint_command_travel_rad']>.75
    np.testing.assert_array_equal(robot.q,before)
    assert control.remaining_steps==13


@pytest.mark.parametrize('gripper,translation,interrupt', [
    ('open',[.003,0,0],True), ('open',[0,0,0],False), ('close',[0,0,0],False),
])
def test_contact_returns_only_open_translation_to_teacher(gripper,translation,interrupt):
    robot,backend=Robot(),Backend()
    control=BoundedCartesianCorrection(robot,backend=backend,execution_protocol=EXTENDED_BURST_PROTOCOL)
    control.begin(correction(steps=20,gripper=gripper,delta_position_m=translation))
    control.next_action()
    tactile={'contract':'robot_gripper_execution_v1','arms':{'left':{'finger_environment_contact':[True,False]}}}
    assert control.interrupt_on_contact(tactile) is interrupt
    assert control.remaining_steps==(0 if interrupt else 19)


@pytest.mark.parametrize("values", [
    {"steps": 0}, {"steps": 6}, {"steps": True}, {"steps": 1.5}, {"arm": "both"},
    {"gripper": "half"}, {"delta_position_m": [0.02, 0.02, 0]},
    {"delta_rotation_axis_angle_rad": [0.1, 0.1, 0]}, {"delta_position_m": [float("nan"), 0, 0]},
    {"delta_position_m": [True, False, False]}, {"delta_position_m": ["0", "0", "0"]},
    {"delta_position_m": [0, 0]}, {"delta_position_m": [0, 0, 0]},
    {"joint_targets": [0]*7},
])
def test_invalid_teacher_corrections_fail_before_execution(values):
    with pytest.raises(CorrectionError):
        validate_correction(correction(**values))


def test_snapshot_documents_measured_state_frame_and_units():
    robot, _, control = adapter()
    state = control.snapshot()
    assert state["frame"] == "robot_base_footprint"
    assert state["frame_link"] == robot.base_footprint_link_name
    assert state["arms"]["right"]["eef_position_m"] == [0.15]*3
    assert state["arms"]["left"]["eef_quaternion_xyzw"] == [0, 0, 0, 1]
    assert state["arms"]["right"]["finger_joint_position_m"] == [0.025]*2
    assert state["limits"]["max_steps"] == 5


def test_upstream_ik_maps_dynamic_indices_and_preserves_other_controls():
    robot, backend, control = adapter()
    previous = np.zeros(23)
    previous[14], previous[22] = -0.9, 0.3
    control.begin(correction(), previous)
    action = control.next_action()
    assert action.shape == (23,) and action.dtype == np.float32
    np.testing.assert_allclose(action[robot.controller_action_idx["trunk"]], robot.q[6:10])
    np.testing.assert_allclose(action[robot.controller_action_idx["arm_right"]], robot.q[17:24])
    np.testing.assert_allclose(action[[14, 22]], [-0.9, 0.3])
    np.testing.assert_allclose(action[:3], 0)
    np.testing.assert_allclose(action[7:14], [0.01, 0, 0, 0, 0, 0, 0], atol=1e-7)
    # Emulate one real environment step: the next solve must use the NEW state.
    robot.q[10:17] = action[7:14]
    second = control.next_action()
    assert second[7] == pytest.approx(0.02)
    assert control.remaining_steps == 0
    assert backend.pose_calls == ["left", "left"]
    with pytest.raises(CorrectionError, match="No pending"):
        control.next_action()


def test_base_frame_rotation_left_multiplies_current_orientation():
    _, backend, control = adapter()
    backend.quaternion = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    captured = {}
    actual = backend.solve
    def spy(**kwargs):
        captured.update(kwargs)
        return actual(**kwargs)
    backend.solve = spy
    control.begin(correction(delta_position_m=[0, 0, 0], delta_rotation_axis_angle_rad=[0.05, 0, 0]))
    action = control.next_action()
    expected = _rotation_from_axis_angle([0.05, 0, 0]) @ captured["ee_mat"][0]
    np.testing.assert_allclose(captured["goal_ori_mat"][0], expected, atol=1e-7)
    assert action[10] == pytest.approx(np.sin(0.05), abs=1e-7)


def test_gripper_only_bypasses_ik_and_does_not_release_other_hand():
    robot, backend, control = adapter()
    backend.solve = lambda **_: pytest.fail("gripper-only correction should not run IK")
    previous = np.zeros(23)
    previous[22] = -1
    control.begin(correction(gripper="open", delta_position_m=[0, 0, 0], steps=1), previous)
    action = control.next_action()
    assert action[14] == 1 and action[22] == -1
    np.testing.assert_allclose(action[7:14], robot.q[10:17])


def upstream_gripper_preprocess():
    """Exercise actual controller clipping/scaling without importing Isaac Sim."""
    path = Path(__file__).resolve().parents[1] / "BEHAVIOR-1K/OmniGibson/omnigibson/controllers/controller_base.py"
    tree = ast.parse(path.read_text())
    controller = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseController")
    method = next(node for node in controller.body if isinstance(node, ast.FunctionDef) and node.name == "_preprocess_command")
    namespace = {"cb": SimpleNamespace(array=np.array)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    state = SimpleNamespace(_command_input_limits=(-np.ones(1), np.ones(1)),
        _command_output_limits=(np.zeros(1), np.full(1, .05)), _command_scale_factor=None)
    return lambda raw: namespace["_preprocess_command"](state, np.array([raw], dtype=np.float64))


@pytest.mark.parametrize("left,right", [(-1.04, 1.025), (1.0000005, -.4), (100., -100.), (.2, -.7)])
def test_keep_preserves_both_grippers_effective_upstream_command(left, right):
    robot, _, control = adapter()
    previous = np.zeros(23);previous[14], previous[22] = left, right
    before = previous.copy();q_before = robot.q.copy()
    control.begin(correction(steps=1), previous)
    action = control.next_action()
    assert np.all(np.abs(action[[14, 22]]) <= 1)
    preprocess = upstream_gripper_preprocess()
    for index in (14, 22):
        np.testing.assert_allclose(preprocess(action[index]), preprocess(previous[index]), atol=1e-9)
    np.testing.assert_equal(previous, before)  # Existing trajectory evidence stays raw.
    np.testing.assert_equal(robot.q, q_before)
    np.testing.assert_allclose(action[15:22], q_before[17:24])


@pytest.mark.parametrize("arm,command,target_index,other_index", [
    ("left", "open", 14, 22), ("left", "close", 14, 22),
    ("right", "open", 22, 14), ("right", "close", 22, 14)])
def test_gripper_override_uses_exact_boundary_and_holds_non_target_hand(arm, command, target_index, other_index):
    robot, backend, control = adapter()
    backend.solve = lambda **_: pytest.fail("Pure gripper correction must not invoke IK")
    previous = np.zeros(23)
    previous[14], previous[22] = -1.03, 1.04
    before = previous.copy()
    control.begin(correction(arm=arm, gripper=command, delta_position_m=[0,0,0], steps=1), previous)
    action = control.next_action()
    assert action[target_index] == (1. if command == "open" else -1.)
    assert action[other_index] == np.clip(previous[other_index], -1., 1.)
    preprocess = upstream_gripper_preprocess()
    np.testing.assert_allclose(preprocess(action[other_index]), preprocess(previous[other_index]), atol=1e-9)
    np.testing.assert_allclose(action[7:14], robot.q[10:17])
    np.testing.assert_allclose(action[15:22], robot.q[17:24])
    np.testing.assert_equal(action[:3], 0)
    np.testing.assert_equal(previous, before)
    assert control.remaining_steps == 0


@pytest.mark.parametrize("index", [0, 7, 14, 22])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_previous_action_still_fails_even_when_target_gripper_will_be_overridden(index, value):
    robot, _, control = adapter()
    previous = np.zeros(23);previous[index] = value;q_before = robot.q.copy()
    with pytest.raises(CorrectionError, match="last executed action must have finite numeric shape"):
        control.begin(correction(arm="right",gripper="close",delta_position_m=[0,0,0],steps=1),previous)
    assert control.remaining_steps == 0 and control.last_diagnostics is None
    np.testing.assert_equal(robot.q, q_before)


@pytest.mark.parametrize("failure", ["singular", "nonfinite", "oversized", "joint_limit", "bad_quaternion"])
def test_bad_ik_solutions_are_not_executed(failure):
    robot, backend, control = adapter()
    if failure == "singular":
        backend.jac = np.c_[np.diag([0, 1, 1, 1, 1, 1]), np.zeros(6)]
    elif failure == "nonfinite":
        backend.solve = lambda **_: np.full((1, 7), np.nan)
    elif failure == "oversized":
        backend.solve = lambda **_: np.full((1, 7), 0.2)
    elif failure == "joint_limit":
        robot.control_limits["position"][1][10] = 0.001
    else:
        backend.quaternion = np.ones(4)
    control.begin(correction())
    with pytest.raises(CorrectionError) as caught:
        control.next_action()
    assert control.remaining_steps == 2
    recoverable = failure in {"singular", "oversized", "joint_limit"}
    assert isinstance(caught.value, RecoverableCorrectionError) is recoverable
    if recoverable:
        assert caught.value.reason_code in {"ik_joint_bounds", "ik_residual"}
        assert all(np.isfinite(value) for key,value in caught.value.diagnostics.items() if key != 'arm')


def test_rejected_ik_keeps_joint_state_and_can_cancel_then_replan():
    robot, backend, control = adapter()
    backend.jac = np.c_[np.diag([0, 1, 1, 1, 1, 1]), np.zeros(6)]
    before = robot.q.copy()
    control.begin(correction())
    with pytest.raises(RecoverableCorrectionError) as caught:control.next_action()
    assert caught.value.reason_code == 'ik_residual'
    diagnostic = caught.value.diagnostics
    assert diagnostic['linearized_position_residual_m'] == pytest.approx(0.01)
    assert diagnostic['linearized_position_limit_m'] == pytest.approx(0.0025)
    assert diagnostic['linearized_orientation_limit_rad'] == pytest.approx(0.01)
    np.testing.assert_equal(robot.q, before)
    assert control.last_diagnostics is None
    control.cancel()
    assert control.remaining_steps == 0
    control.begin(correction(delta_position_m=[0, 0.01, 0], steps=1))
    action = control.next_action()
    assert action[8] == pytest.approx(0.01)
    np.testing.assert_equal(robot.q, before)  # Even accepted preparation only returns a candidate.


def test_invalid_current_joints_remain_hard_error_and_diagnostics_must_be_finite():
    robot, _, control = adapter()
    robot.q[10] = 4
    control.begin(correction())
    with pytest.raises(CorrectionError) as caught:control.next_action()
    assert not isinstance(caught.value, RecoverableCorrectionError)
    with pytest.raises(CorrectionError, match='finite numeric'):
        RecoverableCorrectionError('fixture', reason_code='ik_residual', diagnostics={'arm':'left','residual':np.inf})


@pytest.mark.parametrize("mutation", [
    lambda r: setattr(r, "model", "r1"),
    lambda r: setattr(r, "_action_normalize", True),
    lambda r: r._controller_config["arm_left"].update(use_delta_commands=True),
    lambda r: r._controller_config["arm_left"].update(command_input_limits="default"),
    lambda r: r._controller_config["arm_left"].update(use_impedances=True),
    lambda r: r._controller_config["gripper_left"].update(mode="binary"),
    lambda r: r._controller_config["base"].update(command_input_limits=[[0, 0, 0], [1, 1, 1]]),
    lambda r: r.controller_action_idx.update(gripper_left=np.array([22])),
    lambda r: r.controller_action_idx.update(gripper_left=np.array([14.5])),
    lambda r: r.arm_control_idx.update(left=np.arange(11, 18)),
    lambda r: r.controller_action_idx.update(camera=np.array([0])),
])
def test_unknown_action_mappings_fail_closed(mutation):
    robot = Robot()
    mutation(robot)
    with pytest.raises(CorrectionError):
        BoundedCartesianCorrection(robot, backend=Backend())


def test_pending_burst_cannot_be_overwritten():
    _, _, control = adapter()
    control.begin(correction())
    with pytest.raises(CorrectionError, match="pending"):
        control.begin(correction(arm="right"))


def test_live_jacobian_selects_correct_member_link_and_floating_base_offset():
    backend = object.__new__(_OmniGibsonBackend)
    robot = Robot()
    jac = np.zeros((2, 4, 6, 34))
    expected = np.arange(42).reshape(6, 7)
    jac[1, 2][:, np.arange(16, 23)] = -99  # preceding link must not be selected
    jac[1, 3][:, np.arange(16, 23)] = expected
    backend.views = SimpleNamespace(
        get_member_view_indices=lambda path, members: np.array([1]),
        get_all_relative_jacobians=lambda path: jac,
        get_all_joint_positions=lambda path: np.zeros((2, 28)),
        get_link_index=lambda path, link: 3)
    np.testing.assert_equal(backend.jacobian(robot, "left", np.arange(10, 17)), expected)
    backend.views.get_all_joint_positions = lambda path: np.zeros((2, 29))
    with pytest.raises(CorrectionError, match="floating-base"):
        backend.jacobian(robot, "left", np.arange(10, 17))


def test_fixed_base_jacobian_omits_root_row_and_root_columns():
    backend = object.__new__(_OmniGibsonBackend)
    jac = np.zeros((1, 4, 6, 28))
    expected = np.arange(42).reshape(6, 7)
    jac[0, 2][:, np.arange(10, 17)] = expected
    jac[0, 3][:, np.arange(10, 17)] = -99
    backend.views = SimpleNamespace(
        get_member_view_indices=lambda path, members: np.array([0]),
        get_all_relative_jacobians=lambda path: jac,
        get_all_joint_positions=lambda path: np.zeros((1, 28)),
        get_link_index=lambda path, link: 3)
    np.testing.assert_equal(backend.jacobian(Robot(), 'left', np.arange(10, 17)), expected)
