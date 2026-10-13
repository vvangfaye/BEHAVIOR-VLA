"""Robot-only tactile and executed-time feedback for a collection teacher.

No object identity, object pose, target predicate or native grasp truth is sent.
The clock advances only when a real action has been executed.
"""
from __future__ import annotations
import importlib
import math
import numpy as np

CONTRACT = 'robot_gripper_execution_v1'


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class GripperExecutionFeedback:
    def __init__(self, *, action_hz, grasp_window_seconds, gripper_indices):
        if (type(action_hz) not in (int, float) or not math.isfinite(action_hz) or action_hz <= 0
                or type(grasp_window_seconds) not in (int, float)
                or not math.isfinite(grasp_window_seconds) or not 0 < grasp_window_seconds <= 10):
            raise ValueError('Actual positive finite control rate and grasp window are required')
        if set(gripper_indices) != {'left', 'right'}:
            raise ValueError('Feedback requires both R1Pro gripper action indices')
        indices = {arm: _array(idx).reshape(-1) for arm, idx in gripper_indices.items()}
        if any(v.shape != (1,) or v.dtype.kind not in 'iu' or not 0 <= v[0] < 23 for v in indices.values()):
            raise ValueError('Each gripper must have one valid 23-D action index')
        self.indices = {arm: int(idx[0]) for arm, idx in indices.items()}
        if len(set(self.indices.values())) != 2:
            raise ValueError('The two grippers cannot share an action index')
        self.action_hz = float(action_hz)
        self.grasp_window_seconds = float(grasp_window_seconds)
        self.full_close_steps = dict(left=0, right=0)
        self.executed_control_steps = 0

    @classmethod
    def from_evaluator(cls, evaluator):
        robot = evaluator.robot
        method = robot._handle_assisted_grasping
        module = importlib.import_module(method.__func__.__module__)
        return cls(action_hz=evaluator.env.env_config['action_frequency'],
                   grasp_window_seconds=float(module.m.GRASP_WINDOW),
                   gripper_indices={arm: robot.controller_action_idx[f'gripper_{arm}'] for arm in ('left', 'right')})

    def observe_executed_action(self, action):
        action = _array(action).reshape(-1)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError('Feedback only accepts a finite actually executed 23-D action')
        for arm, idx in self.indices.items():
            # The evaluation smooth-position controller clips [-1,1]; -1 is full close.
            self.full_close_steps[arm] = self.full_close_steps[arm] + 1 if action[idx] <= -0.95 else 0
        self.executed_control_steps += 1

    def snapshot(self, robot):
        arms = {}
        for arm in ('left', 'right'):
            _, by_object = robot._find_gripper_contacts(arm=arm)
            # Deliberately discard the semantic identities in the native contact API.
            touching_links = set().union(*by_object.values()) if by_object else set()
            fingers = list(robot.finger_links[arm])
            if len(fingers) != 2:
                raise ValueError('This feedback contract requires two fingers per gripper')
            q = _array(robot.get_joint_positions())[_array(robot.gripper_control_idx[arm]).astype(int)]
            if q.shape != (2,) or not np.isfinite(q).all():
                raise ValueError('Actual finite finger joint positions are required')
            touching = [link.prim_path in touching_links for link in fingers]
            arms[arm] = dict(consecutive_full_close_control_steps=self.full_close_steps[arm],
                full_close_elapsed_sim_seconds=self.full_close_steps[arm]/self.action_hz,
                finger_link_names=[link.prim_path.rsplit('/',1)[-1] for link in fingers],
                finger_joint_position_m=q.tolist(), finger_environment_contact=touching,
                both_fingers_touch_something=all(touching))
        return dict(contract=CONTRACT, executed_control_steps=self.executed_control_steps,
            control_step_seconds=1/self.action_hz, grasp_window_seconds=self.grasp_window_seconds,
            grasp_contact_window_control_steps=math.ceil(self.grasp_window_seconds*self.action_hz),
            initial_close_settling_advice_control_steps=math.ceil((self.grasp_window_seconds+0.2)*self.action_hz),
            arms=arms, contact_object_identity_disclosed=False, target_grasp_truth_disclosed=False,
            interpretation='Time counts executed simulation controls, never model wall time. Full-close time is not contact duration. '
                           'Tactile flags mean each finger touches something; they do not prove the same object or the target is grasped.')
