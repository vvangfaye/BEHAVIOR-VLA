import json
from types import SimpleNamespace
import numpy as np
import pytest
from codex_gripper_feedback import GripperExecutionFeedback


def fixture():
    links={arm:[SimpleNamespace(prim_path=f'robot/{arm}/{i}') for i in range(2)] for arm in ('left','right')}
    hidden='DO_NOT_DISCLOSE_TARGET_OBJECT_OR_POSITION'
    robot=SimpleNamespace(finger_links=links,gripper_control_idx={'left':np.array([0,1]),'right':np.array([2,3])},
        get_joint_positions=lambda:np.array([0.04,0.03,0.05,0.05]),
        _find_gripper_contacts=lambda arm:(set(),{hidden:{links[arm][0].prim_path}} if arm=='left' else {}))
    clock=GripperExecutionFeedback(action_hz=30,grasp_window_seconds=0.3,gripper_indices={'left':[21],'right':[22]})
    return clock,robot,hidden


def test_only_executed_controls_advance_clock_and_open_resets_one_arm():
    clock,robot,_=fixture();a=np.zeros(23,np.float32);a[21:]=-1
    for _ in range(5):clock.observe_executed_action(a)
    one=clock.snapshot(robot)
    # Re-reading while the model waits or after a rejected correction cannot advance physics.
    for _ in range(20):assert clock.snapshot(robot)==one
    assert one['arms']['left']['full_close_elapsed_sim_seconds']==pytest.approx(5/30)
    assert one['initial_close_settling_advice_control_steps']==15
    a[21]=1;clock.observe_executed_action(a)
    two=clock.snapshot(robot)
    assert two['arms']['left']['consecutive_full_close_control_steps']==0
    assert two['arms']['right']['consecutive_full_close_control_steps']==6
    assert two['executed_control_steps']==6


def test_tactile_snapshot_discards_object_identity_and_never_reports_target_success():
    clock,robot,hidden=fixture();q=clock.snapshot(robot)
    assert q['arms']['left']['finger_environment_contact']==[True,False]
    assert q['arms']['left']['both_fingers_touch_something'] is False
    assert q['contact_object_identity_disclosed'] is False and q['target_grasp_truth_disclosed'] is False
    assert hidden not in json.dumps(q)
    assert all('held_object' not in arm and 'target_position' not in arm for arm in q['arms'].values())


@pytest.mark.parametrize('rate,window',[(0,0.3),(float('nan'),0.3),(30,0),(30,float('inf'))])
def test_invalid_runtime_timing_rejected(rate,window):
    with pytest.raises(ValueError):GripperExecutionFeedback(action_hz=rate,grasp_window_seconds=window,gripper_indices={'left':[21],'right':[22]})


def test_bad_executed_action_does_not_advance_clock():
    clock,robot,_=fixture();before=clock.snapshot(robot)
    for action in (np.zeros(22),np.full(23,np.nan)):
        with pytest.raises(ValueError):clock.observe_executed_action(action)
        assert clock.snapshot(robot)==before
