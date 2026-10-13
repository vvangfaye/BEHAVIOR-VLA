"""Diagnostic joint probes, excluded from data and followed by fresh restoration."""
from __future__ import annotations

import json
import numpy as np

CONTRACT = 'floating_base_jacobian_probe_v1'
RUNTIME_CONTRACT = 'runtime_jacobian_probe_v2'


def assess(rows):
    if len(rows) != 14 or {(r['joint'], r['sign']) for r in rows} != {
            (i, sign) for i in range(7) for sign in (-1, 1)}:
        raise ValueError('Need both finite-difference directions for all seven arm joints')
    total = error = old_error = 0.
    for row in rows:
        actual = np.asarray(row['actual_translation_m'], dtype=float)
        predicted = np.asarray(row['predicted_translation_m'], dtype=float)
        previous = np.asarray(row['old_row_predicted_translation_m'], dtype=float)
        if (any(v.shape != (3,) or not np.isfinite(v).all() for v in (actual,predicted,previous))
                or not np.isfinite(row['actual_joint_motion_rad']) or row['actual_joint_motion_rad'] <= 0
                or row['base_translation_m'] > .0002 or row['base_rotation_rad'] > .002):
            raise ValueError('Invalid probe motion or base drift')
        total += float(actual @ actual)
        error += float((actual-predicted) @ (actual-predicted))
        old_error += float((actual-previous) @ (actual-previous))
    if total <= 1e-8:
        raise ValueError('Probe did not produce enough measured translation')
    relative = float(np.sqrt(error/total))
    if relative > .15 or np.sqrt(error/len(rows)) > .0003:
        raise ValueError('Selected Jacobian does not predict actual joint-probe translation')
    return dict(passed=True, translation_relative_error=relative,
        old_row_translation_relative_error=float(np.sqrt(old_error/total)),
        translation_rmse_m=float(np.sqrt(error/len(rows))), probes=len(rows))


def run(evaluator, plan, cache, materialized, output, *, restore_fn=None):
    from collect_b1k_codex_subtask import _restore
    from codex_collection_actions import BoundedCartesianCorrection, _array
    from codex_subtask_qualification import OriginalActions
    from codex_subtask_collection import NATIVE
    from codex_robot_guidance import pose, rotation_error

    sample = plan['sample']
    protocol = plan['source_plan'].get('source_protocol_by_sample', {}).get(sample['sample_id'], NATIVE)
    original = evaluator.policy
    rows = []
    layout = None
    contract = plan.get('teacher_motion_validation', CONTRACT)
    def restore():
        if restore_fn is None:
            _restore(evaluator, cache, sample, materialized, protocol)
        else:
            restore_fn()
    output.parent.mkdir(parents=True, exist_ok=True)
    def persist(passed, **extra):
        result = dict(contract=contract, plan_fingerprint=plan['fingerprint'],
            diagnostic_only=True, training_eligible=False, model_calls=0,
            probe_controls=len(rows)*3, passed=passed, rows=rows, layout=layout, **extra)
        output.write_text(json.dumps(result, allow_nan=False, indent=2)+'\n')
        return result
    try:
        for joint in range(7):
            for sign in (-1, 1):
                restore()
                control = BoundedCartesianCorrection(evaluator.robot)
                # Start from a zero-motion open-hand action; all other joints hold.
                control.begin(dict(arm='left', delta_position_m=[0.,0.,0.],
                    delta_rotation_axis_angle_rad=[0.,0.,0.], gripper='open', steps=1))
                action = _array(control.next_action()).copy()
                robot, backend = evaluator.robot, control.backend
                before_q = _array(robot.get_joint_positions()).copy()
                before_pos = _array(backend.pose(robot, 'left')[0]).copy()
                before_base = pose(*robot.get_position_orientation())
                full_jac = _array(backend.views.get_all_relative_jacobians(robot.articulation_root_path))
                members = _array(backend.views.get_member_view_indices(
                    robot.articulation_root_path, [robot.articulation_root_path]))
                body = backend.views.get_link_index(robot.articulation_root_path, robot.eef_link_names['left'])
                offset = full_jac.shape[-1] - before_q.size
                row_index = body if offset == 6 else body-1
                layout = dict(body_index=int(body), column_offset=int(offset),
                    selected_row=int(row_index), jacobian_shape=list(full_jac.shape),
                    joint_count=int(before_q.size))
                if offset not in (0,6) or not 0 <= row_index < full_jac.shape[1] or body <= 0:
                    raise ValueError('Unsupported live Jacobian layout: '+str(layout))
                selected = full_jac[int(members[0]), row_index, :3, offset:].copy()
                old = full_jac[int(members[0]), body-1, :3, offset:].copy()
                candidates = full_jac[int(members[0]), :, :3, offset:].copy()
                action[control.action_indices['arm_left'][joint]] += sign*.004
                low, high = robot.control_limits['position']
                desired = action[control.action_indices['arm_left']]
                idx = control.joint_indices['arm_left']
                if np.any(desired < _array(low)[idx]) or np.any(desired > _array(high)[idx]):
                    raise ValueError('Joint probe would exceed native joint bounds')
                evaluator.policy = OriginalActions(np.tile(action, (3,1)), sample['subtask'])
                for _ in range(3):
                    evaluator.external_subtask = sample['subtask']
                    terminated, truncated = evaluator.step()
                    if terminated or truncated:
                        raise ValueError('Probe unexpectedly terminated the environment')
                after_q = _array(robot.get_joint_positions()).copy()
                after_pos = _array(backend.pose(robot, 'left')[0]).copy()
                after_base = pose(*robot.get_position_orientation())
                dq = after_q-before_q
                rows.append(dict(joint=joint, sign=sign, body_index=int(body), selected_row=int(row_index),
                    column_offset=int(offset), jacobian_shape=list(full_jac.shape),
                    commanded_joint_delta_rad=sign*.004,
                    actual_joint_motion_rad=float(np.linalg.norm(dq[idx])),
                    actual_translation_m=(after_pos-before_pos).tolist(),
                    predicted_translation_m=(selected@dq).tolist(),
                    old_row_predicted_translation_m=(old@dq).tolist(),
                    candidate_row_predictions_m=(candidates@dq).tolist(),
                    base_translation_m=float(np.linalg.norm(np.array(after_base['position_m'])-before_base['position_m'])),
                    base_rotation_rad=float(np.linalg.norm(rotation_error(
                        after_base['quaternion_xyzw'], before_base['quaternion_xyzw'])))))
                persist(False)
        return persist(True, assessment=assess(rows))
    except Exception as error:
        persist(False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        evaluator.policy = original
        restore()


def run_initial(evaluator, plan, cache, output):
    """Check actual runtime layout before paying for the long history replay."""
    import copy
    import omnigibson as og
    import eval_b1k_subtask_samples as runner
    from diagnose_b1k_grasp_history import cpu_tree
    from legacy_history_recovery import _counter_state
    saved = dict(state=cpu_tree(runner._dump_full_simulator_state(evaluator.env.scene, og.sim)),
        state_format=runner.FULL_SNAPSHOT_FORMAT, transition_steps=[],
        restoration=dict(diagnostic_only=True, training_eligible=False))
    counters = _counter_state(evaluator.robot)
    def restore():
        runner.restore_materialized_sample_state(evaluator, cache, saved)
        for name, value in counters.items():
            setattr(evaluator.robot, name, copy.deepcopy(value))
    return run(evaluator, plan, cache, None, output, restore_fn=restore)


def verify(path, plan):
    from codex_subtask_collection import file_sha256
    value = json.loads(path.read_text())
    if (value.get('contract') != plan.get('teacher_motion_validation', CONTRACT)
            or value.get('contract') not in (CONTRACT,RUNTIME_CONTRACT)
            or value.get('plan_fingerprint') != plan['fingerprint']
            or value.get('passed') is not True or value.get('diagnostic_only') is not True
            or value.get('training_eligible') is not False or value.get('model_calls') != 0
            or value.get('probe_controls') != 42):
        raise ValueError('Live Jacobian validation missing or inconsistent')
    measured = assess(value['rows'])
    saved = value.get('assessment', {})
    if set(saved) != set(measured) or any(
            not np.isclose(saved[key], measured[key], atol=1e-10, rtol=1e-10) for key in measured):
        raise ValueError('Stored Jacobian assessment differs from the measurements')
    return dict(sha256=file_sha256(path), contract=value['contract'])
