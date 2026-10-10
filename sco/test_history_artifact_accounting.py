import ast
import copy
from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parents[1] / 'scripts')]
from artifact_acceptance import accepted_evidence_kind
from audit_history_assignment_artifacts import CONTRACT, METHOD, BINDING, validate_history_result


def fixture():
    # Reuse the existing summary-contract fixtures without importing GPU test dependencies.
    path = Path(__file__).parents[1] / 'scripts/test_subtask_sample_eval.py'
    functions = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in ('_sample', '_valid_result')]
    scope = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    sample = scope['_sample']()
    result = scope['_valid_result'](sample, 'f' * 64)
    raw = {'sha256': 'a' * 64, 'selected_state_sha256': 'b' * 64,
           'selected_state_size': 6, 'transition_count_before_frame': 1}
    proof = {'contract': CONTRACT, 'method': METHOD, 'sample_id': sample['sample_id'],
             'source_start': sample['start'], 'raw_sha256': raw['sha256'],
             'selected_state_sha256': raw['selected_state_sha256'],
             'recorded_prefix_physics_steps': sample['start'], 'prefix_actions_sha256': 'c' * 64,
             'runtime_history_snapshot_loaded_before_policy': True,
             'future_actions_used_to_prepare_start': False, 'manual_grasp_establishment_used': False,
             'exact_original_grasp_anchor_bytes_verified': False, 'full_robot_joint_positions': [0] * 28,
             'root_position': [0, 0, 0], 'root_orientation': [0, 0, 0, 1], 'recovered_grasps': []}
    result.update(state_recovery_protocol=CONTRACT, grasp_recovery_method=METHOD,
                  exact_original_grasp_anchor_bytes_verified=False, original_sample_scope_preserved=True,
                  production_difficulty_baseline=True, actual_model_control_steps=3,
                  model_action_audit=[{'step': i, 'preceding_observation_state': [0] * 61,
                                       'executed_action': [0] * 23} for i in range(3)])
    result['restoration'].update(history_recovery_source=proof, grasp_recovery_protocol=CONTRACT,
        grasp_recovery_method=METHOD, **{k: proof[k] for k in ('recorded_prefix_physics_steps',
        'prefix_actions_sha256', 'future_actions_used_to_prepare_start', 'manual_grasp_establishment_used',
        'exact_original_grasp_anchor_bytes_verified')})
    result['setup_evidence']['source_role_binding'] = {'contract': BINDING, 'raw_sha256': raw['sha256'],
        'source_state_sha256': raw['selected_state_sha256'], 'sample_id': sample['sample_id'],
        'grasp_recovery_protocol': CONTRACT, 'runtime_native_fields_verified': True,
        'desired_predicate_truth_used': False, 'exact_original_grasp_anchor_bytes_verified': False}
    return result, sample, 'f' * 64, raw, 'c' * 64


def test_approved_history_receipt_is_distinct_from_native_anchor_verification():
    result, *args = fixture()
    assert validate_history_result(result, *args)['exact_original_grasp_anchor_bytes_verified'] is False


@pytest.mark.parametrize('damage', ['raw', 'row', 'prefix', 'start', 'future', 'manual', 'native_claim',
                                   'missing_action', 'nan_action', 'wrong_shape', 'action_order',
                                   'predicted_prompt', 'success_without_monitor', 'changed_predicate'])
def test_changed_source_or_model_evidence_is_rejected(damage):
    result, *args = fixture()
    proof = result['restoration']['history_recovery_source']
    if damage == 'raw': proof['raw_sha256'] = 'd' * 64
    elif damage == 'row': proof['selected_state_sha256'] = 'd' * 64
    elif damage == 'prefix': proof['prefix_actions_sha256'] = 'd' * 64
    elif damage == 'start': proof['source_start'] += 1
    elif damage == 'future': proof['future_actions_used_to_prepare_start'] = True
    elif damage == 'manual': proof['manual_grasp_establishment_used'] = True
    elif damage == 'native_claim': result['exact_original_grasp_anchor_bytes_verified'] = True
    elif damage == 'missing_action': result['model_action_audit'].pop()
    elif damage == 'nan_action': result['model_action_audit'][0]['executed_action'][0] = float('nan')
    elif damage == 'wrong_shape': result['model_action_audit'][0]['executed_action'].pop()
    elif damage == 'action_order': result['model_action_audit'][0]['step'] = 2
    elif damage == 'predicted_prompt': result['observed_subtask_sources'][0] = 'predicted'
    elif damage == 'success_without_monitor': result['monitor']['dwell'] = 0
    else: result['predicate'] = {'kind': 'other'}
    with pytest.raises(ValueError): validate_history_result(result, *args)


def history_assignment():
    return {'expected_samples': 4, 'audited_samples': 4,
            'sample_ids': ['a', 'b', 'c', 'd'], 'episode_keys': [['task', i] for i in range(4)],
            'all_required_video_sha_and_first_frames_passed': True, 'state_recovery_protocol': CONTRACT,
            'original_scope_and_sample_ids_matched': True, 'independent_original_raw_and_prefix_hashes_matched': True,
            'recovery_receipts_and_source_role_bindings_checked': True,
            'all_history_setup_exclusions_checked_against_saved_evidence': True,
            'exact_original_grasp_anchor_bytes_verified': False,
            'independent_all_raw_native_fields_matched': False, 'additional_grasp_physics_replayed_for_audit': False}


def test_full_history_assignment_can_be_accepted_with_explicit_method():
    assert accepted_evidence_kind(history_assignment()) == 'approved_physical_history'


@pytest.mark.parametrize('damage', ['native_claim', 'missing_prefix', 'duplicate', 'missing_h5', 'unknown_protocol'])
def test_history_acceptance_rejects_missing_coverage_or_misstated_evidence(damage):
    row = history_assignment()
    if damage == 'native_claim': row['independent_all_raw_native_fields_matched'] = True
    elif damage == 'missing_prefix': row['independent_original_raw_and_prefix_hashes_matched'] = False
    elif damage == 'duplicate': row['sample_ids'][-1] = row['sample_ids'][0]
    elif damage == 'missing_h5': row['episode_keys'].pop()
    else: row['state_recovery_protocol'] = 'unapproved'
    with pytest.raises(ValueError): accepted_evidence_kind(row)
