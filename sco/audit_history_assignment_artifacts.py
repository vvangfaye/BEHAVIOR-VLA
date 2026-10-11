"""Account for original samples evaluated with the user-approved history protocol.

This checks source bytes, original action prefixes and emitted rollout receipts.
It does not replay the physics or assert equality with unrecorded grasp anchors.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

CONTRACT = 'recorded_actions_before_start_native_grasp_recovery_v1'
METHOD = 'recorded_full_states_with_native_physical_action_prefix_v1'
BINDING = 'recorded_task_scope_with_physical_history_grasp_identity_v1'


def validate_history_result(result, sample, fingerprint, raw, prefix_sha256, *,
                            production_difficulty_baseline=True):
    from summarize_subtask_closed_loop import _validate_result

    errors = _validate_result(result, sample, fingerprint, raw, 2048)
    if errors:
        raise ValueError('; '.join(errors))
    if (result.get('state_recovery_protocol') != CONTRACT
            or result.get('grasp_recovery_method') != METHOD
            or result.get('exact_original_grasp_anchor_bytes_verified') is not False
            or result.get('original_sample_scope_preserved') is not True
            or result.get('production_difficulty_baseline') is not production_difficulty_baseline):
        raise ValueError('History result changed its approved recovery contract')
    restoration = result['restoration']
    proof = restoration.get('history_recovery_source', {})
    expected = {'contract': CONTRACT, 'method': METHOD, 'sample_id': sample['sample_id'],
                'source_start': sample['start'], 'raw_sha256': raw['sha256'],
                'selected_state_sha256': raw['selected_state_sha256'],
                'recorded_prefix_physics_steps': sample['start'],
                'prefix_actions_sha256': prefix_sha256,
                'runtime_history_snapshot_loaded_before_policy': True,
                'future_actions_used_to_prepare_start': False,
                'manual_grasp_establishment_used': False,
                'exact_original_grasp_anchor_bytes_verified': False}
    if any(proof.get(k) != v or type(proof.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError('History result differs from its original source or action prefix')
    for key in ('recorded_prefix_physics_steps', 'prefix_actions_sha256',
                'future_actions_used_to_prepare_start', 'manual_grasp_establishment_used',
                'exact_original_grasp_anchor_bytes_verified'):
        if restoration.get(key) != proof[key]:
            raise ValueError('History restoration and source receipts disagree')
    if restoration.get('grasp_recovery_protocol') != CONTRACT or restoration.get('grasp_recovery_method') != METHOD:
        raise ValueError('History restoration lacks the approved protocol')
    for key, width in [('full_robot_joint_positions', 28), ('root_position', 3), ('root_orientation', 4)]:
        values = np.asarray(proof.get(key, []))
        if values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError('History snapshot has invalid robot fields')
    grasps = proof.get('recovered_grasps')
    if not isinstance(grasps, list) or len(grasps) > 2 or len({g.get('arm_index') for g in grasps}) != len(grasps):
        raise ValueError('History receipt has invalid recovered grasp records')
    for grasp in grasps:
        if grasp.get('arm_index') not in (0, 1) or not all(grasp.get(k) for k in ('target_object', 'target_link_name', 'joint_type')):
            raise ValueError('History receipt has incomplete recovered object identity')
        for key, width in [('parent_frame_pos', 3), ('parent_frame_orn', 4), ('child_frame_pos', 3), ('child_frame_orn', 4)]:
            values = np.asarray(grasp.get(key, []))
            if values.shape != (width,) or not np.isfinite(values).all():
                raise ValueError('History receipt has invalid recovered constraint fields')
    steps = result['steps']
    actions = result.get('model_action_audit')
    if (not isinstance(actions, list) or len(actions) != steps
            or result.get('actual_model_control_steps') != steps
            or [row.get('step') for row in actions] != list(range(steps))):
        raise ValueError('History model action audit is incomplete')
    for row in actions:
        for key, width in [('preceding_observation_state', 61), ('executed_action', 23)]:
            values = np.asarray(row.get(key, []))
            if values.shape != (width,) or not np.isfinite(values).all():
                raise ValueError('History rollout contains invalid policy observations or controls')
    binding = result['setup_evidence'].get('source_role_binding')
    if result['metric_eligible'] or binding is not None:
        expected_binding = {'contract': BINDING, 'raw_sha256': raw['sha256'],
                            'source_state_sha256': raw['selected_state_sha256'],
                            'sample_id': sample['sample_id'], 'grasp_recovery_protocol': CONTRACT,
                            'runtime_native_fields_verified': True, 'desired_predicate_truth_used': False,
                            'exact_original_grasp_anchor_bytes_verified': False}
        if not isinstance(binding, dict) or any(binding.get(k) != v for k, v in expected_binding.items()):
            raise ValueError('History rollout lacks its disclosed source role binding')
    return proof


def audit(config):
    code, root, parent = (Path(config[k]) for k in ('code_root', 'run_root', 'original_run_root'))
    ready = (code / '_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(ready).hexdigest() == config['source_manifest_sha256']
    for name, digest in json.loads(ready)['overlay_files_sha256'].items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest
    sys.path[:0] = [str(code / 'sco'), str(code / 'scripts')]
    import av
    import h5py
    import full100_shards as shards
    import eval_b1k_subtask_samples as runner
    import legacy_history_recovery as history
    from recorded_fixture_state import METHOD as FIXTURE_METHOD, validate_fixture_result_binding
    from summarize_subtask_closed_loop import _group_summary

    assert history.CONTRACT == CONTRACT and history.METHOD == METHOD and history.BINDING == BINDING
    assert config['indices'] and len(set(config['indices'])) == len(config['indices'])
    assignments, subtasks = [], []
    for index in config['indices']:
        panel, assignment, manifest, queue = shards.load_assignment(root, index)
        old_panel, original, _, _ = shards.load_assignment(parent, index)
        assert panel == old_panel and assignment == original
        assert panel['fingerprint'] == config['panel_fingerprint'] and len(assignment['episode_keys']) == 4
        assert manifest['sample_max_steps'] == 2048 and manifest['max_attempts'] == 3
        assert manifest['grasp_recovery_protocol'] == CONTRACT
        assert manifest['child_environment']['B1K_LEGACY_HISTORY_RECOVERY'] == CONTRACT
        jobs = queue.snapshot(readonly=True)['jobs']
        assert all(j['state'] == 'succeeded' and j['max_attempts'] == 3 for j in jobs)
        shard = root / 'shards' / str(index)
        execution = json.loads((shard / 'source_executions' / (config['execution_names'][str(index)] + '.json')).read_text())
        assert execution['code_root'] == str(code) and execution['source_manifest_sha256'] == config['source_manifest_sha256']
        marker = json.loads((shard / '_ASSIGNED_SUCCESS').read_text())
        stored = json.loads((shard / 'result_manifest.json').read_text())
        assert marker['assignment_index'] == index and marker['complete_full100'] is False
        assert marker['sample_ids'] == assignment['sample_ids'] and marker['panel_fingerprint'] == panel['fingerprint']
        assert marker['result_manifest_sha256'] == shards.runner.dynamic._fingerprint(stored)
        by_id = {s['sample_id']: s for s in panel['samples'] if s['sample_id'] in assignment['sample_ids']}
        values, seen = [], []
        for job in manifest['jobs']:
            if [job['task'], job['instance_id']] not in assignment['episode_keys']:
                continue
            paths = queue.validate_artifacts(Path(manifest['results_root']) / 'pi05' / job['task'],
                task=job['task'], instance_id=job['instance_id'], rollout_specs=job['rollout_specs'])
            samples = [s for s in by_id.values() if s['task_instance_id'] == job['instance_id']]
            cache = runner.load_episode_cache(runner.raw_hdf5_path(config['raw_root'], samples[0]), samples,
                                             allow_unverified_grasp_state=True)
            assert cache['grasp_serialization_audit']['legacy_missing_assisted_grasp_state']
            episode_results = [(path, json.loads(path.read_text())) for path, relative in paths if relative.parts[0] == 'json']
            with h5py.File(cache['path'], 'r', locking=False) as file:
                actions = np.asarray(file['data'][cache['details']['demo_group']]['action'][:max(s['start'] for s in samples)], dtype=np.float32)
            assert actions.shape == (max(s['start'] for s in samples), 23) and np.isfinite(actions).all()
            cache['history_identity_grasps'] = {}
            for _, result in episode_results:
                proof = result['restoration']['history_recovery_source']
                if proof['source_start'] in cache['history_identity_grasps']:
                    assert cache['history_identity_grasps'][proof['source_start']] == proof['recovered_grasps']
                cache['history_identity_grasps'][proof['source_start']] = proof['recovered_grasps']
                binding = result.get('setup_evidence', {}).get('source_role_binding') or {}
                for entry in binding.get('bindings', []):
                    event = entry.get('source_demonstration_event')
                    if event:
                        frame = event['event_frame']
                        assert event['recovery_protocol'] == CONTRACT and event['raw_sha256'] == cache['sha256']
                        assert frame == event['event_annotation_end'] - 1 and 0 <= frame < len(cache['states'])
                        row = np.asarray(cache['states'][frame, :int(cache['state_sizes'][frame])], dtype=np.float32)
                        assert hashlib.sha256(row.tobytes()).hexdigest() == event['event_selected_state_sha256']
                        event_grasps = event['event_history_recovered_grasps']
                        if frame in cache['history_identity_grasps']:
                            assert cache['history_identity_grasps'][frame] == event_grasps
                        cache['history_identity_grasps'][frame] = event_grasps
            for path, result in episode_results:
                sample = by_id[result['sample_id']]
                assert sample['task_instance_id'] == job['instance_id']
                raw = {**cache['sample_evidence'][sample['sample_id']], 'sha256': cache['sha256']}
                digest = hashlib.sha256(actions[:sample['start']].tobytes()).hexdigest()
                proof = validate_history_result(result, sample, panel['fingerprint'], raw, digest)
                evidence = result['setup_evidence']
                binding = evidence.get('source_role_binding')
                if binding:
                    expected = history.source_role_bindings(cache, proof, sample['predicate'])
                    assert all(binding.get(k) == v for k, v in expected.items())
                    if any(e['method'] == FIXTURE_METHOD for e in binding['bindings']):
                        validate_fixture_result_binding(binding, sample, proof)
                if result['setup_exclusion_reason'] == 'predicate_unobservable':
                    assert evidence.get('error_type') == 'ValueError'
                    assert evidence.get('error') in {'Source annotation role absent or ambiguous',
                        'Recorded task instance mapping absent', 'Recorded task instance mapping invalid'}
                    try:
                        history.source_role_bindings(cache, proof, sample['predicate'])
                    except ValueError as error:
                        assert str(error) == evidence['error']
                    else:
                        raise AssertionError('An excluded role resolves from the retained recovery receipts')
                if result['metric_eligible']:
                    video = path.parent.parent / 'videos' / (path.stem + '.mp4')
                    assert runner.file_sha256(video) == result['video_sha256']
                    with av.open(str(video)) as stream:
                        frame = next(stream.decode(video=0)).to_ndarray(format='rgb24')
                    assert frame.ndim == 3 and frame.shape[2] == 3 and float(frame.std()) > 1
                seen.append(sample['sample_id']); values.append(result)
        assert len(seen) == len(set(seen)) and set(seen) == set(assignment['sample_ids'])
        eligible = [v for v in values if v['metric_eligible']]
        assignments.append({'index': index, 'tasks': assignment['tasks'], 'sample_ids': assignment['sample_ids'],
            'episode_keys': assignment['episode_keys'], 'expected_samples': len(seen), 'audited_samples': len(values),
            'producing_run_root': str(root), 'producing_execution': execution['execution_name'],
            'producing_code_root': str(code), 'producing_source_manifest_sha256': config['source_manifest_sha256'],
            'original_scope_and_sample_ids_matched': True, 'state_recovery_protocol': CONTRACT,
            'independent_original_raw_and_prefix_hashes_matched': True,
            'recovery_receipts_and_source_role_bindings_checked': True,
            'all_history_setup_exclusions_checked_against_saved_evidence': True,
            'all_required_video_sha_and_first_frames_passed': True,
            'independent_all_raw_native_fields_matched': False,
            'exact_original_grasp_anchor_bytes_verified': False,
            'additional_grasp_physics_replayed_for_audit': False,
            'eligible': len(eligible), 'successes': sum(v['success'] for v in eligible),
            'setup_exclusions': dict(collections.Counter(v['setup_exclusion_reason'] for v in values if not v['metric_eligible'])),
            'complete_full100': False})
        result_by_sample = {v['sample_id']: v for v in values}
        for task in assignment['tasks']:
            scheduled = [s for s in by_id.values() if s['task'] == task]
            excluded = [s for s in panel['exclusions'] if s['task'] == task]
            for prompt in sorted({s['subtask'] for s in scheduled + excluded}):
                counts = _group_summary([s for s in scheduled if s['subtask'] == prompt],
                    [s for s in excluded if s['subtask'] == prompt], result_by_sample)
                subtasks.append({'task': task, 'subtask': prompt, 'assignment_index': index, **counts})
    return {'kind': 'completed_task_subtask_metrics_partial', 'complete_full100': False,
            'observed_at': time.time(), 'assignments': assignments, 'subtasks': subtasks,
            'panel_fingerprint': config['panel_fingerprint'], 'producing_run_root': str(root),
            'state_recovery_protocol': CONTRACT, 'exact_original_grasp_anchor_bytes_verified': False}
