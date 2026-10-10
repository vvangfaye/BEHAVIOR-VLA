"""Independently audit completed assignments using their producing source.

The caller verifies the platform terminal state before running this read-only
audit. Corrective runs must retain the original panel and assignment identities.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys
import time


def audit(config):
    code, root, parent = (Path(config[k]) for k in
                          ('code_root', 'run_root', 'original_run_root'))
    ready = (code / '_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(ready).hexdigest() == config['source_manifest_sha256']
    for name, digest in json.loads(ready)['overlay_files_sha256'].items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest
    sys.path[:0] = [str(code / 'sco'), str(code / 'scripts')]
    import av
    import full100_shards as shards
    import eval_b1k_subtask_samples as runner
    from recorded_native_state import (decode_start, validate_baseline_result,
                                       preview_source_role_bindings)
    from summarize_subtask_closed_loop import _group_summary

    assignments, subtasks = [], []
    indices = config['indices']
    assert indices and len(indices) == len(set(indices))
    for index in indices:
        panel, assignment, manifest, queue = shards.load_assignment(root, index)
        original_panel, original, _, _ = shards.load_assignment(parent, index)
        assert panel == original_panel
        assert panel['fingerprint'] == config['panel_fingerprint']
        assert assignment == original
        assert len(assignment['episode_keys']) == 4
        assert manifest['sample_max_steps'] == 2048 and manifest['max_attempts'] == 3
        jobs = queue.snapshot(readonly=True)['jobs']
        assert all(j['state'] == 'succeeded' and j['max_attempts'] == 3 for j in jobs)
        execution = json.loads((root / 'shards' / str(index) / 'source_executions' /
                                (config['execution_names'][str(index)] + '.json')).read_text())
        assert execution['code_root'] == str(code)
        assert execution['source_manifest_sha256'] == config['source_manifest_sha256']
        marker = json.loads((root / 'shards' / str(index) / '_ASSIGNED_SUCCESS').read_text())
        stored = json.loads((root / 'shards' / str(index) / 'result_manifest.json').read_text())
        assert marker['assignment_index'] == index and marker['complete_full100'] is False
        assert marker['sample_ids'] == assignment['sample_ids']
        assert marker['panel_fingerprint'] == panel['fingerprint']
        assert marker['result_manifest_sha256'] == shards.runner.dynamic._fingerprint(stored)
        by_id = {s['sample_id']: s for s in panel['samples']
                 if s['sample_id'] in assignment['sample_ids']}
        seen, values = [], []
        for job in manifest['jobs']:
            if [job['task'], job['instance_id']] not in assignment['episode_keys']:
                continue
            paths = queue.validate_artifacts(Path(manifest['results_root']) / 'pi05' / job['task'],
                task=job['task'], instance_id=job['instance_id'], rollout_specs=job['rollout_specs'])
            samples = [s for s in by_id.values() if s['task'] == job['task']
                       and s['task_instance_id'] == job['instance_id']]
            cache = runner.load_episode_cache(runner.raw_hdf5_path(config['raw_root'], samples[0]), samples)
            for path, relative in paths:
                if relative.parts[0] != 'json':
                    continue
                result = json.loads(path.read_text())
                sample = by_id[result['sample_id']]
                assert sample['task_instance_id'] == job['instance_id']
                seen.append(sample['sample_id'])
                decoded = decode_start(cache, sample)
                saved = result['restoration']['recorded_native_source']
                assert all(saved.get(k) == v for k, v in decoded.items())
                validate_baseline_result(result, sample, {'sha256': cache['sha256'],
                    'selected_state_sha256': decoded['selected_state_sha256']}, 2048)
                evidence = result.get('setup_evidence', {})
                binding = evidence.get('source_role_binding')
                if binding:
                    expected = preview_source_role_bindings(cache, decoded, sample['predicate'])
                    assert binding['runtime_native_fields_verified'] is True
                    assert all(binding.get(k) == v for k, v in expected.items()
                               if k != 'runtime_native_fields_verified')
                if result.get('setup_exclusion_reason') == 'predicate_unobservable':
                    try:
                        preview_source_role_bindings(cache, decoded, sample['predicate'])
                    except ValueError as error:
                        assert evidence.get('error_type') == 'ValueError'
                        assert str(error) == evidence.get('error')
                        assert str(error) in {'Source annotation role absent or ambiguous',
                            'Recorded task instance mapping absent', 'Recorded task instance mapping invalid'}
                    else:
                        raise AssertionError('Excluded role resolves from its original raw source')
                if result['metric_eligible']:
                    video = path.parent.parent / 'videos' / (path.stem + '.mp4')
                    assert runner.file_sha256(video) == result['video_sha256']
                    with av.open(str(video)) as stream:
                        frame = next(stream.decode(video=0)).to_ndarray(format='rgb24')
                    assert frame.ndim == 3 and frame.shape[2] == 3 and float(frame.std()) > 1
                values.append(result)
        assert len(seen) == len(set(seen)) and set(seen) == set(assignment['sample_ids'])
        eligible = [r for r in values if r['metric_eligible'] is True]
        assignments.append({'index': index, 'tasks': assignment['tasks'],
            'sample_ids': assignment['sample_ids'], 'episode_keys': assignment['episode_keys'],
            'producing_run_root': str(root), 'producing_execution': execution['execution_name'],
            'producing_code_root': str(code), 'producing_source_manifest_sha256': config['source_manifest_sha256'],
            'original_scope_and_sample_ids_matched': True,
            'expected_samples': len(assignment['sample_ids']), 'audited_samples': len(values),
            'independent_all_raw_native_fields_matched': True,
            'provided_source_role_bindings_independently_matched': True,
            'all_role_unobservable_exclusions_rechecked_against_original_raw': True,
            'all_required_video_sha_and_first_frames_passed': True,
            'eligible': len(eligible), 'successes': sum(r['success'] is True for r in eligible),
            'setup_exclusions': dict(collections.Counter(r['setup_exclusion_reason']
                for r in values if r['metric_eligible'] is False)), 'complete_full100': False})
        result_by_sample = {r['sample_id']: r for r in values}
        for task in assignment['tasks']:
            scheduled = [s for s in by_id.values() if s['task'] == task]
            excluded = [s for s in panel['exclusions'] if s['task'] == task]
            for prompt in sorted({s['subtask'] for s in scheduled + excluded}):
                counts = _group_summary([s for s in scheduled if s['subtask'] == prompt],
                    [s for s in excluded if s['subtask'] == prompt], result_by_sample)
                subtasks.append({'task': task, 'subtask': prompt, 'assignment_index': index, **counts})
    return {'kind': 'completed_task_subtask_metrics_partial', 'complete_full100': False,
        'observed_at': time.time(), 'assignments': assignments, 'subtasks': subtasks,
        'panel_fingerprint': config['panel_fingerprint'], 'producing_run_root': str(root)}
