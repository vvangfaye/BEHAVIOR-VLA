"""Independently bind all requested low-rate groups to original baseline/source bytes."""
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run(payload):
    import av
    import h5py
    import numpy as np
    code, output = Path(payload['code_root']), Path(payload['output_root'])
    output.mkdir(parents=True, exist_ok=False)
    source = (code/'_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(source).hexdigest() == payload['source_manifest_sha256']
    sys.path[:0] = [str(code/'scripts'), str(code/'sco')]
    import eval_b1k_subtask_samples as runner
    from subtask_sample_panel import load_panel
    from full_subtask_collection import targets, choose_starts, freeze, NATIVE, HISTORY
    from recorded_native_state import decode_start, validate_baseline_result
    from audit_history_assignment_artifacts import validate_history_result
    from artifact_acceptance import accepted_evidence_kind
    panel = load_panel(payload['panel_path'])
    metrics, acceptance = payload['metrics'], payload['acceptance']
    groups = targets(metrics, acceptance)
    assert len(groups) == payload['expected_groups'] == 299
    assert panel['fingerprint'] == acceptance['panel_fingerprint']
    (output/'parent_panel.json').write_text(json.dumps(panel, indent=2)+'\n')
    assignments = {a['index']: a for a in metrics['assignments']}
    required = {(g['task'], g['subtask']) for g in groups}
    task_rows = {a['tasks'][0]: a for a in metrics['assignments']}
    proofs, values, audit_rows = {}, {}, []
    needed = [s for s in panel['samples'] if (s['task'], s['subtask']) in required]
    for episode_index in sorted({s['episode_index'] for s in needed}):
        samples = [s for s in needed if s['episode_index'] == episode_index]
        a = task_rows[samples[0]['task']]
        kind = accepted_evidence_kind(a)
        root = Path(a.get('producing_run_root') or payload['supplementary_run_roots'][str(a['index'])])
        manifest = json.loads((root/'manifest.json').read_text())
        assert manifest['sample_max_steps'] == 2048
        result_root = Path(manifest['results_root'])/'pi05'/samples[0]['task']
        cache = runner.load_episode_cache(runner.raw_hdf5_path(payload['raw_data_root'], samples[0]),
            samples, allow_unverified_grasp_state=kind == 'approved_physical_history')
        history = cache['grasp_serialization_audit']['legacy_missing_assisted_grasp_state']
        assert history == (kind == 'approved_physical_history')
        actions = None
        if history:
            with h5py.File(cache['path'], 'r', locking=False) as stream:
                actions = np.asarray(stream['data'][cache['details']['demo_group']]['action'][:max(s['start'] for s in samples)], dtype=np.float32)
            assert actions.shape == (max(s['start'] for s in samples), 23) and np.isfinite(actions).all()
        for sample in samples:
            sid = sample['sample_id']
            path = result_root/'json'/(runner._artifact_stem(sample)+'.json')
            result = json.loads(path.read_text())
            raw = dict(cache['sample_evidence'][sid], sha256=cache['sha256'])
            if history:
                prefix = hashlib.sha256(actions[:sample['start']].tobytes()).hexdigest()
                validate_history_result(result, sample, panel['fingerprint'], raw, prefix)
                protocol = HISTORY
            else:
                decoded = decode_start(cache, sample)
                assert all(result['restoration']['recorded_native_source'].get(k) == v for k, v in decoded.items())
                validate_baseline_result(result, sample, raw, 2048)
                prefix, protocol = None, NATIVE
            video_sha = None
            if result['metric_eligible']:
                video = result_root/'videos'/(runner._artifact_stem(sample)+'.mp4')
                video_sha = sha(video)
                assert video_sha == result['video_sha256']
                with av.open(str(video)) as stream:
                    frame = next(stream.decode(video=0)).to_ndarray(format='rgb24')
                assert frame.ndim == 3 and frame.shape[2] == 3 and float(frame.std()) > 1
            proof = dict(raw_sha256=cache['sha256'], selected_state_sha256=raw['selected_state_sha256'],
                baseline_result_sha256=sha(path), baseline_metric_eligible=result['metric_eligible'],
                state_protocol=protocol, prefix_actions_sha256=prefix, raw_sample_evidence=raw)
            proofs[sid], values[sid] = proof, result
            audit_rows.append(dict(sample_id=sid, original_result_path=str(path), video_sha256=video_sha,
                assignment_index=a['index'], original_accepted_assignment_sha256=hashlib.sha256(
                    json.dumps(a, sort_keys=True).encode()).hexdigest(), **proof))
        print('SOURCE_AUDIT_PROGRESS='+json.dumps(dict(episodes=len({s['episode_index'] for s in needed if s['sample_id'] in proofs}),
            samples=len(proofs), total_samples=len(needed))), flush=True)
    for group in groups:
        rows = [values[s['sample_id']] for s in needed if (s['task'], s['subtask']) == (group['task'], group['subtask'])]
        assert sum(r['metric_eligible'] for r in rows) == group['eligible']
        assert sum(r['metric_eligible'] and r['success'] for r in rows) == group['successes']
    additional_panel = None
    additional_proofs = {}
    if payload.get('additional_panel_path'):
        from recorded_native_state import preview_source_role_bindings
        additional_panel = load_panel(payload['additional_panel_path'])
        assert additional_panel['baseline_panel_fingerprint'] == panel['fingerprint']
        (output/'additional_panel.json').write_text(json.dumps(additional_panel,indent=2)+'\n')
        for sample in additional_panel['samples']:
            cache=runner.load_episode_cache(runner.raw_hdf5_path(payload['additional_raw_root'],sample),[sample])
            decoded=decode_start(cache,sample)
            source_entry=additional_panel['source_evidence'][sample['sample_id']]
            assert decoded['raw_sha256']==source_entry['raw_sha256']
            roles=preview_source_role_bindings(cache,decoded,sample['predicate'])
            evidence_entry=dict(source_evidence_kind='additional_training_start_preflight',
                sample_id=sample['sample_id'],raw_sha256=cache['sha256'],
                selected_state_sha256=decoded['selected_state_sha256'],state_protocol=NATIVE,
                literal_source_role_binding=roles,public_training_source=source_entry,
                no_measured_baseline_for_this_start=True)
            data=(json.dumps(evidence_entry,sort_keys=True)+'\n').encode()
            preflight=output/'additional_start_evidence'/ (sample['sample_id']+'.json')
            preflight.parent.mkdir(exist_ok=True);preflight.write_bytes(data)
            additional_proofs[sample['sample_id']]=dict(evidence_entry,
                source_preflight_sha256=hashlib.sha256(data).hexdigest())
            audit_rows.append(dict(evidence_entry,source_preflight_sha256=hashlib.sha256(data).hexdigest()))
    evidence = dict(kind='all_low_rate_original_source_and_baseline_audit_v2', complete=True,
        panel_fingerprint=panel['fingerprint'], source_manifest_sha256=payload['source_manifest_sha256'],
        accepted_input_sha256=payload['accepted_input_sha256'], group_count=len(groups),
        samples=sorted(audit_rows, key=lambda r:r['sample_id']), audited_at=time.time())
    evidence_path = output/'source_evidence.json'
    evidence_path.write_text(json.dumps(evidence, indent=2)+'\n')
    evidence_sha = sha(evidence_path)
    ready, missing = [], []
    for index, group in enumerate(groups):
        starts = choose_starts(panel, group, proofs, payload['held_out_instances'],
                               maximum=payload['starts_per_group'], seed=payload['seed'])
        source_panel,source_proofs,raw_root,panel_file=panel,proofs,payload['raw_data_root'],output/'parent_panel.json'
        if not starts and additional_panel:
            starts=[s for s in additional_panel['samples'] if (s['task'],s['subtask'])==(group['task'],group['subtask'])]
            source_panel,source_proofs,raw_root,panel_file=additional_panel,additional_proofs,payload['additional_raw_root'],output/'additional_panel.json'
        if not starts:
            missing.append(dict(index=index, group=group, reason='no_eligible_original_start_outside_holdouts'))
            continue
        plan = freeze(source_panel, group, starts, source_proofs, payload['checkpoint_identity'], payload['held_out_instances'],
            accepted_input_sha256=payload['accepted_input_sha256'], evidence_manifest_sha256=evidence_sha,
            repeats=payload['repeats'], max_steps=2048, seed=payload['seed']+index)
        directory = output/'groups'/('%03d'%index)
        directory.mkdir(parents=True)
        (directory/'parent_panel.json').symlink_to(panel_file)
        (directory/'collection_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        from subtask_improvement import load_plan
        load_plan(directory/'collection_plan.json', source_panel)
        entry = dict(index=index, state='complete', output_root=str(directory), code_root=str(code),
            source_manifest_sha256=payload['source_manifest_sha256'], raw_data_root=raw_root,
            plan=plan, group=group)
        (directory/'prepared.json').write_text(json.dumps(entry, indent=2)+'\n')
        ready.append(entry)
    report = dict(state='complete' if not missing else 'additional_training_starts_required',
        requested_groups=groups, requested_group_count=len(groups), prepared_group_count=len(ready),
        source_evidence_sha256=evidence_sha, source_manifest_sha256=payload['source_manifest_sha256'],
        code_root=str(code), output_root=str(output), groups=ready, missing_groups=missing,
        expected_trials=sum(r['plan']['expected_trials'] for r in ready), completed_at=time.time())
    (output/'campaign.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
