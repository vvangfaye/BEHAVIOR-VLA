"""Truthful source-fidelity exclusions, distinct from model/setup failures.

No simulator state, predicate value, success measurement or model control is
invented when a recording lacks its assisted-grasp serialization channel.
"""
from __future__ import annotations
import hashlib
import re

CONTRACT = 'missing_recorded_assisted_grasp_state_v1'
STATE = 'source_excluded'
REASON = 'legacy_missing_assisted_grasp_state'
IDENTITY_FIELDS = {'task':'task','instance_id':'task_instance_id','rollout_id':'rollout_id',
 'sample_id':'sample_id','episode_index':'episode_index','raw_episode_id':'raw_episode_id',
 'source_segment_index':'source_segment_index','source_interval_index':'source_interval_index',
 'segment_start':'start','segment_end':'end','segment_frames':'segment_frames',
 'subtask':'subtask','predicate':'predicate'}


def validate_source_exclusion(result, sample=None, fingerprint=None, raw_provenance=None):
    if not isinstance(result, dict) or result.get('result_state') != STATE:
        raise ValueError('Expected an explicit source-excluded result')
    if (result.get('metric_eligible') is not False or result.get('success') is not None or
            type(result.get('steps')) is not int or result['steps'] != 0 or
            result.get('source_exclusion_reason') != REASON or
            result.get('failure_reason') is not None or result.get('setup_exclusion_reason') is not None or
            result.get('q_score') is not None or result.get('task_success') is not None or
            result.get('terminated') is not None or result.get('truncated') is not None or
            result.get('terminal_credit') is not False):
        raise ValueError('An unrestored source cannot claim model success/failure, metrics or steps')
    for key in ('observed_subtasks','observed_subtask_sources','subtask_events'):
        if result.get(key) != []:
            raise ValueError('An excluded source cannot claim policy responses')
    if result.get('monitor') is not None or result.get('restoration') != {'performed':False,'full_snapshot':False}:
        raise ValueError('An excluded source cannot claim a completion monitor or restored snapshot')
    conditioning = result.get('conditioning', {})
    if (conditioning.get('source') != 'not_executed' or conditioning.get('policy_rollout_started') is not False or
            type(conditioning.get('model_action_requests')) is not int or conditioning['model_action_requests'] != 0 or
            conditioning.get('subtask') != result.get('subtask') or
            conditioning.get('official_benchmark') is not False or conditioning.get('training_set_diagnostic') is not True):
        raise ValueError('Source exclusion has inconsistent non-executed conditioning')
    for name in ('instance_id','rollout_id','episode_index','raw_episode_id','source_segment_index',
                 'source_interval_index','segment_start','segment_end','segment_frames'):
        if type(result.get(name)) is not int or result[name] < 0:
            raise ValueError('Source exclusion has invalid numeric identity: '+name)
    if result['segment_end'] <= result['segment_start'] or result['segment_frames'] != result['segment_end'] - result['segment_start']:
        raise ValueError('Source exclusion has invalid source interval')
    if any(not isinstance(result.get(name), str) or not result[name] for name in ('task','sample_id','subtask')):
        raise ValueError('Source exclusion has invalid text identity')
    evidence = result.get('source_evidence', {})
    if (evidence.get('contract') != CONTRACT or evidence.get('sample_id') != result.get('sample_id') or
            evidence.get('selected_frame') != result.get('segment_start') or
            evidence.get('sample_evaluator_created') is not False or evidence.get('sample_restore_performed') is not False):
        raise ValueError('Source exclusion lacks original source identity and non-restoration evidence')
    for name in ('raw_sha256','selected_state_sha256'):
        if not isinstance(evidence.get(name), str) or not re.fullmatch('[0-9a-f]{64}', evidence[name]):
            raise ValueError('Source exclusion lacks frozen raw-file/selected-row hashes')
    size=evidence.get('selected_state_size')
    if type(size) is not int or size <= 0:
        raise ValueError('Source exclusion has invalid selected row size')
    audit=evidence.get('grasp_serialization_audit', {})
    frames=audit.get('inspected_prefix_frames');robots=audit.get('robots')
    if (audit.get('contract') != 'recorded_grasp_serialization_v1' or
            audit.get('legacy_missing_assisted_grasp_state') is not True or
            audit.get('runtime_grasp_restoration_verified') is not False or
            type(audit.get('frames_with_ag_magic')) is not int or audit['frames_with_ag_magic'] != 0 or
            type(frames) is not int or frames < result['segment_start'] + 1 or
            not isinstance(robots, list) or not robots or not all(isinstance(r, dict) for r in robots) or
            not any(r.get('grasping_mode') in ('assisted','sticky') and r.get('legacy_controller_layout') is True and
                    r.get('legacy_missing_assisted_grasp_state') is True for r in robots)):
        raise ValueError('Source exclusion does not prove a legacy missing grasp channel')
    if sample is not None:
        for field, source in IDENTITY_FIELDS.items():
            if result.get(field) != sample[source]:
                raise ValueError('Source exclusion changed frozen sample identity: '+field)
    if fingerprint is not None and result.get('panel_fingerprint') != fingerprint:
        raise ValueError('Source exclusion changed parent panel identity')
    if raw_provenance is not None:
        if (evidence['raw_sha256'] != raw_provenance['sha256'] or size != raw_provenance['selected_state_size']
                or evidence['selected_state_sha256'] != raw_provenance.get('selected_state_sha256')):
            raise ValueError('Source exclusion disagrees with independent raw preflight')


def make_source_exclusion(cache, sample, fingerprint):
    import numpy as np
    audit=cache['grasp_serialization_audit']
    if audit.get('legacy_missing_assisted_grasp_state') is not True:
        raise ValueError('Native/observable recordings cannot be excluded as missing legacy grasps')
    start=sample['start'];size=int(cache['state_sizes'][start])
    row=np.asarray(cache['states'][start,:size])
    result={field:sample[source] for field,source in IDENTITY_FIELDS.items()}
    result.update(panel_fingerprint=fingerprint,result_state=STATE,metric_eligible=False,success=None,
     steps=0,source_exclusion_reason=REASON,setup_exclusion_reason=None,failure_reason=None,q_score=None,
     terminated=None,truncated=None,task_success=None,terminal_credit=False,
     observed_subtasks=[],observed_subtask_sources=[],subtask_events=[],monitor=None,
     restoration={'performed':False,'full_snapshot':False},
     conditioning={'source':'not_executed','subtask':sample['subtask'],'policy_rollout_started':False,
                   'model_action_requests':0,'official_benchmark':False,'training_set_diagnostic':True},
     source_evidence={'contract':CONTRACT,'sample_id':sample['sample_id'],'selected_frame':start,
                      'selected_state_size':size,'raw_sha256':cache['sha256'],
                      'selected_state_sha256':hashlib.sha256(row.tobytes()).hexdigest(),
                      'grasp_serialization_audit':audit,'sample_evaluator_created':False,'sample_restore_performed':False})
    validate_source_exclusion(result,sample,fingerprint)
    return result


def prepublish_source_exclusions(manifest, queue, panel):
    """Exclude proven legacy sources on CPU, before any GPU worker starts."""
    import json
    from pathlib import Path
    import h5py
    import eval_b1k_subtask_samples as runner
    from pi05_eval_queue import atomic_write_json
    from summarize_subtask_closed_loop import _manifest_provenance

    provenance = _manifest_provenance(manifest, panel, manifest['checkpoint'])
    published = []
    for job in manifest['jobs']:
        episode, samples = runner.resolve_episode_samples(panel, job['task'], job['instance_id'])
        path = runner.raw_hdf5_path(manifest['raw_data_root'], samples[0])
        # Metadata inspection avoids loading a modern recording's entire prefix.
        with h5py.File(path, 'r', locking=False) as f:
            config = json.loads(f['data'].attrs['config'])
            scene = json.loads(f['data'].attrs['scene_file'])
        registry = scene.get('state', {}).get('registry', {}).get('object_registry', {})
        robots = config.get('robots', [])
        candidate = any(r.get('grasping_mode') in ('assisted', 'sticky')
                        and 'controllers' in registry.get(r.get('name'), {})
                        and 'controller_groups' not in registry.get(r.get('name'), {}) for r in robots)
        if not candidate:
            continue
        cache = runner.load_episode_cache(path, samples, allow_unverified_grasp_state=True)
        if cache['grasp_serialization_audit']['legacy_missing_assisted_grasp_state'] is not True:
            continue
        records = [make_source_exclusion(cache, sample, panel['fingerprint']) for sample in samples]
        for record, sample in zip(records, samples, strict=True):
            validate_source_exclusion(record, sample, panel['fingerprint'], provenance[sample['sample_id']])
        lease = queue.claim(job['task'], manifest['run_id'] + ':source-preflight:generation-0',
                            manifest['attempt_root'], instance_id=job['instance_id'])
        if lease is None:
            raise ValueError('CPU source exclusion cannot claim its exact pending episode')
        try:
            for sample, record in zip(samples, records, strict=True):
                atomic_write_json(lease.json_dir / (runner._artifact_stem(sample) + '.json'), record)
            queue.publish(lease, Path(manifest['results_root']) / 'pi05' / job['task'],
                rollout_specs=runner.episode_rollout_specs(samples), provenance={
                    'run_id': manifest['run_id'], 'generation':0, 'panel_fingerprint':panel['fingerprint'],
                    'episode_index':episode['episode_index'], 'raw_episode_id':episode['raw_episode_id'],
                    'sample_ids':[sample['sample_id'] for sample in samples],
                })
        except BaseException as error:
            queue.fail(lease, f'{type(error).__name__}: {error}')
            raise
        published.append({'task':job['task'], 'instance_id':job['instance_id'],
                          'raw_sha256':cache['sha256'], 'sample_ids':[s['sample_id'] for s in samples]})
        del cache
    report = {'contract':CONTRACT, 'panel_fingerprint':panel['fingerprint'],
              'source_excluded_episode_jobs':len(published),
              'source_excluded_samples':sum(len(p['sample_ids']) for p in published),
              'sample_evaluators_created':0, 'sample_model_controls':0, 'episodes':published}
    atomic_write_json(Path(manifest['queue_db']).parent / 'source_exclusion_preflight.json', report)
    return report
