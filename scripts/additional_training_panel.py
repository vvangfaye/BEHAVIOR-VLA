"""A separate collection-only panel for additional public training demonstrations."""
import json
import re

from subtask_sample_panel import _canonical_hash, _validate_panel_segment


def validate(panel):
    if (panel.get('schema_version') != 3 or panel.get('kind') != 'training_subtask_closed_loop_panel'
            or panel.get('purpose') != 'additional_training_starts_v1'
            or panel.get('official_benchmark') is not False
            or not re.fullmatch('[0-9a-f]{64}',panel.get('baseline_panel_fingerprint',''))):
        raise ValueError('Unknown additional training-start panel')
    if panel['fingerprint'] != _canonical_hash({k:v for k,v in panel.items() if k!='fingerprint'}):
        raise ValueError('Additional training panel fingerprint mismatch')
    episodes={e['episode_index']:e for e in panel['episodes']}
    if len(episodes)!=len(panel['episodes']) or not episodes:
        raise ValueError('Invalid additional training episodes')
    held={tuple(x) for x in panel['held_out_instances']}
    seen=set()
    for sample in panel['samples']:
        sid,_,_=_validate_panel_segment(sample,episodes,exclusion=False)
        if sid in seen or (sample['task'],sample['task_instance_id']) in held:
            raise ValueError('Additional source duplicates a sample or uses a heldout instance')
        seen.add(sid)
        source=panel['source_evidence'][sid]
        if source['training_episode_metadata']['episode_index']!=sample['episode_index']:
            raise ValueError('Additional start is absent from the public training episode index')
        metadata=source['training_episode_metadata']
        for field in ('task_index','raw_episode_id','task_instance_id'):
            if metadata[field]!=sample[field]:raise ValueError('Additional training episode identity differs')
        if metadata['tasks']!=[sample['task']] or metadata['length']!=sample['episode_length']:
            raise ValueError('Additional training episode scope differs')
        if source['metadata_repo']!='behavior-1k/2026-challenge-demos' or source['raw_repo']!='behavior-1k/2026-challenge-rawdata':
            raise ValueError('Unrecognized additional training source')
        for field in ('annotation_sha256','metadata_sha256','raw_sha256'):
            if not re.fullmatch('[0-9a-f]{64}',source[field]):raise ValueError('Additional source hash missing')
        from gen_subtask_prompts import build_role_aware_segments, _subtask_text
        rows=build_role_aware_segments(source['annotation'],sample['episode_index'],sample['episode_length'],sample['annotation_path'])
        # The parser returns segments, uncovered count, and vocabulary.
        if isinstance(rows,tuple):rows=rows[0]
        match=[r for r in rows if r.get('provenance') == 'annotated' and r['source_segment_index']==sample['source_segment_index'] and
               r['source_interval_index']==sample['source_interval_index']]
        if len(match)!=1:raise ValueError('Additional literal annotation is ambiguous')
        row=match[0]
        # Preserve the exact label format measured by checkpoint 1375000.
        if panel.get('conditioning_label_format') != 'historical_subtask_text_v1':
            raise ValueError('Unknown additional conditioning label format')
        row['subtask']=_subtask_text(source['annotation']['skill_annotation'][sample['source_segment_index']])
        for field in ('start','end','subtask','skill_description','manipulating_object_ids','reference_object_ids','object_ids'):
            if sample[field]!=row[field]:raise ValueError('Additional start changed its literal annotation')
        body=json.dumps(source['annotation'],sort_keys=True,separators=(',',':')).encode()
        import hashlib
        if hashlib.sha256(body).hexdigest()!=source['annotation_canonical_sha256']:
            raise ValueError('Additional source annotation changed')
    if not seen:raise ValueError('Empty additional training panel')
    return panel
