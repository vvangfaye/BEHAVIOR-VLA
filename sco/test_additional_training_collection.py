import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path[:0]=[str(Path(__file__).parents[1]/'scripts'),str(Path(__file__).parent)]
from subtask_sample_panel import _canonical_hash, _segment_record, _sample_id, infer_predicate
from gen_subtask_prompts import build_role_aware_segments, _subtask_text
from additional_training_panel import validate
from full_subtask_collection import freeze, validate as validate_plan, NATIVE


def panel():
    annotation={'skill_annotation':[dict(skill_idx=0,skill_description=['place on'],
        object_id=[['notebook','bookcase']],manipulating_object_id=['notebook'],frame_duration=[2,40])]}
    metadata=dict(episode_index=3,task_index=99,raw_episode_id=990170,task_instance_id=17,
                  length=50,tasks=['sorting_books_on_shelf'])
    path='annotations/task-0099/episode_00990170.json'
    row=build_role_aware_segments(annotation,3,50,path)[0][1]
    row['subtask']=_subtask_text(annotation['skill_annotation'][0]);row['completion_contract']='annotation_roles_v1'
    sample=_segment_record(row,metadata,'sorting_books_on_shelf')
    sample.update(sample_id=_sample_id(sample),predicate=infer_predicate(sample['task'],sample),rollout_id=0)
    episode={k:sample[k] for k in ['task','task_index','episode_index','raw_episode_id','task_instance_id','episode_length','annotation_path']}
    source=dict(training_episode_metadata=metadata,metadata_repo='behavior-1k/2026-challenge-demos',
        raw_repo='behavior-1k/2026-challenge-rawdata',annotation=annotation,annotation_sha256='a'*64,
        metadata_sha256='b'*64,raw_sha256='c'*64,
        annotation_canonical_sha256=hashlib.sha256(json.dumps(annotation,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    p=dict(schema_version=3,kind='training_subtask_closed_loop_panel',purpose='additional_training_starts_v1',
        conditioning_label_format='historical_subtask_text_v1',official_benchmark=False,baseline_panel_fingerprint='f'*64,
        samples=[sample],episodes=[episode],held_out_instances=[['sorting_books_on_shelf',47]],
        source_evidence={sample['sample_id']:source},exclusions=[])
    return dict(p,fingerprint=_canonical_hash(p))


def test_additional_start_uses_literal_training_annotation_and_excludes_holdout():
    p=panel();validate(p)
    p['held_out_instances'].append(['sorting_books_on_shelf',17])
    p['fingerprint']=_canonical_hash({k:v for k,v in p.items() if k!='fingerprint'})
    with pytest.raises(ValueError):validate(p)


@pytest.mark.parametrize('damage',['label','episode','annotation'])
def test_additional_source_changes_are_rejected(damage):
    p=panel();s=p['samples'][0];source=p['source_evidence'][s['sample_id']]
    if damage=='label':s['subtask']='place on hardback to bookcase'
    elif damage=='episode':source['training_episode_metadata']['task_instance_id']=47
    else:source['annotation']['skill_annotation'][0]['object_id']=[['notebook','floor']]
    p['fingerprint']=_canonical_hash({k:v for k,v in p.items() if k!='fingerprint'})
    with pytest.raises(ValueError):validate(p)


def test_additional_raw_preflight_is_not_mislabeled_as_measured_baseline():
    p=panel();s=p['samples'][0]
    proof=dict(source_evidence_kind='additional_training_start_preflight',source_preflight_sha256='1'*64,
        raw_sha256='c'*64,selected_state_sha256='2'*64,state_protocol=NATIVE)
    group=dict(task=s['task'],subtask=s['subtask'],eligible=1,successes=0,rate=0,underpowered=True)
    identity=dict(checkpoint_step=1375000,checkpoint_manifest_sha256='3'*64,checkpoint_source='/checkpoint')
    plan=freeze(p,group,[s],{s['sample_id']:proof},identity,p['held_out_instances'],
        accepted_input_sha256={'metrics':'4'*64},evidence_manifest_sha256='5'*64,repeats=3)
    assert plan['baseline_result_sha256']=={}
    assert plan['additional_start_source_evidence_sha256']=={s['sample_id']:'1'*64}
    assert plan['baseline_rates_panel_fingerprint']==p['baseline_panel_fingerprint']
    validate_plan(plan,{s['sample_id']:s})
