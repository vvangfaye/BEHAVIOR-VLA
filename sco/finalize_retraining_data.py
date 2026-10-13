"""Audit four prepared sources; publish READY only after all data gates pass."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time

from prepare_retraining_data import Preparation, save, sha


def finalize(cache, output):
    prep = Preparation(cache, output)
    sources = prep.inputs['source_archives']
    receipts = [json.loads(p.read_text()) for p in (output/'receipts').glob('*.json')]
    receipts = {r['source_key']:r for r in receipts if r.get('accepted') and r.get('index_target')}
    missing = [r['id'] for r in sources if r['key'] not in receipts]
    for row in sources:
        if row['key'] in receipts:
            assert receipts[row['key']]['source_sha256'] == row['sha256']
    original = output/'original_stage/original_expert'
    transfer = json.loads((original/'transfer_receipt.json').read_text()) if (original/'transfer_receipt.json').exists() else {}
    expert = json.loads((original/'reader_validation.json').read_text()) if (original/'reader_validation.json').exists() else {}
    indexes = {r['root_id']:json.loads((output/'indexes'/(r['root_id']+'.json')).read_text()) for r in receipts.values()}
    seen, duplicates = {}, []
    reserved = set(map(tuple, prep.inputs['held_out_task_instances']))
    for root, descriptor in sorted(indexes.items()):
        # Preserved raw episodes can include excluded holdouts or short trials;
        # audit the actual training selections, not every archived raw record.
        selected=(descriptor.get('action_only_episodes') if 'action_only_episodes' in descriptor
                  else (descriptor.get('index') or {}).get('episodes',[]))
        for episode in selected:
            if episode.get('valid_anchors',1)<=0:continue
            assert (episode['task_name'],episode['task_instance_id']) not in reserved
            content = episode['content_sha256']
            if content in seen: duplicates.append({'first':seen[content],'duplicate':[root,episode['episode_index']]})
            else: seen[content] = [root,episode['episode_index']]
    counts = {}
    for kind in ('subtask','full_episode'):
        accepted = [r for r in receipts.values() if r['kind']==kind]
        counts[kind] = {'accepted_archives':len(accepted), 'expected_archives':sum(r['kind']==kind for r in sources),
            'accepted_episodes':sum(r['acceptance']['episodes'] for r in accepted),
            'accepted_frames':sum(r['acceptance']['frames'] for r in accepted),
            'training_eligible_episodes':sum(r['training_eligible_episodes'] for r in accepted)}
    ready = not missing and not duplicates and expert.get('accepted') is True and transfer.get('metadata_prepared_and_transferred') is True
    manifest = {'kind':'b1k_four_source_data_preparation', 'version':1, 'data_ready':ready,
        'training_allowed':False,'training_executed':False,'created_at':time.time(),
        'source_fingerprint':prep.inputs['fingerprint'],'held_out_task_instances':prep.inputs['held_out_task_instances'],
        'original_expert':transfer,'original_expert_readback':expert,
        'collected_data':counts,'archives':list(receipts.values()),'missing_archives':missing,
        'duplicate_collected_content':duplicates,
        'supervision':{'original_expert_replay':'original_human_subtask_and_action',
            'recovered_expert':'leading_first_label_imputed_with_provenance',
            'subtask':'verified_fixed_subtask_successes',
            'full_episode':'action_only; self_generated_subtask_text_must_not_receive_human_label_CE'},
        'sampling_proposal':{'original_expert_replay':0.50,'recovered_expert':0.25,'subtask':0.20,'full_episode':0.05},
        'sampling_is_not_training_configuration':True,
        'public_collected_instances':'training_seen; not independent evaluation',
        'gpu_training_smoke_test':'not_run_by_user_instruction'}
    if (output/'toolchain_transfer.json').exists():
        manifest['toolchain']=json.loads((output/'toolchain_transfer.json').read_text())
    for filename, field in [('cloud_build_submission.json', 'source_cloud_build'),
                            ('source_cloud_toolchain_manifest.json', 'source_cloud_toolchain'),
                            ('parallel_cloud_launch.json', 'source_parallel_build'),
                            ('parallel_toolchain_manifest.json', 'source_parallel_toolchain')]:
        if (output/filename).exists():
            manifest[field] = json.loads((output/filename).read_text())
    manifest['reader_requirement']={'pyav_timestamp_comparison':'float64','tolerance_s':0.0001,
        'patch_script':'sco/install_reader_timestamp_fix.py'}
    save(output/'data_preparation_manifest.json',manifest)
    if ready:
        target=prep.upload(output/'data_preparation_manifest.json',prep.prefix+'data_preparation_manifest.json')
        marker={'manifest':target,'data_ready':True,'training_allowed':False,'training_executed':False}
        save(output/'_DATA_READY.json',marker)
        prep.upload(output/'_DATA_READY.json',prep.prefix+'_DATA_READY.json')
    return {'data_ready':ready,'counts':counts,'missing_archives':len(missing),'duplicates':len(duplicates),
            'original_readback_accepted':expert.get('accepted',False),'training_executed':False}


if __name__=='__main__':print(json.dumps(finalize(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
