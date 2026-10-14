"""Count every archived episode in all retained rounds, independently of admission.

Includes published episode receipts in running/failed/stopped shards. No cloud
mutation, no new jobs, no training-ready claim. Whole-archive SHA and codec
readback are still required before a training export is accepted.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import time

from restore_full_episode_collection import (BUCKET, accepted_source, get_json,
    not_found, prefix_for, private_json, read_plan, require, validate_episode)


def bc_eligible(receipt):
    result=receipt['result']
    return receipt['success'] is True and result['terminated'] is True and result['truncated'] is False


def episode_identity(row):
    return (row['run_id'],row['sampling_round'],row['task'],row['instance_id'],row['rollout_id'])


def scan_round(root, round_id, client):
    root=Path(root);plan=read_plan(root);original_source=accepted_source(root,plan)
    require(original_source is not None,'Unaccepted round source')
    revision=None
    if (root/'transport-v1/revision.json').exists():
        from full_episode_transport_revision import validate
        revision=validate(root)
    rows=[];missing=0;submitted=0;covered_shards=0
    for path in sorted((root/'submissions').glob('*.json')):
        submission=json.loads(path.read_text())
        if not submission.get('uid'):continue
        submitted+=1;index=submission['index'];shard=plan['shards'][index]
        from full_episode_collection import submission_generation
        generation=submission_generation(root,submission)
        require(submission['plan_fingerprint']==plan['fingerprint'] and submission['task']==shard['task']
                and submission['instance_ids']==shard['instance_ids']
                and submission['name']==f"behavior-full-ep-{plan['fingerprint'][:8]}-a{index:04d}-g{generation}",
                'Submission outside retained frozen scope')
        require(not submission.get('failure_recovery'),'Recovery receipts require explicit provenance merger')
        source=original_source;parts=plan.get('maximum_archive_parts', 512)
        if submission.get('transport_revision_sha256'):
            require(revision and submission['transport_revision_sha256']==revision['revision_sha256']
                    and submission['deployed_source_sha256']==revision['source_manifest_sha256'],
                    'Unaccepted transport source')
            source=revision['source_manifest_sha256'];parts=revision['maximum_parts']
        shard_rows=0
        for instance in shard['instance_ids']:
            key=prefix_for(plan)+f"/episodes/{shard['task']}/{instance}/attempt-1.json"
            receipt=get_json(client,key,optional=True)
            if receipt is None:missing+=1;continue
            validate_episode(receipt,plan,shard,source,maximum_parts=parts)
            head=client.head_object(Bucket=BUCKET,Key=receipt['archive']['object_key'])
            require(head['ContentLength']==receipt['archive']['bytes'] and head['ETag']==receipt['archive']['etag'],
                    'Archive HEAD differs from published receipt')
            result=receipt['result']
            rows.append(dict(run_id=plan['run_id'],sampling_round=round_id,root=str(root.resolve()),
                plan_fingerprint=plan['fingerprint'],shard_index=index,execution_uid=submission['uid'],
                task=shard['task'],instance_id=instance,rollout_id=0,attempt_id=receipt['attempt_id'],
                receipt_key=key,receipt_sha256=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest(),
                object_key=receipt['archive']['object_key'],archive_sha256=receipt['archive']['sha256'],
                raw_sha256=receipt['trajectory_sha256'],source_manifest_sha256=source,
                success=receipt['success'],terminated=result['terminated'],truncated=result['truncated'],
                steps=receipt['steps'],bc_eligible=bc_eligible(receipt),training_seen=True,
                export_readback_accepted=False))
            shard_rows+=1
        if shard_rows==len(shard['instance_ids']):covered_shards+=1
    return dict(sampling_round=round_id,root=str(root.resolve()),submitted_shards=submitted,
        episodes=len(rows),successful_episodes=sum(r['success'] for r in rows),
        bc_eligible=sum(r['bc_eligible'] for r in rows),frames=sum(r['steps'] for r in rows),
        fully_archived_shards=covered_shards,missing_published_episode_receipts=missing,episode_rows=rows)


def report(campaign, client):
    campaign=Path(campaign);auth=json.loads((campaign/'authorization.json').read_text())
    policy=json.loads((campaign/'training_data_policy.json').read_text())
    require(policy['retained_data_roots']==auth['roots'],'Retained data roots differ from frozen scope')
    require(policy['continue_collection_roots']==auth['roots'][:1],'Only original round may continue')
    started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        rounds=list(executor.map(lambda pair:scan_round(pair[1],pair[0],client),enumerate(auth['roots'])))
    rows=[v for r in rounds for v in r.pop('episode_rows')]
    require(len({episode_identity(r) for r in rows})==len(rows),'Duplicate retained episode identity')
    totals={key:sum(r[key] for r in rounds) for key in ('episodes','successful_episodes','bc_eligible','frames','fully_archived_shards')}
    summary=dict(kind='retained_full_episode_campaign_inventory',observed_at=time.time(),query_started_at=started,
        statistics_scope='all_three_retained_rounds_all_published_valid_episode_receipts',rounds=rounds,totals=totals,
        collection_scope='original_round_only_2000',collection_target_episodes=2000,
        active_collection_rounds=[0],retained_rounds=[0,1,2],ready_for_training=False)
    private_json(campaign/'all_collected_data_progress_current.json',summary)
    private_json(campaign/'all_collected_episode_index_current.json',dict(**summary,episode_rows=rows))
    private_json(campaign/'training_candidates_current.json',dict(kind='full_episode_bc_candidates',
        observed_at=summary['observed_at'],selection='environment_success_terminated_untruncated',
        retained_rounds=[0,1,2],episodes=[r for r in rows if r['bc_eligible']],
        required_before_training=['full_archive_SHA256_readback','frame_action_alignment','real_LeRobot_video_readback'],
        all_public_instances_training_seen=True,validation_instances=auth['validation_instances'],ready_for_training=False))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);args=p.parse_args()
    from behavior_aoss_client import client
    with (args.campaign/'.data_inventory.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(report(args.campaign,client())))
