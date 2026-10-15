"""Three independently seeded 1000-instance rounds with the original 1688977 policy."""
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

MODEL='/mnt/afs/260010165/behavior_1k/checkpoints/pi05_b1k_full100_joint_1688977_aoss_20261010'
ROUNDS=(6,7,8)
KIND='native_original_three_repeats_3000'


def digest(raw):return hashlib.sha256(raw).hexdigest()


def validate_original_repeat(plan):
    assert plan['original_repeat_collection'] is True and plan['sampling_round'] in ROUNDS
    assert plan['expected_episodes']==1000 and len(plan['shards'])==201
    assert plan['checkpoint']==MODEL and plan['checkpoint_identity']['checkpoint_step']==1688977
    assert plan['checkpoint_identity']['checkpoint_source']==MODEL
    assert not plan.get('trained_model_export') and not plan.get('paired_evaluation_only')
    assert plan['base_source_manifest_sha256']=='7f2a16738961dfb1ca5b95d280ee61450f996757c5514b8c144e86dce3c982d4'
    assert plan['publish_compact_metrics'] is True and plan['cpu_thread_policy']=='original_unchanged'
    assert plan['parent_plan_fingerprint']=='ddea3ef13b172c231786c8170ec24643967216294e970ef55791df437053991c'
    assert not plan['official_benchmark'] and not plan['automatic_failure_recovery']
    return plan


def validate_campaign(campaign,auth):
    from restore_full_episode_collection import read_plan
    assert auth['kind']==KIND and auth['total_episodes']==3000 and auth['total_shards']==603
    assert auth['rounds']==list(ROUNDS) and len(auth['roots'])==3
    assert auth['checkpoint_step']==1688977 and auth['max_pending_tasks']==5
    assert auth['max_active_gpus'] is None and auth['controller_poll_seconds']==60
    assert auth['deadline_epoch']==auth['inherited_deadline_epoch']
    plans=[read_plan(r) for r in auth['roots']]
    assert [p['fingerprint'] for p in plans]==auth['fingerprints']
    assert [p['sampling_round'] for p in plans]==list(ROUNDS)
    seeds=[s for p in plans for s in p['policy_seeds'].values()]
    assert len(seeds)==len(set(seeds))==3000
    assert all(Path(r).resolve().parent==Path(campaign).resolve() for r in auth['roots'])
    return plans


def gate_allowed(root,plan):
    root=Path(root).resolve();campaign=root.parent;validate_original_repeat(plan)
    raw=(campaign/'authorization.json').read_bytes();auth=json.loads(raw)
    assert auth['kind']==KIND and str(root) in auth['roots']
    index=auth['roots'].index(str(root));assert auth['fingerprints'][index]==plan['fingerprint']
    proof=json.loads((root/'pilot_acceptance_waiver.json').read_text())
    assert proof['authorization_sha256']==digest(raw) and proof['plan_fingerprint']==plan['fingerprint']
    assert proof['prior_user_no_acceptance_instruction_retained'] is True and proof['artifact_acceptance_complete'] is False
    return True


def aggregate(round_reports):
    n=sum(r['completed_episodes'] for r in round_reports);s=sum(r['successful_episodes'] for r in round_reports)
    return dict(completed_episodes=n,target_episodes=3000,missing_episodes=3000-n,successful_episodes=s,
        success_rate=s/n if n else None,
        mean_q_score=sum((r['mean_q_score'] or 0)*r['completed_episodes'] for r in round_reports)/n if n else None,
        competition_sr=sum((r['competition_sr'] or 0)*r['completed_episodes'] for r in round_reports)/n if n else None,
        bc_eligible_episodes=sum(r['bc_eligible_episodes'] for r in round_reports),
        complete=n==3000,denominator='completed_published_episodes_across_independent_rounds')


def observe_metrics(campaign,storage):
    from full_episode_collection import save
    from restore_full_episode_collection import prefix_for
    from trained_full_episode_collection import metric_summary
    campaign=Path(campaign);auth=json.loads((campaign/'authorization.json').read_text());plans=validate_campaign(campaign,auth)
    cache_path=campaign/'metrics_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {};fresh={};reports=[]
    for root,plan in zip(auth['roots'],plans):
        prefix=prefix_for(plan);objects={}
        for page in storage.get_paginator('list_objects_v2').paginate(Bucket='260010165',Prefix=prefix+'/'):
            objects.update({v['Key']:v for v in page.get('Contents',[]) if '/private/' not in v['Key']})
        keys=[k for k in objects if k.startswith(prefix+'/metrics/') and k.endswith('.json')]
        records=[json.loads(f.read_text()) for f in (Path(root)/'submissions').glob('*.json')]
        submitted={(r['task'],i) for r in records if r.get('uid') for i in r['instance_ids']}
        def fetch(k):
            if k in cache and cache[k]['etag']==objects[k]['ETag']:return k,cache[k]
            assert 0<objects[k]['Size']<32768
            response=storage.get_object(Bucket='260010165',Key=k)
            try:raw=response['Body'].read(32769);assert len(raw)<32768
            finally:response['Body'].close()
            return k,dict(etag=objects[k]['ETag'],row=json.loads(raw))
        with ThreadPoolExecutor(max_workers=8) as pool:batch=dict(pool.map(fetch,keys))
        rows=[]
        for item in batch.values():
            row=item['row'];assert row['kind']=='native_original_repeat_episode_metrics_v1'
            assert row['checkpoint_step']==1688977 and row['checkpoint_identity_sha256']==plan['checkpoint_identity_sha256']
            assert row['collection_run_id']==plan['run_id'] and row['sampling_round']==plan['sampling_round']
            assert (row['task'],row['instance_id']) in submitted and row['rollout_id']==0
            assert objects[row['receipt_key']]['Size']>0 and objects[row['archive_key']]['Size']==row['archive_bytes']>0
            assert objects[row['archive_key']]['ETag']==row['archive_etag']
            assert row['repeat_sampling']['server_seed_acknowledged'] is True
            assert row['repeat_sampling']['policy_seed']==plan['policy_seeds'][f"{row['task']}:{row['instance_id']}"]
            rows.append(row)
        report=metric_summary(rows,plan['task_prompts']);report['sampling_round']=plan['sampling_round'];reports.append(report);fresh.update(batch)
    summary=aggregate(reports);summary.update(observed_at=time.time(),checkpoint_step=1688977,rounds=reports,ready_for_training=False)
    save(cache_path,fresh);save(campaign/'metrics_progress_current.json',summary);return summary


def freeze(campaign,parent):
    from full_episode_collection import save
    from restore_full_episode_collection import read_plan
    campaign=Path(campaign).resolve();parent=Path(parent).resolve();old=read_plan(parent)
    assert old['checkpoint']==MODEL and old['sampling_round']==3
    files={name:(parent/'overlay_sources'/name).read_bytes() for name in old['overlay_files_sha256']}
    assert all(digest(body)==old['overlay_files_sha256'][name] for name,body in files.items())
    def patch(name,before,after):
        s=files[name].decode();assert s.count(before)==1,(name,before);files[name]=s.replace(before,after,1).encode()
    patch('scripts/full_episode_repeat_seeding.py','round_id not in (1, 2, 3):','round_id not in (1, 2, 3, 6, 7, 8):')
    patch('sco/run_full_episode_collection.py',
        "        object_storage=payload['object_storage'], official_benchmark=False,",
        "        object_storage=payload['object_storage'], metrics_object_storage=payload['metrics_object_storage'],\n"
        "        checkpoint_identity_sha256=payload['checkpoint_identity_sha256'], official_benchmark=False,")
    patch('scripts/full_episode_checkpoint.py',"    upload_json(receipt, storage['receipt'])\n    return receipt",
        "    upload_json(receipt, storage['receipt'])\n"
        "    import math\n"
        "    q = result['q_score']['final']\n"
        "    assert type(q) in (int, float) and math.isfinite(q) and 0 <= q <= 1\n"
        "    metrics = dict(kind='native_original_repeat_episode_metrics_v1',\n"
        "        collection_run_id=manifest['run_id'].rsplit('-a',1)[0], run_id=manifest['run_id'],\n"
        "        task=result['task'], instance_id=result['instance_id'], rollout_id=result['rollout_id'],\n"
        "        sampling_round=manifest['sampling_round'], checkpoint_step=1688977,\n"
        "        checkpoint_identity_sha256=manifest['checkpoint_identity_sha256'],\n"
        "        success=result['success'], terminated=result['terminated'], truncated=result['truncated'],\n"
        "        q_score_final=q, steps=result['steps'], receipt_key=storage['receipt']['key'],\n"
        "        archive_key=stored['object_key'], archive_bytes=stored['bytes'], archive_etag=stored['etag'],\n"
        "        source_manifest_sha256=manifest['source_manifest_sha256'], repeat_sampling=result['repeat_sampling'])\n"
        "    upload_json(metrics,manifest['metrics_object_storage'][key])\n"
        "    return receipt")
    campaign.mkdir(exist_ok=False);plans=[];roots=[];used=set(old['policy_seeds'].values())
    for rnd in ROUNDS:
        root=campaign/f'round-{rnd}';root.mkdir();source=root/'overlay_sources'
        for name,body in files.items():
            if name.endswith('.py'):compile(body,name,'exec')
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
        hashes={k:digest(v) for k,v in files.items()}
        with tarfile.open(root/'overlay.tar.gz','w:gz') as stream:
            for name,body in sorted(files.items()):
                member=tarfile.TarInfo(name);member.size=len(body);stream.addfile(member,io.BytesIO(body))
            body=json.dumps(hashes,sort_keys=True).encode();member=tarfile.TarInfo('_FULL_EPISODE_OVERLAY.json');member.size=len(body);stream.addfile(member,io.BytesIO(body))
        seeds={}
        for task in sorted(old['task_prompts']):
            for instance in range(301,311):
                seed=int.from_bytes(hashlib.sha256(f'original1688977-20261015-round{rnd}:{task}:{instance}'.encode()).digest()[:4],'big')
                while seed in used:seed=(seed+1)%2**32
                used.add(seed);seeds[f'{task}:{instance}']=seed
        plan=dict(old,run_id=f'native-full1000-original1688977-20261015-r{rnd}',sampling_round=rnd,
            original_repeat_collection=True,publish_compact_metrics=True,cpu_thread_policy='original_unchanged',
            parent_plan_fingerprint=old['fingerprint'],policy_seeds=seeds,
            checkpoint_identity_sha256=digest(json.dumps(old['checkpoint_identity'],sort_keys=True).encode()),
            overlay_files_sha256=hashes,overlay_sha256=digest((root/'overlay.tar.gz').read_bytes()))
        plan.pop('fingerprint');plan['fingerprint']=digest(json.dumps(plan,sort_keys=True).encode());validate_original_repeat(plan)
        save(root/'plan.json',plan);save(root/'held_out_instances.json',old['held_out_instances']);(root/'submissions').mkdir();plans.append(plan);roots.append(root)
    deadline=json.loads((parent.parent/'authorization.json').read_text())['deadline_epoch']
    auth=dict(kind=KIND,total_episodes=3000,total_shards=603,roots=[str(r) for r in roots],fingerprints=[p['fingerprint'] for p in plans],rounds=list(ROUNDS),
        checkpoint_step=1688977,deadline_epoch=deadline,inherited_deadline_epoch=deadline,max_active_gpus=None,max_pending_tasks=5,
        controller_poll_seconds=60,automatic_failure_recovery=False,admission='bounded_spot_queue',
        all_public_instances_training_seen=True,validation_instances=old['held_out_instances'],
        user_instruction='后面用1688977原始权重，不断提交采集轨迹worker；每个实例再采3条，共3000条',authorized_at=time.time())
    save(campaign/'authorization.json',auth)
    for root,plan in zip(roots,plans):
        save(root/'controller_authorization.json',dict(plan_fingerprint=plan['fingerprint'],deadline_epoch=deadline,expected_episodes=1000,automatic_failure_recovery=False))
        save(root/'pilot_acceptance_waiver.json',dict(authorization_sha256=digest((campaign/'authorization.json').read_bytes()),plan_fingerprint=plan['fingerprint'],prior_user_no_acceptance_instruction_retained=True,artifact_acceptance_complete=False))
    policy=json.loads((parent.parent/'queue_policy.json').read_text());policy.update(roots=auth['roots']);save(campaign/'queue_policy.json',policy)
    save(campaign/'HANDOFF.json',dict(state='frozen_original3000_collection',expected_episodes=3000,expected_shards=603,rounds=list(ROUNDS),checkpoint_step=1688977,
        checkpoint=MODEL,controller_required='watch_full_episode_repeats.py',controller_poll_seconds=60,max_pending_tasks=5,
        queue_scope='authorized_campaign_only',old_campaigns_remain_stopped=True,metrics='metrics_progress_current.json',
        independent_policy_seeds=True,cpu_thread_policy='original_unchanged',deadline_epoch=deadline,ready_for_training=False))
    return auth
