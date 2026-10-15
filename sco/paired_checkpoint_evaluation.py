"""Evaluate checkpoint10000 on the exact 962 published baseline instances."""
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

MODEL='/mnt/afs/260010165/behavior_1k/paired_quick20_reused1688977_step10000_20261014/checkpoint10000'
MANIFEST='7b93b0d35d71d577329f8f908a9218d4f753ca79b12300b663dd5587d764332d'
BASE=Path(__file__).resolve().parents[1]
OLD=BASE/'eval_results/sco_full_episode_collection_20261014_final1000_round3'
EXPORT=BASE.parent/'analysis/retraining_20261013/paired_quick20_reused1688977_step10000_20261014/reports/export.json'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_plan(plan):
    assert plan['paired_evaluation_only'] is True and plan['expected_episodes']==962
    assert plan['sampling_round']==5 and plan['checkpoint']==MODEL
    assert plan['checkpoint_identity']['checkpoint_step']==10000
    assert plan['checkpoint_identity']['parent_checkpoint_step']==1688977
    export=plan['trained_model_export']
    assert export['checkpoint']==MODEL and export['manifest_sha256']==MANIFEST
    assert export['state']=='accepted' and export['completed_updates']==10000 and export['full_afs_sha256_readback'] is True
    assert plan['checkpoint_identity']['checkpoint_files']==export['files']
    baseline=plan['baseline_scores']
    assert len(baseline)==962 and len({(r['task'],r['instance_id']) for r in baseline})==962
    assert digest(json.dumps(baseline,sort_keys=True).encode())==plan['baseline_scores_sha256']
    assert plan['baseline_plan_fingerprint']=='ddea3ef13b172c231786c8170ec24643967216294e970ef55791df437053991c'
    assert sum(r['success'] for r in baseline)==82
    assert math.isclose(sum(r['q_score'] for r in baseline)/962,0.23827673683442915,abs_tol=1e-12)
    assert plan['cpu_thread_policy']=='unchanged_baseline' and plan['record_rgb_trajectories'] is False
    assert plan['official_benchmark'] is False and plan['automatic_failure_recovery'] is False
    assert plan['shards'][0]==dict(index=0,task='turning_on_radio',instance_ids=[301],pilot=True)
    pairs=[]
    for index,shard in enumerate(plan['shards']):
        assert shard['index']==index and shard['pilot'] is (index==0)
        assert 1<=len(shard['instance_ids'])<=5
        pairs.extend((shard['task'],i) for i in shard['instance_ids'])
    expected={(r['task'],r['instance_id']) for r in baseline}
    assert len(pairs)==962 and len(set(pairs))==962 and set(pairs)==expected
    assert set(plan['policy_seeds'])=={f'{t}:{i}' for t,i in expected}
    assert len(set(plan['policy_seeds'].values()))==962
    for r in baseline:
        assert type(r['success']) is bool and type(r['q_score']) in (int,float) and math.isfinite(r['q_score']) and 0<=r['q_score']<=1
    return plan


def evaluation_gate_allowed(root,plan):
    root=Path(root).resolve();validate_plan(plan)
    auth=json.loads((root.parent/'authorization.json').read_text())
    assert auth['kind']=='paired_checkpoint_evaluation' and auth['roots']==[str(root)]
    assert auth['fingerprints']==[plan['fingerprint']] and auth['total_episodes']==962
    proof=json.loads((root.parent/'evaluation_gate.json').read_text())
    assert proof['authorization_sha256']==digest((root.parent/'authorization.json').read_bytes())
    assert proof['plan_fingerprint']==plan['fingerprint'] and proof['total_episodes']==962
    assert proof['prior_user_artifact_gate_waiver_retained'] is True
    return True


def score_row(result,manifest,storage):
    """Compact metrics are published only after a whole native episode terminates."""
    import math
    q=result['q_score']['final']
    assert type(result['success']) is bool and type(q) in (int,float) and math.isfinite(q) and 0<=q<=1
    assert result['steps']>0 and (result['terminated'] or result['truncated'])
    assert result['success']==result['final_goal_state']['success']
    assert result['conditioning']['mode']=='whole_task'
    assert not result['conditioning']['oracle'] and not result['conditioning']['privileged']
    proof=result['repeat_sampling'];spec=manifest['policy_seeds'][f"{result['task']}:{result['instance_id']}"]
    assert proof['policy_seed']==spec and proof['server_seed_acknowledged']
    return dict(kind='paired_checkpoint_episode_score',evaluation_run_id=manifest['evaluation_run_id'],
        run_id=manifest['run_id'],task=result['task'],instance_id=result['instance_id'],rollout_id=result['rollout_id'],
        checkpoint_step=10000,checkpoint_manifest_sha256=manifest['checkpoint_manifest_sha256'],
        sampling_round=5,success=result['success'],q_score_final=q,steps=result['steps'],
        terminated=result['terminated'],truncated=result['truncated'],repeat_sampling=proof,
        source_manifest_sha256=manifest['source_manifest_sha256'],result_key=storage['key'])


def aggregate(plan,rows):
    expected={(r['task'],r['instance_id']):r for r in plan['baseline_scores']};seen=set()
    for r in rows:
        key=(r['task'],r['instance_id']);assert key in expected and key not in seen;seen.add(key)
        assert type(r['success']) is bool and math.isfinite(r['q_score_final']) and 0<=r['q_score_final']<=1
    def stats(batch,old=False):
        n=len(batch)
        return dict(episodes=n,successes=sum(r['success'] for r in batch),success_rate=sum(r['success'] for r in batch)/n if n else None,
          competition_sr=sum(r['q_score' if old else 'q_score_final']==1 for r in batch)/n if n else None,
          mean_q_score=sum(r['q_score' if old else 'q_score_final'] for r in batch)/n if n else None)
    base=[expected[r['task'],r['instance_id']] for r in rows];b=stats(base,True);n=stats(rows)
    return dict(completed_episodes=len(rows),target_episodes=962,missing_episodes=962-len(rows),baseline_same_instances=b,candidate=n,
      baseline_all_962=stats(list(expected.values()),True),final_comparison=len(rows)==962,
      success_rate_difference_pp=(n['success_rate']-b['success_rate'])*100 if rows else None,
      mean_q_score_difference=n['mean_q_score']-b['mean_q_score'] if rows else None,
      denominator='identical_task_instance_policy_seed_completed_pairs',
      per_task={t:dict(baseline=stats([expected[r['task'],r['instance_id']] for r in rows if r['task']==t],True),candidate=stats([r for r in rows if r['task']==t])) for t in sorted(plan['task_prompts'])})


def observe_metrics(campaign,storage):
    from full_episode_collection import save
    from restore_full_episode_collection import read_plan,prefix_for
    campaign=Path(campaign);auth=json.loads((campaign/'authorization.json').read_text());root=Path(auth['roots'][0]);plan=read_plan(root)
    cache_path=campaign/'metrics_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {};objects={}
    for page in storage.get_paginator('list_objects_v2').paginate(Bucket='260010165',Prefix=prefix_for(plan)+'/'):
        objects.update({r['Key']:r for r in page.get('Contents',[]) if '/private/' not in r['Key']})
    keys=[k for k in objects if '/metrics/' in k and k.endswith('.json')]
    records=[json.loads(f.read_text()) for f in (root/'submissions').glob('*.json')]
    submitted={(r['task'],i) for r in records if r.get('uid') for i in r['instance_ids']}
    def fetch(k):
        if k in cache and cache[k]['etag']==objects[k]['ETag']:return k,cache[k]
        assert 0<objects[k]['Size']<32768
        response=storage.get_object(Bucket='260010165',Key=k)
        try:raw=response['Body'].read(32769);assert len(raw)<32768
        finally:response['Body'].close()
        return k,dict(etag=objects[k]['ETag'],row=json.loads(raw))
    with ThreadPoolExecutor(max_workers=8) as pool:fresh=dict(pool.map(fetch,keys))
    rows=[]
    for item in fresh.values():
        row=item['row'];assert row['kind']=='paired_checkpoint_episode_score' and row['checkpoint_manifest_sha256']==MANIFEST
        assert row['evaluation_run_id']==plan['run_id'] and row['sampling_round']==5 and row['rollout_id']==0
        assert (row['task'],row['instance_id']) in submitted and objects[row['result_key']]['Size']>0
        assert row['repeat_sampling']['policy_seed']==plan['policy_seeds'][f"{row['task']}:{row['instance_id']}"]
        assert row['repeat_sampling']['server_seed_acknowledged'] is True
        rows.append(row)
    summary=aggregate(plan,rows);summary.update(observed_at=time.time(),checkpoint_step=10000,sampling_round=5)
    save(cache_path,fresh);save(campaign/'metrics_progress_current.json',summary);return summary


WORKER_RUN='''def _run(payload):
    root=Path(payload['assignment_root']);root.mkdir(parents=True,exist_ok=True)
    log=(root/'evaluation_driver.log').open('a',buffering=1)
    os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    target=Path(payload['code']);target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.staging.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);source_sha=stage_source(payload)
    sys.path[:0]=[str(target/'sco'),str(target/'scripts')]
    from pi05_eval_queue import EvaluationQueue
    from collection_checkpoint import request,upload_json
    import run_public_test_assignment as baseline
    manifest=dict(kind='native_full_episode_collection_manifest',run_id=payload['run_id'],
        evaluation_run_id=payload['run_id'].rsplit('-a',1)[0],checkpoint=payload['checkpoint_identity'],
        checkpoint_manifest_sha256=payload['trained_model_export']['manifest_sha256'],
        source_manifest_sha256=source_sha,task_prompts=payload['task_prompts'],
        paired_evaluation_only=True,release_uploaded_scratch=False,
        object_storage=payload['object_storage'],metrics_object_storage=payload['metrics_object_storage'],
        sampling_round=5,policy_seeds=payload['policy_seeds'],parent_plan_fingerprint=payload['parent_plan_fingerprint'])
    manifest_path=root/'evaluation.private.json';manifest_path.write_text(json.dumps(manifest));manifest_path.chmod(0o600)
    queue=EvaluationQueue(root/'queue.sqlite3')
    if not (root/'queue.sqlite3').exists():
        queue.initialize([(payload['task'],i) for i in payload['instance_ids']],max_attempts=1,
            metadata=dict(run_id=payload['run_id'],mode='public_test',num_rollouts=1,require_videos=False,
            checkpoint_step=10000,source_manifest_sha256=source_sha))
    original=baseline.subprocess
    class EvaluationSubprocess:
        def __getattr__(self,name):return getattr(original,name)
        def run(self,command,**kwargs):
            if command[:2]==['bash',str(target/'scripts/eval_b1k.sh')]:
                kwargs['env']=dict(kwargs['env'],PI05_FULL_EPISODE_COLLECTION_MANIFEST=str(manifest_path))
            return original.run(command,**kwargs)
    baseline.subprocess=EvaluationSubprocess()
    baseline.run(dict(payload,source_manifest_sha256=source_sha,pilot=False,write_video=False))
    rows=[]
    for instance in payload['instance_ids']:
        key=f"{payload['task']}:{instance}:0"
        body,_=request(payload['metrics_object_storage'][key]['get']);row=json.loads(body)
        assert row['instance_id']==instance and row['checkpoint_step']==10000
        rows.append(row)
    completion=dict(kind='paired_checkpoint_evaluation_complete',run_id=payload['run_id'],complete=True,
        checkpoint=payload['checkpoint_identity'],source_manifest_sha256=source_sha,
        episodes=len(rows),successes=sum(r['success'] for r in rows),mean_q_score=sum(r['q_score_final'] for r in rows)/len(rows),
        metrics=rows,completed_at=time.time())
    upload_json(completion,payload['completion_object'])
    (root/'_EVALUATION_SUCCESS').write_text(json.dumps(completion))
    print(json.dumps({k:completion[k] for k in ['episodes','successes','mean_q_score']}),flush=True)


'''


def freeze(campaign):
    from full_episode_collection import save
    from restore_full_episode_collection import read_plan
    from trained_full_episode_collection import verify_checkpoint
    campaign=Path(campaign).resolve();oldroot=OLD/'round-3';old=read_plan(oldroot)
    raw=(OLD/'competition_score_final.json').read_bytes();scores=json.loads(raw);assert scores['completed_episodes']==962
    baseline=[{k:r[k] for k in ['task','instance_id','success','q_score','key','etag']} for r in scores['score_rows']]
    baseline.sort(key=lambda r:(r['task'],r['instance_id']));expected={(r['task'],r['instance_id']) for r in baseline}
    export_raw=EXPORT.read_bytes();export=json.loads(export_raw);assert export['manifest_sha256']==MANIFEST
    files={name:(oldroot/'overlay_sources'/name).read_bytes() for name in old['overlay_files_sha256']}
    assert all(digest(body)==old['overlay_files_sha256'][name] for name,body in files.items())
    helper=inspect.getsource(verify_checkpoint).replace("== 20000 and", "== 10000 and").replace('step=20000','step=10000')
    files['sco/trained_checkpoint_verify.py']=helper.encode()
    files['scripts/paired_checkpoint_score.py']=inspect.getsource(score_row).encode()
    def replace(name,before,after):
        text=files[name].decode();assert text.count(before)==1,(name,before);files[name]=text.replace(before,after).encode()
    replace('sco/run_public_test_assignment.py',"    assert json.loads((checkpoint / '_READY').read_text())['checkpoint_step'] == 1688977","    from trained_checkpoint_verify import verify_checkpoint\n    verify_checkpoint(checkpoint,payload['trained_model_export'])")
    replace('sco/run_public_test_assignment.py',"    for row in provenance['checkpoint_files']:","    for row in payload['checkpoint_identity']['checkpoint_files']:")
    files['sco/run_public_test_assignment.py']=files['sco/run_public_test_assignment.py'].replace(b"'checkpoint_step': 1688977",b"'checkpoint_step': 10000")
    replace('scripts/full_episode_repeat_seeding.py','round_id not in (1, 2, 3):','round_id not in (1, 2, 3, 5):')
    replace('scripts/eval_b1k_queue.py','                trajectory = NativeFullEpisodeTrajectory(trajectory_path, identity)',"                if not manifest.get('paired_evaluation_only'):\n                    trajectory = NativeFullEpisodeTrajectory(trajectory_path, identity)")
    replace('scripts/eval_b1k_queue.py','            if trajectory is not None:\n                from full_episode_trajectory import (',
        "            if manifest_path is not None and manifest.get('paired_evaluation_only'):\n"
        "                result['repeat_sampling'] = evaluator.policy.proof()\n"
        "                from paired_checkpoint_score import score_row\n"
        "                from collection_checkpoint import upload_json\n"
        "                key = f'{args.task_name}:{instance_id}:{rollout_id}'\n"
        "                storage = manifest['object_storage'][key]['receipt']\n"
        "                row = score_row(result,manifest,storage)\n"
        "                upload_json(result,storage)\n"
        "                upload_json(row,manifest['metrics_object_storage'][key])\n"
        "            if trajectory is not None:\n                from full_episode_trajectory import (")
    driver=files['sco/run_full_episode_collection.py'].decode();files['sco/run_full_episode_collection.py']=(driver[:driver.index('def _run(payload):')]+WORKER_RUN+driver[driver.index('def run(payload):'):]).encode()
    campaign.mkdir(exist_ok=False);root=campaign/'round-5';root.mkdir();source=root/'overlay_sources'
    for name,body in files.items():
        if name.endswith('.py'):compile(body,name,'exec')
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
    hashes={k:digest(v) for k,v in files.items()}
    with tarfile.open(root/'overlay.tar.gz','w:gz') as stream:
        for name,body in sorted(files.items()):
            member=tarfile.TarInfo(name);member.size=len(body);stream.addfile(member,io.BytesIO(body))
        body=json.dumps(hashes,sort_keys=True).encode();member=tarfile.TarInfo('_FULL_EPISODE_OVERLAY.json');member.size=len(body);stream.addfile(member,io.BytesIO(body))
    shards=[dict(index=0,task='turning_on_radio',instance_ids=[301],pilot=True)]
    for task in sorted(old['task_prompts']):
        instances=sorted(i for t,i in expected if t==task and (t,i)!=('turning_on_radio',301))
        for offset in range(0,len(instances),5):shards.append(dict(index=len(shards),task=task,instance_ids=instances[offset:offset+5],pilot=False))
    plan=dict(old,run_id='paired-eval962-quick10k-20261015-r5',expected_episodes=962,shards=shards,sampling_round=5,
        paired_evaluation_only=True,record_rgb_trajectories=False,collection_scope='exact_962_published_baseline_instances',
        baseline_scores=baseline,baseline_scores_sha256=digest(json.dumps(baseline,sort_keys=True).encode()),
        baseline_plan_fingerprint=old['fingerprint'],baseline_report_sha256=digest(raw),
        parent_plan_fingerprint=old['fingerprint'],policy_seeds={f'{t}:{i}':old['policy_seeds'][f'{t}:{i}'] for t,i in expected},
        checkpoint=MODEL,checkpoint_identity=dict(checkpoint_step=10000,parent_checkpoint_step=1688977,checkpoint_source=MODEL,checkpoint_files=export['files'],manifest_sha256=MANIFEST),
        trained_model_export=export,model_export_report_sha256=digest(export_raw),cpu_thread_policy='unchanged_baseline',
        overlay_files_sha256=hashes,overlay_sha256=digest((root/'overlay.tar.gz').read_bytes()))
    plan.pop('fingerprint');plan['fingerprint']=digest(json.dumps(plan,sort_keys=True).encode());validate_plan(plan)
    save(root/'plan.json',plan);save(root/'held_out_instances.json',old['held_out_instances']);(root/'submissions').mkdir()
    deadline=json.loads((OLD/'authorization.json').read_text())['deadline_epoch']
    auth=dict(kind='paired_checkpoint_evaluation',total_episodes=962,total_shards=len(shards),roots=[str(root)],fingerprints=[plan['fingerprint']],rounds=[5],
        checkpoint_step=10000,baseline_checkpoint_step=1688977,deadline_epoch=deadline,inherited_deadline_epoch=deadline,
        max_pending_tasks=5,max_active_gpus=None,controller_poll_seconds=60,automatic_failure_recovery=False,
        admission='bounded_spot_queue',user_instruction='用1万步的模型在相同实例上评测，比较成功率和Q-score',authorized_at=time.time())
    save(campaign/'authorization.json',auth);save(campaign/'evaluation_gate.json',dict(authorization_sha256=digest((campaign/'authorization.json').read_bytes()),plan_fingerprint=plan['fingerprint'],total_episodes=962,prior_user_artifact_gate_waiver_retained=True,new_model_sha_verified_per_worker=True))
    save(root/'controller_authorization.json',dict(plan_fingerprint=plan['fingerprint'],deadline_epoch=deadline,expected_episodes=962,automatic_failure_recovery=False))
    policy=json.loads((OLD/'queue_policy.json').read_text());policy['roots']=[str(root)];save(campaign/'queue_policy.json',policy)
    save(campaign/'HANDOFF.json',dict(state='frozen_paired_evaluation',expected_episodes=962,expected_shards=len(shards),checkpoint_step=10000,baseline_checkpoint_step=1688977,
        baseline_successes=82,baseline_success_rate=scores['success_rate'],baseline_mean_q_score=scores['mean_q_score'],
        same_policy_seeds=True,cpu_thread_policy='unchanged_baseline',record_rgb_trajectories=False,metrics='metrics_progress_current.json',
        controller_required='watch_full_episode_repeats.py',controller_poll_seconds=60,max_pending_tasks=5,deadline_epoch=deadline,old_campaigns_remain_stopped=True))
    return plan,auth
