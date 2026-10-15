"""Read-only checkpoint/reuse checks and immutable original-artifact publication."""
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

def publish_file(path,route):
    path=Path(path);size=path.stat().st_size
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with path.open('rb') as stream:
            request=urllib.request.Request(route['put'],data=stream,method='PUT',
                headers={'Content-Length':str(size),'If-None-Match':'*'})
            with opener.open(request,timeout=600):pass
    except urllib.error.HTTPError as error:
        if error.code not in (409,412):raise
    actual=hashlib.sha256();count=0
    with opener.open(route['get'],timeout=600) as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            actual.update(block);count+=len(block)
            if count>size:raise ValueError('Original artifact destination too long')
    assert count==size and actual.hexdigest()==digest.hexdigest()
    return dict(bucket='260010165',key=route['key'],bytes=size,sha256=digest.hexdigest(),full_destination_readback=True)

def publish_originals(payload,json_path,video_path):
    return {name:publish_file(path,payload['artifact_routes'][name])
            for name,path in [('json',json_path),('video',video_path)]}

def preflight(payload):
    from worker import load,sha,source_preflight,write
    from native_helpers import verify_video
    plan=payload['plan'];checkpoint=Path(plan['candidate_checkpoint'])
    shared=source_preflight(plan)
    ready=load(checkpoint/'_READY')
    assert ready['state']=='accepted' and ready['completed_updates']==10000
    assert ready['manifest_sha256']==plan['candidate_manifest_sha256']==sha(checkpoint/'SHA256SUMS.inference')
    for row in plan['candidate_files']:
        rel=Path(row['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
        path=checkpoint/rel
        assert not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    assert sha(checkpoint/'assets'/plan['repo_id']/'norm_stats.json')==plan['norm_sha256']
    assert load(checkpoint/'assets/subtask_conditioning.json')['action_conditioning']=='task_and_subtask'
    gate=dict(state='verified',id=payload['id'],protocol_sha256=plan['protocol_sha256'],
        worker_bundle_sha256=payload['worker_bundle_sha256'],checkpoint_manifest_sha256=plan['candidate_manifest_sha256'],
        completed_updates=10000,checkpoint_full_sha256_readback=True,source_preflight=shared)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(payload['checkpoint_gate_put_url'],data=json.dumps(gate).encode(),method='PUT'),timeout=60):pass
    output=Path(plan['afs_root'])/'reuse_audit';output.mkdir(parents=True,exist_ok=False)
    results=[]
    for item in plan['reuse_candidates']:
        report=item['report'];record=report['record'];task=record['task'];instance=record['instance_id']
        assert report['state']=='accepted' and report['protocol_sha256']==plan['parent_protocol_sha256']
        assert report['checkpoint_manifest_sha256']==plan['candidate_manifest_sha256']
        assert record['artifact_accepted'] and record['policy_seed_acknowledged']
        assert record['policy_seed']==record['environment_seed']==0 and [task,instance] in plan['pairs']
        for artifact in report['artifacts']:
            path=Path(artifact['path'])
            assert str(path).startswith('/mnt/afs/260010165/behavior_1k/paired_quick20_reused1688977_step10000_20261014/')
            assert path.stat().st_size==artifact['bytes'] and sha(path)==artifact['sha256']
        native=next(Path(r['path']) for r in report['artifacts'] if r['path'].endswith('/'+task+'_301_0.json'))
        video=next(Path(r['path']) for r in report['artifacts'] if r['path'].endswith('.mp4'))
        row=load(native)
        assert sha(native)==record['result_sha256']
        assert row['task']==task and row['instance_id']==instance and row['rollout_id']==0
        assert row['steps']==record['steps'] and row['success']==record['model_success'] and row['q_score']['final']==record['q_score']
        assert (row['terminated'] or row['truncated']) and row['success']==(row['q_score']['final']==1)
        one=output/task;one.mkdir()
        library=load(Path(plan['supplemental_library']['root'])/'_READY')['library_path']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',
            LD_LIBRARY_PATH=plan['runtime']+'/envs/b1k-eval/lib:'+library+':'+plan['runtime']+'/graphics-runtime/usr/lib/x86_64-linux-gnu')
        verification=verify_video(dict(plan,expected_video_frames=row['steps']),video,env,one)
        index=plan['pairs'].index([task,instance]);id=f'candidate-{index:03d}'
        routes=payload['reuse_artifact_routes'][id]
        originals={name:publish_file(path,routes[name]) for name,path in [('json',native),('video',video)]}
        results.append(dict(record=record,original_report_sha256=item['original_report_sha256'],
            original_id=report['id'],video_validation=verification,object_artifacts=originals,new_rollout_executed=False))
        (output/'progress.json').write_text(json.dumps(dict(verified=len(results),expected=20)))
    assert len(results)==20
    return dict(state='accepted',checkpoint_manifest_sha256=plan['candidate_manifest_sha256'],
        completed_updates=10000,reused_candidate_results=results,new_rollouts_executed=0,shared=shared)
