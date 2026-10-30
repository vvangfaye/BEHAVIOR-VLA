"""Submit one finite task/instance shard; never rerun completed cases."""
import argparse
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from cluster_helpers import HERE, OLD, aoss_client, signed, submit, describe, write
from recovery_archives import completed_archives
from checkpoint_inventory import verified_checkpoints


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True);parser.add_argument('--recover',action='store_true');args=parser.parse_args()
    plan=json.loads((HERE/'plan.json').read_text());shard=next(x for x in plan['shards'] if x['shard']==args.shard);task=next(x for x in plan['tasks'] if x['task_id']==shard['task_id']);tid=task['task_id']
    source=json.loads((HERE/'source_upload.json').read_text());data=json.loads((HERE/'data_bundle_upload.json').read_text());small=json.loads((HERE/'small_upload.json').read_text());weights=json.loads((HERE/'checkpoint_inventory.json').read_text()) if (HERE/'checkpoint_inventory.json').exists() else json.loads((HERE/'checkpoints_upload.json').read_text());assert data['complete'] and small['complete']
    expected_weights=[x for x in json.loads((HERE/'checkpoint_manifest.json').read_text())['items'] if f'task-{tid:04d}/' in x['destination']]
    assert {x['destination'] for x in expected_weights}<={x['destination'] for x in weights['uploaded']},'Task checkpoint files are not all verified'
    manifest=json.loads((HERE/'preflight.json').read_text());assert manifest['passed'] and manifest['source_sha256']==source['source_sha256']
    checkpoint_folders={'image':f'pi_behavior_b1k_fast--50t_lora--task-{tid:04d}','serf':f'pi_serf_behavior_b1k_fast--4d_env_robot_feat_map--50t_lora--task-{tid:04d}'}
    states=[dict(relative=x['destination'].removeprefix('instances/'),size=x['size'],sha256=x['sha256']) for x in small['uploaded'] if task['task_name'] in x['destination'] and x['destination'].endswith('-tro_state.json')]
    assert len(states)==20
    maps=[dict(instance=int(Path(x['destination']).stem.removeprefix('scene_')),**x) for x in small['uploaded'] if f'/task-{tid:04d}/eval/' in x['destination']]
    assert {x['instance'] for x in maps}==set(task['instance_ids'])
    map_models=[x for x in small['uploaded'] if f'maps/map_models/task-{tid:04d}/' in x['destination']]
    legacy=json.loads((OLD/'legacy_stage01_result.json').read_text());tracker=json.loads((OLD/'cotracker_upload.json').read_text());deps=json.loads((OLD/'policydeps03_result.json').read_text())
    receipt_path=HERE/f'shard{args.shard}_submission.json';old=None;generation=1;seconds=plan['max_seconds_per_shard'];completed={};archives=[]
    c=aoss_client()
    try:
        if args.recover:
            old=json.loads(receipt_path.read_text());actual=describe(old['name']);assert actual['uid']==old['uid'] and actual['state'] in {'FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}
            workers=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','get-workers',old['name'],'--workspace-name=share-space'],capture_output=True,text=True,check=True,timeout=45)
            assert old['uid'] in workers.stdout and 'Running' not in workers.stdout,'Old worker is not confirmed stopped'
            generation=old.get('generation',1)+1;assert generation<=plan['max_generation_per_shard']
            # Queue time and time spent stopped do not consume GPU runtime.
            # Retain a conservative wall-time fallback if platform dates are absent.
            if actual.get('start_time') and actual.get('complete_time'):
                runtime_used=max(0,(datetime.fromisoformat(actual['complete_time'].replace('Z','+00:00'))-
                    datetime.fromisoformat(actual['start_time'].replace('Z','+00:00'))).total_seconds())
            else:
                runtime_used=time.time()-old['submitted_at_unix']
            seconds=int(old['max_seconds']-runtime_used);assert seconds>=1800
            chain=[old]
            for previous in range(generation-2,0,-1):
                path=HERE/f'shard{args.shard}_generation{previous}.json'
                if path.exists():chain.append(json.loads(path.read_text()))
            cases=[(task['instance_ids'][i],condition) for i in shard['indices'] for condition in ['image','serf']]
            completed,archives=completed_archives(c,'260010165',chain,cases,lambda key:signed(c,'get_object',key))
        else:
            migration=plan.get('scaling8',{})
            chain=[x['receipt'] for x in migration.get('old_jobs',[]) if x['task_id']==tid and (x.get('episode_archive_count',0) or x.get('completed_episodes',0))]
            if chain:
                cases=[(task['instance_ids'][i],condition) for i in shard['indices'] for condition in ['image','serf']]
                completed,archives=completed_archives(c,'260010165',chain,cases,lambda key:signed(c,'get_object',key))
        prefix=f'behavior1k/serf-multitask-20261029/layout8/shard{args.shard}/generation{generation}/'
        cfg=dict(task_id=tid,output=f'/tmp/serf-mt8-1029-s{args.shard}-a{generation}',result_key=prefix+'status.json',mode='full',stage=legacy['root'],checks_file=legacy['root']+'/stage_result.json',asset_preflight_passed=True,
            data_bundle_get=signed(c,'get_object',data['key']),data_bundle_sha256=data['sha256'],
            source_get=signed(c,'get_object',source['key']),source_sha256=source['source_sha256'],
            cotracker_get=signed(c,'get_object',tracker['key']),cotracker_sha256=tracker['sha256'],
            site_path=OLD.as_posix().replace('/mnt/data1/faye/code/Behavior-1k Challenge/analysis','/mnt/afs/260010165/behavior_1k')+'/dev01/site',
            policy_dependency_path=deps['site'],check_serf_construction=True,
            background_assets=json.loads((OLD/'background_stage_result.json').read_text())['verified_background'],
            instance_states=states,maps_root='resolved-on-node',
            maps=[dict(instance=x['instance'],size=x['size'],sha256=x['sha256'],url=signed(c,'get_object',x['key'])) for x in maps],
            map_model_assets=[dict(relative=x['destination'].removeprefix('maps/'),size=x['size'],sha256=x['sha256'],url=signed(c,'get_object',x['key'])) for x in map_models],
            checkpoint_folders=checkpoint_folders,
            checkpoints=[dict(relative=x['destination'].removeprefix('checkpoints/'),size=x['size'],sha256=x['sha256'],url=signed(c,'get_object',x['key'])) for x in weights['uploaded'] if any('/'+f+'/' in x['destination'] for f in checkpoint_folders.values())],
            indices=shard['indices'],instance_ids=task['instance_ids'],resume_archives=archives,
            conditions_by_index={str(i):[condition for condition in ['image','serf'] if f'{task["instance_ids"][i]}_{condition}' not in completed] for i in shard['indices']},
            episode_puts={f'{task["instance_ids"][i]}_{condition}':signed(c,'put_object',prefix+f'episodes/{task["instance_ids"][i]}_{condition}.tar.gz') for i in shard['indices'] for condition in ['image','serf']},
            result_put=signed(c,'put_object',prefix+'status.json'),artifacts_put=signed(c,'put_object',prefix+'artifacts.tar.gz'))
        if old:
            receipt_path.rename(HERE/f'shard{args.shard}_generation{generation-1}.json')
        cfg['startup_diagnostics']=bool(tid==26 and args.recover)
        try:
            receipt=submit(f'b1k-serfmt8-1029-s{args.shard}-a{generation}',HERE/'gpu_worker.py',cfg,receipt_path,seconds,True)
        except BaseException:
            # Before an intent is written, submit has not called the create API.
            # After intent creation, preserve it for exact-name reconciliation.
            if old and not receipt_path.exists():
                write(receipt_path,old)
            raise
        receipt.update(shard=args.shard,task_id=tid,generation=generation,preserved_episodes=list(completed),previous_uid=old['uid'] if old else None);write(receipt_path,receipt)
    finally:c.close()


if __name__=='__main__':main()
