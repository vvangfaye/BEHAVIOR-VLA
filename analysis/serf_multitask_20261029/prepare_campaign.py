"""Pin public releases and stage only two new tasks for paired evaluation."""
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import os
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
OLD=HERE.parent/'serf_reproduction_20261025'
spec=importlib.util.spec_from_file_location('download_assets',OLD/'download_assets.py')
download=importlib.util.module_from_spec(spec);spec.loader.exec_module(download)
sys.path.insert(0,str(PROJECT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def item(repo,revision,record,dest,dataset=False):
    name=record.get('rfilename',record.get('path'));lfs=record.get('lfs') or {}
    sha=lfs.get('sha256',lfs.get('oid'))
    return dict(url=f"https://huggingface.co/{'datasets/' if dataset else ''}{repo}/resolve/{revision}/{name}",destination=dest,size=record['size'],sha256=sha,git_blob_sha1=None if sha else record.get('blobId',record.get('oid')))


def prepare():
    public=json.loads((OLD/'public_asset_manifest.json').read_text());extra=json.loads((OLD/'extra_asset_manifest.json').read_text())
    rows=list(csv.DictReader((OLD/'assets/instances/metadata/test_instances.csv').open()))
    tasks=[]
    for tid in [21,26]:
        row=next(x for x in rows if int(x['Task ID'])==tid)
        tasks.append(dict(task_id=tid,task_name=row['Task'],instance_ids=[int(x.strip()) for x in row['Public Test Instance IDs'].split(',')]))
    episodes=[json.loads(x) for x in (OLD/'assets/instances/metadata/episodes.jsonl').read_text().splitlines()]
    for task in tasks:
        lengths=[e['length'] for e in episodes if e['episode_index']//10000==task['task_id']]
        task['native_max_steps']=int(sum(lengths)/len(lengths)*2)
    plan_path=HERE/'plan.json'
    if not plan_path.exists():
        save(plan_path,dict(created_at=time.time(),authorization='User requested more SERF reproduction tasks; standing authorization covers routine research evaluations and necessary uploads.',tasks=tasks,new_episodes=80,total_episodes_with_reused_task22=120,reused_task22=str(HERE.parent/'serf_eval20_20261026/comparison.json'),seed=0,max_concurrent_gpus=2,quota_preference='Use free owner Reserved first; Spot when standard allocation is occupied.',gpu_hour_budget=288,max_seconds_per_shard=259200,max_generation_per_shard=2,shards=[dict(shard=i,task_id=t['task_id'],indices=list(range(part*10,(part+1)*10))) for i,(t,part) in enumerate((t,part) for part in [0,1] for t in tasks)],source_revisions=dict(model=public['model']['data']['sha'],maps=extra['map_info']['sha'],instances=extra['instances']['sha']),selection='All 20 official public test instances, in official CSV order; no score-based selection.',metrics=['paired mean final Q difference','whole task success','paired wins/ties/losses','paired bootstrap 95% CI per task','equal-task macro-average including original task22'],comparison_controls='Same task instance, shared saved physical initial state, seed 0, native task time limit, original author policy settings; restore completed cases only.',storage='Large weights, maps and runtime outputs on worker local scratch and private AOSS; no new large AFS copies.',scope='No training, no new human demonstrations, no task22 reruns.'))
    small=[];weights=[]
    for task in tasks:
        tid=task['task_id'];name=task['task_name'];ids=set(task['instance_ids'])
        for r in extra['instances']['siblings']:
            path=r['rfilename']
            selected=path.startswith('metadata/') or (f'_task_{name}_' in path and (path.endswith(f'_task_{name}_0_0_template.json') or path.endswith(f'_task_{name}_0_0_template-partial_rooms.json') or any(path.endswith(f'_0_{i}_template-tro_state.json') for i in ids)))
            if selected and not any(x['destination']=='instances/'+path for x in small):small.append(item(extra['instances']['id'],extra['instances']['sha'],r,'instances/'+path,True))
        for label in ['maps','map_models']:
            for r in json.loads((HERE/f'{label}_{tid}.json').read_text()):
                if r['type']!='file':continue
                path=r['path']
                selected=label=='maps' or path.endswith(('/config.yaml','/instance_id_to_name.json','/neural_points/robot/sampler.npz','/neural_points/robot/robot_neural_points.pt'))
                if selected:small.append(item('suk063/SERF',extra['map_info']['sha'],r,'maps/'+path,True))
        for r in public['model']['data']['siblings']:
            if f'task-{tid:04d}/' in r['rfilename']:weights.append(item('byeonghyunpak/SERF-VLA',public['model']['data']['sha'],r,'checkpoints/'+r['rfilename']))
    assert len({x['destination'].split('/')[1] for x in weights})==4
    for folder in {x['destination'].split('/')[1] for x in weights}:
        assert any(x['destination']==f'checkpoints/{folder}/params/_METADATA' for x in weights)
    assert sum('/eval/scene_' in x['destination'] for x in small)==40
    save(HERE/'small_manifest.json',{'items':small,'bytes':sum(x['size'] for x in small)})
    save(HERE/'checkpoint_manifest.json',{'items':weights,'bytes':sum(x['size'] for x in weights)})
    return small,weights


def fetch_and_upload(items,label):
    root=HERE/'assets';root.mkdir(exist_ok=True)
    client=aoss_client();completed=[]
    old_items=json.loads((OLD/'download_manifest.json').read_text())['items']
    by_dest={i['destination']:i for i in old_items}
    def one(i):
        target=root/i['destination'];legacy=OLD/'assets'/i['destination']
        if i['destination'] in by_dest and download.digest_file(legacy,i) and not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(legacy.resolve())
        if not target.is_symlink() and not download.digest_file(target,i):
            if i['size']>=8*1024**2:
                target.parent.mkdir(parents=True,exist_ok=True)
                partial=target.with_name(target.name+'.part')
                command=['aria2c','--console-log-level=error','--summary-interval=0','--download-result=hide','--allow-overwrite=true','--auto-file-renaming=false','--file-allocation=none','--continue=true','--max-connection-per-server=8','--split=8','--min-split-size=8M','--timeout=60','--connect-timeout=30','--max-tries=5','--retry-wait=2','--dir='+str(target.parent),'--out='+partial.name]
                proxy=os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
                if proxy:command.append('--all-proxy='+proxy)
                command.append('--ca-certificate=/etc/ssl/certs/ca-certificates.crt')
                command.append(i['url'])
                result=subprocess.run(command,capture_output=True,text=True,timeout=7200)
                if result.returncode:raise RuntimeError('Parallel checkpoint download failed: '+i['destination']+' exit='+str(result.returncode))
                assert download.digest_file(partial,i),'Parallel checkpoint checksum mismatch'
                partial.replace(target)
            else:download.fetch(root,i)
        assert download.digest_file(target,i)
        hsh=hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):hsh.update(block)
        sha=hsh.hexdigest()
        key='behavior1k/serf-multitask-20261029/assets/'+i['destination']
        exists=False
        try:
            h=client.head_object(Bucket='260010165',Key=key);exists=h['ContentLength']==i['size'] and h.get('Metadata',{}).get('sha256')==sha
        except client.exceptions.ClientError as e:
            if e.response['Error']['Code'] not in {'404','NoSuchKey'}:raise
        if not exists:client.upload_file(str(target),'260010165',key,ExtraArgs={'Metadata':{'sha256':sha}})
        h=client.head_object(Bucket='260010165',Key=key);assert h['ContentLength']==i['size'] and h['Metadata']['sha256']==sha
        return dict(destination=i['destination'],key=key,size=i['size'],sha256=sha)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(one,items):
                completed.append(result);save(HERE/f'{label}_upload.json',dict(complete=len(completed)==len(items),uploaded=completed));print(label,len(completed),'/',len(items),result['destination'],flush=True)
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['small','checkpoints','manifest'],default='small');args=parser.parse_args();small,weights=prepare()
    if args.part!='manifest':fetch_and_upload(small if args.part=='small' else weights,args.part)
