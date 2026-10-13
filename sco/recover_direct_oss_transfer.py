"""One diagnosed bootstrap replacement; keep the original allocation evidence."""
import ast,base64,json,re,shlex,subprocess,time,zlib
from pathlib import Path
from launch_retraining_parallel import environment,reserved_admission
from submit_retraining_cloud_probe import client,CLI
from launch_direct_oss_transfer import sign_items,enrich
from relay_retraining_stream import accepted_keys,collect_new
from launch_retraining_cloud import write


def recover(out):
    path=out/'direct_oss_launch.json';old=json.loads(path.read_text());assert old['uid']=='703ff0ae6f74479492cbcea48562c707' and old.get('generation')==1
    result=subprocess.run([CLI,'acp','jobs','describe',old['name'],'--workspace-name=share-space','-o','json'],env=environment(),capture_output=True,text=True,check=True,timeout=40)
    job=json.loads(result.stdout);assert job['uid']==old['uid'] and job['state']=='FAILED'
    workers=subprocess.run([CLI,'acp','jobs','get-workers',old['name'],'--workspace-name=share-space'],env=environment(),capture_output=True,text=True,check=True,timeout=40).stdout
    assert re.search(re.escape('pt-'+old['uid']+'-worker-0')+r'[^\n]*\|\s*Failed\s*\|',workers)
    from recover_retraining_parallel import startup_payload
    payload=startup_payload(job['roles'][0]['startup_script'])
    public,internal=client(),client(internal=True)
    launch=json.loads((out/'parallel/parallel_launch.json').read_text());frozen=json.loads((out/'source_archives.json').read_text())
    proofs={'by_key':{},'seen_objects':set()};collect_new(public,launch,frozen,proofs)
    accepted={r['source_key'] for r in accepted_keys(out)}
    active=set()
    for p in (out/'stream_transfers').glob('*.json'):
        value=json.loads(p.read_text())
        if value.get('phase')!='accepted':active.add(value['source_key'])
    candidates=[r for k,r in proofs['by_key'].items() if k not in accepted|active]
    assert candidates
    pilot=max(candidates,key=lambda r:r['target']['bytes'])
    queue={'fingerprint':frozen['fingerprint'],'training_allowed':False,'all_sources_prepared':False,'items':sign_items([enrich(pilot,public,internal,old['prefix'])])}
    public.put_object(Bucket='260010165',Key=old['queue_key'],Body=json.dumps(queue).encode());write(Path('/private/tmp/behavior-direct-oss-private/queue.json'),queue)
    status_key=old['prefix']+'generation-2/status.json'
    payload['root']=old['root'].removesuffix('-g1')+'-g2'
    payload['control']['status_put']=internal.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':status_key},ExpiresIn=43200)
    record={**old,'name':old['name'].removesuffix('-g1')+'-g2','root':payload['root'],'status_key':status_key,'generation':2,
            'attempt_history':old.get('attempt_history',[])+[{k:v for k,v in old.items() if k!='attempt_history'}],
            'diagnosis':'OSS signature includes Content-Type; urllib added an unsigned default. Independent old request reproduced HTTP403 and corrected headers HTTP200 with full byte readback. New queue signs and sends application/octet-stream consistently.'}
    record.pop('uid');record.pop('platform_state',None)
    source=Path(__file__).with_name('bootstrap_direct_oss_transfer.py').read_text()+'\nrun('+repr(payload)+')\n'
    encoded=base64.b64encode(zlib.compress(source.encode())).decode();command='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))'
    record['admission']=reserved_admission([],additional_gpus=0);record['state']='submission_intent';write(path,record)
    args=[CLI,'acp','jobs','create','--workspace-name=share-space','--aec2-name=share-cluster','--name='+record['name'],'--job-name='+record['name'],'--training-framework=pytorch','--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630','--worker-spec=n6ls.iu.i40.8c32g','--worker-nodes=1','--quota-type=spot','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--command=timeout --signal=TERM --kill-after=30 43200 python3 -c '+shlex.quote(command)]
    r=subprocess.run(args,env=environment(),capture_output=True,text=True,timeout=90);record.update(state='created_needs_describe' if r.returncode==0 else 'submission_unknown',cli_exit_code=r.returncode);write(path,record)
    if r.returncode:raise RuntimeError('Reconcile exact new job before another action')
    r=subprocess.run([CLI,'acp','jobs','describe',record['name'],'--workspace-name=share-space','-o','json'],env=environment(),capture_output=True,text=True,check=True,timeout=40);j=json.loads(r.stdout)
    record.update(uid=j['uid'],state='submitted',platform_state=j['state']);write(path,record)
    print(json.dumps({k:record[k] for k in ['name','uid','platform_state','gpu_count']}))


if __name__=='__main__':recover(Path('/Users/faye/code/Behavior-1k Challenge/analysis/retraining_20261013/preparation'))
