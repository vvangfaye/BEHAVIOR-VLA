"""Submit the fixed six episodes only after the 5.1 initialization gate passes."""
import argparse,base64,hashlib,json,os,shlex,subprocess,sys,zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
from public_test_admission import query_standard_capacity
cli='/home/faye/.local/bin/sco'
receipt=HERE/'eval02_submission.json'
parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,choices=[0,1],required=True);args=parser.parse_args()
if args.rank==0:
    assert not receipt.exists(),'Resolve existing evaluation jobs before resubmission'
else:
    current=json.loads(receipt.read_text());assert [x['rank'] for x in current['workers']]==[0]
    ready=json.loads((HERE/'eval02_policy_ready.json').read_text())
    assert ready['passed'] is True and ready['worker_uid']==current['workers'][0]['uid']
gate=json.loads((HERE/'init02_result.json').read_text());assert gate['complete'] is True
init=json.loads((HERE/'init02_submission.json').read_text())
actual_init=json.loads(subprocess.check_output([cli,'acp','jobs','describe',init['job'],'--workspace-name=share-space','-o','json'],text=True))
assert actual_init['uid']==init['uid'] and actual_init['ownership']['user_name']=='260010165'
assert actual_init['state'] in {'SUCCEEDED','FAILED','SUSPENDED','TERMINATED','STOPPED','CANCELED'}, 'Wait for the initialization worker to finish before allocating two evaluation GPUs'
previous=json.loads((HERE/'eval01_submission.json').read_text())
for old in previous['workers']:
    old_actual=json.loads(subprocess.check_output([cli,'acp','jobs','describe',old['name'],'--workspace-name=share-space','-o','json'],text=True))
    assert old_actual['uid']==old['uid'] and old_actual['ownership']['user_name']=='260010165'
    assert old_actual['state'] in {'FAILED','SUCCEEDED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}, 'Previous worker is still active'
    old_result=json.loads((HERE/'retrieved'/('eval01_worker'+str(old['rank'])+'_status.json')).read_text())
    assert not old_result['conditions'], 'Completed cases need an explicit resume assignment'
deps=json.loads((HERE/'policydeps01_result.json').read_text());assert deps['passed'] is True
assert deps['source_sha256']==init['source_sha256']
source=HERE/'runtime51_source.tar.gz';sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert sha==init['source_sha256'],'Source changed after the simulation gate'
uploaded=json.loads((HERE/'checkpoint_upload_progress.json').read_text())
assert uploaded['complete'] is True and len(uploaded['uploaded'])==32
stage=json.loads((HERE/'legacy_stage01_result.json').read_text());assert stage['probes']['sim']['passed']
assert json.loads((HERE/'dev01_result.json').read_text())['probes']['policy']['passed']
tracker=json.loads((HERE/'cotracker_upload.json').read_text())
background=json.loads((HERE/'background_stage_result.json').read_text())['verified_background']
assert {item['file'] for item in background}=={'sky.jpg','checkerboard.png'}
intent=current if args.rank==1 else {'scope':'3 paired configurations / 6 complete episodes','workers':[],'gpu_count':2,'max_worker_seconds':27000,'previous_attempt':'eval01_submission.json','policy_dependency_proof':'policydeps01_result.json','source_sha256':sha}
receipt.write_text(json.dumps(intent,indent=2)+'\n')
c=aoss_client()
try:
    def signed(method,key):return c.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=172800)
    source_key='behavior1k/serf-reproduction-20261025/init02/source-'+sha+'.tar.gz'
    assert c.head_object(Bucket='260010165',Key=source_key)['ContentLength']==source.stat().st_size
    checkpoints=[{'relative':x['key'].split('/checkpoints/',1)[1],'size':x['size'],'sha256':x['sha256'],'url':signed('get_object',x['key'])} for x in uploaded['uploaded']]
    for rank,indices in [(args.rank,[[0,2],[1]][args.rank])]:
        name=f'b1k-serf-eval-1025-w{rank}-a02'
        prefix=f'behavior1k/serf-reproduction-20261025/eval02/worker{rank}/'
        root=f'/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/eval02/worker{rank}'
        config={'stage':stage['root'],'checks_file':stage['root']+'/stage_result.json','site_path':'/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/dev01/site',
            'background_assets':background,'policy_dependency_path':deps['site'],
            'output':root,'mode':'full','indices':indices,'gate_file':init['root']+'/status.json',
            'source_get':signed('get_object',source_key),'source_sha256':sha,
            'cotracker_get':signed('get_object',tracker['key']),'cotracker_sha256':tracker['sha256'],
            'result_put':signed('put_object',prefix+'status.json'),'artifacts_put':signed('put_object',prefix+'artifacts.tar.gz'),
            'checkpoints':checkpoints,'checkpoint_folders':{'image':'pi_behavior_b1k_fast--50t_lora--task-0022','serf':'pi_serf_behavior_b1k_fast--4d_env_robot_feat_map--50t_lora--task-0022'}}
        code=(HERE/'gpu_init_worker.py').read_text()+'\nrun('+repr(config)+')\n'
        packed=base64.b64encode(zlib.compress(code.encode())).decode()
        command='timeout --signal=TERM --kill-after=60s 26900s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        capacity=query_standard_capacity(cli,dict(os.environ));(HERE/f'eval02_capacity_{rank}.json').write_text(json.dumps(capacity,indent=2))
        quota='reserved' if capacity['used_gpus']<capacity['standard_limit'] else 'spot'
        entry={'name':name,'rank':rank,'indices':indices,'quota_type':quota,'output':root,'result_key':prefix+'status.json','submission_attempted':True}
        intent['workers'].append(entry);receipt.write_text(json.dumps(intent,indent=2)+'\n')
        p=subprocess.run([cli,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=b1k-repeat1688-1019-batch-00',
            '--name='+name,'--job-name='+name,'--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type='+quota,
            '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--command='+command],
            capture_output=True,text=True,timeout=90)
        d=json.loads(subprocess.check_output([cli,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],text=True))
        assert d['ownership']['user_name']=='260010165' and d['resource_pool']['name']=='computing-cluster-5090-01g'
        entry.update(uid=d['uid'],state=d['state'],returncode=p.returncode);receipt.write_text(json.dumps(intent,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
finally:c.close()
