"""Read one bounded snapshot of this trial; never submit, retry, or stop a job."""
from pathlib import Path
import argparse
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
import importlib.util,json,os,re,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
R=Path('/Users/faye/Library/Caches/behavior-quality-taskbalanced-20261016')
s=importlib.util.spec_from_file_location('observer_dispatch',R/'dispatch_support.py')
d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
c=d.module('observer_oss',R/'formal_stage/bundle/nebula/submit_oss.py').OSSClient()
out={'observed_at':time.time(),'read_only':True,'stages':{}}
launch=d.read(R/'controller_launch.json')
probe=subprocess.run(['ps','-p',str(launch['pid']),'-o','command='],capture_output=True,text=True,timeout=10)
live=probe.returncode==0 and str(R/'launch_quality_training.py') in probe.stdout
out['controller']={'launch':launch,'live':live,'status':d.read(R/'gated_training_status.json')}
for stage,folder in __STAGE_SELECTION__:
 rp=R/folder/'receipt.json'
 if not rp.exists():continue
 receipt=d.read(rp)
 assert receipt['run_id'].startswith('b1k-quality-taskbalanced-1688977-20261016')
 assert len(receipt['identifiers'])==1 and receipt['identifiers'][0].isdigit()
 status=d.status(receipt['identifiers'][0])
 prefix=c.key(receipt['output'])+'/'
 assert prefix.startswith('wangfaye/result/behavior-challenge/quality_taskbalanced/b1k-quality-taskbalanced-1688977-20261016')
 item={'receipt':receipt,'status':status,'artifacts':{},'metadata':{}}
 def scan(p):
  page=c.bucket.list_objects(prefix=p,delimiter='/',max_keys=100)
  return {x.key:x.size for x in page.object_list if x.size>0}
 files=scan(prefix)
 runprefix=prefix
 if stage=='formal':
  job=d.read(R/'formal_stage/bundle/nebula/worker_job.json')
  assert job['attempt_id']=='stage5000-g0'
  runprefix=prefix+'attempts/'+job['attempt_id']+'/'
  files.update(scan(runprefix))
 for rank in range(2):files.update(scan(runprefix+'rank-'+str(rank)+'/'))
 keep={'started.json','hardware.json','runtime.json','checkpoint_source_verified.json',
       'original_cache_space.json','original_cache_verified.json','data_staged.json',
       '_SMOKE_SUCCESS.json','_TRAINING_STAGE_SUCCESS.json','_RANK_SUCCESS.json',
       'failure.json','checkpoint_restore_ok.json'}
 for key,size in files.items():
  rel=key[len(prefix):];item['artifacts'][rel]=size
  if key.rsplit('/',1)[-1] in keep and size<1024**2:
   item['metadata'][rel]=json.loads(c.read(key))
  elif key.endswith(('trainer.log','smoke_metrics.jsonl','training_metrics.jsonl')):
   res=c.bucket.get_object(key,byte_range=(max(0,size-12000),size-1))
   try:tail=res.read().decode(errors='replace')
   finally:res.resp.response.close()
   item['metadata'][rel+'_tail']=tail
 if not any(key.endswith('/trainer.log') for key in files):
  native=subprocess.run(['/Users/faye/.local/bin/ai-hub-cli','train','status','--iid',receipt['identifiers'][0],
                         '--detail','--token='+os.environ['AI_HUB_TOKEN']],capture_output=True,text=True,timeout=45)
  if '{' in native.stdout:
   obj=json.loads(native.stdout[native.stdout.index('{'):])
   item['metadata']['platform_worker_hosts']=[{'rank':w.get('rank'),'host':w.get('host_name')}
       for w in obj.get('data',{}).get('containers',[]) if w.get('role')=='worker']
   for worker in obj.get('data',{}).get('containers',[]):
    if worker.get('role')!='worker':continue
    path=worker.get('log_storage_path','')
    if not path.startswith('oss://ados-nebula-v2/nebula-one-proxy/remote_log/'):continue
    cmd=['/Users/faye/.local/bin/ai-hub-cli','train','logview','oss','--path='+path+'/stdout',
         '--bufferSize=8192','--fromHead=false','--token='+os.environ['AI_HUB_TOKEN']]
    try:
     log=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
     body=json.loads(log.stdout[log.stdout.index('{'):])
     item['metadata']['platform_worker_'+str(worker.get('rank'))+'_stdout_tail']=body.get('content','')[-7000:]
    except (ValueError,subprocess.TimeoutExpired):
     item['metadata']['platform_worker_'+str(worker.get('rank'))+'_log_status']='temporarily_unavailable'
 out['stages'][stage]=item
text=json.dumps(out,ensure_ascii=False)
for key in ('AI_HUB_TOKEN','OSS_ACCESS_ID','OSS_ACCESS_KEY'):
 if os.environ.get(key):text=text.replace(os.environ[key],'[REDACTED]')
text=re.sub(r'https?://[^\s"\\]+\?[^\s"\\]+','[signed-url-redacted]',text)
print(text)
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verified-ipv6', action='store_true')
    p.add_argument('--formal-only', action='store_true', help='Skip the already accepted smoke run.')
    args = p.parse_args()
    command = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=8', '-oStrictHostKeyChecking=yes']
    if args.verified_ipv6:
        command += ['-oHostKeyAlias=MBP16.local', '-oHostName=fe80::100e:8e6c:1f72:2de%%en7']
    command += ['macpro', 'cd "/Users/faye/code/Behavior-1k Challenge/BEHAVIOR-VLAs" && '
                'source nebula/config.sh >/dev/null 2>&1 && '
                '/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/env/bin/python -']
    stages = [('formal', 'formal_stage')] if args.formal_only else [('smoke', 'smoke'), ('formal', 'formal_stage')]
    remote = REMOTE.replace('__STAGE_SELECTION__', repr(stages))
    cp = subprocess.run(command, input=remote, capture_output=True, text=True, timeout=120)
    if cp.returncode:
        raise RuntimeError('Read-only SSH snapshot failed: ' + cp.stderr[-800:])
    snapshot = json.loads(cp.stdout)
    history = ROOT / 'observations'
    history.mkdir(exist_ok=True)
    text = json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n'
    (history / (str(int(snapshot['observed_at'])) + '.json')).write_text(text)
    tmp = ROOT / 'latest_runtime_observation.json.tmp'
    tmp.write_text(text)
    tmp.replace(ROOT / 'latest_runtime_observation.json')
    brief = {}
    for name, value in snapshot['stages'].items():
        details = {}
        for key, data in value['metadata'].items():
            if key.endswith('_tail'):
                details[key] = data[-2200:]
            elif not key.endswith(('hardware.json', 'started.json', 'runtime.json')):
                details[key] = data
        brief[name] = {'iid': value['status']['iid'], 'status': value['status']['application_status'],
                       'artifact_names': list(value['artifacts']), 'progress': details}
    print(json.dumps({'observed_at': snapshot['observed_at'], 'controller_live': snapshot['controller']['live'],
                      'controller_state': snapshot['controller']['status'].get('state'),
                      'stages': brief}, ensure_ascii=False))


if __name__ == '__main__':
    main()
