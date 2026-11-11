"""Finite read-only acceptance watcher. Never submits, restarts, or trains."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
PREFIX='wangfaye/result/behavior-challenge/two_source_rft/b1k-two-source-cpu-20261015-1624/'
REMOTE='''import configparser,json,sys,time
from pathlib import Path
import oss2
request=json.load(sys.stdin)
assert request['prefix']=='wangfaye/result/behavior-challenge/two_source_rft/b1k-two-source-cpu-20261015-1624/'
c=configparser.ConfigParser(interpolation=None);c.read(Path.home()/'.ossutilconfig');v=c['Credentials']
b=oss2.Bucket(oss2.Auth(v['accesskeyid'],v['accesskeysecret']),'https://oss-cn-zhangjiakou.aliyuncs.com','traffic-model',connect_timeout=10)
result={}
for name in ['bootstrap_result.json','bootstrap_failure.json','bootstrap_process_failure.json','bootstrap_transport_failure.json','started.json','batch_progress.json','data_staged.json','original_cache_verified.json']:
 try:
  response=b.get_object(request['prefix']+name)
  try:
   body=response.read(2*1024**2+1);assert len(body)<=2*1024**2;result[name]=json.loads(body)
  finally:response.resp.response.close()
 except oss2.exceptions.NoSuchKey:pass
print(json.dumps(dict(observed_at=time.time(),files=result)))
'''


def save(path,value):
    target=HERE/path;temp=target.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(target)


def validate(proof):
    assert proof['passed'] and proof['real_two_source_batches']==2
    assert proof['gpus_requested']==0 and proof['model_loaded'] is False and proof['optimizer_initialized'] is False
    compiled=json.loads((HERE/'launch_bundle/inputs/compiled/preparation.json').read_text())
    assert proof['mixture_fingerprint']==compiled['mixture_fingerprint']
    assert len(proof['batches'])==2
    for batch in proof['batches']:
        assert set(batch['source_counts'])=={'0','3'} and min(batch['source_counts'].values())>0
        assert sum(batch['source_counts'].values())==192 and batch['full_episode_text_target_tokens']==0
        assert batch['supervised_text_target_tokens']>0 and batch['actions_shape']==[192,32,32]
        assert len(batch['images'])==3 and all(shape==[192,224,224,3] for shape in batch['images'].values())


def main():
    deadline=time.time()+10800
    with (HERE/'cpu_preflight_watcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save('cpu_preflight_watcher_launch.json',dict(pid=os.getpid(),started_at=time.time(),deadline_epoch=deadline,
            iid=238330,purpose='finite read-only receipts watcher',training_allowed=False,automatic_submission=False))
        failures=0
        while time.time()<deadline:
            if (HERE/'STOP').exists():return
            try:
                cp=subprocess.run(['ssh','-S','none','-o','BatchMode=yes','-o','ConnectTimeout=8','macpro',
                    shlex.join(['/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/env/bin/python','-c',REMOTE])],
                    input=json.dumps(dict(prefix=PREFIX)),capture_output=True,text=True,timeout=45)
                if cp.returncode:raise ConnectionError('Scoped CPU proof read failed')
                observed=json.loads(cp.stdout);files=observed['files'];failures=0
                save('cpu_preflight_observation.json',observed)
                for name,record in files.items():save('cpu_probe/'+name,record)
                errors=[name for name in files if 'failure' in name]
                if errors:
                    save('cpu_preflight_status.json',dict(iid=238330,state='needs_diagnostic',errors=errors,
                        observed_at=time.time(),training_started=False,automatic_resubmit=False));return
                if 'bootstrap_result.json' in files:
                    proof=files['bootstrap_result.json'];validate(proof)
                    accepted=dict(proof,accepted_at=time.time(),iid=238330)
                    save('gpu_smoke/cpu_batch_preflight.json',accepted)
                    save('cpu_preflight_status.json',dict(iid=238330,state='passed',real_two_source_batches=2,
                        observed_at=time.time(),training_started=False,GPU_smoke_pending=True))
                    handoff=json.loads((HERE/'HANDOFF.json').read_text());handoff.update(updated_at=time.time(),
                        real_decoded_CPU_batch_verified=True,CPU_probe_state='passed',new_GPU_training_started=False,
                        training_integration_ready=False)
                    save('HANDOFF.json',handoff);return
                save('cpu_preflight_status.json',dict(iid=238330,state='running' if files else 'queueing_or_starting',
                    observed_at=time.time(),available_reports=list(files),training_started=False))
            except Exception as e:
                failures+=1;save('cpu_preflight_observation_error.json',dict(error_type=type(e).__name__,
                    consecutive=failures,observed_at=time.time(),training_started=False))
                if failures>=6:return
            time.sleep(60)
        save('cpu_preflight_status.json',dict(iid=238330,state='observation_deadline',training_started=False))


if __name__=='__main__':main()
