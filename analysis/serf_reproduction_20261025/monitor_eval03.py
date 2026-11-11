"""Bounded controller: finish this six-episode attempt, never auto-retry failures."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    lock=(HERE/'eval03_monitor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=json.loads((HERE/'eval03_submission.json').read_text())
    expected_runner=receipt['runner_sha256']
    expected_launcher=hashlib.sha256((HERE/'launch_eval03.py').read_bytes()).hexdigest()
    started=time.time();deadline=started+8*3600
    status={'pid':os.getpid(),'started':started,'deadline':deadline,'phase':'monitoring','complete':False,'completed_episodes':0}
    c=aoss_client();failures=0
    try:
        while time.time()<deadline:
            try:
                receipt=json.loads((HERE/'eval03_submission.json').read_text())
                reports={}
                for worker in receipt['workers']:
                    key=f"behavior1k/serf-reproduction-20261025/eval03/worker{worker['rank']}/status.json"
                    try:data=c.get_object(Bucket='260010165',Key=key)['Body'].read()
                    except c.exceptions.ClientError as e:
                        if e.response['Error']['Code'] in {'NoSuchKey','404'}:continue
                        raise
                    report=json.loads(data);reports[worker['rank']]=report
                    write(HERE/'retrieved'/f"eval03_worker{worker['rank']}_status.json",report)
                    if report.get('error') or report.get('termination_signal'):
                        raise RuntimeError(f"Worker {worker['rank']} stopped: "+str(report.get('error',report.get('termination_signal'))))
                count=sum(len(r.get('conditions',{})) for r in reports.values())
                status.update(updated=time.time(),completed_episodes=count,workers={str(k):{'phase':v.get('phase'),'complete':v.get('complete'),'heartbeat':v.get('heartbeat',{})} for k,v in reports.items()})
                write(HERE/'eval03_monitor_status.json',status)
                if count==6 and len(reports)==2 and all(r.get('complete') for r in reports.values()):
                    artifacts_ready=True
                    for rank in [0,1]:
                        try:c.head_object(Bucket='260010165',Key=f'behavior1k/serf-reproduction-20261025/eval03/worker{rank}/artifacts.tar.gz')
                        except c.exceptions.ClientError as e:
                            if e.response['Error']['Code'] in {'NoSuchKey','404'}:artifacts_ready=False
                            else:raise
                    if not artifacts_ready:
                        status['phase']='waiting_for_artifact_upload';write(HERE/'eval03_monitor_status.json',status)
                        time.sleep(30)
                        continue
                    status['phase']='collecting';write(HERE/'eval03_monitor_status.json',status)
                    with (HERE/'eval03_collection.log').open('w') as log:
                        subprocess.run([sys.executable,str(HERE/'collect_results.py'),'--artifacts'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                        subprocess.run([sys.executable,str(HERE/'compare_results.py'),str(HERE/'retrieved'),'--output',str(HERE/'comparison.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
                    result=json.loads((HERE/'comparison.json').read_text());assert result['complete']
                    status.update(phase='complete',complete=True,comparison='comparison.json');write(HERE/'eval03_monitor_status.json',status)
                    return
                if len(receipt['workers'])==1 and 0 in reports:
                    report=reports[0];progress={}
                    try:progress=json.loads(report.get('heartbeat',{}).get('progress.json','{}'))
                    except (json.JSONDecodeError,TypeError):pass
                    if progress.get('steps',0)>=100 or report.get('conditions'):
                        first=receipt['workers'][0]
                        actual=json.loads(subprocess.check_output(['/home/faye/.local/bin/sco','acp','jobs','describe',first['name'],'--workspace-name=share-space','-o','json'],text=True,timeout=40))
                        assert actual['uid']==first['uid'] and actual['ownership']['user_name']=='260010165'
                        if actual['state'] not in {'RUNNING','STARTING','SUCCEEDED'}:raise RuntimeError('First worker is no longer running')
                        assert hashlib.sha256((HERE/'gpu_init_worker.py').read_bytes()).hexdigest()==expected_runner
                        assert hashlib.sha256((HERE/'launch_eval03.py').read_bytes()).hexdigest()==expected_launcher
                        write(HERE/'eval03_policy_ready.json',{'passed':True,'worker_uid':first['uid'],'verified_at':time.time(),'evidence':progress or list(report['conditions'])})
                        status['phase']='launching_worker1';write(HERE/'eval03_monitor_status.json',status)
                        with (HERE/'eval03_worker1_launch.log').open('w') as log:
                            subprocess.run([sys.executable,str(HERE/'launch_eval03.py'),'--rank','1'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
                        status['phase']='monitoring'
                # Periodic authoritative state checks catch abrupt pod deletion even
                # if the last persisted heartbeat has no Python error.
                if int((time.time()-started)/30)%4==0:
                    for worker in receipt['workers']:
                        actual=json.loads(subprocess.check_output(['/home/faye/.local/bin/sco','acp','jobs','describe',worker['name'],'--workspace-name=share-space','-o','json'],text=True,timeout=40))
                        assert actual['uid']==worker['uid']
                        if actual['state'] in {'FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}:
                            raise RuntimeError(f"Worker {worker['rank']} is {actual['state']}; no automatic resubmission")
                failures=0
            except (OSError,subprocess.TimeoutExpired) as e:
                failures+=1;status.update(transient_error=str(e),consecutive_transient_errors=failures)
                write(HERE/'eval03_monitor_status.json',status)
                if failures>=3:raise
            time.sleep(30)
        raise TimeoutError('Eight-hour controller deadline reached; no jobs resubmitted')
    except BaseException:
        status.update(phase='needs_attention',error=traceback.format_exc(),updated=time.time());write(HERE/'eval03_monitor_status.json',status)
        raise
    finally:c.close()


if __name__=='__main__':main()
