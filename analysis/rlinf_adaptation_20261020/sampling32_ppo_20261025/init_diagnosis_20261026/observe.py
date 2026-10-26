"""Observe the single diagnostic; download reports, never submit or restart jobs."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def main():
    lock=(HERE/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'observer_process.json').write_text(json.dumps({'pid':os.getpid(),'started_at':time.time()})+'\n')
    receipt=json.loads((HERE/'submission.json').read_text())
    end=datetime.fromisoformat(receipt['submitted_at']).timestamp()+7200
    previous=None
    client=aoss_client()
    try:
        while time.time()<end:
            try:
                raw=subprocess.check_output(['/home/faye/.local/bin/sco','acp','jobs','describe',
                    receipt['job'],'--workspace-name=share-space','-o','json'],text=True,timeout=40)
                job=json.loads(raw)
                assert job['uid']==receipt['uid'] and job['ownership']['user_name']=='260010165'
                state={'cloud_state':job['state'],'observed_at':datetime.now(timezone.utc).isoformat()}
                try:
                    state['progress']=json.loads(client.get_object(Bucket='260010165',
                        Key=receipt['aoss_prefix']+'status.json')['Body'].read())
                except client.exceptions.ClientError as error:
                    if error.response['Error']['Code'] not in ('NoSuchKey','404'):raise
                (HERE/'latest_observation.json').write_text(json.dumps(state,indent=2)+'\n')
                marker=json.dumps({k:v for k,v in state.items() if k!='observed_at'},sort_keys=True)
                if marker!=previous:print(json.dumps(state),flush=True);previous=marker
                if job['state'] in ('SUCCEEDED','FAILED','SUSPENDED','TERMINATED','STOPPED','CANCELED'):
                    response=client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'report.tar.gz')
                    assert response['ContentLength']<1024**3
                    path=HERE/'report.tar.gz'
                    with path.open('wb') as out:
                        for block in iter(lambda:response['Body'].read(1024**2),b''):out.write(block)
                    with tarfile.open(path) as tar:
                        for member in tar:
                            rel=Path(member.name)
                            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
                            if rel.suffix!='.json':continue
                            out=HERE/'retrieved'/rel;out.parent.mkdir(parents=True,exist_ok=True)
                            out.write_bytes(tar.extractfile(member).read())
                    result=HERE/'retrieved/result.json'
                    if result.exists():
                        data=json.loads(result.read_text())
                        print(json.dumps({'trials':[{k:r.get(k) for k in ('name','ready','elapsed','pre_abort_evidence')}
                            for r in data['trials']],'controls':data['controls_executed'],'updates':data['optimizer_steps']}),flush=True)
                    return
            except Exception as error:
                print(json.dumps({'observer_error':type(error).__name__,'detail':str(error)[:200]}),flush=True)
            threading.Event().wait(45)
    finally:client.close()


if __name__=='__main__':main()
