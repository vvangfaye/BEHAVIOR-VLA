"""Wait locally for two free standard slots, submit one fixed two-worker job."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from public_test_admission import query_standard_capacity


def write(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    if (HERE/'CANCELLED_BY_USER').exists():
        print('Cancelled by user; automatic submission is disabled.',flush=True)
        return 0
    lock=(HERE/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=HERE/'queue.json'
    queue=json.loads(path.read_text()) if path.exists() else {'created_at':time.time(),'deadline':time.time()+86400}
    queue.update(pid=os.getpid(),state='waiting_standard',requested_gpus=2,cloud_job_submitted=False)
    write(path,queue)
    previous=None
    while time.time()<queue['deadline']:
        if (HERE/'CANCELLED_BY_USER').exists():
            queue.update(state='cancelled_by_user',auto_submit_enabled=False)
            write(path,queue)
            return 0
        if (HERE/'submission_pair.json').exists():
            receipt=json.loads((HERE/'submission_pair.json').read_text())
            if receipt.get('uid'):
                queue.update(state='submitted',cloud_job_submitted=True,job=receipt['job'],uid=receipt['uid'])
                write(path,queue);break
            queue.update(state='submission_needs_inspection');write(path,queue)
            print('Existing submission intent needs inspection; no duplicate submission.',flush=True);return 1
        try:
            capacity=query_standard_capacity('/home/faye/.local/bin/sco',dict(os.environ))
            write(HERE/'latest_capacity.json',capacity)
            free=capacity['standard_limit']-capacity['used_gpus']
            queue.update(checked_at=datetime.now(timezone.utc).isoformat(),free_standard_gpus=free)
            write(path,queue)
            if free!=previous:
                print(json.dumps({'state':'waiting_standard','free_standard_gpus':free,'required':2}),flush=True)
                previous=free
            if free>=2:
                reply=subprocess.run([sys.executable,str(HERE/'deploy_pair.py')],capture_output=True,text=True,timeout=240)
                if reply.stdout:print(reply.stdout.strip(),flush=True)
                if reply.returncode==75:
                    continue
                if reply.returncode!=0:
                    queue.update(state='submission_needs_inspection',exit_code=reply.returncode)
                    write(path,queue)
                    (HERE/'submission_error.log').write_text(reply.stderr[-6000:])
                    return 1
                continue
        except Exception as error:
            queue.update(last_check_error=type(error).__name__)
            write(path,queue)
            print(json.dumps({'capacity_check_error':type(error).__name__,'no_submission':True}),flush=True)
        threading.Event().wait(300)
    else:
        queue.update(state='wait_deadline_reached');write(path,queue)
        print('Local queue deadline reached; no GPU reserved.',flush=True);return 0
    active_path=HERE.parent.parent/'active_run.json'
    active=json.loads(active_path.read_text())
    workers=[]
    for rank in (0,1):
        receipt=json.loads((HERE/f'r{rank}/submission.json').read_text())
        workers.append({'rank':rank,'name':receipt['job'],'uid':receipt['uid'],'state':receipt['state'],'quota':'RESERVED'})
    plan=json.loads((HERE/'plan.json').read_text())
    active.update(status='scheduling',workers=workers,local_run_directory=str(HERE),plan_path=str(HERE/'plan.json'),
        remote_run_directory=str(Path(plan['output']).parent),queue_path=str(path),
        max_worker_seconds=plan['max_job_seconds'],completed_training_episodes=39,completed_optimizer_steps=17,
        recovery_status='Restore 39 completed episodes; restart only missing or preempted episodes; 13 PPO updates remain')
    write(active_path,active)
    print(json.dumps({'state':'submitted','workers':workers}),flush=True)
    return subprocess.call([sys.executable,'-u',str(HERE/'observe_and_collect.py')])


if __name__=='__main__':raise SystemExit(main())
