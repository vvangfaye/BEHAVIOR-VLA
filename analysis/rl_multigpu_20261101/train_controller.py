"""Own only the four learner jobs; existing controllers keep their evaluations."""
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import fcntl
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
from deployment import HERE, COMPARISON, WATER, CLI, ENV, describe, save, submit, aoss_client, register_with_evaluator

TERMINAL = {'SUCCEEDED', 'FAILED', 'SUSPENDED', 'STOPPED', 'CANCELED', 'TERMINATED'}
VARIANTS = ['control', 'gamma1_ppo', 'balanced_ppo', 'group_outcome']


def workers_exited(row):
    job = describe(row['job'])
    assert job['uid'] == row['uid']
    if job['state'] not in TERMINAL:
        return False
    reply = subprocess.run([CLI, 'acp', 'jobs', 'get-workers', row['job'], '--workspace-name=share-space'],
                           env=ENV, capture_output=True, text=True, timeout=40)
    assert reply.returncode == 0
    return not any(word in reply.stdout for word in ['Running', 'Pending', 'Terminating'])


def all_active_gpus():
    receipts = {}
    for root in [HERE, COMPARISON, WATER]:
        for path in (root/'jobs').glob('*/submission.json'):
            row = json.loads(path.read_text())
            if 'uid' in row:
                receipts[row['uid']] = row
    for path in HERE.glob('communication_probe_submission*.json'):
        row=json.loads(path.read_text())
        if 'uid' in row:
            receipts[row['uid']]=row
    cache_path=HERE/'usage_cache.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def read(row):
        cached=cache.get(row['uid'])
        if cached and cached['state'] in ('SUCCEEDED','FAILED'):
            return row,cached
        job=describe(row['job'])
        return row,{k:job.get(k) for k in ['name','uid','state','start_time','complete_time','update_time']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        queried=list(pool.map(read,receipts.values()))
    total = 0
    seconds = 0.0
    for row,job in queried:
        assert job['uid'] == row['uid']
        cache[row['uid']]=job
        if job['state'] not in TERMINAL:
            total += row.get('gpu_count', 1)
            seconds += row.get('max_seconds', 43200)*row.get('gpu_count',1)
        elif job.get('start_time'):
            parse=lambda s:datetime.fromisoformat(s.replace('Z','+00:00'))
            end=job.get('complete_time') or job.get('update_time')
            if end:
                seconds += max(0.,(parse(end)-parse(job['start_time'])).total_seconds())*row.get('gpu_count',1)
    save(cache_path,cache)
    return total, seconds


def may_retry(client, row):
    if row['state']=='SUSPENDED':
        return True
    if row.get('progress',{}).get('error_type') not in ('KeyboardInterrupt','ExecutionStopped'):
        return False
    try:
        response=client.get_object(Bucket='260010165',Key=row['aoss_prefix']+'result.tar.gz')
        assert response['ContentLength']<256*1024**2
        with tarfile.open(fileobj=io.BytesIO(response['Body'].read()),mode='r:gz') as archive:
            for item in archive:
                if item.isfile() and item.name.startswith('failures/') and f"_a{row['attempt']}_rank" in item.name:
                    failure=json.loads(archive.extractfile(item).read())
                    if failure['type'] not in ('KeyboardInterrupt','ExecutionStopped'):
                        return False
        return True
    except client.exceptions.ClientError:
        return False


def main():
    with (HERE/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        handoff=json.loads((HERE/'handoff_complete.json').read_text())
        assert handoff['old_workers_exited']
        precompleted=set(handoff.get('precompleted',[]))
        client = aoss_client()
        try:
            while True:
                rows = []
                for path in (HERE/'jobs').glob('t-*/submission.json'):
                    row = json.loads(path.read_text())
                    job = describe(row['job'])
                    if 'uid' not in row:
                        row['uid']=job['uid']
                        save(path,row)
                    assert job['uid'] == row['uid']
                    register_with_evaluator(row,json.loads(path.with_name('worker_plan.json').read_text()))
                    row['state'] = job['state']
                    try:
                        response = client.get_object(Bucket='260010165', Key=row['aoss_prefix']+'status.json')
                        row['progress'] = json.loads(response['Body'].read())
                    except client.exceptions.ClientError as error:
                        if error.response['Error']['Code'] not in ['NoSuchKey', '404']:
                            raise
                    save(path.with_name('latest.json'), row)
                    rows.append(row)
                latest = {v: max((r for r in rows if r['variant'] == v), key=lambda r: r['attempt'], default=None)
                          for v in VARIANTS}
                for variant in precompleted:
                    latest[variant]={'state':'SUCCEEDED','variant':variant,'reused_existing_complete_training':True}
                save(HERE/'status.json', {'at': time.time(), 'learners': latest,
                    'active_training_gpus': sum(4 for r in latest.values() if r and r['state'] not in TERMINAL)})
                if all(r and r['state'] == 'SUCCEEDED' for r in latest.values()):
                    return
                launched = False
                for variant, row in latest.items():
                    if row and row['state'] not in TERMINAL or row and row['state'] == 'SUCCEEDED':
                        continue
                    if row:
                        if row['attempt'] >= 4 or not may_retry(client,row):
                            continue
                        if not workers_exited(row):
                            continue
                    active, seconds = all_active_gpus()
                    if active+4 > 32:
                        break
                    if seconds+4*43200 > 512*3600:
                        save(HERE/'budget_stop.json', {'at':time.time(),'gpu_hours_bound':seconds/3600})
                        return
                    submit(variant, 3 if row is None else 4)
                    launched = True
                    break
                if not launched:
                    time.sleep(30)
        finally:
            client.close()


if __name__ == '__main__':
    deadline=time.time()+48*3600
    while time.time()<deadline:
        try:
            main()
            break
        except BlockingIOError:
            raise
        except Exception as error:
            save(HERE/'controller_error.json', {'at':time.time(),'type':type(error).__name__})
            time.sleep(30)
