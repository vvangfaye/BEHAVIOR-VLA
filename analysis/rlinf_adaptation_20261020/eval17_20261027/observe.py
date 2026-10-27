"""Read-only finite observer for two evaluation jobs; never submit or resume training."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
from eval_utils import aggregate

TERMINAL = {'SUCCEEDED','FAILED','SUSPENDED','TERMINATED','CANCELED','CANCELLED','STOPPED'}


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def status(client):
    workers = []
    for rank in (0,1):
        receipt = json.loads((HERE/f'r{rank}/submission.json').read_text())
        p = subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','describe',receipt['job'],
                            '--workspace-name=share-space','-o','json'],
                           capture_output=True,text=True,check=True,timeout=40)
        job = json.loads(p.stdout)
        assert job['uid']==receipt['uid'] and job['ownership']['user_name']=='260010165'
        row = {k:receipt[k] for k in ('job','uid','rank','quota')}
        row['state'] = job['state']
        try:
            response = client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'status.json')
            assert response['ContentLength'] < 2*1024**2
            row['progress'] = json.loads(response['Body'].read())
        except client.exceptions.ClientError as error:
            if error.response['Error']['Code'] not in ('NoSuchKey','404'):
                raise
        workers.append(row)
    pairs = [p for w in workers for p in w.get('progress',{}).get('comparison',{}).get('pairs',[])]
    assert len({(p['instance'],p['seed']) for p in pairs})==len(pairs)
    pairs.sort(key=lambda x:(x['instance'],x['seed']))
    def summary(selected, expected):
        return {'expected_pairs':expected,'completed_pairs':len(selected),
                'comparison_complete':len(selected)==expected,
                'comparison_valid':bool(selected) and all(p['initial_state_valid'] for p in selected),
                'baseline':aggregate([p['baseline'] for p in selected]),
                'final':aggregate([p['final'] for p in selected]),
                'success_to_failure':sum(p['success_delta']<0 for p in selected),
                'failure_to_success':sum(p['success_delta']>0 for p in selected),
                'pairs':selected}
    report = {'observed_at':datetime.now(timezone.utc).isoformat(),'workers':workers,
              'new_optimizer_steps':0,'checkpoint_updates':17,
              'comparison':summary(pairs,30),
              'first_six':summary([p for p in pairs if p['instance'] in (301,302)],6)}
    write(HERE/'latest_status.json',report)
    write(HERE/'comparison.json',report['comparison'])
    write(HERE/'first_six_comparison.json',report['first_six'])
    return report


def collect(client, rank):
    marker=HERE/f'r{rank}/collection.json'
    if marker.exists():return
    receipt=json.loads((HERE/f'r{rank}/submission.json').read_text())
    response=client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'result.tar.gz')
    assert response['ContentLength'] < 256*1024**2
    archive_path=HERE/f'r{rank}/result.tar.gz'
    data=response['Body'].read()
    archive_path.write_bytes(data)
    target_root=HERE/'retrieved'/f'rank{rank}'
    for_extract=[]
    with tarfile.open(archive_path,'r:gz') as archive:
        for item in archive:
            relative=Path(item.name)
            assert not relative.is_absolute() and '..' not in relative.parts and item.isfile()
            assert relative.suffix=='.json'
            contents=archive.extractfile(item).read()
            target=target_root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==contents
            else:target.write_bytes(contents)
            for_extract.append(str(relative))
    write(marker,{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'files':for_extract})


def main():
    lock=(HERE/'observer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipts=[json.loads((HERE/f'r{r}/submission.json').read_text()) for r in (0,1)]
    deadline=max(datetime.fromisoformat(r['submitted_at']).timestamp()+r['max_seconds']+1800 for r in receipts)
    write(HERE/'observer_process.json',{'pid':os.getpid(),'started_at':time.time(),
          'read_only':True,'automatic_submissions':False,'deadline':deadline})
    client=aoss_client()
    previous=None
    try:
        while time.time()<deadline:
            try:
                report=status(client)
                compact=[{'rank':w['rank'],'state':w['state'],
                          'stage':w.get('progress',{}).get('stage'),
                          'completed':w.get('progress',{}).get('completed_new_episodes',0)} for w in report['workers']]
                encoded=json.dumps(compact,sort_keys=True)
                if encoded!=previous:
                    print(json.dumps({'at':report['observed_at'],'workers':compact,
                                      'paired':report['comparison']['completed_pairs']},ensure_ascii=False),flush=True)
                    previous=encoded
                for worker in report['workers']:
                    if worker['state'] in TERMINAL:
                        try:collect(client,worker['rank'])
                        except client.exceptions.ClientError as error:
                            if error.response['Error']['Code'] not in ('NoSuchKey','404'):raise
                if all(w['state'] in TERMINAL for w in report['workers']):
                    write(HERE/'completion.json',{'state':'complete' if report['comparison']['comparison_complete'] else 'incomplete',
                          'workers':compact,'comparison':report['comparison'],
                          'artifacts_collected':[r for r in (0,1) if (HERE/f'r{r}/collection.json').exists()]})
                    return
            except Exception as error:
                print(json.dumps({'observation_error':type(error).__name__}),flush=True)
            threading.Event().wait(45)
        write(HERE/'observer_deadline.json',{'at':time.time(),'no_resubmissions':True})
    finally:
        client.close()


if __name__=='__main__':
    main()
