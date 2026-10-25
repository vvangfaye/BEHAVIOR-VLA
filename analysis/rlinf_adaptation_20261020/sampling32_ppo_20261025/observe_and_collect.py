"""Finite single observer. Stop only a failed experiment's dependent peer."""
from datetime import datetime
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

from status import HERE, aoss_client, status


def main():
    lock=(HERE/'observer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'observer_process.json').write_text(json.dumps({'pid':os.getpid(),'started_at':time.time(),
        'scope':'exact two recorded jobs; collect artifacts; stop dependent peer only after terminal failure; no new submissions'})+'\n')
    receipts=[json.loads((HERE/'attempt02'/f'r{i}/submission.json').read_text()) for i in (0,1)]
    deadline=max(datetime.fromisoformat(r['submitted_at']).timestamp()+r['max_seconds']+1800 for r in receipts)
    terminal={'SUCCEEDED','FAILED','SUSPENDED','TERMINATED','CANCELED','STOPPED'}
    previous=None; stop_requested=set()
    while time.time()<deadline:
        try:
            result=status(); rows=result['workers']
            compact=[{k:r.get(k) for k in ['rank','state','progress']} for r in rows]
            encoded=json.dumps(compact,sort_keys=True)
            if encoded!=previous:
                print(json.dumps({'at':result['observed_at'],'workers':compact},ensure_ascii=False),flush=True)
                previous=encoded
            if any(r['state'] in terminal-{'SUCCEEDED'} for r in rows):
                for row in rows:
                    if row['state'] in terminal or row['uid'] in stop_requested:
                        continue
                    # Recheck exact identity and state immediately before stopping.
                    checked=status()['workers'][row['rank']]
                    assert checked['uid']==row['uid']
                    if checked['state'] in terminal:
                        continue
                    reply=subprocess.run([str(Path.home()/'.local/bin/sco'),'acp','jobs','stop',row['name'],
                        '--workspace-name=share-space'],capture_output=True,text=True,timeout=45)
                    stop_requested.add(row['uid'])
                    (HERE/f'stop_peer_r{row["rank"]}.json').write_text(json.dumps({'uid':row['uid'],
                        'reason':'paired experiment has a terminal failure','returncode':reply.returncode})+'\n')
            if all(r['state'] in terminal for r in rows):
                client=aoss_client(); archives=[]
                try:
                    for receipt in receipts:
                        rank=receipt['rank'];key=receipt['aoss_prefix']+f'result_rank{rank}.tar.gz'
                        try:
                            response=client.get_object(Bucket='260010165',Key=key)
                        except client.exceptions.ClientError as error:
                            if error.response['Error']['Code'] in ('NoSuchKey','404'):
                                archives.append({'rank':rank,'missing':True});continue
                            raise
                        assert response['ContentLength'] < 1024**3
                        path=HERE/f'result_rank{rank}.tar.gz';digest=hashlib.sha256()
                        with path.open('wb') as target:
                            for block in iter(lambda:response['Body'].read(8*1024**2),b''):
                                target.write(block);digest.update(block)
                        with tarfile.open(path,'r:gz') as archive:
                            for item in archive:
                                relative=Path(item.name)
                                assert not relative.is_absolute() and '..' not in relative.parts and item.isfile()
                                if relative.suffix!='.json':continue
                                target=HERE/'retrieved'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                                target.write_bytes(archive.extractfile(item).read())
                        archives.append({'rank':rank,'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
                finally:
                    client.close()
                comparison=HERE/'retrieved/comparison.json'
                (HERE/'collection_receipt.json').write_text(json.dumps({'archives':archives,'workers':rows,
                    'comparison_present':comparison.is_file()},ensure_ascii=False,indent=2)+'\n')
                if comparison.exists():
                    from analyze import analyze
                    print(json.dumps({'analysis':analyze()},ensure_ascii=False),flush=True)
                return
        except Exception as error:
            print(json.dumps({'observation_error':type(error).__name__,'detail':str(error)[:500]}),flush=True)
        threading.Event().wait(40)
    print(json.dumps({'observer_deadline':True,'no_jobs_submitted':True}),flush=True)


if __name__=='__main__':
    main()
