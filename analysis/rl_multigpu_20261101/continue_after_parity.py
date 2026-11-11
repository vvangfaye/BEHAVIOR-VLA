"""Perform the authorized handoff only after the real-model parity job passes."""
import fcntl
import io
import json
import subprocess
import sys
import tarfile
import time
from deployment import HERE, ENV, describe, aoss_client, save, worker_plan


def main():
    with (HERE/'parity_continuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'handoff_complete.json').exists():
            return
        receipt=json.loads((HERE/'jobs/parity-a1/submission.json').read_text())
        client=aoss_client()
        try:
            deadline=time.time()+3900
            while time.time()<deadline:
                job=describe(receipt['job'])
                assert job['uid']==receipt['uid']
                if job['state']=='SUCCEEDED':
                    response=client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'result.tar.gz')
                    assert response['ContentLength']<32*1024**2
                    with tarfile.open(fileobj=io.BytesIO(response['Body'].read()),mode='r:gz') as archive:
                        member=archive.getmember('parity_gate.json')
                        result=json.loads(archive.extractfile(member).read())
                    assert result['passed'] and result['weights_restored'] and len(result['hardware'])==4
                    assert all(report['passed'] for report in result['reports'].values())
                    frozen=json.loads((HERE/'jobs/parity-a1/worker_plan.json').read_text())
                    assert {item['rank'] for item in result['hardware']}==set(range(4))
                    assert all(item['allocated_bytes']>1024**3 and item['cold_hash']==frozen['source_identity']['cold_trainable_hash']
                               for item in result['hardware'])
                    result['runtime_signature']=frozen['distributed']['runtime_signature']
                    assert worker_plan('control',1,True)['distributed']['runtime_signature']==result['runtime_signature']
                    result.update(job=receipt['job'],uid=receipt['uid'],verified_at=time.time())
                    save(HERE/'parity_result.json',result)
                    subprocess.run([sys.executable,str(HERE/'handoff.py')],env=ENV,check=True)
                    return
                if job['state'] in ('FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'):
                    save(HERE/'parity_failed.json',{'state':job['state'],'job':receipt['job'],'at':time.time()})
                    return
                time.sleep(20)
            save(HERE/'parity_failed.json',{'reason':'parity deadline','at':time.time()})
        finally:
            client.close()


if __name__=='__main__':main()
