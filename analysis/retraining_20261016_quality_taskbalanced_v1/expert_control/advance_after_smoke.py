"""Finite single-writer handoff: accepted smoke -> one expert-control training job."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'smoke/bundle'))
import dispatch_support as dispatch
from nebula.submit_oss import OSSClient
from verify_smoke import verify_smoke

BAD={'FAIL','FAILED','KILLED','STOPPED','CANCELED','CANCELLED'}
GOOD={'FINISHED','SUCCESS','SUCCEEDED'}


def main():
    with (ROOT/'advance.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        launch_path=ROOT/'advance_launch.json'
        if launch_path.exists():
            launch=dispatch.read(launch_path)
            if launch.get('finished'):
                raise RuntimeError('Completed handoff cannot be relaunched')
        else:
            launch={'created_at':time.time(),'deadline':time.time()+86400,'smoke_iid':'240212'}
        launch.update(pid=os.getpid(),started_at=time.time())
        dispatch.save(launch_path,launch)
        client=OSSClient()
        receipt=dispatch.read(ROOT/'smoke/receipt.json')
        assert receipt['identifiers']==['240212']
        prefix=client.key(receipt['output'])
        while time.time()<launch['deadline']:
            if (ROOT/'STOP').exists():
                dispatch.save(ROOT/'advance_status.json',{'state':'stopped_by_file','cloud_jobs_unchanged':True})
                return
            if (ROOT/'formal_stage/intent.json').exists() or (ROOT/'formal_stage/receipt.json').exists():
                dispatch.save(ROOT/'advance_status.json',{'state':'formal_submission_already_attempted','observed_at':time.time()})
                return
            try:
                native=dispatch.status('240212')
                state=native['application_status']
                dispatch.save(ROOT/'advance_status.json',{'state':'waiting_for_smoke','native':native,'observed_at':time.time()})
                if state in BAD:
                    dispatch.save(ROOT/'advance_status.json',{'state':'smoke_failed_no_formal_submission','native':native})
                    return
                if state in GOOD:
                    proof=verify_smoke(client,prefix)
                    dispatch.save(ROOT/'smoke/verified_gpu_smoke.json',proof)
                    # This checks native status, proof identity, all source objects, and
                    # a single-use intent again under its own independent submit lock.
                    import submit_formal
                    submit_formal.main()
                    launch['finished']=True
                    dispatch.save(launch_path,launch)
                    dispatch.save(ROOT/'advance_status.json',{'state':'formal_submitted',
                        'receipt':dispatch.read(ROOT/'formal_stage/receipt.json'),'observed_at':time.time()})
                    return
            except Exception as error:
                # Observation errors are not terminal and never authorize a retry.
                dispatch.save(ROOT/'advance_error.json',{'error_type':type(error).__name__,'observed_at':time.time()})
                if (ROOT/'formal_stage/intent.json').exists():
                    dispatch.save(ROOT/'advance_status.json',{'state':'formal_outcome_requires_reconciliation','observed_at':time.time()})
                    return
            time.sleep(60)
        dispatch.save(ROOT/'advance_status.json',{'state':'deadline_reached_no_submission','observed_at':time.time()})


if __name__=='__main__':main()
