"""Run fixed task replications sequentially after the existing modem campaign."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
MODEM=HERE.parent/'modem_parallel_20261029'

def write(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def main():
    lock=(HERE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((HERE/'campaign.json').read_text())
    write(HERE/'coordinator_process.json',{'pid':os.getpid(),'started_at':time.time()})
    def status(**kw):write(HERE/'status.json',{'observed_at':datetime.now(timezone.utc).isoformat(),**kw})
    # The modem experiment retains its authorized 8-GPU concurrency. Never
    # allocate another RL campaign in parallel with it.
    while True:
        if (MODEM/'outcome.json').is_file():break
        if (MODEM/'stopped.json').is_file():
            # An unsuccessful modem run must not prevent independent task tests.
            # Confirm every submitted modem worker is terminal before proceeding.
            sys.path.insert(0,str(MODEM))
            from deploy import describe
            receipts=[json.loads(p.read_text()) for p in (MODEM/'jobs').glob('*/submission.json')]
            terminal=True
            for receipt in receipts:
                current=describe(receipt['job'])
                assert current['uid']==receipt['uid']
                terminal &= current['state'] in {'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}
            if terminal:
                write(HERE/'modem_incomplete_at_dispatch.json',json.loads((MODEM/'stopped.json').read_text()))
                break
            status(state='waiting_for_modem_workers_to_stop',new_task_jobs_submitted=0)
        else:status(state='queued_after_modem',new_task_jobs_submitted=0)
        threading.Event().wait(30)
    for task in plan['preregistered_tasks']:
        folder=Path(task['directory'])
        if (folder/'outcome.json').is_file():continue
        if (folder/'stopped.json').is_file():
            status(state='needs_attention',task=task['task'],reason='Previous task attempt stopped');return
        status(state='running',task=task['task'])
        with (folder/'controller.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,str(folder/'controller.py')],cwd=folder,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
        while child.poll() is None:threading.Event().wait(30)
        if child.returncode or not (folder/'outcome.json').is_file():
            status(state='needs_attention',task=task['task'],returncode=child.returncode);return
    outcomes={x['slug']:json.loads((Path(x['directory'])/'outcome.json').read_text()) for x in plan['preregistered_tasks']}
    status(state='complete',finished=True,outcomes=outcomes)


if __name__=='__main__':main()
