"""Wait for the identified collector, then run the frozen CPU score analysis."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE=Path(__file__).resolve().parent
ROOT=HERE/'native_score_ada01'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[19]
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--collector-pid',type=int,required=True)
    args=p.parse_args();start_id=identity(args.collector_pid)
    if start_id is None:raise RuntimeError('Expected collector is not live at supervisor startup')
    if read(ROOT/'collection_state.json')['pid']!=args.collector_pid:
        raise RuntimeError('Collector PID disagrees with live state')
    lock=ROOT/'supervisor.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    child=None;started=time.monotonic()
    def save(name,**values):
        (ROOT/name).write_text(json.dumps(dict(observed_at=datetime.now(timezone.utc).isoformat(),
            supervisor_pid=os.getpid(),collector_pid=args.collector_pid,collector_start_ticks=start_id,
            **values),indent=2)+'\n')
    try:
        save('supervisor_state.json',stage='waiting_for_collector')
        while not (ROOT/'collection_complete.json').exists():
            if identity(args.collector_pid)!=start_id:
                raise RuntimeError('Identified collector exited without completion receipt')
            state=read(ROOT/'collection_state.json')
            if state.get('stage')=='failed':raise RuntimeError('Collector reported failure')
            if time.monotonic()-started>11100:raise TimeoutError('Collector wait cap')
            time.sleep(10)
        complete=read(ROOT/'collection_complete.json')
        if not complete['complete'] or complete['plan_sha256']!=sha(ROOT/'plan.json'):
            raise RuntimeError('Invalid collection completion receipt')
        for record in complete['contexts']:
            if sha(ROOT/record['file'])!=record['file_sha256']:
                raise RuntimeError('Collected file checksum mismatch')
        protocol=read(ROOT/'estimator_protocol.json')
        if protocol['script_sha256']!=sha(HERE/'fit_native_scores.py'):
            raise RuntimeError('Frozen estimator source changed')
        if (ROOT/'estimator_result.json').exists():
            raise RuntimeError('Estimator already has a result; refusing duplicate fit')
        environment=dict(os.environ);environment['CUDA_VISIBLE_DEVICES']=''
        with (ROOT/'estimator.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(HERE/'fit_native_scores.py')],
                env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save('supervisor_state.json',stage='fitting_cpu_score',estimator_pid=child.pid)
            print(json.dumps(dict(event='fitting_cpu_score',pid=child.pid)),flush=True)
            returncode=child.wait(timeout=1800)
        if returncode!=0:raise RuntimeError(f'Estimator exited with status {returncode}')
        result=read(ROOT/'estimator_result.json')
        if not result['complete']:raise RuntimeError('Estimator result is incomplete')
        save('supervisor_complete.json',complete=True,estimator_result_sha256=sha(ROOT/'estimator_result.json'),
             native_moment_gate_passed=result['native_moment_gate_passed'],
             elapsed_seconds=time.monotonic()-started,goal_complete=False)
        save('supervisor_state.json',stage='complete')
        print(json.dumps(dict(event='complete',native_moment_gate_passed=result['native_moment_gate_passed'])),flush=True)
    except Exception as exc:
        save('supervisor_state.json',stage='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,15)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,9)
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
