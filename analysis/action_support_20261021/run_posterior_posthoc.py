"""One bounded inference-only control on the idle local GPU0; no retry."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE/'posterior_posthoc_control_ada01'


def main():
    plan_path=ROOT/'plan.json';plan=json.loads(plan_path.read_text())
    if (ROOT/'launch.json').exists():raise ValueError('Control was already launched')
    live=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'],text=True).strip()
    if live:raise RuntimeError('GPU0 is not idle')
    record=dict(controller_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
                plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),gpu=0,
                max_seconds=plan['max_seconds'],no_retry=True)
    with (ROOT/'launch.json').open('x') as f:json.dump(record,f,indent=2)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    start=time.monotonic()
    with (ROOT/'worker.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(HERE/'evaluate_posterior_posthoc.py'),
                                '--plan',str(plan_path)],cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record['worker_pid']=child.pid
        (ROOT/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        try:code=child.wait(timeout=plan['max_seconds'])
        except subprocess.TimeoutExpired:
            child.terminate()
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();code=child.wait()
    result_path=ROOT/'result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else {}
    accepted=(code==0 and result.get('complete') is True and result.get('noise_streams_equal') is True
              and result.get('replay_max_abs_difference',1)>-1 and result.get('replay_max_abs_difference',1)<=2e-5)
    receipt=dict(**record,returncode=code,accepted=accepted,elapsed_seconds=time.monotonic()-start,
                 result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest() if result else None)
    (ROOT/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if not accepted:raise SystemExit(1)


if __name__=='__main__':main()
