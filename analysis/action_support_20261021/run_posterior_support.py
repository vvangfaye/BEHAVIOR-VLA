"""One immutable local sequence: two optimizer smokes, then two matched runs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE/'posterior_support_pilot_ada01'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def main():
    lock=(ROOT/'controller.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not all(read(HERE/'posterior_support_formula_check.json')[k] for k in
               ('mean_preserved','padding_unchanged','all_active_matches_native','autograd_gradcheck')):
        raise ValueError('Formula prerequisite failed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    # GPU1 must have no compute process. The existing baseline uses GPU0.
    usage=subprocess.check_output(['nvidia-smi','--id=1',
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if usage:raise RuntimeError('GPU1 already has a compute process: '+usage)
    for stage in ('smoke','formal'):
        plan_path=ROOT/('smoke_plan.json' if stage=='smoke' else 'plan.json')
        plan=read(plan_path)
        if digest(HERE/'posterior_support_formula_check.json') != plan['formula_receipt_sha256']:
            raise ValueError('Formula check receipt changed')
        for name,expected in plan['source_sha256'].items():
            if digest(HERE/name)!=expected:raise ValueError('Frozen source changed: '+name)
        for method in plan['methods']:
            out=ROOT/(stage+'_'+method);out.mkdir(exist_ok=False)
            started=time.monotonic()
            record=dict(stage=stage,method=method,controller_pid=os.getpid(),gpu=1,
                        started_at=datetime.now(timezone.utc).isoformat(),
                        plan_sha256=digest(plan_path),max_seconds=plan['max_seconds']+180,
                        no_retry=True)
            write(out/'launch.json',record)
            with (out/'worker.log').open('x') as stream:
                child=subprocess.Popen([sys.executable,'-u',str(HERE/'train_posterior_support.py'),
                    '--plan',str(plan_path),'--method',method,'--seed','17','--output',str(out)],
                    cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT)
                record['worker_pid']=child.pid
                write(out/'launch.json',record);write(ROOT/'state.json',dict(running=record))
                print(json.dumps(dict(event='launch',**record)),flush=True)
                try:code=child.wait(timeout=record['max_seconds'])
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:code=child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();code=child.wait()
            result=read(out/'result.json') if (out/'result.json').exists() else {}
            accepted=(code==0 and result.get('complete') is True
                      and result.get('completed_steps')==plan['steps']
                      and result.get('first_expert_attention_max_change',0)>0)
            if accepted:
                run=read(out/'run.json')
                accepted=(run['method']==method and run['seed']==17
                          and run['plan_sha256']==record['plan_sha256'])
            receipt=dict(**record,returncode=code,accepted=accepted,
                         elapsed_seconds=time.monotonic()-started,
                         finished_at=datetime.now(timezone.utc).isoformat(),
                         result_sha256=digest(out/'result.json') if result else None)
            write(out/'completion.json',receipt)
            write(ROOT/'state.json',dict(last_finished=receipt,running=None))
            print(json.dumps(dict(event='finished',**receipt)),flush=True)
            if not accepted:raise SystemExit(1)
    write(ROOT/'completion.json',dict(complete=True,methods=plan['methods'],
          final_steps=plan['steps'],finished_at=datetime.now(timezone.utc).isoformat(),
          scope='Matched offline VLA pilot only; no simulator evaluation.'))


if __name__=='__main__':main()
