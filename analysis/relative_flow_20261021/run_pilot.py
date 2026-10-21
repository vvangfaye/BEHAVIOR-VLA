"""Two local GPUs, four immutable arms, smoke prerequisites, no retry."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE/'pilot_ada01'
LOCK=threading.Lock()
RUNNING={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def run(stage,method,gpu):
    plan_path=ROOT/('smoke_plan.json' if stage=='smoke' else 'plan.json')
    plan=read(plan_path)
    for path,expected in plan['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('Frozen source changed: '+path)
    out=ROOT/(stage+'_'+method)
    if out.exists():raise ValueError('Refusing to retry an existing arm')
    out.mkdir()
    record=dict(stage=stage,method=method,gpu=gpu,controller_pid=os.getpid(),
                started_at=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(plan_path),
                max_seconds=plan['max_seconds']+180,no_retry=True)
    started=time.monotonic()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (out/'worker.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),'--plan',str(plan_path),
                                '--method',method,'--output',str(out/'output')],
                               cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record['worker_pid']=child.pid
        write(out/'launch.json',record)
        with LOCK:
            RUNNING[method]=record;write(ROOT/'state.json',dict(running=dict(RUNNING)))
            print(json.dumps(dict(event='launch',**record)),flush=True)
        try:code=child.wait(timeout=record['max_seconds'])
        except subprocess.TimeoutExpired:
            child.terminate()
            try:code=child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();code=child.wait()
    result_path=out/'output/result.json'
    result=read(result_path) if result_path.exists() else {}
    accepted=(code==0 and result.get('complete') is True and result.get('completed_steps')==plan['steps']
              and result.get('first_attention_max_change',0)>0
              and result.get('training_expert_forwards')==plan['steps']*plan['groups_per_update']*5
              and result.get('training_prefix_forwards')==plan['steps']*plan['groups_per_update']*3
              and result.get('plan_sha256')==record['plan_sha256'])
    receipt=dict(**record,returncode=code,accepted=accepted,elapsed_seconds=time.monotonic()-started,
                 finished_at=datetime.now(timezone.utc).isoformat(),
                 result_sha256=sha(result_path) if result_path.exists() else None)
    write(out/'completion.json',receipt)
    with LOCK:
        RUNNING.pop(method,None);write(ROOT/'state.json',dict(running=dict(RUNNING),last_finished=receipt))
        print(json.dumps(dict(event='finished',**receipt)),flush=True)
    return receipt


def main():
    ROOT.mkdir(exist_ok=True)
    path=ROOT/'controller_launch.json'
    with path.open('x') as stream:
        json.dump(dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()),stream)
    for gpu in [0,1]:
        rows=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'],text=True).strip()
        if rows:raise RuntimeError('GPU is occupied: '+str(gpu))
    if not read(HERE/'loss_check.json')['passed'] or not read(HERE/'mathematical_check.json')['passed']:
        raise ValueError('Construction prerequisite failed')
    for stage in ['smoke','formal']:
        for wave in [('extra_fm','relative_difference'),('mean_difference','deltafm_aux')]:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run,stage,method,gpu) for gpu,method in enumerate(wave)]
                receipts=[f.result() for f in futures]
            if not all(r['accepted'] for r in receipts):raise SystemExit(1)
        results=[read(ROOT/(stage+'_'+method)/'output/result.json') for method in read(ROOT/'plan.json')['methods']]
        runs=[read(ROOT/(stage+'_'+method)/'output/run.json') for method in read(ROOT/'plan.json')['methods']]
        for key in ['data_trace_sha256','noise_trace_sha256','training_expert_forwards','training_prefix_forwards']:
            if len({x[key] for x in results})!=1:raise ValueError('Arms differ: '+key)
        if len({x['initial_attention_sha256'] for x in runs})!=1:raise ValueError('Initial weights differ')
        write(ROOT/(stage+'_matched_protocol.json'),dict(passed=True,stage=stage,
              initial_attention_sha256=runs[0]['initial_attention_sha256'],
              **{key:results[0][key] for key in ['data_trace_sha256','noise_trace_sha256',
                                               'training_expert_forwards','training_prefix_forwards']}))
    write(ROOT/'completion.json',dict(complete=True,finished_at=datetime.now(timezone.utc).isoformat(),
          scope='Four-arm exploratory offline VLA training only; no simulation.'))


if __name__=='__main__':main()
