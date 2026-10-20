from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]


def save(state):
    p=HERE/'execution_state.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)


def main():
    plan=json.loads((HERE/'plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
    state={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'checkpoints':{}}
    save(state)
    rt=PROJECT/'analysis/organizer_rehearsal_1688977_20261016/runtime/policy-bundle/rootfs/mnt/afs/260010165/behavior_1k/runtime_20261007_1306'
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='0',
        CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',MKL_NUM_THREADS='16',
        LD_LIBRARY_PATH=str(rt/'envs/b1k-pi05/lib'),OPENPI_DATA_HOME=str(rt/'openpi-cache'),
        PYTHONPATH=str(PROJECT/'BEHAVIOR-VLAs/openpi/src')+':'+str(PROJECT/'BEHAVIOR-VLAs/openpi/packages/openpi-client/src'))
    try:
        for name,checkpoint in plan['checkpoints'].items():
            target=HERE/(name+'.json')
            if target.exists():raise ValueError('Result exists; inspect instead of rerunning')
            state['checkpoints'][name]={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat()};save(state)
            with (HERE/(name+'.log')).open('x') as log:
                p=subprocess.Popen(['taskset','-c','0-15',str(rt/'envs/b1k-pi05/bin/python3.11'),
                    str(HERE/'run_local_cpu.py'),'--checkpoint',checkpoint,'--out',str(target)],
                    env=env,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=p.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    raise TimeoutError(name+' exceeded frozen runtime')
            if code:raise RuntimeError(name+' nonzero exit '+str(code))
            result=json.loads(target.read_text());assert result['status']=='complete'
            state['checkpoints'][name].update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat())
            save(state)
        state['status']='complete'
    except Exception as e:
        state.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:save(state)


if __name__=='__main__':main()
