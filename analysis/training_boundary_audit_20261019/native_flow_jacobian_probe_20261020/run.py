"""Run one Jacobian check under the same bounded local CPU environment."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
RUNTIME=PROJECT/'analysis/organizer_rehearsal_1688977_20261016/runtime/policy-bundle/rootfs/mnt/afs/260010165/behavior_1k/runtime_20261007_1306'
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='0',
    CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',MKL_NUM_THREADS='16',
    LD_LIBRARY_PATH=str(RUNTIME/'envs/b1k-pi05/lib'),OPENPI_DATA_HOME=str(RUNTIME/'openpi-cache'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    PYTHONPATH=str(PROJECT/'BEHAVIOR-VLAs/openpi/src')+':'+str(PROJECT/'BEHAVIOR-VLAs/openpi/packages/openpi-client/src'))
entry=('import asyncio,runpy,sys;sys.path.insert(0,'+repr(str(HERE.parent/'flow_endpoint_screen_20261020'))+');'
       'from run_local_cpu import PeriodicWakeupPolicy;asyncio.set_event_loop_policy(PeriodicWakeupPolicy());'
       'runpy.run_path('+repr(str(HERE/'probe.py'))+",run_name='__main__')")
receipt=HERE/'execution.json'
status={'kind':'native_noise_jacobian_conditioning_check','cpus':list(range(16)),
        'maximum_seconds':600,'training':False,'simulation':False,'uploads':False,'status':'starting'}
with receipt.open('x') as stream: json.dump(status,stream)
start=time.monotonic()
with (HERE/'run.log').open('x') as log:
    process=subprocess.Popen([str(RUNTIME/'envs/b1k-pi05/bin/python3.11'),'-u','-c',entry],
        cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status.update(status='running',pid=process.pid)
    receipt.write_text(json.dumps(status,indent=2)+'\n')
    try:
        rc=process.wait(timeout=600)
        status.update(status='complete' if rc==0 else 'failed',exit_code=rc)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        status.update(status='timed_out',exit_code=process.returncode)
status['elapsed_seconds']=time.monotonic()-start
receipt.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status))
sys.exit(0 if status['status']=='complete' else 1)
