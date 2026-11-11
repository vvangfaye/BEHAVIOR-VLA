"""Bounded offline CPU launcher for the real-policy replay validation."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
RUNTIME=PROJECT/'analysis/organizer_rehearsal_1688977_20261016/runtime/policy-bundle/rootfs/mnt/afs/260010165/behavior_1k/runtime_20261007_1306'
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='0',
    CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',MKL_NUM_THREADS='16',
    LD_LIBRARY_PATH=str(RUNTIME/'envs/b1k-pi05/lib'),OPENPI_DATA_HOME=str(RUNTIME/'openpi-cache'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    PYTHONPATH=str(PROJECT/'BEHAVIOR-VLAs/openpi/src')+':'+str(PROJECT/'BEHAVIOR-VLAs/openpi/packages/openpi-client/src'))
command=[str(RUNTIME/'envs/b1k-pi05/bin/python3.11'),
    str(HERE.parent/'native_input_screen_20261020/run_local_cpu.py')]
# Reuse only the known asyncio setup, then execute the new diagnostic script.
entry="import asyncio,runpy,sys;sys.path.insert(0,"+repr(str(HERE.parent/'flow_endpoint_screen_20261020'))+");from run_local_cpu import PeriodicWakeupPolicy;asyncio.set_event_loop_policy(PeriodicWakeupPolicy());runpy.run_path("+repr(str(HERE/'neural_branch_parity.py'))+",run_name='__main__')"
command=[command[0],'-u','-c',entry]
with (HERE/'neural_branch_parity.log').open('x') as log:
    process=subprocess.Popen(command,env=env,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=process.wait(timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise TimeoutError('Local validation exceeded 900 seconds')
print(json.dumps({'exit_code':code,'result_exists':(HERE/'neural_branch_parity_result.json').exists()}))
sys.exit(code)
