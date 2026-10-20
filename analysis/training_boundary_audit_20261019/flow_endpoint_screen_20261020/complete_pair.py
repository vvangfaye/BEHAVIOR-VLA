#!/usr/bin/env python3
"""Complete the frozen CPU pair once its existing prerequisites finish."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from compare_screens import compare

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    with (HERE/'complete_pair.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status':'waiting_for_existing_prerequisites', 'started_at_utc':datetime.now(timezone.utc).isoformat(),
                 'training':False, 'simulation':False, 'cpu_count':16,
                 'maximum_prerequisite_wait_seconds':1900, 'maximum_expert_runtime_seconds':900}
        status_path = HERE/'complete_pair.status.json'
        write(status_path,state)
        try:
            plan=json.loads((HERE/'screen_plan.json').read_text())
            for filename, expected in plan['source_sha256'].items():
                if hashlib.sha256((HERE/filename).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Frozen collection source changed')
            deadline=time.monotonic()+1900
            while True:
                download=json.loads((HERE/'checkpoint_download_status.json').read_text())
                baseline=json.loads((HERE/'baseline1688977.status.json').read_text())
                if download['status'] in ('failed','signing_failed'):
                    raise RuntimeError('Existing download failed; no automatic restart')
                if (baseline.get('returncode', 0) not in (0, None)
                        or baseline['status'] in ('failed', 'timeout', 'timed_out', 'error')):
                    raise RuntimeError('Existing baseline did not finish successfully: '+baseline['status'])
                marker=HERE/'expert1000_checkpoint/_LOCAL_VERIFIED.json'
                if download['status']=='complete' and marker.exists() and (HERE/'baseline1688977.json').exists():
                    break
                if time.monotonic()>deadline:
                    raise TimeoutError('Prerequisites exceeded wait budget')
                time.sleep(5)
            baseline_doc=json.loads((HERE/'baseline1688977.json').read_text())
            if baseline_doc['status']!='screen_complete':
                raise ValueError('Baseline incomplete')
            receipt=json.loads(marker.read_text())
            expected=hashlib.sha256((HERE/'checkpoint_download_plan.json').read_bytes()).hexdigest()
            if not receipt['all_content_verified'] or receipt['plan_sha256']!=expected:
                raise ValueError('Checkpoint verification receipt mismatch')
            runtime_root=PROJECT/'analysis/organizer_rehearsal_1688977_20261016/runtime/policy-bundle/rootfs/mnt/afs/260010165/behavior_1k'
            runtime=runtime_root/'runtime_20261007_1306'
            venv=runtime/'envs/b1k-pi05'
            env=os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='0',
                CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='16', OPENBLAS_NUM_THREADS='16', MKL_NUM_THREADS='16',
                LD_LIBRARY_PATH=str(venv/'lib'), OPENPI_DATA_HOME=str(runtime/'openpi-cache'),
                PYTHONPATH=str(PROJECT/'BEHAVIOR-VLAs/openpi/src')+':'+str(PROJECT/'BEHAVIOR-VLAs/openpi/packages/openpi-client/src'))
            output=HERE/'expert1000.json'
            if output.exists():
                raise ValueError('Expert result already exists; inspect instead of rerunning')
            command=['taskset','-c','0-15',str(venv/'bin/python3.11'),str(HERE/'run_local_cpu.py'),
                '--cache-manifest',str(HERE.parent/'pilot_three_tasks_cache_v1/cache_manifest.json'),
                '--checkpoint',str(HERE/'expert1000_checkpoint'),
                '--norm-asset-id','b1k_full100_joint_subtask_v1_drop_episode_s1',
                '--limit','3','--draws','8','--selection','first-per-task','--rng-seed','0',
                '--max-token-len','512','--out',str(output)]
            state.update(status='expert_running', expert_started_at_utc=datetime.now(timezone.utc).isoformat())
            write(status_path,state)
            with (HERE/'expert1000.log').open('x') as log:
                process=subprocess.Popen(command, env=env, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    returncode=process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                    raise TimeoutError('Expert screen exceeded frozen 900-second budget')
            if returncode:
                raise RuntimeError('Expert screen nonzero exit: '+str(returncode))
            result=compare(baseline_doc,json.loads(output.read_text()))
            result['input_sha256']={name:hashlib.sha256((HERE/(name+'.json')).read_bytes()).hexdigest()
                                    for name in ['baseline1688977','expert1000']}
            write(HERE/'paired_comparison.json',result)
            state.update(status='complete', comparison_file='paired_comparison.json')
        except Exception as error:
            state.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            state['updated_at_utc']=datetime.now(timezone.utc).isoformat()
            write(status_path,state)


if __name__=='__main__':
    main()
