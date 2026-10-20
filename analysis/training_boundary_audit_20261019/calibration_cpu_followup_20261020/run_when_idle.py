"""Reuse a validated 36-state CPU baseline after testing current-runtime parity."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
CPU_AFFINITY='16-31'


def save(value):
    target=HERE/'execution_state.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(target)


def execute(name,checkpoint,limit,seconds,env,python):
    output=HERE/(name+'.json')
    if output.exists():raise ValueError('Result already exists: '+name)
    args=['taskset','-c',CPU_AFFINITY,str(python),str(HERE/'eval_local.py'),
          '--cache-manifest',str(HERE.parent/'pilot_three_tasks_cache_v1/cache_manifest.json'),
          '--pi05-ckpt',checkpoint,'--out',str(output),'--max-token-len','512',
          '--norm-asset-id','b1k_full100_joint_subtask_v1_drop_episode_s1','--save-raw-actions']
    if limit:args.extend(['--limit',str(limit)])
    with (HERE/(name+'.log')).open('x') as stream:
        p=subprocess.Popen(args,env=env,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise TimeoutError(name+' exceeded frozen time bound')
    if code:raise RuntimeError(name+' nonzero exit '+str(code))
    return output


def main():
    plan=json.loads((HERE/'plan.json').read_text())
    state={'status':'waiting_for_current_cpu_probe','started_at_utc':datetime.now(timezone.utc).isoformat()}
    save(state)
    try:
        for path,expected in plan['source_sha256'].items():
            assert hashlib.sha256((PROJECT/path).read_bytes()).hexdigest()==expected
        limit=time.monotonic()+1900
        previous=HERE.parent/'native_input_screen_20261020/execution_state.json'
        while plan.get('wait_for_current_native_probe',True) and json.loads(previous.read_text())['status']=='running':
            if time.monotonic()>limit:raise TimeoutError('Previous CPU probe did not finish within wait bound')
            time.sleep(5)
        rt=PROJECT/'analysis/organizer_rehearsal_1688977_20261016/runtime/policy-bundle/rootfs/mnt/afs/260010165/behavior_1k/runtime_20261007_1306'
        python=rt/'envs/b1k-pi05/bin/python3.11'
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='0',
            CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',MKL_NUM_THREADS='16',
            LD_LIBRARY_PATH=str(rt/'envs/b1k-pi05/lib'),OPENPI_DATA_HOME=str(rt/'openpi-cache'),
            PYTHONPATH=str(PROJECT/'BEHAVIOR-VLAs/openpi/src')+':'+str(PROJECT/'BEHAVIOR-VLAs/openpi/packages/openpi-client/src'))
        state['status']='baseline_one_parity';save(state)
        check=execute('baseline_one_parity',plan['checkpoints']['baseline1688977'],1,300,env,python)
        old=json.loads((PROJECT/plan['baseline_result']).read_text());new=json.loads(check.read_text())
        row=new['sample_keys'][0];key=(row['episode_index'],row['frame_index'])
        old_row=next(r for r in old['sample_keys'] if (r['episode_index'],r['frame_index'])==key)
        for field in ('action_noise_sha256','policy_input_sha256','ground_truth_sha256'):
            assert row[field]==old_row[field],field
        # Both runs use the same CPU backend, 16 threads and unchanged source.
        errors=[abs(a-b) for x,y in zip(row['raw_actions']['prediction'],old_row['raw_actions']['prediction'],strict=True)
                for a,b in zip(x,y,strict=True)]
        assert max(errors)<1e-6
        (HERE/'baseline_reuse_receipt.json').write_text(json.dumps({'sample':key,'max_prediction_difference':max(errors),
            'old_result_sha256':hashlib.sha256((PROJECT/plan['baseline_result']).read_bytes()).hexdigest(),
            'full_source_hashes_checked':True,'rechecked_samples':1,'reused_samples':36,
            'scope':'One-sample runtime parity plus unchanged code/config and previously verified full checkpoint; not a rerun of all baseline outputs.'},indent=2)+'\n')
        state['status']='expert36_running';save(state)
        candidate=execute('expert1000_36_raw',plan['checkpoints']['expert1000'],0,900,env,python)
        state['status']='analyzing';save(state)
        result=subprocess.run([str(python),str(HERE.parent/'action_change_calibration.py'),
            '--baseline',str(PROJECT/plan['baseline_result']),'--candidate',str(candidate),
            '--output',str(HERE/'calibration.json')],env=env,cwd=PROJECT,timeout=120,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Calibration analysis failed: '+result.stderr[-1000:])
        state['status']='complete'
    except Exception as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        state['updated_at_utc']=datetime.now(timezone.utc).isoformat();save(state)


if __name__=='__main__':main()
