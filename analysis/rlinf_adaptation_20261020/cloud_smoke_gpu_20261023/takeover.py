"""Transfer this allocation to the GPU continuation, preserving prior evidence.

The original Python supervisor is held while its CPU stage is stopped. The
replacement uploads all results before ending the held supervisor, so no second
GPU allocation or two competing result writers are required.
"""
import hashlib
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

OLD = Path('/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/retry07')
HERE = OLD / 'gpu_continuation'


def write(name, value):
    p=HERE/name
    q=p.with_suffix('.tmp')
    q.write_text(json.dumps(value,indent=2)+'\n')
    q.replace(p)


def process_roles():
    roles={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try: args=(path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if str(OLD/'run/driver.py').encode() in args:roles['driver']=int(path.name)
        if str(OLD/'run/rollout.py').encode() in args and b'jax' in args:roles['baseline']=int(path.name)
    return roles


def running_group(pgid):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:s=(p/'stat').read_text().split()
        except (FileNotFoundError,PermissionError):continue
        if int(s[4])==pgid and s[2]!='Z':found.append(int(p.name))
    return found


def main():
    probe=json.loads((OLD/'results/gpu_fp32_probe/result.json').read_text())
    assert probe['passed'] and all(probe['checks'].values()),'GPU probe not accepted'
    assert not (OLD/'results/jax/302/result.json').exists(),'CPU reference already completed; preserve it'
    status=json.loads((OLD/'results/status.json').read_text())
    assert status['stage']=='baseline' and status['state']=='running'
    roles=process_roles()
    assert set(roles)=={'driver','baseline'},roles
    st=(Path('/proc')/str(roles['baseline'])/'stat').read_text().split()
    assert int(st[3])==roles['driver'] and os.getpgid(roles['baseline'])==roles['baseline']
    assert os.getpgrp()!=roles['baseline']
    env_raw=(Path('/proc')/str(roles['driver'])/'environ').read_bytes()
    env={k.decode():v.decode() for item in env_raw.split(b'\0') if b'=' in item for k,v in [item.split(b'=',1)]}
    env['PYTHONPATH']=env['PYTHONPATH'].replace(str(OLD/'run'),str(HERE/'run'))
    env.pop('JAX_PLATFORMS',None)
    env['JAX_DEFAULT_MATMUL_PRECISION']='highest'
    os.sched_setaffinity(0,os.sched_getaffinity(roles['driver']))
    plan=json.loads((HERE/'run/plan.json').read_text())
    for relative,expected in plan['reuse_files_sha256'].items():
        assert hashlib.sha256((OLD/'results'/relative).read_bytes()).hexdigest()==expected,relative
    partial=json.loads((OLD/'results/jax/302/progress.json').read_text())
    receipt={'at':time.time(),'job':'b1k-rlinf-1023-r07','uid':'cc2919cc6b5045d3ad07009a842762d7',
             'roles':roles,'old_result_root':str(OLD/'results'),'new_result_root':plan['output'],
             'partial_cpu_episode':partial,'partial_episode_replaced_for_device_switch':True,
             'completed_jax301_preserved':True,'added_gpus':0,'probe_passed':True,
             'normal_platform_exit_after_handoff':'Original supervisor is ended after replacement archive upload; terminal job state does not replace artifact acceptance.'}
    write('handoff_receipt.json',receipt)
    os.kill(roles['driver'],signal.SIGSTOP)
    held=True
    try:
        os.killpg(roles['baseline'],signal.SIGTERM)
        deadline=time.monotonic()+30
        while running_group(roles['baseline']) and time.monotonic()<deadline:time.sleep(0.2)
        if running_group(roles['baseline']):os.killpg(roles['baseline'],signal.SIGKILL)
        assert not running_group(roles['baseline']),'Old CPU/simulator stage still active'
        write('takeover_state.json',{'stage':'gpu_continuation_running','at':time.time(),'roles':roles})
        with (HERE/'driver.log').open('x') as log:
            result=subprocess.run([plan['runtime']+'/envs/b1k-pi05/bin/python',str(HERE/'run/driver.py')],
                                  env=env,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
        status_path=Path(plan['output'])/'status.json'
        write('takeover_state.json',{'stage':'continuation_finished','returncode':result.returncode,
                                    'result':json.loads(status_path.read_text()) if status_path.exists() else None,
                                    'at':time.time()})
    except BaseException as error:
        write('takeover_state.json',{'stage':'takeover_failed','error':str(error),'traceback':traceback.format_exc(),'at':time.time()})
        raise
    finally:
        if held:
            # Never resume the old supervisor: it would overwrite the new archive.
            os.kill(roles['driver'],signal.SIGKILL)


if __name__=='__main__':main()
