"""Queue the proven H100 trainer, dispatch RTX only once H100 is running.

No training settings or outcomes are selected here. Each role has one frozen
submission receipt. The coordinator persists if the interactive turn closes.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from submit import cli,write

HERE=Path(__file__).resolve().parent
TERMINAL={'FAILED','SUCCEEDED','SUSPENDED','STOPPED','DELETED','CANCELLED'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--ready-proof-run',default='')
    parser.add_argument('--prewarm',action='store_true')
    parser.add_argument('--quota-type',choices=('spot','reserved'),default='spot')
    args=parser.parse_args()
    root=HERE/'split_trial_submissions'/args.method/args.run_id
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def event(stage,**extra):
        item=dict(stage=stage,observed_at=datetime.now(timezone.utc).isoformat(),**extra)
        write(root/'coordinator_status.json',item)
        print(json.dumps(item),flush=True)
    command=[sys.executable,str(HERE/'submit_split_trial.py'),'--method',args.method,
             '--run-id',args.run_id,
             '--quota-type',args.quota_type,'--submit']
    if args.ready_proof_run:command+=['--ready-proof-run',args.ready_proof_run]
    if args.prewarm:command+=['--prewarm']
    def submit_role(role):
        deadline=time.monotonic()+48*3600
        while time.monotonic()<deadline:
            response=subprocess.run(command+['--role',role],capture_output=True,text=True)
            if response.stdout:print(response.stdout,end='',flush=True)
            if response.returncode==0:return
            receipt_path=root/role/'submission.json'
            if receipt_path.exists() and json.loads(receipt_path.read_text()).get('submission_attempted'):
                raise RuntimeError('Submission was attempted; resolve the exact receipt before any retry')
            retryable=('standard-resource occupancy','Active-job detail unavailable','No unused standard quota')
            if not any(message in response.stderr for message in retryable):
                raise RuntimeError('Submission preparation failed: '+response.stderr[-1200:])
            event('submission_preflight_retry',role=role,cloud_submission_attempted=False)
            time.sleep(30)
        raise TimeoutError('Submission preflight deadline reached')
    if args.prewarm:
        event('submit_renderer_for_current_code_verification')
        submit_role('renderer')
        deadline=time.monotonic()+48*3600
        while time.monotonic()<deadline:
            try:
                observed=subprocess.run([sys.executable,str(HERE/'observe_split_trial.py'),
                    '--method',args.method,'--run-id',args.run_id],capture_output=True,text=True,timeout=110)
            except (subprocess.TimeoutExpired,OSError):
                event('observation_retry',reason='Renderer read timed out; no terminal state inferred')
                time.sleep(30);continue
            if observed.returncode:
                event('observation_retry',reason='Renderer readiness read failed')
                time.sleep(30);continue
            state=json.loads((root/'renderer/job_status.json').read_text())
            if state['state'] in TERMINAL:
                event('renderer_terminal_before_ready',state=state['state']);return 1
            path=root/'renderer/progress.json'
            progress=json.loads(path.read_text()) if path.exists() else {}
            if progress.get('stage')=='prewarmed':
                event('current_code_initial_observation_verified')
                submit_role('trainer')
                event('watching_24_episodes',quota_type=args.quota_type)
                result=subprocess.run([sys.executable,str(HERE/'watch_trial.py'),
                    '--method',args.method,'--run-id',args.run_id])
                event('watch_finished',returncode=result.returncode);return result.returncode
            event('waiting_for_renderer_initial_observation',state=state['state'],renderer_stage=progress.get('stage'))
            time.sleep(30)
        event('renderer_queue_deadline');return 2
    event('submit_trainer_after_verified_observation')
    submit_role('trainer')
    receipt=json.loads((root/'trainer/submission.json').read_text())
    if args.quota_type=='reserved':
        event('submit_reserved_renderer_with_paired_quota')
        submit_role('renderer')
        event('watching_24_episodes',quota_type='reserved')
        result=subprocess.run([sys.executable,str(HERE/'watch_trial.py'),
                               '--method',args.method,'--run-id',args.run_id])
        event('watch_finished',returncode=result.returncode)
        return result.returncode
    last=None
    deadline=time.monotonic()+48*3600
    while time.monotonic()<deadline:
        try:
            response=cli('acp','jobs','describe',receipt['job_name'],'--workspace-name=share-space','-o','json')
        except (subprocess.TimeoutExpired,OSError):
            event('observation_retry',reason='Trainer query timed out; no terminal state inferred')
            time.sleep(30);continue
        if response.returncode:
            event('observation_retry',reason='Trainer query failed; no terminal state inferred')
            time.sleep(30);continue
        job=json.loads(response.stdout)
        if job['uid']!=receipt['uid']:raise ValueError('Trainer identity changed')
        state=job['state']
        if state!=last:event('waiting_for_trainer_allocation',state=state,uid=job['uid']);last=state
        if state=='RUNNING':
            event('submit_renderer',trainer_uid=job['uid'])
            submit_role('renderer')
            event('watching_24_episodes')
            result=subprocess.run([sys.executable,str(HERE/'watch_trial.py'),
                                   '--method',args.method,'--run-id',args.run_id])
            event('watch_finished',returncode=result.returncode)
            return result.returncode
        if state in TERMINAL:
            event('trainer_terminal_before_renderer',state=state)
            return 1
        time.sleep(20)
    event('queue_deadline',cloud_state_inferred=False)
    return 2


if __name__=='__main__':raise SystemExit(main())
