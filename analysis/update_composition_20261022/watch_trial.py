"""Persist exact-job observations until the requested training run finishes.

Read-only with respect to cloud jobs: failures are reported, never silently
restarted. Local episode summaries survive a disconnected chat or terminal.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--max-seconds',type=int,default=26*3600)
    args=parser.parse_args()
    root=HERE/'split_trial_submissions'/args.method/args.run_id
    plan=json.loads((root/'plan.json').read_text())
    required=plan['max_training_episodes']
    deadline=time.monotonic()+args.max_seconds
    observations={}
    report_path=root/'episode_report.json'
    if report_path.exists():
        for row in json.loads(report_path.read_text()).get('episodes',[]):
            observations[(row['iteration'],row['slot'])]=row
    previous=None
    while time.monotonic()<deadline:
        stamp=datetime.now(timezone.utc).isoformat()
        try:
            completed=subprocess.run([sys.executable,str(HERE/'observe_split_trial.py'),
                '--method',args.method,'--run-id',args.run_id],capture_output=True,text=True,timeout=110)
            if completed.returncode:
                raise RuntimeError('State or artifact read failed; cloud terminal status not inferred')
            roles={}
            for role in ('renderer','trainer'):
                status=root/role/'job_status.json'
                roles[role]=json.loads(status.read_text()) if status.exists() else {'state':'NOT_SUBMITTED'}
            progress_path=root/'trainer/progress.json'
            progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
            for iteration in progress.get('completed',[]):
                for slot,row in enumerate(iteration['episodes']):
                    observations[(iteration['iteration'],slot)]=dict(row,iteration=iteration['iteration'],slot=slot)
            if progress.get('stage')=='episode_finished':
                key=(progress['iteration'],progress['slot'])
                observations[key]=dict(progress['episode'],iteration=key[0],slot=key[1])
            rows=[observations[key] for key in sorted(observations)]
            result=dict(observed_at=stamp,required_episodes=required,observed_episodes=len(rows),
                successes=sum(bool(row['success']) for row in rows),episodes=rows,
                updates=progress.get('completed',[]),roles=roles,
                scope='Training episodes only; this is not a controlled efficacy comparison')
            write(report_path,result)
            trainer_result_path=root/'trainer/result.json'
            trainer_result=json.loads(trainer_result_path.read_text()) if trainer_result_path.exists() else {}
            status='running'
            if trainer_result.get('complete') and trainer_result.get('training_episodes')==required:
                status='training_complete' if len(rows)==required else 'training_complete_outcomes_need_retrieval'
            elif any(value['state'] in ('FAILED','DELETED','SUSPENDED','STOPPED') for value in roles.values()):
                status='worker_terminal_before_completion'
            elif trainer_result.get('worker_returncode',0)!=0:
                status='training_worker_failed'
            state=dict(observed_at=stamp,status=status,episodes=len(rows),required=required,
                       successes=result['successes'],roles={k:v['state'] for k,v in roles.items()},
                       training_stage=progress.get('stage'),iteration=progress.get('iteration'),slot=progress.get('slot'))
            write(root/'watch_status.json',state)
            signature={k:v for k,v in state.items() if k!='observed_at'}
            if signature!=previous:
                print(json.dumps(state),flush=True)
                previous=signature
            if status!='running':
                return 0 if status.startswith('training_complete') else 1
        except (subprocess.TimeoutExpired,RuntimeError,OSError,ValueError) as error:
            state=dict(observed_at=stamp,status='observation_retry',error_type=type(error).__name__,
                       error=str(error)[:300],cloud_state_inferred=False)
            write(root/'watch_status.json',state)
            print(json.dumps(state),flush=True)
        time.sleep(30)
    write(root/'watch_status.json',dict(status='watch_deadline',cloud_state_inferred=False))
    return 2


if __name__=='__main__':
    raise SystemExit(main())
