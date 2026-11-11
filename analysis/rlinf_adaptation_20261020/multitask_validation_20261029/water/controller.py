"""One task, fixed phase order and paired evaluation; never select a best checkpoint."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

from deploy import HERE,BASE,describe,submit,write,aoss_client
from eval_utils import paired_report
from recovery_policy import attempt_cap

import control_support as shared
# Shared methods import the task-local deploy module and therefore use this HERE.
assert shared.HERE==HERE


def finalize(plan):
    roots=sorted((HERE/'retrieved').iterdir());evidence=HERE/'final_evidence'
    for phase in ('baseline','final'):
        for instance,seed in plan['evaluation']['cases']:
            case=f'i{instance}_s{seed}'
            folders=[r/phase/case for r in roots if all((r/phase/case/n).is_file()
                     for n in ['result.json','initial.json','spec.json','actions.json','goal_summary.json'])]
            assert folders,(phase,case)
            for name in ['result.json','initial.json','spec.json','goal_summary.json']:
                bodies=[(f/name).read_bytes() for f in folders]
                assert all(b==bodies[0] for b in bodies),'Conflicting completed case'
                p=evidence/phase/case/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(bodies[0])
    report={key:paired_report(evidence,plan['evaluation'][cases]) for key,cases in
             [('heldout_instances','heldout_cases'),('training_instances','training_cases')]}
    assert all(x['comparison_complete'] and x['comparison_valid'] for x in report.values())
    ready=list((HERE/'retrieved').glob('*/checkpoints/round_01/ready.json'))
    assert ready
    checkpoint=json.loads(ready[-1].read_text());assert checkpoint['optimizer_steps']==17
    report.update(task=plan['task'],rounds=1,total_optimizer_steps=17,
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  limitation='One training seed; 20 independent held-out instances. Preliminary task replication, not a seed-robustness claim.')
    write(HERE/'outcome.json',report)
    return report


def main():
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import faulthandler,signal
    faulthandler.register(signal.SIGUSR1)
    plan=json.loads((HERE/'plan.json').read_text());client=aoss_client()
    manifest=json.loads((HERE/'execution_manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    deadline=time.time()+plan['max_wall_hours']*3600
    baseline_only=HERE/'BASELINE_ONLY.json'
    try:
        while time.time()<deadline:
            try:
                write(HERE/'controller_activity.json',{'at':time.time(),'operation':'inspect'})
                rows=shared.inspect(client)
                for row in rows:
                    if row['state']=='SUCCEEDED':
                        write(HERE/'controller_activity.json',{'at':time.time(),'operation':'collect_results','job':row['job']})
                        shared.collect(client,row)
                        task=f"{plan['worker_prefix']}{row['phase']}_r{row['rank']}_a{row['attempt']}"
                        proof=HERE/'retrieved'/f"{row['phase']}_r{row['rank']}_a{row['attempt']}"/'checks'/f'{task}.json'
                        assert proof.is_file() and json.loads(proof.read_text())['frozen_unchanged']
                write(HERE/'controller_activity.json',{'at':time.time(),'operation':'select_phase'})
                phase,current=shared.next_pending(plan,rows)
                write(HERE/'latest_status.json',{'observed_at':datetime.now(timezone.utc).isoformat(),
                      'experiment':plan['experiment'],'state':'running','next_phase':phase,'workers':rows,'finished':False})
                if phase is None:
                    report=finalize(plan)
                    write(HERE/'latest_status.json',{'observed_at':datetime.now(timezone.utc).isoformat(),
                          'experiment':plan['experiment'],'state':'complete','finished':True,'outcome':report})
                    return
                if phase.startswith('train') and (HERE/'TRAINING_HANDOFF.json').exists():
                    threading.Event().wait(20);continue
                if phase!='baseline':
                    proof=HERE/'training_storage_clearance.json'
                    clearance=json.loads(proof.read_text()) if proof.is_file() else {}
                    from capacity_admission import acceptable
                    if not acceptable(clearance,plan):
                        write(HERE/'latest_status.json',{'observed_at':datetime.now(timezone.utc).isoformat(),
                              'experiment':plan['experiment'],'state':'waiting_for_training_capacity',
                              'next_phase':phase,'workers':rows,'finished':False,
                              'baseline_complete':True})
                        threading.Event().wait(30);continue
                active=sum(r['state'] not in shared.TERMINAL for r in rows)
                launch_limit=plan['max_live_gpus']
                control=HERE/'continuation_control.json'
                if control.is_file():
                    launch_limit=min(launch_limit,int(json.loads(control.read_text())['max_live_gpus']))
                if baseline_only.is_file() and not control.is_file():
                    launch_limit=min(launch_limit,int(json.loads(baseline_only.read_text())['max_live_gpus']))
                for rank,row in zip(shared.phase_ranks(phase),current):
                    if phase=='baseline' and rank>0:
                        first=shared.latest(rows,'baseline',0)
                        if not first or first['state']!='SUCCEEDED':continue
                    if row and row['state']=='SUCCEEDED':continue
                    if row and row['state'] in shared.TERMINAL:
                        if row['attempt']>=attempt_cap(plan,phase,HERE) or not shared.retryable(row):
                            shared.stop_active(rows,'nonretryable_task_failure')
                            write(HERE/'stopped.json',{'reason':'phase_failed','worker':row})
                            return
                    if active<launch_limit and (row is None or row['state'] in shared.TERMINAL):
                        submit(phase,rank,1 if row is None else row['attempt']+1)
                        break
                else:
                    threading.Event().wait(30)
            except Exception as error:
                # Preserve transient control-plane errors, then reconcile exact names.
                write(HERE/'controller_last_error.json',{'at':time.time(),'type':type(error).__name__,'error':str(error)[:700]})
                if isinstance(error,AssertionError):raise
                threading.Event().wait(30)
        raise TimeoutError('Task campaign deadline')
    finally:client.close()


if __name__=='__main__':main()
