"""Maintain the authorized baseline -> sampling -> PPO -> evaluation sequence."""
from datetime import datetime,timezone
import fcntl,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'capacity_repair_20261030'))
from read_quota_exact import query_quota
from capacity_admission import evidence


def write(p,j):
    tmp=p.with_name(p.name+'.supervisor.tmp');tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def controller_running(folder):
    with (folder/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False

def start_controller(slug):
    folder=ROOT/slug
    if controller_running(folder) or (folder/'outcome.json').exists() or (folder/'stopped.json').exists():return
    with (folder/'controller.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,str(folder/'controller.py')],cwd=folder,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                           env={**{k:v for k,v in os.environ.items() if k.lower() not in {'http_proxy','https_proxy','all_proxy'}},'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
    write(folder/'controller_process.json',{'pid':p.pid,'at':time.time(),'generation':'capacity-continuation'})

def baseline_completed(state):
    latest={}
    for worker in state.get('workers',[]):
        if worker.get('phase')!='baseline':continue
        rank=worker['rank']
        if rank not in latest or worker['attempt']>latest[rank]['attempt']:latest[rank]=worker
    return sum((w.get('progress') or {}).get('completed_evaluations',0) for w in latest.values())

def main():
    with (ROOT/'continuation_supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(ROOT/'continuation_supervisor_process.json',{'pid':os.getpid(),'at':time.time()})
        deadline=datetime.fromisoformat(json.loads((ROOT/'campaign.json').read_text())['created_at']).timestamp()+7*86400
        quota=None;refresh_at=0;last_error=None
        while time.time()<deadline:
            task_states={s:json.loads((ROOT/s/'latest_status.json').read_text()) for s in ['trash','water']}
            done={s:(ROOT/s/'outcome.json').is_file() for s in task_states}
            eligible='water' if done['trash'] else 'trash'
            suite_limit=int(json.loads((ROOT/'campaign.json').read_text())['max_live_gpus'])
            assert suite_limit in (8,16)
            if time.time()>=refresh_at:
                try:
                    quota=query_quota();write(ROOT/'capacity_repair_20261030/quota_latest.json',quota)
                    last_error=None
                except Exception as e:
                    quota=None;last_error={'type':type(e).__name__,'at':time.time()}
                refresh_at=time.time()+180
            admissions={}
            for slug,state in task_states.items():
                other='water' if slug=='trash' else 'trash'
                other_active=any(w.get('state') not in {'SUCCEEDED','FAILED','STOPPED','SUSPENDED','TERMINATED','CANCELED'}
                                 for w in task_states[other].get('workers',[]))
                other_baseline=task_states[other].get('next_phase')=='baseline' and not done[other]
                write(ROOT/slug/'continuation_control.json',{'max_live_gpus':8 if suite_limit==16 else (4 if other_active or other_baseline else 8),
                      'total_suite_max_live_gpus':suite_limit,'training_slot':eligible,'updated_at':time.time()})
                plan=json.loads((ROOT/slug/'plan.json').read_text())
                proof=evidence(plan,quota,eligible==slug,other_baseline) if quota else {
                    'complete':False,'passed':False,'experiment':plan['experiment'],'observed_at':time.time(),
                    'reason':'Exact quota refresh failed; running work retained'}
                write(ROOT/slug/'training_storage_clearance.json',proof);admissions[slug]=proof
                start_controller(slug)
            # Preserve the completed task in AOSS before admitting the next if needed.
            if done['trash'] and not done['water'] and quota and not admissions['water']['passed']:
                target=ROOT/'capacity_repair_20261030/archive_completed_trash_submission.json'
                if not target.exists():
                    from archive_completed import start_completed_archive
                    try:start_completed_archive('trash')
                    except Exception as e:last_error={'type':type(e).__name__,'stage':'completed_task_archive','at':time.time()}
            summary={'observed_at':datetime.now(timezone.utc).isoformat(),'state':'complete' if all(done.values()) else 'automatic_continuation',
                     'finished':all(done.values()),'training_slot':eligible,'max_total_gpus':suite_limit,
                     'quota_error':last_error,'tasks':{s:{'state':task_states[s].get('state'),
                       'next_phase':task_states[s].get('next_phase'),'storage_passed':admissions[s]['passed'],
                       'baseline_completed':baseline_completed(task_states[s]),
                       'controller_running':controller_running(ROOT/s)} for s in task_states}}
            write(ROOT/'status.json',summary)
            if all(done.values()):return
            time.sleep(30)
        raise TimeoutError('Original seven-day campaign deadline')

if __name__=='__main__':main()
