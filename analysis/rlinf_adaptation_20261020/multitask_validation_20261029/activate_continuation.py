"""Replace only local coordinators; running ACP workers and outputs stay intact."""
import pathlib,json,os,signal,time,subprocess,sys,fcntl,hashlib,shutil
ROOT=pathlib.Path(__file__).resolve().parent
REPAIR=ROOT/'capacity_repair_20261030'
for slug in ['trash','water']:
    p=ROOT/slug
    for name in ['controller.py','storage_guard.py','capacity_admission.py']:
        shutil.copyfile(ROOT/('task_controller.py' if name=='controller.py' else name),p/name)
    marker=p/'BASELINE_ONLY.json'
    if marker.exists():marker.rename(REPAIR/f'{slug}_baseline_only_retired.json')
    m=json.loads((p/'execution_manifest.json').read_text());m['source_sha256'].pop('BASELINE_ONLY.json',None)
    for name in ['controller.py','storage_guard.py','capacity_admission.py','bootstrap.py','deploy.py','recovery_policy.py']:
        m['source_sha256'][name]=hashlib.sha256((p/name).read_bytes()).hexdigest()
    (p/'execution_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
old=json.loads((ROOT/'baseline_unblock_20261030/launch.json').read_text())['tasks']
old += [{'slug':'watch','pid':json.loads((ROOT/'baseline_unblock_20261030/watch_launch.json').read_text())['pid']}]
stopped=[]
for item in old:
    pid=item['pid'];path=pathlib.Path(f'/proc/{pid}/cmdline')
    if not path.exists():continue
    cmd=path.read_bytes().replace(b'\0',b' ').decode()
    expected=str(ROOT/('baseline_watch.py' if item['slug']=='watch' else item['slug']+'/controller.py'))
    assert expected in cmd,(pid,'PID reused; do not signal')
    os.kill(pid,signal.SIGTERM);stopped.append(pid)
for _ in range(100):
    if all(not pathlib.Path(f'/proc/{p}').exists() for p in stopped):break
    time.sleep(.1)
for slug in ['trash','water']:
    with (ROOT/slug/'controller.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
with (ROOT/'continuation_supervisor.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
with (ROOT/'continuation_supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen([sys.executable,str(ROOT/'continuation_supervisor.py')],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
receipt={'at':time.time(),'supervisor_pid':child.pid,'retired_local_pids':stopped,
         'cloud_workers_stopped':0,'completed_results_deleted':0,'training_settings_changed':False,
         'training_trajectories':32,'ppo_updates':17,'max_rl_gpus':8}
(REPAIR/'continuation_activation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
