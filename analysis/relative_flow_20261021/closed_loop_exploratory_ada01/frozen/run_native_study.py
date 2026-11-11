"""Separate fixed-checkpoint, paired native task-outcome exploration.

The original offline screen stays failed. This study evaluates the user-facing
endpoint directly and does not label an offline failure as a promotion.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE/'closed_loop_exploratory_ada01'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)


def schedule(plan):
    rows=[]
    for task,instance in plan['pairs']:
        methods=plan['protocol']['methods']
        if instance==304:methods=list(reversed(methods))
        rows.extend((method,task,instance) for method in methods)
    assert len(rows)==12 and len(set(rows))==12
    return rows


def collect(plan):
    rows=[];pairs=[]
    for task,instance in plan['pairs']:
        found={}
        for method in plan['protocol']['methods']:
            folder=Path(plan['afs_root'])/method/f'{task}-{instance}'
            if not folder.exists():continue
            if not (folder/'receipt.json').exists():continue
            receipt=read(folder/'receipt.json')
            if not receipt['accepted']:raise ValueError('Native episode failed: '+str(folder))
            if receipt['plan_sha256']!=sha(ROOT/'plan.json'):raise ValueError('Receipt plan mismatch')
            native=folder/'results/json'/f'{task}_{instance}_0.json'
            if sha(native)!=receipt['result_sha256']:raise ValueError('Native result changed')
            found[method]=receipt
            rows.append(dict(method=method,**receipt['record']))
        if len(found)==2:
            a,b=(found[m] for m in plan['protocol']['methods'])
            if a['initial_state']['physics_sha256']!=b['initial_state']['physics_sha256']:
                raise ValueError('Paired initial physics differ')
            pairs.append(dict(task=task,instance=instance,
                 control=a['record'],candidate=b['record'],initial_physics_identical=True))
    methods={}
    for method in plan['protocol']['methods']:
        group=[r for r in rows if r['method']==method]
        methods[method]=dict(episodes=len(group),successes=sum(r['success'] for r in group),
                            mean_score=sum(r['q_score'] for r in group)/len(group) if group else None)
    differences=[p['candidate']['q_score']-p['control']['q_score'] for p in pairs]
    return dict(complete=len(rows)==12,episodes=len(rows),paired_instances=len(pairs),
                methods=methods,by_instance=pairs,
                paired_score_mean_delta=sum(differences)/len(differences) if differences else None,
                success_gains=sum(p['candidate']['success'] and not p['control']['success'] for p in pairs),
                success_losses=sum(p['control']['success'] and not p['candidate']['success'] for p in pairs),
                plan_sha256=sha(ROOT/'plan.json'),
                scope='Exploratory six-instance, three-task, one-training-seed study; not confirmation of efficacy or novelty.',
                offline_screen_still_failed=True,goal_complete=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    plan=read(ROOT/'plan.json')
    for name,expected in plan['source_sha256'].items():
        if sha(ROOT/'frozen'/name)!=expected:raise ValueError('Native study source changed: '+name)
    if not args.execute:
        print(json.dumps(dict(episodes=schedule(plan),launched=False,
                             waiting_for_four_arm_completion=not (HERE/'pilot_ada01/completion.json').exists())));return
    if not (HERE/'pilot_ada01/completion.json').exists() or not read(HERE/'pilot_ada01/completion.json')['complete']:
        print(json.dumps(dict(launched=False,reason='Wait for the current frozen four-arm experiment to finish.')));return
    if (ROOT/'launch.json').exists():raise ValueError('Study has already been launched; no automatic retry')
    # Cooperate with the original local GPU lease; never steal a running GPU.
    lease=(HERE.parent/'action_support_20261021/local_vla_pilot_ada01/gpu0.lock').open('a')
    fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'],text=True).strip()
    if apps:raise RuntimeError('GPU0 is occupied')
    write(ROOT/'launch.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
                                 plan_sha256=sha(ROOT/'plan.json'),max_hours=20,no_retries=True))
    deadline=time.monotonic()+20*3600;child=None
    def interrupted(*_):raise InterruptedError('Study controller terminated')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for method,task,instance in schedule(plan):
            if time.monotonic()>=deadline:raise TimeoutError('Study total runtime cap reached')
            if (Path(plan['afs_root'])/method/f'{task}-{instance}').exists():
                raise ValueError('Episode directory already exists; inspect instead of retrying')
            label=f'{method}-{task}-{instance}'
            record=dict(method=method,task=task,instance=instance,started_at=datetime.now(timezone.utc).isoformat())
            with (ROOT/(label+'.log')).open('x') as stream:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'frozen/closed_loop_worker.py'),
                    '--plan',str(ROOT/'plan.json'),'--method',method,'--task',task,'--instance',str(instance)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),cwd=ROOT/'frozen',stdout=stream,stderr=subprocess.STDOUT)
                record['pid']=child.pid;write(ROOT/'state.json',dict(running=record))
                print(json.dumps(dict(event='launch',**record)),flush=True)
                try:code=child.wait(timeout=min(plan['resource_bound']['max_episode_seconds']+60,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=45)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise
            child=None
            if code!=0:raise RuntimeError('Native worker failed: '+label)
            summary=collect(plan);write(ROOT/'results.json',summary)
            write(ROOT/'state.json',dict(running=None,completed_episodes=summary['episodes']))
            print(json.dumps(dict(event='completed',episodes=summary['episodes'],paired_instances=summary['paired_instances'])),flush=True)
        write(ROOT/'completion.json',dict(complete=True,finished_at=datetime.now(timezone.utc).isoformat(),
                                         results_sha256=sha(ROOT/'results.json')))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        lease.close()


if __name__=='__main__':main()
