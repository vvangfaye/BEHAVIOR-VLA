"""Replace only the four authorized single-GPU learners after parity passes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from deployment import HERE, COMPARISON, WATER, CLI, ENV, describe, save, aoss_client


def spawn(root, name, record_name):
    log=(root/(name+'.log')).open('ab', buffering=0)
    process=subprocess.Popen([str(Path(sys.executable).absolute()), '-u', str(root/(name+'.py'))],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=ENV)
    log.close()
    save(root/record_name, {'pid':process.pid, 'at':time.time(), 'generation':'four-gpu-spot'})


def main():
    assert json.loads((HERE/'parity_result.json').read_text())['passed']
    root=WATER.parent
    paused=[]
    for file in [COMPARISON/'controller_process.json', root/'continuation_supervisor_process.json', WATER/'controller_process.json']:
        record=json.loads(file.read_text());pid=record['pid'];path=Path(f'/proc/{pid}/cmdline')
        if path.exists():
            command=path.read_bytes().replace(b'\0', b' ').decode()
            assert 'controller.py' in command or 'continuation_supervisor.py' in command
            os.kill(pid, signal.SIGSTOP)
            paused.append({'pid':pid, 'command':command})
    save(HERE/'paused_controllers.json', {'processes':paused})
    marker={'owner':str(HERE/'train_controller.py'), 'gpus_per_learner':4, 'quota':'spot', 'at':time.time()}
    save(COMPARISON/'TRAINING_HANDOFF.json', marker);save(WATER/'TRAINING_HANDOFF.json', marker)
    file=COMPARISON/'controller.py';source=file.read_text()
    source=source.replace("    if ready:\n        for variant in plan['variants']:",
        "    if ready and not (HERE/'TRAINING_HANDOFF.json').exists():\n        for variant in plan['variants']:",1)
    source=source.replace("return float(receipt['max_seconds'])", "return float(receipt['max_seconds'])*receipt.get('gpu_count',1)")
    source=source.replace("return max(0.,(parse(end)-parse(start)).total_seconds())", "return max(0.,(parse(end)-parse(start)).total_seconds())*receipt.get('gpu_count',1)")
    source=source.replace("if len(active)>=allowed_new:", "if sum(r.get('gpu_count',1) for r in active)>=allowed_new:")
    source=source.replace("'active_jobs':len(active)", "'active_jobs':len(active),'active_gpus':sum(r.get('gpu_count',1) for r in active)")
    file.write_text(source)
    file=COMPARISON/'deployment.py';source=file.read_text()
    source=source.replace("if not (water/'outcome.json').exists():", "if not (water/'outcome.json').exists() and not campaign.get('combined_training_storage',False):")
    file.write_text(source)
    file=COMPARISON/'plan.json';plan=json.loads(file.read_text())
    plan['combined_training_storage']=True
    plan['storage_bounds']['comparison_total_budget_bytes']=52*1024**3
    plan['storage_bounds']['training_worker_limit_bytes']=16*1024**3
    save(file,plan)
    verified=json.loads((COMPARISON/'implementation_verified.json').read_text())
    verified['plan_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();save(COMPARISON/'implementation_verified.json',verified)
    file=WATER/'controller.py';source=file.read_text()
    source=source.replace("                if phase!='baseline':", "                if phase.startswith('train') and (HERE/'TRAINING_HANDOFF.json').exists():\n                    threading.Event().wait(20);continue\n                if phase!='baseline':",1)
    file.write_text(source)
    file=WATER/'plan.json';plan=json.loads(file.read_text());plan['runtime_storage']['max_persistent_bytes']=28*1024**3;save(file,plan)
    manifest=json.loads((WATER/'execution_manifest.json').read_text())
    for name in ['controller.py','plan.json']:
        manifest['source_sha256'][name]=hashlib.sha256((WATER/name).read_bytes()).hexdigest()
    manifest['distributed_execution']={**marker,'algorithm':'same loss, rewards, global batch and 17 updates'}
    save(WATER/'execution_manifest.json',manifest)
    paths=[COMPARISON/'jobs'/f't-{v}-a2/submission.json' for v in ['gamma1-ppo','balanced-ppo','group-outcome']]
    paths.append(WATER/'jobs/train1_r0_a2/submission.json')
    old=[]
    precompleted=[]
    for path in paths:
        row=json.loads(path.read_text());job=describe(row['job']);assert job['uid']==row['uid']
        if job['state']=='SUCCEEDED':
            precompleted.append('control' if path.parent.name.startswith('train1') else row['variant'])
        if job['state'] not in {'SUCCEEDED','FAILED','SUSPENDED','STOPPED','CANCELED','TERMINATED'}:
            reply=subprocess.run([CLI,'acp','jobs','stop',row['job'],'--workspace-name=share-space'],env=ENV,capture_output=True,text=True,timeout=45)
            assert reply.returncode==0
        old.append(row)
    save(HERE/'old_learners.json', {'jobs':old})
    deadline=time.time()+300
    for row in old:
        while True:
            job=describe(row['job'])
            reply=subprocess.run([CLI,'acp','jobs','get-workers',row['job'],'--workspace-name=share-space'],env=ENV,capture_output=True,text=True,timeout=40)
            if job['state'] in {'SUCCEEDED','FAILED','SUSPENDED','STOPPED','CANCELED','TERMINATED'} and reply.returncode==0 and not any(w in reply.stdout for w in ['Running','Pending','Terminating']):
                break
            assert time.time()<deadline, 'Old learner still owns a worker; no replacement started'
            time.sleep(10)
    save(HERE/'handoff_complete.json', {'at':time.time(),'old_workers_exited':True,'old_jobs':old,'precompleted':precompleted})
    for row in paused:
        if Path(f"/proc/{row['pid']}").exists():
            os.kill(row['pid'],signal.SIGTERM);os.kill(row['pid'],signal.SIGCONT)
    time.sleep(1)
    spawn(root,'continuation_supervisor','continuation_supervisor_process.json')
    spawn(COMPARISON,'controller','controller_process.json')
    spawn(HERE,'train_controller','controller_process.json')


if __name__=='__main__':main()
