"""Prepare two finite fixed20 campaigns, without creating cloud jobs."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def hashes(folder):return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.py'))}

def main():
 old=ROOT/'paired_fixed20_step5000';batch=ROOT/'paired_fixed20_step5000_5x4'
 export=ROOT/'expert_fixed20_step1000';expert=ROOT/'expert_fixed20_step1000_5x4'
 if not export.exists():
  for dest,source in [(export,old),(expert,batch)]:
   for part in ('frozen','intents','reports','private','observations','telemetry'):(dest/part).mkdir(parents=True,exist_ok=True)
   # Only source and immutable historical comparison inputs, never old intents/results.
   for p in (source/'frozen').glob('*.py'):
    text=p.read_text().replace('5000','1000').replace('b1k-quality-taskbalanced-1688977-20261016','b1k-expert-control-1688977-20261016').replace('/quality_taskbalanced/', '/expert_control/')
    (dest/'frozen'/p.name).write_text(text)
   text=(source/'controller.py').read_text().replace('paired_fixed20_step5000','expert_fixed20_step1000').replace('5000','1000').replace('b1k-qbal5k-1016','b1k-expert1k-1016').replace('239707','240478')
   (dest/'controller.py').write_text(text)
   save(dest/'frozen_sha256.json',hashes(dest/'frozen'))
   plan=read(source/'plan.json');plan.update(training_iid=240478,training_native_id='01a1445fb3c339ac1c5be3b248259d12',run_id='b1k-expert-control-1688977-20261016',training_target_unchanged=1000,candidate_step=1000,
    afs_root='/mnt/afs/260010165/behavior_1k/expert_fixed20_step1000_20261016'+('_5x4' if dest==expert else ''),
    object_prefix='behavior1k/expert-fixed20-step1000-20261016'+('-5x4' if dest==expert else ''),
    formal_contract=read(ROOT/'expert_control/formal_stage/bundle/inputs/formal_run_contract.json'))
   if dest==expert:
    plan['candidate_checkpoint']='/mnt/afs/260010165/behavior_1k/expert_fixed20_step1000_20261016/checkpoint1000'
    plan['base_controller_sha256']=hashlib.sha256((export/'controller.py').read_bytes()).hexdigest()
   save(dest/'plan.json',plan)
 # Replace the expert probe's known attempt only. Its checkpoint/commit proof remains mandatory.
 p=export/'frozen/remote_probe.py';text=p.read_text().replace('stage1000-g0','stage1000-g0')
 # Numeric replacement already converted stage5000-g0 and max_authorized_updates to 1000.
 assert "'/expert_control/'" in text or '/expert_control/' in text
 save(export/'frozen_sha256.json',hashes(export/'frozen'))
 for folder in (old,batch,export,expert):
  for sub in ('private','intents','reports','observations'):(folder/sub).mkdir(exist_ok=True)
  plan=read(folder/'plan.json');plan['deadline_epoch']=time.time()+72*3600
  plan['human_authorization']='混合数据的5000步模型赶紧开始评测，纯专家训练完也直接用同样20组评测，跑完给我完整对比，不用再确认。'
  save(folder/'plan.json',plan)
 print(json.dumps({'campaigns':[str(x.name) for x in (batch,expert)],'pairs':20,'jobs_each':5,'max_simultaneous_gpu_jobs':5,'cloud_jobs_created':0}))

if __name__=='__main__':main()
