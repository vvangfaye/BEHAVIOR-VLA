"""Two finite fixed20 campaigns, each capped at five Spot jobs; no training mutations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.chmod(0o600);q.replace(p)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def suites():
 return [(ROOT/'paired_fixed20_step5000',ROOT/'paired_fixed20_step5000_5x4'),(ROOT/'expert_fixed20_step1000',ROOT/'expert_fixed20_step1000_5x4')]

def probe(base,plan):
 cmd=['ssh','-oBatchMode=yes','-oConnectTimeout=8','-oStrictHostKeyChecking=yes','macpro',
  'cd "/Users/faye/code/Behavior-1k Challenge/BEHAVIOR-VLAs" && source nebula/config.sh >/dev/null 2>&1 && export OSS_ENDPOINT=https://oss-cn-zhangjiakou.aliyuncs.com && export PYTHONPATH="/Users/faye/code/Behavior-1k Challenge/BEHAVIOR-VLAs" && /Users/faye/Library/Caches/behavior-data-prep-20261013-v1/env/bin/python -']
 cp=subprocess.run(cmd,input=(base.ROOT/'frozen/remote_probe.py').read_text(),text=True,capture_output=True,timeout=150)
 if cp.returncode:raise RuntimeError('Checkpoint snapshot unavailable; SSH exit '+str(cp.returncode))
 snap=json.loads(cp.stdout)
 # Never publish signed URLs in status output.
 save(base.ROOT/'private/last_probe.json',snap)
 if snap.get('state')!='committed_checkpoint'+str(plan['candidate_step']):return None
 if snap['formal_contract']!=plan['formal_contract']:raise ValueError('Committed checkpoint formal contract differs')
 return snap

def native_rows(m,folder):
 intents=[read(p) for p in sorted((folder/'intents').glob('*.json'))]
 with ThreadPoolExecutor(max_workers=5) as pool:states=list(pool.map(m.observe,intents))
 for intent,state in zip(intents,states):save(folder/'observations'/(intent['id']+'.json'),state)
 return intents,states

def comparison():
 from compare_candidate import validate,summary,paired
 hist=read(ROOT/'evaluation_history/manifest.json');pairs=list(map(tuple,hist['pairs']));models={}
 source=hist['models']['1688977'];raw=(ROOT/'evaluation_history'/source['file']).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('Baseline changed')
 models['baseline1688977']=validate(json.loads(raw),pairs,hist['protocol_sha256'],baseline=True)
 for name,folder in [('mixed1000',ROOT/'paired_fixed20_step1000_5x4'),('mixed5000',suites()[0][1]),('expert1000',suites()[1][1])]:
  plan=read(folder/'plan.json');bundle=hashlib.sha256((folder/'frozen_sha256.json').read_bytes()).hexdigest();rows=[]
  for i in range(20):
   p=folder/'reports'/f'candidate-{i:02d}.json'
   if not p.exists():break
   report=read(p)
   if report.get('state')!='accepted':break
   if report.get('worker_bundle_sha256')!=bundle or report.get('checkpoint_manifest_sha256')!=plan['expected_checkpoint_manifest_sha256']:raise ValueError('Wrong evaluated checkpoint or bundle')
   rows.append(report['record'])
  if len(rows)==20:models[name]=validate({'protocol_sha256':plan['protocol_sha256'],'records':rows},pairs,hist['protocol_sha256'])
 result={'observed_at':time.time(),'complete':len(models)==4,'scope':'same fixed20, seeds0; training-seen regression panel',
  'generalization_proven':False,'summaries':{n:summary(v,pairs) for n,v in models.items()},'paired_comparisons':{}}
 for name,rows in models.items():
  if name!='baseline1688977':result['paired_comparisons'][name+'_vs_baseline']=paired(models['baseline1688977'],rows,pairs)
 if 'expert1000' in models:
  for name in ('mixed1000','mixed5000'):
   if name in models:result['paired_comparisons']['expert1000_vs_'+name]=paired(models[name],models['expert1000'],pairs)
 save(ROOT/'dual_experiment_comparison.json',result)
 if result['complete']:
  from build_final_comparison import main as publish_final_comparison
  publish_final_comparison()
 return result['complete']

def tick():
 # The old early evaluation writers must already be terminal; no duplicate GPU queue.
 recon=read(ROOT/'paired_fixed20_step1000_5x4/readonly_reconciliation.json')
 if not recon['all_jobs_terminal'] or not recon['comparison']['complete']:raise RuntimeError('Original early1000 panel is not finished')
 loaded=[];active=0
 for n,(export,batch) in enumerate(suites()):
  base=module('export_'+str(n),export/'controller.py');ctl=module('batch_'+str(n),batch/'controller.py')
  plan=read(export/'plan.json');bplan=read(batch/'plan.json')
  if bplan['pairs']!=read(ROOT/'paired_fixed20_step1000_5x4/plan.json')['pairs']:raise ValueError('Fixed20 changed')
  groups=bplan['batch_groups']
  if len(groups)!=5 or any(len(g['indices'])!=4 for g in groups) or sorted(i for g in groups for i in g['indices'])!=list(range(20)):raise ValueError('Expected exact five-by-four partition')
  if bplan['max_active_gpu_jobs']!=5 or bplan['protocol_sha256']!=read(ROOT/'evaluation_history/manifest.json')['protocol_sha256']:raise ValueError('Protocol or resource budget changed')
  for folder in (export,batch):
   expected=read(folder/'frozen_sha256.json');actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'frozen').glob('*.py')}
   if actual!=expected:raise ValueError('Frozen source changed')
  intents,states=native_rows(ctl,batch);active+=sum(s['state'] not in ctl.TERMINAL for s in states)
  loaded.append((base,ctl,export,batch,plan,bplan,intents,states))
 from mixed_recovery import tick as recovery_tick
 mixed_active=sum(s['state'] not in loaded[0][1].TERMINAL for s in loaded[0][7])
 recovery_active=recovery_tick(mixed_active)
 active+=recovery_active
 from stalled_recovery import tick as stalled_tick
 stalled_active=stalled_tick([mixed_active+recovery_active,
     sum(s['state'] not in loaded[1][1].TERMINAL for s in loaded[1][7])])
 active+=sum(stalled_active)
 recovery_active+=stalled_active[0]
 if mixed_active+recovery_active>5:raise ValueError('Mixed-model GPU evaluation concurrency exceeds five')
 if active>10:raise ValueError('Combined GPU evaluation concurrency exceeds ten')
 output=[]
 for base,ctl,export,batch,plan,bplan,intents,states in loaded:
  client=base.aoss_client();epath=export/'intents/export.json';ready=None
  if not epath.exists():
   snap=probe(base,plan)
   if snap is not None:
    save(export/'private/export_snapshot.json',snap)
    base.submit(client,plan,'export','export',snapshot=snap)
  if epath.exists():
   intent=read(epath);native=base.observe(intent);save(export/'observations/export.json',native)
   ready=base.read_report(client,plan,'export')
   if ready:
    save(export/'reports/export.json',ready)
    if ready.get('state')!='accepted':raise RuntimeError('Export failed for '+batch.name)
    if ready.get('completed_updates')!=plan['candidate_step'] or not ready.get('full_afs_sha256_readback'):raise ValueError('Incomplete export proof')
   elif native['state'] in base.TERMINAL:raise RuntimeError('Export terminal without accepted report; reconcile before retry')
  # Source identity is read from an existing owned template, no capacity polling.
  own_active=sum(s['state'] not in ctl.TERMINAL for s in states)
  campaign_active=own_active+(recovery_active if batch.name=='paired_fixed20_step5000_5x4' else stalled_active[1])
  if ready and len(intents)<5 and campaign_active<5:
   template=base.cli_json(['acp','jobs','describe',bplan['gpu_copy_template'],'--workspace-name=share-space','-o','json'])
   if template['uid']!=bplan['gpu_copy_template_uid'] or template['ownership']['user_name']!='260010165':raise ValueError('Evaluation template ownership changed')
   if any(r['image_path']!=bplan['image'] for r in template['roles']):raise ValueError('Evaluation template image changed')
   bplan.update(expected_checkpoint_manifest_sha256=ready['manifest_sha256'],final_checkpoint_restore_verified=True,execution_enabled=True,launch_preflight_complete=True)
   save(batch/'plan.json',bplan)
   existing={i['id'] for i in intents}
   available=5-campaign_active
   for group in bplan['batch_groups']:
    if group['id'] not in existing and available>0:
     ctl.submit(client,bplan,group);available-=1
   intents,states=native_rows(ctl,batch);active=active-own_active+sum(s['state'] not in ctl.TERMINAL for s in states)
  reports=[]
  for i in range(20):
   canonical=batch/'reports'/f'candidate-{i:02d}.json'
   report=None
   if canonical.exists():
    cached=read(canonical)
    if cached.get('state')=='accepted' and cached.get('recovery_provenance'):
     if (cached.get('checkpoint_manifest_sha256')!=bplan['expected_checkpoint_manifest_sha256']
       or cached.get('worker_bundle_sha256')!=hashlib.sha256((batch/'frozen_sha256.json').read_bytes()).hexdigest()
       or cached.get('protocol_sha256')!=bplan['protocol_sha256']):raise ValueError('Invalid recovered canonical report')
     report=cached
   if report is None:
    report=ctl.get_report(client,bplan,f'candidate-{i:02d}')
   if report:reports.append(report)
  accepted=sum(r.get('state')=='accepted' for r in reports)
  output.append({'campaign':batch.name,'export_accepted':bool(ready),'created_gpu_jobs':len(intents),'accepted_episodes':accepted,'states':[s['state'] for s in states]})
  if len(intents)==5 and all(s['state'] in ctl.TERMINAL for s in states) and accepted!=20:
   output[-1]['requires_recovery']=True
 done=comparison();status={'observed_at':time.time(),'complete':done,'campaigns':output,'training_modified':False}
 save(ROOT/'dual_evaluation_status.json',status);print(json.dumps(status),flush=True)
 return done and active==0

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
 with (ROOT/'dual_evaluation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  save(ROOT/'dual_evaluation_launch.json',{'pid':os.getpid(),'started_at':time.time(),'entry':str(Path(__file__).resolve())})
  deadline=min(read(b/'plan.json')['deadline_epoch'] for a,b in suites())
  while time.time()<deadline:
   if (ROOT/'STOP_DUAL_EVALUATION').exists():return
   try:
    if tick():return
   except Exception as exc:
    save(ROOT/'dual_evaluation_error.json',{'observed_at':time.time(),'error_type':type(exc).__name__,'message':str(exc)[:220]})
    if args.once:raise
   if args.once:return
   time.sleep(60)

if __name__=='__main__':main()
