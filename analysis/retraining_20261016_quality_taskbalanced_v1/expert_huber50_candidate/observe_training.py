"""Read-only native and bounded artifact observation for Huber 50, iid 241869."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import importlib.util,json,re,sys,time
r=Path('/Users/faye/code/Behavior-1k Challenge/analysis/huber50_submission_20261017_g0')
p=r/'expert_huber50_candidate/submit_selected.py'
spec=importlib.util.spec_from_file_location('h50_observer_source',p)
wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
shared=wrapper.shared;shared.load_credentials()
dispatch=shared.module('h50_observer_dispatch',r/'expert_control/dispatch_support.py')
stage=r/'expert_huber50_candidate/selected_stage'
sys.path.insert(0,str(stage/'bundle'))
from nebula.submit_oss import OSSClient
client=OSSClient();job=dispatch.read(stage/'bundle/nebula/worker_job.json')
receipt=dispatch.read(stage/'receipt.json')
assert receipt['identifiers']==['241869'] and job['run_id']=='b1k-expert-huber50-1688977-20261017'
prefix=client.key(job['output'])+'/attempts/'+job['attempt_id']
checkpoint_prefix=client.key(job['output'])+'/checkpoints/pi05_b1k/'+job['run_id']+'/1000'
checkpoint_page=client.bucket.list_objects(prefix=checkpoint_prefix,max_keys=1000)
checkpoint_progress={'listed_objects':len(checkpoint_page.object_list),
                     'listed_bytes':sum(item.size for item in checkpoint_page.object_list),
                     'listing_truncated':checkpoint_page.is_truncated}
metadata_key=checkpoint_prefix+'/_CHECKPOINT_METADATA'
metadata_size=client.head(metadata_key)
if metadata_size and metadata_size<=16384:
 checkpoint_progress['commit_timestamp_nsecs']=json.loads(client.read(metadata_key)).get('commit_timestamp_nsecs')
outputs={}
bootstrap_page=client.bucket.list_objects(prefix=client.key(job['output'])+'/transport_failure_',max_keys=32)
bootstrap_failures=[]
for item in bootstrap_page.object_list:
 if item.size>524288:raise ValueError('Unexpected oversized bootstrap failure')
 value=json.loads(client.read(item.key))
 bootstrap_failures.append({k:v for k,v in value.items() if k in ('stage','error_type','observed_at')})
names=['rank-0/started.json','rank-1/started.json','rank-0/effective_training.json',
             'rank-1/effective_training.json','rank-0/_RANK_SUCCESS.json','rank-1/_RANK_SUCCESS.json',
             'checkpoint_restore_ok.json','_TRAINING_STAGE_SUCCESS.json','rank-0/failure.json','rank-1/failure.json']
names += [f'rank-{rank}/{name}' for rank in (0,1) for name in
          ('hardware.json','checkpoint_source_verified.json','original_cache_space.json',
           'original_cache_verified.json','data_staged.json','runtime.json')]
for name in names:
 key=prefix+'/'+name;n=client.head(key)
 if n is None:continue
 if n>524288:raise ValueError('Unexpected oversized JSON artifact')
 value=json.loads(client.read(key))
 if name.endswith('failure.json'):
  value={k:v for k,v in value.items() if k in ('error_type','observed_at','formal_training')}
 elif name.endswith('effective_training.json'):
  arguments=value.get('argv',[])
  actual_delta=(arguments[arguments.index('--model.action-loss-huber-delta')+1]
                if '--model.action-loss-huber-delta' in arguments else None)
  value={k:v for k,v in value.items() if k!='argv'}
  value['action_loss_huber_delta_argument']=actual_delta
 elif name.endswith('runtime.json'):
  value={k:v for k,v in value.items() if k!='stdout'}
 outputs[name]=value
trainer_logs={}
for rank in (0,1):
 key=prefix+f'/rank-{rank}/trainer.log';size=client.head(key)
 if size is None:continue
 row={'bytes':size}
 if size:
  response=client.bucket.get_object(key,byte_range=(max(0,size-65536),size-1))
  try:tail=response.read(65537).decode(errors='replace')
  finally:response.resp.response.close()
  row.update(train_state_marker_in_tail='Initialized train state:' in tail,
             data_loader_marker_in_tail='Initialized data loader:' in tail,
             checkpoint_wait_marker_in_tail='Waiting for checkpoint manager to finish' in tail,
             exact_restore_comparison_marker_in_tail='Comparing all restored parameter, EMA and optimizer leaves exactly' in tail,
             traceback_marker='Traceback (most recent call last)' in tail,
             exception_types=sorted(set(re.findall(r'\b(ValueError|TypeError|RuntimeError|KeyError|AssertionError|XlaRuntimeError|FileNotFoundError|ModuleNotFoundError|ResourceExhaustedError):',tail))))
 trainer_logs[str(rank)]=row
key=prefix+'/training_metrics.jsonl';n=client.head(key);metrics=[]
if n:
 start=max(0,n-262144);response=client.bucket.get_object(key,byte_range=(start,n-1))
 try:raw=response.read(262145)
 finally:response.resp.response.close()
 lines=raw.splitlines()
 if start:lines=lines[1:]
 for line in lines:
  try:row=json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
  if row.get('kind')=='training_update':metrics.append(row)
receipts={name:dispatch.read(stage/name) for name in ['intent.json','receipt.json','runtime_receipt.json',
          'checkpoint_source_preflight.json','original_source_preflight.json'] if (stage/name).exists()}
print(json.dumps({'observed_at':time.time(),'native':dispatch.status('241869'),'receipts':receipts,
                  'checkpoint_progress':checkpoint_progress,
                  'bootstrap_failures':bootstrap_failures,'bootstrap_failure_listing_truncated':bootstrap_page.is_truncated,
                  'outputs':outputs,'trainer_log_metadata':trainer_logs,'latest_metric':metrics[-1] if metrics else None}))
'''


def main():
    command = shlex.quote('/Users/faye/code/Behavior-1k Challenge/.venv/bin/python') + ' -c ' + shlex.quote(REMOTE)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-o', 'ConnectionAttempts=1', 'macpro', command],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Read-only training observation failed; no submission attempted')
    observed = json.loads(result.stdout)
    if observed['native']['iid'] != '241869':
        raise ValueError('Unexpected task identity')
    (ROOT / 'current_training_observation.json').write_text(json.dumps(observed, indent=2) + '\n')
    receipt_dir = ROOT / 'submission_receipts'
    receipt_dir.mkdir(exist_ok=True)
    for name, data in observed['receipts'].items():
        (receipt_dir / name).write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'native': observed['native'], 'outputs_present': sorted(observed['outputs']),
                      'bootstrap_failures': observed['bootstrap_failures'],
                      'trainer_log_metadata': observed['trainer_log_metadata'],
                      'latest_metric': observed['latest_metric']}))


if __name__ == '__main__':
    main()
