"""Read only the exact active run's OSS metrics and checkpoint assets."""
import collections
import hashlib
import json
import math
import statistics
import time
from nebula.submit_oss import OSSClient

RUN='b1k-four-source-finetune-20261014_080449-20k'
ROOT='wangfaye/result/behavior-challenge/four_source_finetune/'+RUN
ATTEMPT=ROOT+'/attempts/stage20000-g0'
c=OSSClient()

def read_json(key,limit=128*1024**2):
    n=c.head(key)
    if not n or n>limit:raise ValueError('Missing or oversized audit object')
    body=c.bucket.get_object(key)
    try:raw=body.read(n+1)
    finally:body.resp.response.close()
    if len(raw)!=n:raise ValueError('Audit object length changed')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()

def expected_lr(clock):
    peak=2.5e-6;floor=5e-7;warm=200;end=20000;initial=peak/(warm+1)
    if clock<warm:return initial+(peak-initial)*clock/warm
    return floor+.5*(peak-floor)*(1+math.cos(math.pi*min(max((clock-warm)/(end-warm),0),1)))

key=ATTEMPT+'/training_metrics.jsonl';head=c.bucket.head_object(key);body=c.bucket.get_object(key)
try:rows=[json.loads(line) for line in body.read().decode().splitlines() if line.strip()]
finally:body.resp.response.close()
previous=0;nonmonotonic=[];lr_errors=[];nonfinite=[];clock_errors=[];mask_errors=[];lr_increases=[]
previous_lr=None
for row in rows:
    step=row['completed_updates'];m=row['metrics'];clock=m['optimizer_internal_step'];rate=m['learning_rate']
    if step<=previous:nonmonotonic.append(step)
    if not math.isclose(rate,expected_lr(clock),rel_tol=2e-5,abs_tol=1e-12):
        lr_errors.append(dict(step=step,observed=rate,expected=expected_lr(clock),clock=clock))
    if m.get('nonfinite_count',0)!=0:nonfinite.append(step)
    if m.get('lr_clock_matches_step')!=1:clock_errors.append(step)
    if m.get('subtask_loss/full_episode',0)!=0:mask_errors.append(step)
    if step>210 and previous_lr is not None and rate>previous_lr*(1+2e-6):lr_increases.append(step)
    previous=step;previous_lr=rate
last=rows[-1];latest=last['metrics'];counts={k:latest.get('source_draws_total/'+k) for k in ('replay','recovered','collected','full_episode')}
total=sum(v for v in counts.values() if v is not None)
def window(lo,hi):
    selected=[r for r in rows if lo<r['completed_updates']<=hi]
    names=['loss','action_loss','subtask_loss','subtask_token_accuracy','grad_norm']
    names += [f'{metric}/{kind}' for metric in ('action_loss','subtask_loss') for kind in counts]
    return {'rows':len(selected),'from_exclusive':lo,'to_inclusive':hi,
            'mean_metrics':{n:statistics.mean(r['metrics'][n] for r in selected if n in r['metrics']) for n in names if any(n in r['metrics'] for r in selected)}}
result={'observed_at':time.time(),'run_id':RUN,'latest_step':last['completed_updates'],'metrics_last_modified_epoch':head.last_modified,
        'metrics_rows':len(rows),'cumulative_draws':counts,'cumulative_total':total,
        'expected_draw_total':last['completed_updates']*192,'actual_fractions':{k:v/total for k,v in counts.items()},
        'last_learning_rate':latest['learning_rate'],'peak_logged_learning_rate':max(r['metrics']['learning_rate'] for r in rows),
        'lr_mismatch_rows':lr_errors[:10],'lr_mismatch_count':len(lr_errors),'lr_increases_after_warmup':lr_increases,
        'nonmonotonic_steps':nonmonotonic,'nonfinite_steps':nonfinite,'clock_error_steps':clock_errors,
        'full_episode_text_loss_nonzero_steps':mask_errors,'milestones':[],
        'loss_windows':[window(500,1000),window(4500,5000),window(9500,10000),window(last['completed_updates']-500,last['completed_updates'])]}
for step in (1,100,200,1000,5000,10000,last['completed_updates']):
    selected=next((r for r in rows if r['completed_updates']==step),None)
    if selected:result['milestones'].append({'completed_updates':step,'clock':selected['metrics']['optimizer_internal_step'],
                    'actual_lr':selected['metrics']['learning_rate'],'expected_lr':expected_lr(selected['metrics']['optimizer_internal_step'])})
result['effective_training']=[]
for rank in (0,1):
    data,digest=read_json(ATTEMPT+f'/rank-{rank}/effective_training.json',4*1024**2)
    argv=data['argv'];selected={}
    for i,arg in enumerate(argv[:-1]):
        if arg.startswith(('--lr-schedule.','--optimizer.')) or arg in ('--num-train-steps','--batch-size','--seed','--ema-decay',
            '--weight-loader.params-path','--data.subtask-whole-task-prob','--data.subtask-mixture-path','--learning-rate-contract-path'):
            selected[arg]=argv[i+1]
    result['effective_training'].append({'rank':rank,'sha256':digest,'global_batch':data['global_batch'],'gpus':data['gpus'],
        'processes':data['processes'],'stage_end_step':data['stage_end_step'],'fresh_from1688977':data['fresh_from1688977'],
        'learning_rate_decay_end':data['learning_rate_decay_end'],'selected_argv':selected})
result['checkpoint_assets']={}
for step in (5000,10000):
    prefix=ROOT+'/checkpoints/pi05_b1k/'+RUN+'/'+str(step)
    assets={}
    for name in ('learning_rate_contract.json','formal_run_contract.json','subtask_mixture_execution.json','realized_source_counts.json',
                 'subtask_conditioning.json','frame_memory.json'):
        key=prefix+'/assets/'+name
        if c.head(key):
            data,digest=read_json(key,4*1024**2);assets[name]={'data':data,'sha256':digest}
    meta,_=read_json(prefix+'/_CHECKPOINT_METADATA',4*1024**2)
    assets['committed_at']=meta.get('commit_timestamp_nsecs')
    norm,_=read_json(prefix+'/assets/b1k_full100_joint_subtask_v1_drop_episode_s1/norm_stats.json',4*1024**2)
    assets['norm_stats_file_sha256']=hashlib.sha256(c.read(prefix+'/assets/b1k_full100_joint_subtask_v1_drop_episode_s1/norm_stats.json')).hexdigest()
    result['checkpoint_assets'][str(step)]=assets
mix,mixsha=read_json(ROOT+'/checkpoints/pi05_b1k/'+RUN+'/10000/assets/subtask_mixture.json')
result['consumed_mixture']={'step':mix['step'],'consumed_samples':mix['consumed_samples'],'sha256':mixsha,
                          'fingerprint':mix['fingerprint'],'semantic_plan':{k:v for k,v in mix['semantic_plan'].items() if k!='phases'},
                          'phases':[],'source_census':{}}
for phase in mix['semantic_plan']['phases']:
    result['consumed_mixture']['phases'].append({'start_step':phase['start_step'],'end_step':phase['end_step'],
        'sources':[{'kind':s['kind'],'weight':s['weight'],'groups':len(s['group_weights']),
                    'tasks':len({g['task_name'] for g in s['group_weights']}),'min_group_weight':min(g['weight'] for g in s['group_weights']),
                    'max_group_weight':max(g['weight'] for g in s['group_weights'])} for s in phase['sources']]})
for index in mix['indexes']:
    kind=index['source_kind'];dest=result['consumed_mixture']['source_census'].setdefault(kind,
        {'indexes':0,'episodes':0,'episode_frames':0,'intervals':0,'legal_anchors':0,'tasks':set(),'groups':set(),
         'intervals_by_origin':collections.Counter(),'anchors_by_origin':collections.Counter(),'example_schema':{}})
    dest['indexes']+=1;dest['episodes']+=len(index['episodes']);dest['intervals']+=len(index['intervals'])
    dest['episode_frames']+=sum(e['length'] for e in index['episodes'])
    dest['tasks'].update(e['task_name'] for e in index['episodes'])
    ep={e['episode_index']:e for e in index['episodes']}
    for interval in index['intervals']:
        anchors=interval['end']-interval['start']-32+1;dest['legal_anchors']+=anchors
        dest['groups'].add((ep[interval['episode_index']]['task_name'],interval['subtask']))
        origin=interval.get('label_origin',interval.get('provenance','unspecified'))
        if not isinstance(origin,str):origin='structured_provenance'
        dest['intervals_by_origin'][origin]+=1;dest['anchors_by_origin'][origin]+=anchors
    if not dest['example_schema']:
        dest['example_schema']={'index_keys':list(index),'episode_keys':list(index['episodes'][0]),
          'interval_example':index['intervals'][0],'provenance_keys':list(index.get('provenance',{})),
          'small_provenance':{k:v for k,v in index.get('provenance',{}).items() if isinstance(v,(str,int,float,bool))}}
for value in result['consumed_mixture']['source_census'].values():
    value['tasks']=sorted(value['tasks']);value['groups']=len(value['groups'])
    value['intervals_by_origin']=dict(value['intervals_by_origin']);value['anchors_by_origin']=dict(value['anchors_by_origin'])
print(json.dumps(result,ensure_ascii=False))
