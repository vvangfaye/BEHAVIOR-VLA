from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
paths={'base':ROOT/'analysis/training_boundary_audit_20261019/cpu_raw_feasibility_20261020/baseline36_raw.json','ft':ROOT/'analysis/training_boundary_audit_20261019/calibration_cpu_followup_20261020/expert1000_36_raw.json'}
docs={k:json.load(open(p)) for k,p in paths.items()};maps={k:{(r['episode_index'],r['frame_index']):r for r in d['sample_keys']} for k,d in docs.items()}
rows=[]
for key,b in maps['base'].items():
 f=maps['ft'][key];br=b['raw_actions'];fr=f['raw_actions']
 assert br['valid_mask']==fr['valid_mask'];gt=np.array(br['ground_truth']);assert np.array_equal(gt,np.array(fr['ground_truth']))
 if not np.array(br['valid_mask'])[:16].all():continue
 gt=gt[:16];flat=np.ptp(gt,axis=0)==0
 r={'episode':key[0],'frame':key[1],'task':b['task'],'flat':flat.tolist()}
 for name,raw in [('base',br),('ft',fr)]:
  pred=np.array(raw['prediction'])[:16];r[name]={'mse':np.mean((pred-gt)**2,axis=0).tolist(),'jitter':np.var(pred,axis=0).tolist()}
 rows.append(r)
blocks=[]
for task in sorted(set(r['task'] for r in rows)):
 for j in range(23):
  chosen=[r for r in rows if r['task']==task and r['flat'][j]]
  if len(chosen)<3:continue
  blocks.append({'task':task,'coordinate':j,'n':len(chosen),**{m:{name:float(np.mean([r[m][name][j] for r in chosen])) for name in ['mse','jitter']} for m in ['base','ft']}})
result={'input_hashes':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},'complete_prefixes':len(rows),'constant_channel_blocks':len(blocks),'ft_jitter_increases':sum(r['ft']['jitter']>r['base']['jitter'] for r in blocks),'ft_mse_increases':sum(r['ft']['mse']>r['base']['mse'] for r in blocks),'blocks':blocks,'limits':['One draw per observation; at most nine episodes, three selected tasks','Coordinates not independent','Exact constant target commands are not proof of a stationary physical joint','No closed-loop or causal intervention']}
Path(__file__).with_name('flat_prediction_result.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['blocks','input_hashes']})
