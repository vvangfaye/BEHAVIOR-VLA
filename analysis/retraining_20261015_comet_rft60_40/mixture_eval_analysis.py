"""Read-only evidence analysis, separate from the frozen live evaluation code."""
from pathlib import Path
import importlib.util,json,statistics,time

P=Path(__file__).resolve().parent
E=P/'paired_fixed20_step5000'
spec=importlib.util.spec_from_file_location('frozen_eval_scorer',E/'frozen/scorer.py')
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)

def analyze():
 plan=json.loads((E/'plan.json').read_text());pairs=[tuple(x) for x in plan['pairs']]
 selected=json.loads((P/'selected_full_episodes.json').read_text())['episodes']
 full_tasks={e['task_name'] for e in selected};full_pairs={(e['task_name'],e['task_instance_id']) for e in selected}
 groups={'all':pairs,'task_has_full_success':[x for x in pairs if x[0] in full_tasks],
  'task_has_no_full_success':[x for x in pairs if x[0] not in full_tasks],
  'exact_instance_has_full_success':[x for x in pairs if x in full_pairs],
  'exact_instance_has_no_full_success':[x for x in pairs if x not in full_pairs]}
 refs=[('1688977','baseline_accepted.json','baseline'),('previous_5000','step5000_candidate_accepted.json','candidate'),
  ('previous_10000','step10000_candidate_accepted.json','candidate'),('previous_20000','step20000_candidate_accepted.json','candidate')]
 model_rows={}
 for name,file,role in refs:model_rows[name]=scorer.validate(json.loads((E/file).read_text()),plan['pairs'],role)
 accepted=E/'candidate_accepted.json'
 if accepted.exists():model_rows['rft60_40_5000']=scorer.validate(json.loads(accepted.read_text()),plan['pairs'],'candidate')
 group_rows={}
 for name,keys in groups.items():
  group_rows[name]={'count':len(keys),'pairs':keys,'models':{}}
  for label,rows in model_rows.items():
   group_rows[name]['models'][label]=dict(episodes=len(keys),successes=sum(rows[k]['model_success'] for k in keys),
    mean_q=statistics.mean(rows[k]['q_score'] for k in keys) if keys else None)
 current={}
 for file in (E/'reports').glob('candidate-*.json'):
  data=json.loads(file.read_text())
  if data.get('state')!='accepted':continue
  row=data['record'];key=(row['task'],row['instance_id']);assert key in pairs and key not in current
  assert data['protocol_sha256']==plan['protocol_sha256'] and row['artifact_accepted'] and row['policy_seed_acknowledged']
  current[key]=row
 matched={}
 paired={}
 if current:
  for group,keys in groups.items():
   keys=[k for k in keys if k in current]
   if not keys:continue
   allmodels=dict(model_rows,rft60_40_5000=current)
   matched[group]={label:{'episodes':len(keys),'successes':sum(rows[k]['model_success'] for k in keys),'mean_q':statistics.mean(rows[k]['q_score'] for k in keys)} for label,rows in allmodels.items()}
  for label,rows in model_rows.items():
   if label=='rft60_40_5000':continue
   changes=[]
   for key in pairs:
    if key not in current:continue
    old,new=rows[key],current[key]
    changes.append(dict(task=key[0],instance_id=key[1],previous_q=old['q_score'],current_q=new['q_score'],
     delta_q=new['q_score']-old['q_score'],previous_success=old['model_success'],current_success=new['model_success'],
     task_has_full_success=key[0] in full_tasks,exact_instance_has_full_success=key in full_pairs))
   paired[label]=dict(matched_episodes=len(changes),complete=len(changes)==len(pairs),
    mean_delta_q=statistics.mean(x['delta_q'] for x in changes),
    improved=sum(x['delta_q']>1e-12 for x in changes),regressed=sum(x['delta_q']< -1e-12 for x in changes),
    tied=sum(abs(x['delta_q'])<=1e-12 for x in changes),
    new_successes=sum(x['current_success'] and not x['previous_success'] for x in changes),
    lost_successes=sum(x['previous_success'] and not x['current_success'] for x in changes),records=changes)
 result=dict(observed_at=time.time(),analysis_scope='fixed20, training-seen regression; no generalization claim',
  current_accepted_episodes=len(current),candidate_complete=accepted.exists(),groups=group_rows,matched_partial_results=matched,
  paired_comparisons=paired,
  model_selection_warning='Do not compare a partial candidate mean against a full20 historical mean; matched subset only',
  average_sampling_expectation={'original_episode_draws':960000*.6/15949,'full_episode_draws':960000*.4/270,
   'task_with_full_success_probability':.6/100+.4/52,'task_without_full_success_probability':.6/100,
   'within_source_sampler':'uniform task then instance then episode, legal frame anchor; averages are not per-episode guarantees'},
  next_training_launched=False,current_training_budget_completed=5000)
 (P/'mixture_effects_current.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'accepted_current':len(current),'groups':{k:{'count':v['count'],'models':v['models']} for k,v in group_rows.items()},'matched_partial':matched},ensure_ascii=False))

if __name__=='__main__':analyze()
