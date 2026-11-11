"""Inspect logged predicates and reconstruct rewards using the frozen formula."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYS = ('toggled_on', 'nextto_tv', 'shared_tv_cabinet')
data = json.loads((ROOT/'traces.json').read_text())
episodes, all_rows = [], []
for item in data:
    receipt = item['receipt']
    rows = [json.loads(line) for line in item['trace'].splitlines()]
    assert len(rows) == receipt['chunks']
    previous = receipt['goal_summary']['initial']
    changes = []
    sums = [0.0, 0.0]
    discounted_shaping = 0.0
    chunks = []
    for n, row in enumerate(rows):
        done = n == len(rows)-1
        for key in KEYS:
            if row[key] != previous[key]:
                changes.append({'step': row['step'], 'goal': key, 'to': row[key], 'chunk': n+1})
        task = (row['potential'] - previous['potential'])*10/3
        shaping = .99*(0 if done else row['potential']) - previous['potential']
        discounted_shaping += .99**n*shaping
        sums[0] += task
        sums[1] += shaping
        derived = {'episode': item['episode'], 'instance': receipt['instance'], 'seed': receipt['seed'],
                   'chunk':n+1, 'step':row['step'], 'phi':row['potential'],
                   **{key:row[key] for key in KEYS}, 'score':row['official']['score'],
                   'success':row['official']['success'], 'done':done,
                   'task_reward_reconstructed':task, 'shaping_reward_reconstructed':shaping,
                   'reward_reconstructed':task+shaping}
        all_rows.append(derived)
        chunks.append(derived)
        previous = row
    assert abs(sums[0]-receipt['task_reward']) < 1e-5
    assert abs(sums[1]-receipt['shaping_reward']) < 1e-7
    assert abs(discounted_shaping) < 1e-7
    quick_reversals=[]
    for key in KEYS:
        events=[c for c in changes if c['goal']==key]
        for a,b in zip(events,events[1:]):
            if b['step']-a['step'] <=64:
                quick_reversals.append({'goal':key,'from_step':a['step'],'to_step':b['step'],'first_value':a['to']})
    runs=[]
    start=0
    state=receipt['goal_summary']['initial']
    for c in chunks:
        if any(c[key] != state[key] for key in KEYS):
            runs.append({'from_step':start,'to_step':c['step'],'length_steps':c['step']-start,
                         'phi':state['potential'] if 'potential' in state else state['phi']})
            start=c['step']
            state=c
    runs.append({'from_step':start,'to_step':rows[-1]['step'],'length_steps':rows[-1]['step']-start,
                 'phi':state['potential'] if 'potential' in state else state['phi']})
    episodes.append({'episode':item['episode'],'instance':receipt['instance'],'seed':receipt['seed'],
                     **receipt['status'],'chunks':len(rows),'changes':changes,'quick_reversals_64steps':quick_reversals,
                     'losses':sum(not c['to'] for c in changes),'task_reward':sums[0],'shaping_reward':sums[1],
                     'last_reward':chunks[-1]['reward_reconstructed'],'no_change_runs':runs,
                     'observation_dt':item['initial']['observation_dt'],'videos':item['videos']})
summary={'scope':'32 first-round modem training rollouts, original checkpoint; no Astra judgments',
         'reward_source':'per-chunk reconstruction from frozen code and physical-goal traces; sums verified against saved receipts',
         'episode_count':len(episodes),'successes':sum(e['success'] for e in episodes),'total_chunks':len(all_rows),
         'episodes_with_goal_loss':[e['episode'] for e in episodes if e['losses']],
         'episodes_with_quick_reversal_64steps':[e['episode'] for e in episodes if e['quick_reversals_64steps']],
         'episodes_without_any_goal_change':[e['episode'] for e in episodes if not e['changes']],
         'total_goal_changes':sum(len(e['changes']) for e in episodes),
         'total_goal_losses':sum(e['losses'] for e in episodes),
         'reward_histogram':dict(sorted(Counter(round(c['reward_reconstructed'],3) for c in all_rows).items())),
         'episodes':episodes}
(ROOT/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (ROOT/'chunk_rewards_reconstructed.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
print(json.dumps({k:v for k,v in summary.items() if k!='episodes'},ensure_ascii=False,indent=2))
for e in episodes:
    print('episode',e['episode'],'instance',e['instance'],'success',e['success'],'changes',e['changes'])
    if e['quick_reversals_64steps']:print('QUICK',e['quick_reversals_64steps'])
    if not e['success']:print('LONGEST',max(e['no_change_runs'],key=lambda r:r['length_steps']))
