"""Read-only analysis of accepted repetitions, including incomplete groups."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import statistics
import time

ROOT=Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    plan=json.loads((ROOT/'plan.json').read_text())
    status=json.loads((ROOT/'status.json').read_text())
    reports={};proof={};failures=[]
    for index in range(30):
        p=ROOT/'reports'/f'baseline-{index:02d}.json'
        if not p.exists():continue
        raw=p.read_bytes();d=json.loads(raw)
        if d['state']!='accepted':failures.append({'index':index,'state':d['state'],'error_type':d.get('error_type')});continue
        assert d['protocol_sha256']==plan['protocol_sha256']
        assert d['worker_bundle_sha256']==hashlib.sha256((ROOT/'frozen_sha256.json').read_bytes()).hexdigest()
        assert d['checkpoint_manifest_sha256']==hashlib.sha256(json.dumps(plan['baseline_files'],sort_keys=True).encode()).hexdigest()
        r=d['record'];assert [r['task'],r['instance_id']]==plan['pairs'][index]
        assert r['policy_seed']==r['environment_seed']==0 and r['policy_seed_acknowledged']
        assert d['repeat']==index%3
        reports[index]=d;proof[p.name]=hashlib.sha256(raw).hexdigest()
    rows=[];flat=[]
    for group,(task,instance) in enumerate(plan['protocol']['unique_pairs']):
        ds=[reports.get(group*3+r) for r in range(3)]
        known=[d for d in ds if d]
        qs=[d['record']['q_score'] for d in known]
        ss=[d['record']['model_success'] for d in known]
        row=dict(task=task,instance=instance,completed=len(known),
            scores=[d['record']['q_score'] if d else None for d in ds],
            successes=[d['record']['model_success'] if d else None for d in ds],
            steps=[d['record']['steps'] if d else None for d in ds],
            complete=len(known)==3,score_range=max(qs)-min(qs) if len(qs)>=2 else None,
            score_std=statistics.pstdev(qs) if len(qs)>=2 else None,
            success_flip=len(set(ss))>1 if len(ss)>=2 else None,
            initial_physics_identical=len({d['initial_state']['physics_sha256'] for d in known})==1 if len(known)>=2 else None,
            initial_observation_identical=len({digest(d['initial_state']['observation']) for d in known})==1 if len(known)>=2 else None,
            action_trace_identical=len({d['action_trace']['sha256'] for d in known})==1 if len(known)>=2 else None)
        max_position=max_joint=0.
        for a,b in itertools.combinations(known,2):
            x=a['initial_state']['objects'];y=b['initial_state']['objects'];assert x.keys()==y.keys()
            for name in x:
                max_position=max(max_position,math.dist(x[name]['position'],y[name]['position']))
                if 'joint_positions' in x[name]:
                    max_joint=max(max_joint,max((abs(u-v) for u,v in zip(x[name]['joint_positions'],y[name]['joint_positions'])),default=0.))
        row['max_initial_position_difference']=max_position if len(known)>=2 else None
        row['max_initial_joint_difference']=max_joint if len(known)>=2 else None
        rows.append(row)
        rr=dict(task=task,instance=instance,completed=len(known))
        for r in range(3):
            rr[f'repeat{r+1}_score']=row['scores'][r]
            rr[f'repeat{r+1}_success']=row['successes'][r]
            rr[f'repeat{r+1}_steps']=row['steps'][r]
        rr.update({k:row[k] for k in ['score_range','score_std','success_flip','initial_physics_identical','initial_observation_identical','action_trace_identical']})
        flat.append(rr)
    complete=[r for r in rows if r['complete']]
    out=dict(analyzed_at=time.time(),status_observed_at=status['observed_at'],accepted_episodes=len(reports),expected_episodes=30,
        complete_groups=len(complete),expected_groups=10,failures=failures,
        complete_group_score_changed=sum(r['score_range']>1e-10 for r in complete),
        complete_group_success_flipped=sum(r['success_flip'] for r in complete),
        groups=rows,source_report_sha256=proof,
        caveat='Incomplete groups are displayed but excluded from complete-group statistics. Different RGB hashes do not quantify image difference magnitude.')
    (ROOT/'current_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'current_by_instance.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({k:v for k,v in out.items() if k not in ['groups','source_report_sha256']},ensure_ascii=False))
    for r in rows:print(json.dumps(r,ensure_ascii=False))


if __name__=='__main__':main()
