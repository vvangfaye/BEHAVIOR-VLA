"""Aggregate only completed, state-matched paired episodes per task."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from cluster_helpers import HERE, aoss_client, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifacts',action='store_true');args=parser.parse_args()
    plan=json.loads((HERE/'plan.json').read_text());records={};roots={};c=aoss_client()
    try:
        for shard in plan['shards']:
            path=HERE/f"shard{shard['shard']}_submission.json"
            if not path.exists():continue
            receipt=json.loads(path.read_text());report=json.loads(c.get_object(Bucket='260010165',Key=receipt['result_key'])['Body'].read())
            for value in report.get('conditions',{}).values():records[(shard['task_id'],value['instance'],value['condition'])]=value
            if args.artifacts:
                assert report['complete']
                root=HERE/'retrieved'/f"shard{shard['shard']}";root.mkdir(parents=True,exist_ok=True);archive=root.parent/f"shard{shard['shard']}.tar.gz"
                if not archive.exists():c.download_file('260010165',receipt['result_key'].replace('status.json','artifacts.tar.gz'),str(archive))
                with tarfile.open(archive) as tf:
                    for member in tf:
                        rel=Path(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
                        dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                        with tf.extractfile(member) as source,dest.open('wb') as output:
                            import shutil;shutil.copyfileobj(source,output)
                for i in shard['indices']:roots[(shard['task_id'],i)]=root
    finally:c.close()
    results=[];all_rows=[]
    for task in plan['tasks']:
        tid=task['task_id'];rows=[]
        for i,instance in enumerate(task['instance_ids']):
            a=records.get((tid,instance,'image'));b=records.get((tid,instance,'serf'))
            row={'task_id':tid,'task_name':task['task_name'],'instance':instance,'image':a,'serf':b,'score_delta':b['score']-a['score'] if a and b else None}
            if args.artifacts and a and b:
                root=roots[(tid,i)];initials=[json.loads((root/f'{instance}_{cond}/initial.json').read_text()) for cond in ['image','serf']]
                assert initials[0]['initial_state_sha256']==initials[1]['initial_state_sha256']
                assert all(x['max_initial_state_delta']<1e-4 for x in initials)
                assert hashlib.sha256((root/f'shared_initial_{i}/initial_state.npz').read_bytes()).hexdigest()==initials[0]['initial_state_sha256']
                for cond in ['image','serf']:
                    log=(root/f'{instance}_{cond}/server.log').read_text();assert 'POLICY_STEP_ALIGNMENT_PASSED' in log and 'Policy action counter mismatch' not in log
                    assert 'keepalive ping failed' not in (root/f'{instance}_{cond}/evaluation.log').read_text()
            rows.append(row)
        paired=[x for x in rows if x['score_delta'] is not None]
        summary={'task_id':tid,'task_name':task['task_name'],'paired_instances':len(paired),'complete':len(paired)==20,'rows':rows}
        if paired:
            delta=np.array([x['score_delta'] for x in paired]);rng=np.random.default_rng(20261029+tid);means=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
            summary.update(image_mean_score=float(np.mean([x['image']['score'] for x in paired])),serf_mean_score=float(np.mean([x['serf']['score'] for x in paired])),mean_score_delta=float(delta.mean()),paired_bootstrap_95_ci=np.percentile(means,[2.5,97.5]).tolist(),wins=int((delta>1e-8).sum()),ties=int((np.abs(delta)<=1e-8).sum()),losses=int((delta<-1e-8).sum()),image_successes=sum(x['image']['success'] for x in paired),serf_successes=sum(x['serf']['success'] for x in paired))
        results.append(summary);all_rows+=rows
    old=json.loads(Path(plan['reused_task22']).read_text())
    final={'required_new_episodes':80,'completed_new_episodes':len(records),'complete':len(records)==80,'artifact_checks_passed':args.artifacts,'new_tasks':results,'reused_task22':{k:v for k,v in old.items() if k!='rows'}}
    if final['complete']:
        final['three_task_macro_image_mean']=float(np.mean([old['image_mean_score']]+[s['image_mean_score'] for s in results]))
        final['three_task_macro_serf_mean']=float(np.mean([old['serf_mean_score']]+[s['serf_mean_score'] for s in results]))
    write(HERE/'comparison.json',final)
    with (HERE/'paired_results.csv').open('w',newline='') as f:
        cols=['task_id','task_name','instance','baseline_score','serf_score','baseline_success','serf_success'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for row in all_rows:w.writerow({k:row[k] for k in cols[:3]}|{'baseline_score':row['image']['score'] if row['image'] else None,'serf_score':row['serf']['score'] if row['serf'] else None,'baseline_success':row['image']['success'] if row['image'] else None,'serf_success':row['serf']['success'] if row['serf'] else None})
    print(json.dumps({k:v for k,v in final.items() if k not in ['new_tasks','reused_task22']}))


if __name__=='__main__':main()
