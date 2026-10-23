"""Assess the completed, fixed 24-episode pilot without claiming a control win."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    root=args.directory
    plan=json.loads((root/'plan.json').read_text())
    report=json.loads((root/'episode_report.json').read_text())
    result_path=root/'trainer/result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else {}
    rows=report['episodes']
    required=plan['max_training_episodes']
    if not result.get('complete') or result.get('training_episodes')!=required or len(rows)!=required:
        print(json.dumps(dict(complete=False,observed=len(rows),required=required,
                              verdict='Insufficient completed outcomes; no method verdict')))
        return
    expected={(i,j):(item['task'],instance) for i,item in enumerate(plan['iterations'])
              for j,instance in enumerate(item['instances'])}
    if len(expected)!=required or len({(r['iteration'],r['slot']) for r in rows})!=required:
        raise ValueError('Repeated or missing episode identities')
    groups=defaultdict(list)
    for row in rows:
        if expected[(row['iteration'],row['slot'])]!=(row['task'],row['instance']):
            raise ValueError('Observed episode differs from fixed plan')
        groups[(row['task'],'mixed' if row['mixed'] else 'all_updated')].append(row)
    summaries=[]
    for (task,mode),items in sorted(groups.items()):
        scores=[x['score'] for x in items]
        summaries.append(dict(task=task,mode=mode,n=len(items),successes=sum(x['success'] for x in items),
                              mean_score=statistics.mean(scores),min_score=min(scores),max_score=max(scores)))
    updates=report.get('updates',[])
    if len(updates)!=len(plan['iterations']):
        raise ValueError('Missing final optimizer-update records')
    nonzero_gradients=sum(x['gradient_norm']>0 for x in updates)
    changed_updates=sum(x['adapter_delta_norm']>0 for x in updates)
    if changed_updates==0:
        decision='Do not scale this sparse-success training recipe: it produced no learned initialization update.'
    else:
        decision='Training is executable and received a learning signal; only a matched control evaluation can establish its extra value.'
    assessment=dict(complete=True,episode_count=len(rows),successes=sum(x['success'] for x in rows),
        groups=summaries,nonzero_gradient_updates=nonzero_gradients,changed_parameter_updates=changed_updates,
        decision=decision,efficacy_proven=False,
        limitation='These are on-policy training outcomes. Temporal ordering and mixed/all-updated differences do not establish causal improvement.')
    (root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    print(json.dumps(assessment,indent=2))


if __name__=='__main__':main()
