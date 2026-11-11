"""Compare independent historical runs of checkpoint1688977.

These are NOT certified identical-runtime or identical-seed replicates.
"""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVAL = ROOT / 'BEHAVIOR-VLAs/eval_results'
A = EVAL / 'sco_test2000_1688977_20261010'
B = EVAL / 'sco_submission1000_1688977_20261013'
sources = {}


def read(p):
    sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


old = read(A / 'submission1000_progress_current.json')
a = {(r['task'], r['instance_id']): r for r in old['results']}
assert len(a) == len(old['results'])
b = {}
for p in sorted((B / 'artifacts').glob('*.json')):
    r = read(p)
    assert r['state'] == 'complete' and r['verified']
    key = (r['task'], r['instance_id'])
    assert key not in b
    b[key] = r

source_b = read(B / 'preparation_progress.json')['source_manifest']
weights_a = {r['relative_path']: r['sha256'] for r in read(A / 'aoss_checkpoint_upload_plan.json')['files']}
weights_b = {r['relative_path']: r['sha256'] for r in source_b['checkpoint_files']}
assert weights_a == weights_b and len(weights_a) == 21
source_a = read(A / 'source_overlay_sha256.json')
assert all(source_b['files_sha256'].get(k) == v for k, v in source_a.items())
scope_a, plan_b = read(A / 'scope.json'), read(B / 'plan.json')
assert scope_a['production_write_video'] is False and plan_b['write_video'] is True

rows = []
for key in sorted(a.keys() & b.keys()):
    x, y = a[key], b[key]
    rows.append(dict(task=key[0], instance=key[1], score_a=x['q_score'], score_b=y['q_score'],
                     score_delta=y['q_score']-x['q_score'], success_a=x['success'], success_b=y['model_success'],
                     success_flip=x['success'] != y['model_success']))


def summary(rr):
    n = len(rr)
    delta = [abs(r['score_delta']) for r in rr]
    aa = sum(r['success_a'] for r in rr)
    bb = sum(r['success_b'] for r in rr)
    both = sum(r['success_a'] and r['success_b'] for r in rr)
    flips = sum(r['success_flip'] for r in rr)
    return dict(n=n,mean_score_a=statistics.mean(r['score_a'] for r in rr),
                mean_score_b=statistics.mean(r['score_b'] for r in rr),successes_a=aa,successes_b=bb,
                success_rate_a=aa/n,success_rate_b=bb/n,score_changed=sum(d>1e-10 for d in delta),
                mean_absolute_score_difference=statistics.mean(delta),
                absolute_score_difference_at_least_half=sum(d>=.5-1e-10 for d in delta),
                success_flips=flips,success_flip_fraction=flips/n,both_success=both,
                only_a_success=aa-both,only_b_success=bb-both,both_failure=n-aa-bb+both,
                at_least_one_success=aa+bb-both,success_repeat_fraction_given_a_success=both/aa if aa else None)


by_task = defaultdict(list)
for r in rows:
    by_task[r['task']].append(r)
task_rows = [dict(task=task,**summary(rr)) for task,rr in by_task.items()]
task_rows.sort(key=lambda r: abs(r['success_rate_b']-r['success_rate_a']),reverse=True)
result = dict(checkpoint=1688977,run_a=str(A.relative_to(ROOT)),run_b=str(B.relative_to(ROOT)),
              matched=summary(rows),instance301=summary([r for r in rows if r['instance']==301]),
              task_count=len(task_rows),tasks_with_success_rate_change=sum(r['successes_a']!=r['successes_b'] for r in task_rows),
              tasks_with_success_rate_change_at_least_20pp=sum(abs(r['success_rate_b']-r['success_rate_a'])>=.2-1e-10 for r in task_rows),
              verified=dict(checkpoint_files_identical=21,compared_evaluation_source_files_identical=len(source_a)),
              limitations=['Run A production video disabled; run B video enabled.',
                           'Launchers, batching and recovery execution differ; per-episode RNG states are not certified identical.',
                           'These describe historical rerun disagreement, not pure simulator nondeterminism under fixed runtime conditions.',
                           'Two incomplete tasks have fewer than 10 paired episodes; all comparisons use only common completed instances.',
                           'No new evaluation, simulator, or cloud job was started.'],
              source_sha256=sources)
for name, rr in [('per_instance.csv',rows),('per_task.csv',task_rows)]:
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import Counter
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),gridspec_kw={'width_ratios':[1,1,1.45]})
    counts=Counter((r['score_a'],r['score_b']) for r in rows)
    ax=axes[0]
    for (x,y),n in counts.items():
        ax.scatter(x,y,s=18+4*n,alpha=.65,c='#498eb8' if x==y else '#c36d41',edgecolor='white',lw=.4)
    ax.plot([0,1],[0,1],ls='--',color='#77818c',lw=1)
    ax.set(xlabel='Run A: Q',ylabel='Run B: Q',xlim=(-.05,1.05),ylim=(-.05,1.05),aspect='equal',title='458 / 997 episodes changed Q')
    matrix=[[result['matched']['both_failure'],result['matched']['only_b_success']],
            [result['matched']['only_a_success'],result['matched']['both_success']]]
    ax=axes[1];ax.imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(matrix[i][j]),ha='center',va='center',fontsize=20,color='white' if matrix[i][j]>400 else '#162c40')
    ax.set(xticks=[0,1],xticklabels=['Fail','Success'],yticks=[0,1],yticklabels=['Fail','Success'],xlabel='Run B',ylabel='Run A',title='Success / failure: 92 flips')
    ax=axes[2];top=task_rows[:10]
    yy=list(range(len(top)))
    ax.barh([y-.17 for y in yy],[r['success_rate_a'] for r in top],height=.32,color='#6b9cbc',label='Run A')
    ax.barh([y+.17 for y in yy],[r['success_rate_b'] for r in top],height=.32,color='#cc8451',label='Run B')
    ax.set(yticks=yy,yticklabels=[r['task'] for r in top],xlim=(0,1),title='Largest task success-rate changes')
    ax.invert_yaxis();ax.legend(frameon=False)
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Same checkpoint 1688977: 997 common instances\nHistorical reruns; video mode and per-episode RNG controls are not matched',fontsize=14,y=1.06)
    fig.tight_layout()
    for extension in ['png','svg']:fig.savefig(OUT/f'rerun_disagreement.{extension}',dpi=180,bbox_inches='tight')
