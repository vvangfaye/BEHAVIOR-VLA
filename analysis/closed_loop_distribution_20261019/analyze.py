"""Rebuild descriptive closed-loop statistics from saved evaluation evidence.

Run with Python 3.10+. Plotting additionally requires numpy and matplotlib.
No training, simulator, cloud, or source-result mutations are performed.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
QUALITY = ROOT / 'analysis/retraining_20261016_quality_taskbalanced_v1'
EVAL = ROOT / 'BEHAVIOR-VLAs/eval_results'
rows, sources, panels, checks = [], {}, {}, []


def read(path):
    path = Path(path)
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    path = Path(path)
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def boolean(value):
    if isinstance(value, bool):
        return value
    assert str(value).lower() in ('true', 'false', '1', '0'), value
    return str(value).lower() in ('true', '1')


def add(panel, model, task, instance, q, success, source):
    q = float(q)
    assert math.isfinite(q) and 0 <= q <= 1, (panel, model, task, q)
    rows.append(dict(panel=panel, model=model, task=task, instance=int(instance),
                     score=q, success=boolean(success), source=str(Path(source).relative_to(ROOT))))


def add_csv(panel, path, mappings, expected, note):
    panels[panel] = dict(expected_per_model=expected, note=note)
    for r in read_csv(path):
        for prefix, model in mappings.items():
            add(panel, model, r['task'], r.get('instance', r.get('instance_id')),
                r[prefix + '_q'], r[prefix + '_success'], path)


# Full benchmark reference: final, independently verified 1,000 episode run.
panel = 'full1000'
panels[panel] = dict(expected_per_model=1000, note='100 tasks x 10 instances (301-310), final verified evaluation.')
baseline = {}
for p in sorted((EVAL / 'sco_submission1000_1688977_20261013/artifacts').glob('*.json')):
    r = read(p)
    assert r['verified'] and r['state'] == 'complete'
    assert r['metrics']['q_score'] == r['q_score']
    assert r['metrics']['success'] == r['model_success']
    key = (r['task'], r['instance_id'])
    assert key not in baseline
    baseline[key] = r
    add(panel, 'Baseline 1688977', *key, r['q_score'], r['model_success'], p)
assert len(baseline) == 1000
final_reference = read(EVAL / 'sco_submission1000_1688977_20261013/summary1000_current.json')

# The stopped 2M run is described only over completed, verified episodes.
panel = 'checkpoint2m_completed185'
panels[panel] = dict(expected_per_model=1000, note='Stopped campaign: 185 completed pairs; 15 failed artifacts excluded. Same instances, historical RNG streams not proven identical.')
for p in sorted((EVAL / 'sco_submission1000_2000000_20261013/artifacts').glob('*.json')):
    r = read(p)
    if not r.get('verified') or r.get('state') != 'complete':
        continue
    b = baseline[(r['task'], r['instance_id'])]
    add(panel, 'Baseline 1688977', b['task'], b['instance_id'], b['q_score'], b['model_success'], EVAL / 'sco_submission1000_1688977_20261013/artifacts')
    add(panel, 'Checkpoint 2000000', r['task'], r['instance_id'], r['q_score'], r['model_success'], p)

# Historical fixed20 and newer task-balanced runs use identical task/instance pairs.
panel = 'fixed20'
panels[panel] = dict(expected_per_model=20, note='Training-seen regression panel: 20 tasks, one instance301 per task, seed0. Historical baseline seed inferred from source; not an unseen test set.')
p = ROOT / 'analysis/retraining_20261013/paired_quick20_reused1688977_step20000_20261015/comparison.json'
d = read(p)
names = {'1688977': 'Baseline 1688977', '5000': 'Original mix 5k', '10000': 'Original mix 10k', '20000': 'Original mix 20k'}
for r in d['per_instance']:
    for ck, v in r['checkpoints'].items():
        add(panel, names[ck], r['task'], r['instance_id'], v['q'], v['success'], p)
p = QUALITY / 'final_comparison.csv'
for r in read_csv(p):
    b = [v for v in rows if v['panel'] == panel and v['model'] == 'Baseline 1688977' and v['task'] == r['task']][0]
    assert b['score'] == float(r['baseline1688977_q']) and b['success'] == boolean(r['baseline1688977_success'])
    for prefix, name in [('mixed1000', 'Balanced mix 1k'), ('mixed5000', 'Balanced mix 5k'), ('expert1000', 'Expert MSE 1k')]:
        add(panel, name, r['task'], r['instance'], r[prefix + '_q'], r[prefix + '_success'], p)

# 60:40 run stopped with 18 pairs. Keep its own matched denominator.
panel = 'rft60_40_completed18'
panels[panel] = dict(expected_per_model=20, note='18 complete pairs. Two user-cancelled pairs excluded from all models.')
p = ROOT / 'analysis/retraining_20261015_comet_rft60_40/completed18_comparison_20261016/comparison.json'
for r in read(p)['per_instance']:
    for model, v in r['models'].items():
        add(panel, 'Baseline 1688977' if model == '1688977' else 'RFT 60:40 5k' if model == 'rft60_40_5000' else names[model.replace('previous_', '')],
            r['task'], r['instance_id'], v['q_score'], v['model_success'], p)

# The 100-task 10k campaign stopped after 97 valid comparisons.
panel = 'task100_completed97'
panels[panel] = dict(expected_per_model=100, note='97 complete pairs of a 100-task instance301 regression panel; three incomplete pairs excluded.')
p = ROOT / 'analysis/retraining_20261015_step10000_task100/partial_comparison.json'
partial97 = read(p)
for r in partial97['per_instance']:
    for prefix, model in [('baseline', 'Baseline 1688977'), ('candidate', 'Original mix 10k')]:
        add(panel, model, r['task'], r['instance_id'], r[prefix + '_q'], r[prefix + '_success'], p)

# Unseen-by-current-finetuning instances. The third model shares exact references.
add_csv('heldout20', QUALITY / 'huber50_heldout20/comparison.csv',
        {'baseline': 'Baseline 1688977', 'expert_mse': 'Expert MSE 1k', 'huber50': 'Expert Huber50'}, 20,
        'Same 20 task categories as fixed20; different instances excluded from current finetuning indexes. Baseline pretraining exclusion not proven.')
add_csv('confirmation20', QUALITY / 'confirmation_selected_vs_baseline/comparison.csv',
        {'baseline': 'Baseline 1688977', 'expert': 'Expert MSE 1k'}, 20,
        'Separate reserved confirmation panel; candidate selected before inspecting this panel. Baseline pretraining exclusion not proven.')

# Additional completed-trajectory baseline and the stopped 57-pair rerun.
p = EVAL / 'sco_paired_eval962_step10000_20261015/round-5/plan.json'
plan = read(p)
b962 = {(r['task'], r['instance_id']): r for r in plan['baseline_scores']}
panels['collection_baseline962'] = dict(expected_per_model=962, note='Usable baseline episodes from the separate full-trajectory collection run. Do not pool with the final1000 run.')
for r in b962.values():
    add('collection_baseline962', 'Collection baseline 1688977', r['task'], r['instance_id'], r['q_score'], r['success'], p)
cache_path = EVAL / 'sco_paired_eval962_step10000_20261015/metrics_cache.json'
panels['rerun_completed57'] = dict(expected_per_model=962, note='Stopped after 57 pairs in a separate collection protocol. Pairing by task/instance; baseline per-episode RNG evidence not revalidated here.')
for entry in read(cache_path).values():
    r = entry['row']
    b = b962[(r['task'], r['instance_id'])]
    add('rerun_completed57', 'Collection baseline 1688977', b['task'], b['instance_id'], b['q_score'], b['success'], p)
    add('rerun_completed57', 'Original mix 10k', r['task'], r['instance_id'], r['q_score_final'], r['success'], cache_path)


def quantile(x, p):
    x = sorted(x)
    index = (len(x) - 1) * p
    lo = int(index)
    return x[lo] + (x[min(lo + 1, len(x) - 1)] - x[lo]) * (index - lo)


def wilson(k, n):
    z = 1.959963984540054
    center = (k / n + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def bin_id(q):
    return 0 if q == 0 else 1 if q < .25 else 2 if q < .5 else 3 if q < .75 else 4 if q < 1 else 5


groups = defaultdict(list)
for r in rows:
    groups[(r['panel'], r['model'])].append(r)
summaries, task_rows = [], []
for (panel, model), group in groups.items():
    assert len({(r['task'], r['instance']) for r in group}) == len(group), (panel, model)
    n = len(group)
    q = [r['score'] for r in group]
    successes = sum(r['success'] for r in group)
    bins = Counter(bin_id(x) for x in q)
    summary = dict(panel=panel, model=model, completed=n, expected=panels[panel]['expected_per_model'],
                   tasks=len({r['task'] for r in group}), successes=successes, success_rate=successes/n,
                   success_wilson95=wilson(successes, n), mean_score=statistics.mean(q), median_score=statistics.median(q),
                   q25=quantile(q,.25), q75=quantile(q,.75), q90=quantile(q,.90),
                   zero_score=bins[0], score_at_least_half=sum(x>=.5 for x in q), perfect_score=bins[5],
                   bins=[bins[i] for i in range(6)])
    assert sum(summary['bins']) == n
    summaries.append(summary)
    tasks = defaultdict(list)
    for r in group:
        tasks[r['task']].append(r)
    for task, rr in tasks.items():
        task_rows.append(dict(panel=panel, model=model, task=task, episodes=len(rr),
                             successes=sum(r['success'] for r in rr),
                             success_rate=statistics.mean(r['success'] for r in rr),
                             mean_score=statistics.mean(r['score'] for r in rr)))
    mismatches = [dict(task=r['task'], instance=r['instance'], score=r['score'], success=r['success'])
                  for r in group if (r['score'] == 1) != r['success']]
    if mismatches:
        checks.append(dict(panel=panel, model=model, score1_success_mismatches=mismatches))

by_summary = {(r['panel'],r['model']):r for r in summaries}
s = by_summary[('full1000','Baseline 1688977')]
assert s['successes'] == final_reference['successes'] == 77
assert math.isclose(s['mean_score'], final_reference['q_completed'], abs_tol=1e-12)
s = by_summary[('task100_completed97','Original mix 10k')]
assert s['successes'] == partial97['completed_pair_metrics']['candidate_successes']
assert math.isclose(s['mean_score'], partial97['completed_pair_metrics']['candidate_mean_q'], abs_tol=1e-12)
for panel, path, mapping in [
    ('fixed20', QUALITY / 'dual_experiment_comparison.json',
     {'baseline1688977':'Baseline 1688977','mixed1000':'Balanced mix 1k','mixed5000':'Balanced mix 5k','expert1000':'Expert MSE 1k'}),
    ('heldout20', QUALITY / 'huber50_heldout20/comparison.json',
     {'baseline':'Baseline 1688977','expert_mse':'Expert MSE 1k','huber50':'Expert Huber50'}),
    ('confirmation20', QUALITY / 'confirmation_selected_vs_baseline/comparison.json',
     {'baseline':'Baseline 1688977','candidate':'Expert MSE 1k'}),
]:
    reference = read(path)['summaries']
    for key, model in mapping.items():
        actual = by_summary[(panel, model)]
        assert actual['completed'] == reference[key]['episodes']
        assert actual['successes'] == reference[key]['successes']
        assert math.isclose(actual['mean_score'], reference[key]['mean_q'], abs_tol=1e-12)

paired = []
for panel in panels:
    base_name = 'Collection baseline 1688977' if panel == 'rerun_completed57' else 'Baseline 1688977'
    if (panel, base_name) not in groups:
        continue
    b = {(r['task'], r['instance']):r for r in groups[(panel, base_name)]}
    for (p, model), group in groups.items():
        if p != panel or model == base_name:
            continue
        a = {(r['task'], r['instance']):r for r in group}
        assert a.keys() == b.keys(), (panel,model)
        changes = [(key,a[key]['score']-b[key]['score']) for key in a]
        gains = [list(key) for key in a if a[key]['success'] and not b[key]['success']]
        losses = [list(key) for key in a if b[key]['success'] and not a[key]['success']]
        discordant = len(gains)+len(losses)
        exact_p = min(1., 2*sum(math.comb(discordant,k) for k in range(min(len(gains),len(losses))+1))/2**discordant) if discordant else 1.
        paired.append(dict(panel=panel,model=model,baseline=base_name,n=len(a),
                           mean_score_delta=statistics.mean(x for _,x in changes),
                           score_improved=sum(x>1e-12 for _,x in changes),
                           score_regressed=sum(x < -1e-12 for _,x in changes),
                           score_unchanged=sum(abs(x)<=1e-12 for _,x in changes),
                           success_gains=gains,success_losses=losses,
                           success_both=sum(a[k]['success'] and b[k]['success'] for k in a),
                           exact_mcnemar_two_sided_p=exact_p))


def write_csv(name, records):
    with (OUT / name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


write_csv('per_episode.csv', rows)
write_csv('per_task.csv', task_rows)
write_csv('summary.csv', [{**s,'success_wilson95':json.dumps(s['success_wilson95']),'bins':json.dumps(s['bins'])} for s in summaries])
audit = dict(score_definition='Saved native evaluator q_score / Q, range 0..1. Success uses saved boolean, never inferred from Q.',
             score_bins=['Q=0','0<Q<0.25','0.25<=Q<0.5','0.5<=Q<0.75','0.75<=Q<1','Q=1'],
             panels=panels,summaries=summaries,paired=paired,validation=dict(unique_episode_keys=True,all_scores_in_range=True,
             summary_crosschecks_passed=True,score1_success_mismatches=checks),source_sha256=sources)
(OUT / 'analysis.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')

for s in summaries:
    print(f"{s['panel']:26} {s['model']:28} n={s['completed']:4} SR={s['successes']:3}/{s['completed']:<4} mean={s['mean_score']:.6f} median={s['median_score']:.4f} zero={s['zero_score']:4} bins={s['bins']}")
print('BASELINE TASK SUCCESS COUNTS', dict(sorted(Counter(t['successes'] for t in task_rows if t['panel']=='full1000').items())))
print('SCORE1/SUCCESS MISMATCH GROUPS',len(checks))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    colors=['#d9dee5','#bed6ed','#7fb8dd','#478dbc','#26608e','#183f50']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    labels=['Q = 0','0 < Q < .25','.25 <= Q < .5','.5 <= Q < .75','.75 <= Q < 1','Q = 1']
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
        fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def bars(ax, panel, models):
        y=np.arange(len(models));left=np.zeros(len(models))
        ss=[by_summary[(panel,m)] for m in models]
        for i,c in enumerate(colors):
            widths=np.array([s['bins'][i]/s['completed'] for s in ss])
            ax.barh(y,widths,left=left,color=c,label=labels[i],height=.62)
            for yy,w,l,s in zip(y,widths,left,ss):
                if w>=.075:ax.text(l+w/2,yy,str(s['bins'][i]),ha='center',va='center',fontsize=9,color='white' if i>=3 else '#182333')
            left+=widths
        ax.set(yticks=y,yticklabels=models,xlim=(0,1));ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    fig,ax=plt.subplots(1,3,figsize=(16,4.4),gridspec_kw={'width_ratios':[1.2,1,1.35]})
    qq=[r['score'] for r in groups[('full1000','Baseline 1688977')]]
    ax[0].hist(qq,bins=np.linspace(0,1,21),color='#478dbc',edgecolor='white')
    ax[0].set(title='Episode score distribution',xlabel='Q score',ylabel='Episodes')
    tt=[r for r in task_rows if r['panel']=='full1000'];cc=Counter(t['successes'] for t in tt)
    ax[1].bar(np.arange(11),[cc[i] for i in range(11)],color='#26608e')
    for i in range(11):
        if cc[i]:ax[1].text(i,cc[i]+.7,str(cc[i]),ha='center')
    ax[1].set(title='Success rates across 100 tasks',xlabel='Successful episodes out of 10',ylabel='Tasks',xticks=np.arange(11))
    top=sorted(tt,key=lambda r:(r['successes'],r['mean_score']),reverse=True)[:12]
    ax[2].barh(np.arange(len(top)),[r['success_rate'] for r in top],color='#26608e')
    ax[2].set(yticks=np.arange(len(top)),yticklabels=[r['task'] for r in top],xlim=(0,1),title='Tasks with most successes')
    ax[2].invert_yaxis();ax[2].xaxis.set_major_formatter(PercentFormatter(1))
    for i,r in enumerate(top):ax[2].text(r['success_rate']+.015,i,f"{r['successes']}/10",va='center',fontsize=9)
    fig.suptitle('Baseline 1688977 | 1,000 verified episodes | 100 tasks x 10 instances',fontsize=15,y=1.05)
    fig.tight_layout();save(fig,'baseline1000_distribution')

    fig,axes=plt.subplots(3,2,figsize=(14,12),gridspec_kw={'width_ratios':[2.5,1.5],'height_ratios':[2.3,1.2,1]})
    panel_specs=[('fixed20',['Baseline 1688977','Original mix 5k','Original mix 10k','Original mix 20k','Balanced mix 1k','Balanced mix 5k','Expert MSE 1k'],'Fixed20: training-seen instance301'),
                 ('heldout20',['Baseline 1688977','Expert MSE 1k','Expert Huber50'],'Heldout20: different instances, same task categories'),
                 ('confirmation20',['Baseline 1688977','Expert MSE 1k'],'Confirmation20: separate reserved panel')]
    for row,(panel,models,title) in enumerate(panel_specs):
        bars(axes[row,0],panel,models);axes[row,0].set_title(title,loc='left',pad=13)
        ss=[by_summary[(panel,m)] for m in models];y=np.arange(len(models))
        axes[row,1].barh(y,[s['success_rate'] for s in ss],color='#bd6f39',height=.62)
        axes[row,1].set(yticks=y,yticklabels=[],xlim=(0,.42));axes[row,1].invert_yaxis();axes[row,1].xaxis.set_major_formatter(PercentFormatter(1))
        for yy,s in enumerate(ss):axes[row,1].text(s['success_rate']+.01,yy,f"{s['successes']}/{s['completed']}   mean Q {s['mean_score']:.3f}",va='center',fontsize=9)
        axes[row,1].set_title('Whole-task success',loc='left',pad=13)
    handles,_=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.suptitle('Fine-tuning: compare models within each panel',fontsize=16,y=1.015)
    fig.tight_layout(h_pad=2.7);save(fig,'finetuning_distribution')

    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for ax,panel,title in zip(axes,['task100_completed97','heldout20','confirmation20'],['Original mix 10k: 97 completed pairs','Expert MSE: heldout20','Expert MSE: confirmation20']):
        candidate='Original mix 10k' if panel=='task100_completed97' else 'Expert MSE 1k'
        bb={(r['task'],r['instance']):r for r in groups[(panel,'Baseline 1688977')]}
        aa={(r['task'],r['instance']):r for r in groups[(panel,candidate)]}
        counts=Counter((bb[k]['score'],aa[k]['score']) for k in aa)
        for (x,y),n in counts.items():
            ax.scatter(x,y,s=50+26*n,c='#287fa8' if y>x else '#c36a40' if y<x else '#aeb5be',alpha=.75,edgecolor='white')
            if n>1:ax.text(x,y,str(n),ha='center',va='center',fontsize=8)
        ax.plot([0,1],[0,1],color='#657181',ls='--',lw=1)
        pair=next(p for p in paired if p['panel']==panel and p['model']==candidate)
        ax.set(xlabel='Baseline Q',ylabel='Fine-tuned Q',xlim=(-.06,1.06),ylim=(-.06,1.06),aspect='equal')
        ax.set_title(title + '\n' + f"Q up / down / tied: {pair['score_improved']} / {pair['score_regressed']} / {pair['score_unchanged']}", fontsize=11)
    fig.suptitle('Paired score changes | bubble labels show duplicate episode counts',fontsize=14,y=1.04)
    fig.tight_layout();save(fig,'paired_score_changes')


if __name__=='__main__':
    try:
        plot()
    except ImportError as e:
        print('Statistics saved; plotting dependencies unavailable:',e)
