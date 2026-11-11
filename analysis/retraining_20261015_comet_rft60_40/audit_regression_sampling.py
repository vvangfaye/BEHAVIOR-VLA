"""Read-only replay of frozen metadata sampler; never opens data or starts training."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'formal_stage/bundle'
sys.path.insert(0, str(BUNDLE / 'openpi/src'))
from openpi.training.subtask_mixture import MixturePlan


def quantiles(values):
    values = sorted(values)
    def q(p):
        n = (len(values)-1)*p
        return values[int(n)]*(1-n%1)+values[math.ceil(n)]*(n%1)
    return {name:q(p) for name,p in [('min',0),('p10',.1),('median',.5),('p90',.9),('max',1)]}


def main():
    plan = MixturePlan.load(BUNDLE/'inputs/compiled/mixture.json')
    selected = json.loads((ROOT/'selected_full_episodes.json').read_text())['episodes']
    generation = {x['root_id']:x['generation'] for x in selected}
    rows, sources, segments_by_episode, source_label_sets = [], {}, {}, {}
    for source in plan._phases[0]['sources']:
        kind = source['kind']
        tasks = source['balanced_tasks']
        source_rows, label_probs = [], Counter()
        for task, instances in tasks:
            for instance, episodes in instances:
                for (root, number), segments in episodes:
                    rec = plan.episode_records_by_root[root][number]
                    anchors = sum(s['end']-s['start']-31 for s in segments)
                    prob = 1/(len(tasks)*len(instances)*len(episodes))
                    label_mass = Counter()
                    for s in segments:
                        label_mass[s['subtask']] += s['end']-s['start']-31
                    row = dict(source=kind, root_id=root, episode_index=number, task=task,
                        instance_id=instance, frames=rec['length'], legal_anchors=anchors,
                        intervals=len(segments), within_source_probability=prob,
                        expected_draws=960000*source['weight']*prob,
                        largest_label_anchor_share=max(label_mass.values())/anchors,
                        longest_label_interval_frames=max(s['end']-s['start'] for s in segments),
                        generation=generation.get(root,'original'), actual_draws=0)
                    source_rows.append(row)
                    segments_by_episode[(root,number)] = segments
                    for s in segments:
                        label_probs[(task,s['subtask'])] += prob*(s['end']-s['start']-31)/anchors
        assert abs(sum(r['within_source_probability'] for r in source_rows)-1) < 1e-8
        long_share = {}
        for seconds in (30,60,120):
            long_share[str(seconds)] = sum(r['within_source_probability']*sum(
                s['end']-s['start']-31 for s in segments_by_episode[(r['root_id'],r['episode_index'])]
                if s['end']-s['start']>=seconds*30)/r['legal_anchors'] for r in source_rows)
        probs = sorted((r['within_source_probability'] for r in source_rows),reverse=True)
        sources[kind] = dict(tasks=len(tasks), episodes=len(source_rows),
            task_instances=sum(len(instances) for _,instances in tasks),
            frames=sum(r['frames'] for r in source_rows),
            anchors=sum(r['legal_anchors'] for r in source_rows),
            frames_quantiles=quantiles([r['frames'] for r in source_rows]),
            expected_draws_per_episode_quantiles=quantiles([r['expected_draws'] for r in source_rows]),
            episode_probability_ESS=1/sum(p*p for p in probs),
            top10_episodes_within_source_mass=sum(probs[:10]),
            probability_of_interval_at_least_seconds=long_share,
            episodes_with_label_over80pct=sum(r['largest_label_anchor_share']>.8 for r in source_rows))
        source_label_sets[kind] = label_probs
        rows.extend(source_rows)
    # Deterministic references only: no image decoder/model/optimizer imported.
    counts, anchors, anchor_bins = Counter(), defaultdict(set), defaultdict(set)
    lookup = {(r['root_id'],r['episode_index']):r for r in rows}
    for i in range(960000):
        ref = plan.reference(i)
        kind = ref['source_kind']
        key = (ref['root_id'],ref['episode_index'])
        counts[kind] += 1
        lookup[key]['actual_draws'] += 1
        anchors[kind].add((*key,ref['frame']))
        # Fixed 32-frame bins are a correlation proxy, not an independent-sample count.
        anchor_bins[kind].add((*key,ref['frame']//32))
    runtime = json.loads((ROOT/'final_mixture_runtime_audit.json').read_text())
    assert dict(counts)=={'replay':runtime['actual_draws']['original'],
                          'full_episode':runtime['actual_draws']['full_episode']}
    for kind, n in counts.items():
        source_rows = [r for r in rows if r['source']==kind]
        sources[kind].update(actual_draws=n,actual_draws_per_episode_quantiles=quantiles([r['actual_draws'] for r in source_rows]),
            unique_exact_anchors=len(anchors[kind]), repeated_exact_anchor_draws=n-len(anchors[kind]),
            unique_fixed_32frame_bins=len(anchor_bins[kind]))
    comparison = json.loads((ROOT/'completed18_comparison_20261016/comparison.json').read_text())
    full_tasks = {r['task'] for r in rows if r['source']=='full_episode'}
    task_eval = []
    for x in comparison['per_instance']:
        task = x['task']
        full = [r for r in rows if r['source']=='full_episode' and r['task']==task]
        task_eval.append(dict(task=task,instance=x['instance_id'],full_training_episodes=len(full),
            full_same_instance_episodes=sum(r['instance_id']==x['instance_id'] for r in full),
            expected_total_task_draws=960000*(.6/100+(.4/52 if full else 0)),
            baseline_q=x['models']['1688977']['q_score'],new_q=x['models']['rft60_40_5000']['q_score'],
            delta=x['models']['rft60_40_5000']['q_score']-x['models']['1688977']['q_score']))
    grouped = {}
    for covered in (False,True):
        group = [r for r in task_eval if bool(r['full_training_episodes'])==covered]
        grouped[str(covered)] = dict(tasks=len(group),baseline_mean_q=sum(r['baseline_q'] for r in group)/len(group),
            new_mean_q=sum(r['new_q'] for r in group)/len(group),mean_delta=sum(r['delta'] for r in group)/len(group),
            regressions=sum(r['delta']<0 for r in group),improvements=sum(r['delta']>0 for r in group))
    generation_mass = Counter()
    for r in rows:
        if r['source']=='full_episode':
            generation_mass[r['generation']] += r['within_source_probability']
    sha = {}
    for f in ['inputs/compiled/mixture.json','openpi/src/openpi/training/subtask_mixture.py',
              'openpi/src/openpi/models/pi0.py','openpi/src/openpi/transforms.py']:
        sha[f] = hashlib.sha256((BUNDLE/f).read_bytes()).hexdigest()
    result = dict(kind='frozen_960000_metadata_draw_regression_audit',observed_at=time.time(),
        mixture_fingerprint=plan.fingerprint,input_sha256=sha,sources=sources,
        exact_sampler_counts_match_persisted_runtime=True,
        full_sampling_mass_with_label_text_absent_from_same_task_original=sum(p for label,p in source_label_sets['full_episode'].items()
            if label not in source_label_sets['replay']),
        full_generation_sampling_mass=dict(generation_mass),
        all_task_counts=dict(covered=len(full_tasks),uncovered=100-len(full_tasks)),
        eval_by_full_data_task_coverage=grouped,eval_per_task=task_eval,
        highest_exposure_full_episodes=sorted([r for r in rows if r['source']=='full_episode'],key=lambda r:r['actual_draws'],reverse=True)[:15],
        objective=dict(action='0.6 E_expert(flow MSE) + 0.4 E_full(flow MSE)',
            language='E_expert(per-example token CE), weight1, batch-renormalized by B / supervised_count',
            full_action_context='recorded self-generated subtask retained, human verification absent',
            full_CE=0,language_diluted_by_action_only_examples=False,
            no_stop_gradient_between_action_suffix_and_language_prefix=True,
            causal_gradient_conflict_measured=False),
        limitations=['Metadata only, no new raw video/action semantic scan.',
            'Exact references reproduce frozen sampler; compare counts to persisted runtime counters separately.',
            'Fixed32frame bins and long text intervals are correlation/label-duration proxies, not proven independent samples or failed actions.',
            'Outcome comparisons are18 single-seed training-seen pairs, with2 user-stopped exclusions; no expert-only control.'],
        new_training_jobs=0,training_data_or_parameters_changed=False)
    (ROOT/'deep_regression_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'deep_regression_episode_exposure.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k in ['sources','eval_by_full_data_task_coverage','full_generation_sampling_mass','highest_exposure_full_episodes']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
