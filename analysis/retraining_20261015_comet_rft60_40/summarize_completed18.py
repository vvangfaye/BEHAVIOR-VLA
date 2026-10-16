"""Audit and compare only the18 accepted, user-stopped-panel results."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean
import time

ROOT = Path(__file__).resolve().parent
EVAL = ROOT/'paired_fixed20_step5000'
OUT = ROOT/'completed18_comparison_20261016'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = load(EVAL/'plan.json')
    spec = importlib.util.spec_from_file_location('frozen_scorer18', EVAL/'frozen/scorer.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    rows, inputs, identities = {}, {}, set()
    references = [
        ('1688977', 'baseline_accepted.json', 'baseline_accepted_sha256', 'baseline'),
        ('previous_5000', 'step5000_candidate_accepted.json', 'previous_candidate_accepted_sha256', 'candidate'),
        ('previous_10000', 'step10000_candidate_accepted.json', 'additional_candidate_accepted_sha256', 'candidate'),
        ('previous_20000', 'step20000_candidate_accepted.json', 'extra_candidate_accepted_sha256', 'candidate')]
    for name, filename, field, role in references:
        path = EVAL/filename
        assert digest(path) == plan[field], 'Historical evidence changed'
        bundle = load(path)
        assert bundle['protocol_sha256'] == plan['protocol_sha256']
        rows[name] = scorer.validate(bundle, plan['pairs'], role)
        identities.add(bundle['checkpoint_manifest_sha256'])
        inputs[filename] = digest(path)
    export = load(EVAL/'reports/export.json')
    assert export['state'] == 'accepted'
    assert export['completed_updates'] == 5000 and export['full_afs_sha256_readback'] is True
    inputs['reports/export.json'] = digest(EVAL/'reports/export.json')
    candidate = {}
    for path in sorted((EVAL/'reports').glob('candidate-*.json')):
        report = load(path)
        assert report['state'] == 'accepted' and report['role'] == 'candidate'
        assert report['protocol_sha256'] == plan['protocol_sha256']
        assert report['checkpoint_manifest_sha256'] == export['manifest_sha256']
        intent = load(EVAL/'intents'/(report['id']+'.json'))
        assert report['worker_bundle_sha256'] == intent['worker_bundle_sha256']
        index = int(report['id'].removeprefix('candidate-'))
        row = report['record']
        key = (row['task'], row['instance_id'])
        assert key == tuple(plan['pairs'][index]) and key not in candidate
        assert row['artifact_accepted'] and row['policy_seed_acknowledged']
        assert row['policy_seed'] == row['environment_seed'] == 0
        assert row['steps'] > 0 and (row['terminated'] or row['truncated'])
        assert type(row['model_success']) is bool and math.isfinite(row['q_score'])
        assert 0 <= row['q_score'] <= 1 and row['model_success'] == (row['q_score'] == 1)
        result = [a for a in report['artifacts'] if a['path'].endswith('/'+row['task']+'_301_0.json')]
        video = [a for a in report['artifacts'] if a['path'].endswith('/'+row['task']+'_301_0.mp4')]
        assert len(result) == len(video) == 1
        assert result[0]['sha256'] == row['result_sha256']
        v = report['video_validation']
        assert video[0]['sha256'] == v['video_sha256'] and v['frames'] == row['steps'] and v['fps'] == 30
        assert set(v['decoded_frames']) == {0, row['steps']//2, row['steps']-1}
        candidate[key] = row
        inputs[str(path.relative_to(EVAL))] = digest(path)
    assert len(candidate) == 18
    identities.add(export['manifest_sha256'])
    assert len(identities) == 5
    stopped = load(ROOT/'user_stop_receipt.json')
    stopped_keys = {tuple(plan['pairs'][int(a['id'].removeprefix('candidate-'))]) for a in stopped['actions']}
    pairs = [tuple(key) for key in plan['pairs'] if tuple(key) in candidate]
    assert set(map(tuple, plan['pairs'])) - set(pairs) == stopped_keys and len(stopped_keys) == 2
    rows['rft60_40_5000'] = candidate
    summaries = {name: dict(episodes=18, successes=sum(data[k]['model_success'] for k in pairs),
                          success_rate=mean(data[k]['model_success'] for k in pairs),
                          mean_q=mean(data[k]['q_score'] for k in pairs)) for name, data in rows.items()}
    comparisons = {}
    for name, data in rows.items():
        if name == 'rft60_40_5000':
            continue
        delta = {key:candidate[key]['q_score']-data[key]['q_score'] for key in pairs}
        comparisons[name] = dict(mean_q_delta=mean(delta.values()),
            improved=[key for key,d in delta.items() if d > 1e-9],
            regressed=[key for key,d in delta.items() if d < -1e-9],
            unchanged=[key for key,d in delta.items() if abs(d) <= 1e-9],
            success_gains=[key for key in pairs if candidate[key]['model_success'] and not data[key]['model_success']],
            success_losses=[key for key in pairs if data[key]['model_success'] and not candidate[key]['model_success']])
    result = dict(kind='completed18_same_instance_comparison', created_at=time.time(),
        protocol_sha256=plan['protocol_sha256'], summaries=summaries, comparisons=comparisons,
        cancelled_excluded=sorted(stopped_keys),
        cancelled_historical_results=[dict(task=k[0],instance_id=k[1],models={name:data[k] for name,data in rows.items() if name!='rft60_40_5000'}) for k in sorted(stopped_keys)],
        per_instance=[dict(task=k[0],instance_id=k[1],models={name:data[k] for name,data in rows.items()}) for k in pairs],
        input_sha256=inputs, training_or_evaluation_restarted=False,
        limitations=['18completed pairs only; the2user-stopped pairs are excluded from every model, not scored as failures.',
          'Single native-seed rollout per instance; baseline seed has historical source evidence instead of a server ACK.',
          'Training-seen regression panel, not held-out generalization. Comparisons do not isolate the causal effect of ratio alone.'])
    OUT.mkdir(exist_ok=True)
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'per_instance.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        columns=['task','instance_id']+[name+suffix for name in rows for suffix in ('_q_percent','_success')]+['delta_vs_1688977_percentage_points']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for key in pairs:
            item={'task':key[0],'instance_id':key[1]}
            for name,data in rows.items():
                item[name+'_q_percent']=f"{100*data[key]['q_score']:.6f}"
                item[name+'_success']=int(data[key]['model_success'])
            item['delta_vs_1688977_percentage_points']=f"{100*(candidate[key]['q_score']-rows['1688977'][key]['q_score']):.6f}"
            writer.writerow(item)
    print(json.dumps({'summaries':summaries,'comparisons':comparisons,'excluded':sorted(stopped_keys),'output':str(OUT)},ensure_ascii=False))


if __name__ == '__main__':
    main()
