"""Publish three-model comparison only after all candidate episodes validate."""
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / 'heldout20_baseline_vs_expert1k'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_module():
    spec = importlib.util.spec_from_file_location('h50_report_reference', REFERENCE / 'build_report.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def collect():
    ref = reference_module()
    old_plan, pairs, old_models, proofs, missing = ref.collect()
    if missing:
        raise ValueError('Reference episodes missing')
    reference_native = ref.verify_native_completion(old_plan)
    exclusion = ref.verify_exclusion(pairs)
    plan = load(ROOT / 'plan.json')
    protocol = load(ROOT / 'protocol.json')
    canonical = json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()
    if (hashlib.sha256(canonical).hexdigest() != plan['protocol_sha256']
            or plan['protocol_sha256'] != old_plan['protocol_sha256']
            or plan['pairs'] != [list(pair) for pair in pairs]
            or digest(REFERENCE / 'plan.json') != plan['reference_plan_sha256']):
        raise ValueError('Candidate and reference protocols differ')
    bundle = digest(ROOT / 'frozen_sha256.json')
    for name, expected in load(ROOT / 'frozen_sha256.json').items():
        if digest(ROOT / 'frozen' / name) != expected:
            raise ValueError('Frozen evaluation source changed')
    records, candidate_proofs, missing = [], [], []
    for index, pair in enumerate(pairs):
        identity = f'candidate-{index:02d}'
        path = ROOT / 'reports' / (identity + '.json')
        if not path.exists():
            missing.append(identity)
            continue
        report = load(path)
        if report.get('state') != 'accepted':
            missing.append(identity)
            continue
        expected = {'id': identity, 'role': 'candidate', 'worker_bundle_sha256': bundle,
                    'protocol_sha256': plan['protocol_sha256'],
                    'checkpoint_manifest_sha256': plan['expected_checkpoint_manifest_sha256']}
        if any(report.get(k) != v for k, v in expected.items()):
            raise ValueError('Candidate identity mismatch: ' + identity)
        row = report['record']
        if (row['task'], row['instance_id']) != pair:
            raise ValueError('Candidate instance mismatch')
        if row['truncated'] and row['steps'] != plan['native_task_budgets'][pair[0]] + 1:
            raise ValueError('Candidate stopped before native budget')
        records.append(row)
        candidate_proofs.append({'file': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    if missing:
        return {'missing': missing}
    status = load(ROOT / 'status.json')
    expected_jobs = {}
    for group in plan['groups']:
        intent = load(ROOT / 'intents' / (group['id'] + '.json'))
        expected_jobs[intent['name']] = intent['uid']
    states = status['states']
    terminal = {'SUCCEEDED', 'FAILED', 'STOPPED', 'CANCELLED', 'DELETED', 'SUSPENDED'}
    if (not status.get('complete') or status['accepted'] != 20
            or len(states) != len(expected_jobs) or {s['name'] for s in states} != set(expected_jobs)
            or any(s['uid'] != expected_jobs[s['name']] or s['state'] not in terminal for s in states)):
        raise ValueError('Candidate native jobs are not all terminal')
    candidate = ref.validate({'protocol_sha256': plan['protocol_sha256'], 'records': records}, pairs, plan['protocol_sha256'])
    models = {'baseline': old_models['baseline'], 'expert_mse': old_models['candidate'], 'huber50': candidate}
    stats = {name: ref.summary(rows, pairs) for name, rows in models.items()}
    contrasts = {name: ref.paired(models[name], candidate, pairs) for name in ('baseline', 'expert_mse')}
    comparison = load(ROOT / 'comparison.json')
    if (not comparison.get('complete') or comparison['summaries'] != stats
            or comparison['huber50_vs_baseline'] != json.loads(json.dumps(contrasts['baseline']))
            or comparison['huber50_vs_expert_mse'] != json.loads(json.dumps(contrasts['expert_mse']))
            or comparison['candidate_report_sha256'] != {Path(p['file']).stem: p['sha256'] for p in candidate_proofs}
            or comparison['reference_proofs'] != proofs or comparison['native_jobs'] != states):
        raise ValueError('Controller comparison differs from validated source reports')
    return dict(pairs=pairs, models=models, stats=stats, contrasts=contrasts, missing=[],
                reference_proofs=proofs, candidate_proofs=candidate_proofs,
                reference_native=reference_native, exclusion=exclusion,
                candidate_status_sha256=digest(ROOT / 'status.json'), protocol_sha256=plan['protocol_sha256'])


def main():
    data = collect()
    if data['missing']:
        print(json.dumps({'published': False, 'missing_episodes': len(data['missing'])}))
        return
    labels = {'baseline': '原模型 1688977', 'expert_mse': '纯专家 MSE 1,000 步', 'huber50': '纯专家 Huber50 1,000 步'}
    csv_rows, table = [], []
    for pair in data['pairs']:
        row = {'task': pair[0], 'instance': pair[1]}
        cells = [html.escape(pair[0]), str(pair[1])]
        for name in labels:
            result = data['models'][name][pair]
            row.update({name + '_success': result['model_success'], name + '_q': result['q_score'], name + '_steps': result['steps']})
            cells.append(('成功' if result['model_success'] else '未成功') + f" / {result['q_score']:.4f}")
        row['huber50_minus_expert_mse_q'] = row['huber50_q'] - row['expert_mse_q']
        row['huber50_minus_baseline_q'] = row['huber50_q'] - row['baseline_q']
        cells.append(f"{row['huber50_minus_expert_mse_q']:+.4f}")
        csv_rows.append(row)
        table.append('<tr>' + ''.join('<td>' + cell + '</td>' for cell in cells) + '</tr>')
    with (ROOT / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summaries = ''.join('<tr><td>' + labels[name] + '</td><td>' + str(data['stats'][name]['successes'])
                        + '/20</td><td>' + f"{data['stats'][name]['mean_q']:.4f}" + '</td></tr>' for name in labels)
    contrasts = []
    for name, contrast in data['contrasts'].items():
        low, high = contrast['paired_q_bootstrap95']
        contrasts.append('<p>Huber50 相对 ' + labels[name] + f"：平均 Q 差 {contrast['mean_q_delta']:+.4f}，"
                         + f"配对 bootstrap 95% 区间 [{low:+.4f}, {high:+.4f}]；新增成功/失去成功 "
                         + str(len(contrast['success_gains'])) + '/' + str(len(contrast['success_losses'])) + '。</p>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Huber50 三模型配对评测</title>'
                '<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:8px;border:1px solid #ddd;text-align:left}th{background:#eef3f7}</style>'
                '<h1>Huber50 三模型配对评测</h1><p>三个模型各 20 项。同一任务实例、种子 0、原生帧预算及推理协议；原模型和 MSE 结果复用已经核验的完整评测。</p>'
                '<table><tr><th>模型</th><th>成功数</th><th>平均任务进度 Q</th></tr>' + summaries + '</table>'
                + ''.join(contrasts)
                + '<p>这些实例从本轮微调训练索引排除；初始模型预训练是否见过这些实例仍未证明。单种子小面板结果用于选择候选，不能独立证明总体能力提升；选定后应在预留面板确认。</p>'
                '<table><tr><th>任务</th><th>实例</th><th>原模型：结果 / Q</th><th>MSE：结果 / Q</th><th>Huber50：结果 / Q</th><th>Huber50 − MSE</th></tr>'
                + ''.join(table) + '</table></html>')
    (ROOT / 'report.html').write_text(document)
    proof = {key: value for key, value in data.items() if key not in ('pairs', 'models', 'missing')}
    proof.update(complete=True, plan_sha256=digest(ROOT / 'plan.json'),
                 report_html_sha256=digest(ROOT / 'report.html'), csv_sha256=digest(ROOT / 'comparison.csv'))
    (ROOT / 'report_proof.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'published': True, 'episodes': 60, 'summaries': data['stats']}))


if __name__ == '__main__':
    main()
