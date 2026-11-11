"""Publish the paired heldout report only after all 40 episodes validate."""
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from compare_candidate import paired, summary, validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    plan = json.loads((ROOT / 'plan.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    canonical = json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != plan['protocol_sha256'] or protocol['pairs'] != plan['pairs']:
        raise ValueError('Protocol changed')
    bundle = digest(ROOT / 'frozen_sha256.json')
    for name, expected in json.loads((ROOT / 'frozen_sha256.json').read_text()).items():
        if digest(ROOT / 'frozen' / name) != expected:
            raise ValueError('Frozen source changed')
    pairs = list(map(tuple, plan['pairs']))
    if len(pairs) != 20 or len(set(pairs)) != 20:
        raise ValueError('Panel is not 20 unique pairs')
    missing, models, proofs = [], {}, []
    for role in ('baseline', 'candidate'):
        records = []
        expected_checkpoint = (plan['expected_checkpoint_manifest_sha256'] if role == 'candidate' else
            hashlib.sha256(json.dumps(plan['baseline_files'], sort_keys=True).encode()).hexdigest())
        for index, pair in enumerate(pairs):
            identity = role + '-' + f'{index:02d}'
            path = ROOT / 'reports' / (identity + '.json')
            if not path.exists():
                missing.append(identity)
                continue
            report = json.loads(path.read_text())
            if report.get('state') != 'accepted':
                missing.append(identity)
                continue
            expected = {'id': identity, 'role': role, 'worker_bundle_sha256': bundle,
                        'protocol_sha256': plan['protocol_sha256'], 'checkpoint_manifest_sha256': expected_checkpoint}
            if any(report.get(k) != v for k, v in expected.items()):
                raise ValueError('Episode identity mismatch: ' + identity)
            row = report['record']
            if (row['task'], row['instance_id']) != pair:
                raise ValueError('Episode pair mismatch')
            if row['truncated'] and row['steps'] != plan['native_task_budgets'][pair[0]] + 1:
                raise ValueError('Episode stopped before its native budget')
            records.append(row)
            proofs.append({'file': str(path.relative_to(ROOT)), 'sha256': digest(path)})
        if len(records) == 20:
            models[role] = validate({'protocol_sha256': plan['protocol_sha256'], 'records': records}, pairs, plan['protocol_sha256'])
    return plan, pairs, models, proofs, missing


def verify_exclusion(pairs):
    source = ROOT.parent / 'heldout20_candidate_panel.json'
    manifest = json.loads(source.read_text())
    if pairs != list(map(tuple, manifest['pairs'])):
        raise ValueError('Predeclared holdout panel changed')
    chosen = set(pairs)
    for proof in manifest['source_proofs']:
        path = ROOT.parent / proof['path']
        if digest(path) != proof['sha256']:
            raise ValueError('Training index changed')
        index = json.loads(path.read_text())
        if not chosen.issubset(set(map(tuple, index['held_out_instances']))):
            raise ValueError('Panel is not predeclared heldout data')
        if any((row['task_name'], row['task_instance_id']) in chosen for row in index['episodes']):
            raise ValueError('A heldout pair appears in training')
    return {'file': str(source.relative_to(ROOT.parent)), 'sha256': digest(source),
            'indexed_episode_exclusion_rechecked': True, 'baseline_pretraining_exclusion_proven': False}


def verify_native_completion(plan):
    expected, recoveries = {}, []
    for group in plan['groups']:
        intent = json.loads((ROOT / 'intents' / (group['id'] + '.json')).read_text())
        expected[intent['name']] = intent['uid']
    for path in sorted((ROOT / 'intents').glob('recovery-*.json')):
        intent = json.loads(path.read_text())
        directory = ROOT / ('recovery_g' + str(intent['generation']))
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        stopped_path = directory / 'old_writer_stopped.json'
        stopped = json.loads(stopped_path.read_text())
        if (intent['id'] != manifest['id'] or intent['name'] != manifest['name'] or
                intent['members'] != manifest['members'] or
                manifest['original_plan_sha256'] != digest(ROOT / 'plan.json') or
                manifest['worker_bundle_sha256'] != digest(ROOT / 'frozen_sha256.json') or
                stopped['old_uid'] != manifest['old_uid'] or stopped.get('worker_absent') is not True):
            raise ValueError('Recovery provenance differs')
        if digest(directory / 'private/startup.py') != manifest['startup_sha256']:
            raise ValueError('Recovery startup source changed')
        for identity, expected_hash in manifest['preserved_reports'].items():
            if digest(ROOT / 'reports' / (identity + '.json')) != expected_hash:
                raise ValueError('A preserved accepted episode changed')
        if intent['name'] in expected:
            raise ValueError('Duplicate native job identity')
        expected[intent['name']] = intent['uid']
        recoveries.append({'manifest_sha256': digest(manifest_path),
                           'old_writer_stop_sha256': digest(stopped_path),
                           'members': manifest['members'], 'old_uid': manifest['old_uid'],
                           'new_uid': intent['uid']})
    status_path = ROOT / 'status.json'
    status = json.loads(status_path.read_text())
    states = status['states']
    terminal = {'SUCCEEDED', 'FAILED', 'STOPPED', 'CANCELLED', 'DELETED', 'SUSPENDED'}
    if (status.get('complete') is not True or status['accepted'] != {'baseline': 20, 'candidate': 20}
            or len(states) != len(expected) or {s['name'] for s in states} != set(expected)
            or any(s['uid'] != expected[s['name']] or s['state'] not in terminal for s in states)):
        raise ValueError('Wait for every original and recovery job to become terminal')
    return {'status_sha256': digest(status_path), 'native_jobs': states, 'recoveries': recoveries}


def main():
    plan, pairs, models, proofs, missing = collect()
    if missing:
        print(json.dumps({'published': False, 'missing_episodes': len(missing)}))
        return
    native = verify_native_completion(plan)
    exclusion = verify_exclusion(pairs)
    comparison = json.loads((ROOT / 'comparison.json').read_text())
    stats = {role: summary(rows, pairs) for role, rows in models.items()}
    contrast = paired(models['baseline'], models['candidate'], pairs)
    serialized_contrast = json.loads(json.dumps(contrast))
    if not comparison.get('complete') or comparison['summaries'] != stats or comparison['candidate_vs_baseline'] != serialized_contrast:
        raise ValueError('Coordinator comparison differs from validated reports')
    if comparison['report_sha256'] != {Path(p['file']).stem: p['sha256'] for p in proofs}:
        raise ValueError('Coordinator report sources changed')
    with (ROOT / 'comparison.csv').open(encoding='utf-8-sig') as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != 20 or [(r['task'], int(r['instance'])) for r in csv_rows] != pairs:
        raise ValueError('CSV panel differs')
    for row, pair in zip(csv_rows, pairs):
        old, new = models['baseline'][pair], models['candidate'][pair]
        if (row['baseline_success'] != str(old['model_success']) or
                row['expert_success'] != str(new['model_success']) or
                float(row['baseline_q']) != old['q_score'] or float(row['expert_q']) != new['q_score'] or
                float(row['q_delta']) != new['q_score']-old['q_score']):
            raise ValueError('CSV metrics differ from accepted reports')
    labels = {'baseline': '原模型 1688977', 'candidate': '纯专家 1,000 步'}
    head = ''.join('<tr><td>'+labels[r]+'</td><td>'+str(stats[r]['successes'])+'/20</td><td>'+
                   f"{stats[r]['mean_q']:.4f}"+'</td></tr>' for r in labels)
    rows = []
    for pair in pairs:
        old, new = models['baseline'][pair], models['candidate'][pair]
        cells = [html.escape(pair[0]), str(pair[1])]
        for row in (old, new):
            cells.append(('成功' if row['model_success'] else '未成功')+f" / {row['q_score']:.4f}")
            cells.append(str(row['steps']))
        cells.append(f"{new['q_score']-old['q_score']:+.4f}")
        rows.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>')
    interval = contrast['paired_q_bootstrap95']
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>微调留出实例配对评测</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#17212b}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:8px;border:1px solid #ddd;text-align:left}th{background:#eef3f7}</style><h1>微调留出实例配对评测</h1><p>原模型与纯专家模型各20组，所有40个原生回合已通过任务实例、权重、源码、种子和终止状态校验。每组使用相同实例、种子0和原生帧预算。</p><table><tr><th>模型</th><th>成功数</th><th>平均任务进度 Q</th></tr>'''+head+'''</table><p>这些实例已从本轮微调训练索引中排除，并重新核验了索引。初始模型预训练是否见过这些实例仍未证明。本结果为单种子、小面板对照，不与先前实例301的20组混合计分。</p><p>纯专家相对原模型的平均Q差：'''+f"{contrast['mean_q_delta']:+.4f}"+'''；配对bootstrap 95%区间：'''+f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"+'''；新增成功/失去成功：'''+str(len(contrast['success_gains']))+'/'+str(len(contrast['success_losses']))+'''。</p><table><tr><th>任务</th><th>实例</th><th>原模型：结果 / Q</th><th>步数</th><th>纯专家：结果 / Q</th><th>步数</th><th>Q差</th></tr>'''+''.join(rows)+'''</table></html>'''
    if native['recoveries']:
        document += '<p>仿真停滞的回合在确认旧worker退出后单独补跑；已完成回合保留。原作业及恢复作业均已进入终态，恢复时沿用相同权重、实例、种子和帧预算。</p>'
    output = ROOT / 'report.html'
    output.write_text(document)
    proof = {'complete': True, 'protocol_sha256': plan['protocol_sha256'], 'summaries': stats,
             'candidate_vs_baseline': contrast, 'input_proofs': proofs, 'exclusion': exclusion,
             'native_completion': native,
             'plan_sha256': digest(ROOT / 'plan.json'), 'report_html_sha256': digest(output),
             'csv_sha256': digest(ROOT / 'comparison.csv')}
    (ROOT / 'report_proof.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'published': True, 'episodes': 40, 'summaries': stats}))


if __name__ == '__main__':
    main()
