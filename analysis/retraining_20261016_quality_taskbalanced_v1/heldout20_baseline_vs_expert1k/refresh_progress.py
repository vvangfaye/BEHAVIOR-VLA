"""Validate and snapshot partial results without publishing a full comparison."""
import json
from pathlib import Path
import time

import build_report

ROOT = Path(__file__).resolve().parent


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    plan, pairs, _, proofs, missing = build_report.collect()
    old_path = ROOT / 'accepted_snapshot.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else {'episodes': []}
    prior = {row['id']: row['sha256'] for row in old['episodes']}
    episodes, by_role = [], {'baseline': {}, 'candidate': {}}
    for proof in proofs:
        report = json.loads((ROOT / proof['file']).read_text())
        identity, row = report['id'], report['record']
        pair = row['task'], row['instance_id']
        build_report.validate({'protocol_sha256': plan['protocol_sha256'], 'records': [row]},
                              [pair], plan['protocol_sha256'])
        if identity in prior and prior[identity] != proof['sha256']:
            raise ValueError('Previously accepted report changed: ' + identity)
        by_role[report['role']][pair] = row
        episodes.append({'id': identity, 'role': report['role'], 'record': row, 'sha256': proof['sha256']})
    if not set(prior).issubset({row['id'] for row in episodes}):
        raise ValueError('Previously accepted report disappeared')
    matches = []
    for pair in pairs:
        if pair in by_role['baseline'] and pair in by_role['candidate']:
            baseline, candidate = by_role['baseline'][pair], by_role['candidate'][pair]
            matches.append({'pair': pair, 'baseline': baseline, 'candidate': candidate,
                            'q_delta': candidate['q_score'] - baseline['q_score']})
    now = time.time()
    save(old_path, {'observed_at': now, 'episodes': episodes})
    save(ROOT / 'paired_progress.json', {'observed_at': now, 'matched_pairs': matches,
         'full_panel_complete': not missing, 'missing_episodes': missing, 'overall_improvement_proven': False})
    print(json.dumps({'accepted': {role: len(rows) for role, rows in by_role.items()},
         'matched_pairs': len(matches), 'q_gains': sum(row['q_delta'] > 1e-9 for row in matches),
         'q_regressions': sum(row['q_delta'] < -1e-9 for row in matches),
         'full_panel_complete': not missing}))


if __name__ == '__main__':
    main()
