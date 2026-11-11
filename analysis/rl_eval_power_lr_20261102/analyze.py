"""Final (or interim) analysis for rl_eval_power_lr_20261102. Read-only; writes report.json and REPORT.md."""
from collections import defaultdict
import json
from math import comb
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / 'plan.json').read_text())
CASES = [tuple(c) for c in plan['eval_cases']]
REPEAT = [tuple(c) for c in plan['repeat_cases']]
ARMS = [a for a in plan['eval_arms'] if (HERE / 'evidence' / a).exists()]


def result(arm, case):
    path = HERE / 'evidence' / arm / f'i{case[0]}_s{case[1]}' / 'result.json'
    return json.loads(path.read_text()) if path.exists() else None


def mcnemar(gains, losses):
    n = gains + losses
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n)


def sign_flip(diffs, iterations=50000):
    values = [d for d in diffs if abs(d) > 1e-12]
    if not values:
        return 1.0
    observed, rng = abs(sum(values)), random.Random(0)
    hits = sum(abs(sum(v if rng.random() < .5 else -v for v in values)) >= observed - 1e-12 for _ in range(iterations))
    return (hits + 1) / (iterations + 1)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5 / (1 + z * z / n)
    return [center - half, center + half]


def compare(reference, arm, cases):
    pairs = [(c, result(reference, c), result(arm, c)) for c in cases]
    pairs = [(c, a, b) for c, a, b in pairs if a and b]
    n = len(pairs)
    if not n:
        return {'reference': reference, 'arm': arm, 'pairs': 0}
    gains = sum(b['success'] and not a['success'] for _, a, b in pairs)
    losses = sum(a['success'] and not b['success'] for _, a, b in pairs)
    base = sum(a['success'] for _, a, _ in pairs)
    new = sum(b['success'] for _, _, b in pairs)
    diffs = [b['score'] - a['score'] for _, a, b in pairs]
    # Paired difference CI (normal approximation on per-pair success differences).
    d = [int(b['success']) - int(a['success']) for _, a, b in pairs]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / max(1, n - 1)
    half = 1.96 * (var / n) ** .5
    return {'reference': reference, 'arm': arm, 'pairs': n,
            'reference_success_rate': base / n, 'arm_success_rate': new / n,
            'success_rate_delta': (new - base) / n, 'delta_95ci': [mean - half, mean + half],
            'arm_success_wilson95': wilson(new, n), 'gains': gains, 'losses': losses,
            'discordant_fraction': (gains + losses) / n, 'mcnemar_p': mcnemar(gains, losses),
            'reference_mean_score': sum(a['score'] for _, a, _ in pairs) / n,
            'arm_mean_score': sum(b['score'] for _, _, b in pairs) / n,
            'score_delta': sum(diffs) / n, 'score_sign_flip_p': sign_flip(diffs)}


def per_instance(arm):
    rows = defaultdict(lambda: [0, 0])
    for c in CASES:
        r = result(arm, c)
        if r:
            rows[c[0]][0] += r['success']
            rows[c[0]][1] += 1
    return {str(i): f'{s}/{n}' for i, (s, n) in sorted(rows.items())}


report = {'completed_per_arm': {a: sum(result(a, c) is not None for c in (REPEAT if a == 'origrep' else CASES)) for a in ARMS}}
report['noise_floor_AA'] = compare('orig', 'origrep', REPEAT)
report['vs_original'] = {a: compare('orig', a, CASES) for a in ARMS if a not in ('orig', 'origrep')}
report['vs_old17'] = {a: compare('old17', a, CASES) for a in ARMS if a not in ('orig', 'origrep', 'old17')}
report['per_instance_success'] = {a: per_instance(a) for a in ARMS if a != 'origrep'}
training = {}
for path in sorted((HERE / 'checkpoints').glob('*.json')):
    r = json.loads(path.read_text())['training_result']
    steps = r['update_steps']
    training[path.stem] = {'actor_lr': json.loads(path.read_text())['actor_lr'], 'accepted': r['optimizer_steps'],
                           'final_kl': r['final_batch_kl']['mean'],
                           'final_outside_clip_fraction': r['final_batch_kl'].get('outside_clip_fraction'),
                           'max_ppo_clip_fraction': max((s.get('ppo_clip_fraction') or 0) for s in steps),
                           'first_step_kl': steps[0]['candidate_kl']}
report['training'] = training
(HERE / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

lines = ['# rl_eval_power_lr_20261102 结果', '', '## 训练端（更新是否真的发生）', '',
         '| actor LR | 接受更新 | 首步 KL | 最终 KL | 最大 PPO clip 比例 | 最终越界比例 |', '|---|---:|---:|---:|---:|---:|']
for arm, t in sorted(training.items(), key=lambda kv: kv[1]['actor_lr']):
    note = '（首步超过 0.01，已回滚）' if t['accepted'] == 0 else ''
    lines.append(f"| {t['actor_lr']:.0e} | {t['accepted']}/17{note} | {t['first_step_kl']:.5f} | {t['final_kl']:.5f} | "
                 f"{t['max_ppo_clip_fraction']:.3f} | {(t['final_outside_clip_fraction'] or 0):.3f} |")
aa = report['noise_floor_AA']
lines += ['', '## 噪声底（原模型 vs 原模型重跑）', '',
          f"{aa.get('pairs')} 对：翻转 {aa.get('gains')}+{aa.get('losses')}，翻转率 {aa.get('discordant_fraction', 0):.1%}，"
          f"McNemar p={aa.get('mcnemar_p', 1):.3f}", '', '## 各组 vs 原模型（同一 case 配对）', '',
          '| 组 | 配对数 | 原模型成功率 | 该组成功率 | 差值 (95% CI) | 修复/退化 | McNemar p | 分数差 | 分数 p |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for arm, c in report['vs_original'].items():
    if c.get('pairs'):
        lines.append(f"| {arm} | {c['pairs']} | {c['reference_success_rate']:.1%} | {c['arm_success_rate']:.1%} | "
                     f"{c['success_rate_delta']:+.1%} ({c['delta_95ci'][0]:+.1%}, {c['delta_95ci'][1]:+.1%}) | "
                     f"{c['gains']}/{c['losses']} | {c['mcnemar_p']:.3f} | {c['score_delta']:+.3f} | {c['score_sign_flip_p']:.3f} |")
(HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
if '--json' in sys.argv:
    print(json.dumps(report['completed_per_arm']))
