"""Read immutable rollouts and PPO buffers; never instantiate or update a policy."""
from collections import Counter, defaultdict
import gc
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import torch


def request(url):
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=90) as response:
        return response.read()


def put(url, value):
    body = json.dumps(value, allow_nan=False).encode()
    req = urllib.request.Request(url, data=body, method='PUT')
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=90) as response:
        assert response.status in [200, 201, 204]


def mc(reward, gamma):
    out = np.empty(len(reward))
    value = 0.
    for i in range(len(reward) - 1, -1, -1):
        value = float(reward[i]) + gamma * value
        out[i] = value
    return out


def gae(reward, value, gamma):
    out = np.empty(len(reward))
    advantage = 0.
    for i in range(len(reward) - 1, -1, -1):
        following = value[i + 1] if i + 1 < len(value) else 0.
        advantage = reward[i] + gamma * following - value[i] + gamma * .95 * advantage
        out[i] = advantage
    return out


def stat(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'min': float(x.min()), 'max': float(x.max()), 'std': float(x.std())}


def load_buffer(item):
    if 'npz_get' in item:
        with np.load(io.BytesIO(request(item['npz_get'])), allow_pickle=False) as data:
            return {k: data[k].astype(float).reshape(-1) for k in data.files}, None
    if 'path' in item:
        data = torch.load(item['path'], map_location='cpu', weights_only=True)
        snapshot = {'source': item['path'], 'immutable_final': True}
    else:
        root = Path(item['latest_root'])
        for attempt in range(5):
            try:
                marker = json.loads((root / 'resume/latest.json').read_text())
                candidate = root / marker['relative_path']
                if not candidate.exists():
                    candidate = root / 'checkpoints/round_01'
                with (candidate / 'buffers.pt').open('rb') as stream:
                    data = torch.load(stream, map_location='cpu', weights_only=True)
                snapshot = {'accepted': marker['state']['accepted'], 'identity': marker['identity'],
                            'source': str(candidate), 'read_committed_only': True}
                break
            except FileNotFoundError:
                if attempt == 4:
                    raise
    return {k: v.detach().float().reshape(-1).numpy().astype(float)
            for k, v in data.items() if torch.is_tensor(v)}, snapshot


def run(config):
    torch.set_num_threads(1)
    started = time.time()
    output = {}
    for task_name, task in config['tasks'].items():
        buffers = {}
        snapshots = {}
        for variant, item in task['buffers'].items():
            buffers[variant], snapshots[variant] = load_buffer(item)
        names = task['goal_names']
        episodes = []
        phases = []
        success_flags = []
        raw_all = defaultdict(list)
        returns_all = defaultdict(list)
        offset = 0
        for receipt in task['receipts']:
            e = torch.load(receipt['path'], map_location='cpu', weights_only=True)
            rows = e['transitions']
            n = len(rows)
            assert n == receipt['chunks'] and e['status']['success'] == receipt['status']['success']
            def goals(signals):
                return signals['goals'] if 'goals' in signals else {k: signals[k] for k in names}
            before = [goals(r['goals_before']) for r in rows]
            after = [goals(r['goals_after']) for r in rows]
            phi = np.array([sum(g.values()) for g in before], dtype=float)
            next_phi = np.array([0. if r['terminal'] else sum(g.values()) for r, g in zip(rows, after)])
            task_reward = np.array([r['task_reward'] for r in rows], dtype=float)
            reward99 = task_reward + .99 * next_phi - phi
            reward1 = task_reward + next_phi - phi
            assert np.max(np.abs(reward99 - np.array([r['reward'] for r in rows]))) < 1e-5
            events = [{'chunk': i, 'task_reward': float(task_reward[i]), 'before': a, 'after': b,
                       'terminal': bool(rows[i]['terminal'])}
                      for i, (a, b) in enumerate(zip(before, after)) if a != b or task_reward[i]]
            phase = [''.join(str(int(g[k])) for k in names) for g in before]
            report = {
                'episode': receipt['episode'], 'instance': receipt['instance'], 'seed': receipt['seed'],
                'success': bool(e['status']['success']), 'score': float(e['status']['score']), 'chunks': n,
                'initial_goals': before[0], 'final_goals': after[-1], 'task_reward_sum': float(task_reward.sum()),
                'task_reward_positive_sum': float(np.maximum(task_reward, 0).sum()),
                'task_reward_negative_sum': float(np.minimum(task_reward, 0).sum()),
                'events': events, 'goal_phase_counts': dict(Counter(phase)),
                'first_achieved': {k: next((i for i, g in enumerate(after) if g[k]), None) for k in names},
                'goal_losses': {k: sum(a[k] and not b[k] for a, b in zip(before, after)) for k in names},
                'counterfactual_returns': {}, 'variants': {},
            }
            for label, gamma, reward in [('gamma99', .99, reward99), ('gamma1', 1., reward1)]:
                G = mc(reward, gamma)
                task_G = mc(task_reward, gamma)
                assert np.max(np.abs(G - (task_G - phi))) < 1e-8
                report['counterfactual_returns'][label] = {'start': float(G[0]), 'start_task_only': float(task_G[0])}
            for variant, data in buffers.items():
                gamma = task['buffers'][variant]['gamma']
                reward = reward1 if gamma == 1. else reward99
                sl = slice(offset, offset + n)
                A, V, W = data['advantages'][sl], data['prev_values'][sl], data['weights'][sl]
                raw = gae(reward, V, gamma)
                G = mc(reward, gamma)
                report['variants'][variant] = {
                    'start_value': float(V[0]), 'start_return': float(G[0]), 'mean_adv': float(A.mean()),
                    'positive_fraction': float((A > 0).mean()), 'weight': float(W[0]),
                    'positive_mass': float((W * np.maximum(A, 0)).sum()),
                    'negative_mass': float((W * np.maximum(-A, 0)).sum()),
                    'abs_mass': float((W * np.abs(A)).sum()),
                    'head10_mean_adv': float(A[:max(1, n // 10)].mean()),
                    'tail10_mean_adv': float(A[-max(1, n // 10):].mean()),
                }
                raw_all[variant].append(raw)
                returns_all[variant].append(G)
            episodes.append(report)
            phases.extend(phase)
            success_flags.extend([report['success']] * n)
            offset += n
            put(config['status_put'], {'task': task_name, 'episodes': len(episodes), 'elapsed': time.time() - started})
            del e, rows
            gc.collect()
        success_mask = np.array(success_flags)
        phases = np.array(phases)
        summaries = {}
        for variant, data in buffers.items():
            A, V, W = data['advantages'], data['prev_values'], data['weights']
            assert len(A) == offset
            raw = np.concatenate(raw_all[variant])
            G = np.concatenate(returns_all[variant])
            normalized = (raw - raw.mean()) / (raw.std(ddof=1) + 1e-5)
            reconstruction_error = float(np.max(np.abs(normalized - A)))
            assert reconstruction_error < 1e-4, (task_name, variant, reconstruction_error)
            R = data['returns']
            gae_target_error = float(np.max(np.abs(R - (raw + V))))
            assert gae_target_error < 1e-4, (task_name, variant, gae_target_error)
            entry = {
                'advantage_reconstruction_max_error': reconstruction_error,
                'target_reconstruction_max_error': gae_target_error,
                'value_mc_mse': float(np.mean((G - V)**2)),
                'value_mc_explained_variance': float(1 - np.var(G - V) / np.var(G)),
                'mean_initial_value': float(np.mean([e['variants'][variant]['start_value'] for e in episodes])),
                'success_negative_mean_adv': sum(e['success'] and e['variants'][variant]['mean_adv'] < 0 for e in episodes),
                'failure_positive_mean_adv': sum(not e['success'] and e['variants'][variant]['mean_adv'] > 0 for e in episodes),
                'outcomes': {}, 'phases': {},
            }
            for outcome, mask in [('success', success_mask), ('failure', ~success_mask)]:
                entry['outcomes'][outcome] = {
                    'chunks': int(mask.sum()), 'weight_fraction': float(W[mask].sum() / W.sum()),
                    'abs_adv_mass_fraction': float((W[mask] * np.abs(A[mask])).sum() / (W * np.abs(A)).sum()),
                    'positive_mass': float((W[mask] * np.maximum(A[mask], 0)).sum()),
                    'negative_mass': float((W[mask] * np.maximum(-A[mask], 0)).sum()),
                    'mean_adv': float(A[mask].mean()), 'positive_fraction': float((A[mask] > 0).mean()),
                }
            for phase in sorted(set(phases)):
                mask = phases == phase
                entry['phases'][phase] = {'chunks': int(mask.sum()), 'weight_fraction': float(W[mask].sum() / W.sum()),
                    'mean_adv': float(A[mask].mean()), 'positive_fraction': float((A[mask] > 0).mean()),
                    'abs_mass_fraction': float((W[mask] * np.abs(A[mask])).sum() / (W * np.abs(A)).sum())}
            summaries[variant] = entry
        pairs = []
        for instance in sorted({e['instance'] for e in episodes}):
            group = [e for e in episodes if e['instance'] == instance]
            if len({e['success'] for e in group}) != 2:
                continue
            good = next(e for e in group if e['success'])
            bad = next(e for e in group if not e['success'])
            pair = {'instance': instance, 'success_episode': good['episode'], 'failure_episode': bad['episode']}
            for gamma in ['gamma99', 'gamma1']:
                pair[gamma] = {'success_return': good['counterfactual_returns'][gamma]['start'],
                               'failure_return': bad['counterfactual_returns'][gamma]['start']}
            pair['advantages'] = {v: {'success_mean': good['variants'][v]['mean_adv'],
                                     'failure_mean': bad['variants'][v]['mean_adv']} for v in buffers}
            pairs.append(pair)
        output[task_name] = {'episodes': episodes, 'variant_statistics': summaries, 'mixed_pairs': pairs,
                             'goal_names': names, 'buffer_snapshots': snapshots, 'chunks': offset}
    result = {'tasks': output, 'read_only': True, 'model_updates': 0, 'elapsed_seconds': time.time() - started}
    put(config['result_put'], result)
    put(config['status_put'], {'stage': 'complete', 'elapsed': time.time() - started})
    print('COMPARATIVE_READ_ONLY_AUDIT_COMPLETE', flush=True)
