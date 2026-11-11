"""Frozen six-case baseline, five PPO rounds, then the same six evaluation cases."""
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from common import Simulator, env_observation, label_text, torch_model, write
from training import build_optimizer, hash_parameters, load_warm_start, optimizer_to, save_checkpoint, update_policy


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, plan, phase, publish):
    root = Path(plan['output'])/phase
    before = hash_parameters((n, p) for n, p in model.named_parameters() if p.requires_grad)
    rows = []
    for instance, seed in plan['evaluation']['cases']:
        output = root/f'i{instance}_s{seed}'
        case_plan = {**plan, 'seed': seed}
        seed_all(seed)
        publish(stage=phase, instance=instance, seed=seed, completed_cases=len(rows))
        sim = Simulator(case_plan, output, instance, 'public_test', plan['eval_steps'])
        rng = np.random.default_rng(seed)
        actions_log = []
        try:
            while sim.current['status']['steps'] < plan['eval_steps']:
                raw = sim.current['observation']
                noise = torch.from_numpy(rng.standard_normal((1, 32, 32)).astype(np.float32)).to(model.device)
                with torch.no_grad():
                    actions, result = model.predict_action_batch(env_observation(raw), mode='eval', noise=noise)
                text = label_text(model, result)
                action = actions[0].cpu().numpy()
                assert action.shape == (16, 23) and np.isfinite(action).all()
                remaining = plan['eval_steps']-sim.current['status']['steps']
                actions_log.append({'subtask': text, 'actions': action[:remaining].tolist()})
                sim.step(action[:remaining], text)
                write(output/'progress.json', sim.current['status'])
                if sim.current['done']:break
            row = {'instance': instance, 'seed': seed, **sim.current['status']}
            write(output/'result.json', row)
            write(output/'actions.json', actions_log)
            rows.append(row)
        finally:
            sim.close()
        publish(stage=phase, instance=instance, seed=seed, completed_cases=len(rows), last_result=rows[-1])
    after = hash_parameters((n, p) for n, p in model.named_parameters() if p.requires_grad)
    assert after == before
    write(root/'summary.json', rows)
    write(root/'checks.json', {'weights_unchanged': True, 'trainable_sha_before': before, 'trainable_sha_after': after,
                               'cases': plan['evaluation']['cases'], 'evaluation_policy': plan['eval_policy']})
    return rows


def collect(model, plan, round_index, publish):
    seed = plan['train_seeds'][round_index-1]
    seed_all(seed)
    output = Path(plan['output'])/'training'/f'round_{round_index:02d}'
    sim = Simulator({**plan, 'seed': seed}, output/'rollout', plan['train_instance'], 'train', plan['train_steps_per_round'])
    transitions = []
    try:
        while sim.current['status']['steps'] < plan['train_steps_per_round']:
            with torch.no_grad():
                actions, result = model.predict_action_batch(env_observation(sim.current['observation']), mode='train')
            previous = {k: {n: t.detach().cpu() for n, t in v.items()} if isinstance(v, dict) else v.detach().cpu()
                        for k, v in result.items() if k != 'model_actions'}
            remaining = plan['train_steps_per_round']-sim.current['status']['steps']
            current = sim.step(actions[0].cpu().numpy()[:remaining], label_text(model, result))
            previous.update(reward=float(sum(current['rewards']))*10., terminal=current['terminal'], done=current['done'])
            transitions.append(previous)
            write(output/'progress.json', {'stage': 'rollout', 'round': round_index, 'chunks': len(transitions), **current['status']})
            if current['done']:break
        if transitions[-1]['terminal']:
            bootstrap = torch.zeros(1)
        else:
            with torch.no_grad():
                _, end = model.predict_action_batch(env_observation(sim.current['observation']), mode='train')
            bootstrap = end['prev_values'].reshape(1).cpu()
        final_status = sim.current['status']
    finally:
        sim.close()
    publish(stage='training_update', round=round_index, rollout=final_status, chunks=len(transitions))
    return transitions, bootstrap, final_status, output


def compare(plan, baseline, final):
    assert len(baseline) == len(final) == len(plan['evaluation']['cases'])
    pairs = []
    for left, right in zip(baseline, final):
        assert (left['instance'], left['seed']) == (right['instance'], right['seed'])
        case = f"i{left['instance']}_s{left['seed']}"
        a = json.loads((Path(plan['output'])/'baseline'/case/'initial.json').read_text())
        b = json.loads((Path(plan['output'])/'final'/case/'initial.json').read_text())
        assert a['physical_state'].keys() == b['physical_state'].keys()
        error = max(float(np.max(np.abs(np.asarray(a['physical_state'][k][part])-np.asarray(b['physical_state'][k][part]))))
                    for k in a['physical_state'] for part in (0,1))
        assert all(a[key] == b[key] for key in ('physics_dt','rendering_dt','observation_dt'))
        pairs.append({'instance': left['instance'], 'seed': left['seed'], 'baseline': left, 'final': right,
                      'initial_max_error': error, 'score_delta': right['score']-left['score'],
                      'success_delta': int(right['success'])-int(left['success'])})
    def aggregate(rows):
        return {'episodes': len(rows), 'mean_score': sum(r['score'] for r in rows)/len(rows),
                'successes': sum(r['success'] for r in rows), 'success_rate': sum(r['success'] for r in rows)/len(rows)}
    before, after = aggregate(baseline), aggregate(final)
    valid = all(p['initial_max_error'] <= plan['evaluation']['paired_initial_pose_tolerance'] for p in pairs)
    ds, dp = after['mean_score']-before['mean_score'], after['success_rate']-before['success_rate']
    if not valid: conclusion = 'Initial-state pairing check failed; do not attribute the difference to PPO.'
    elif ds > 0 and dp > 0: conclusion = 'Improvement observed on this fixed six-case pilot only; broader reliability remains untested.'
    elif ds == 0 and dp == 0: conclusion = 'No improvement observed on the fixed evaluation panel.'
    elif ds < 0 or dp < 0: conclusion = 'At least one aggregate metric regressed on the fixed evaluation panel.'
    else: conclusion = 'Only one aggregate metric improved; evidence is mixed.'
    return {'comparison_valid': valid, 'baseline': before, 'final': after, 'mean_score_delta': ds,
            'success_rate_delta': dp, 'success_to_failure': sum(p['success_delta'] < 0 for p in pairs),
            'failure_to_success': sum(p['success_delta'] > 0 for p in pairs), 'pairs': pairs,
            'conclusion': conclusion, 'scope': 'Two fixed instances, three paired seeds, one training run; no checkpoint selection.'}


def run(plan, publish):
    root = Path(plan['output'])
    seed_all(plan['model_seed'])
    model = torch_model(plan)
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    frozen_before = hash_parameters(frozen)
    publish(stage='baseline', completed_cases=0)
    baseline = evaluate(model, plan, 'baseline', publish)
    optimizer = build_optimizer(model, plan)
    warm_hash = load_warm_start(model, optimizer, plan)
    write(root/'warm_start.json', {'passed': True, 'trainable_sha256': warm_hash, 'source': plan['warm_start']})
    previous = None
    rounds = []
    for index in range(1, plan['rounds']+1):
        optimizer_to(optimizer, 'cpu')
        gc.collect(); torch.cuda.empty_cache()
        publish(stage='training_rollout', round=index)
        transitions, bootstrap, rollout, output = collect(model, plan, index, publish)
        optimizer_to(optimizer, model.device)
        update, advantages, returns = update_policy(model, optimizer, transitions, bootstrap, plan)
        torch.save({'transitions': transitions, 'advantages': advantages, 'returns': returns}, output/'rollout.pt')
        optimizer_to(optimizer, 'cpu')
        checkpoint = root/'checkpoints'/f'round_{index:02d}'
        saved = save_checkpoint(model, optimizer, checkpoint)
        assert saved['trainable_sha256'] != (rounds[-1]['checkpoint']['trainable_sha256'] if rounds else warm_hash)
        report = {'round': index, 'seed': plan['train_seeds'][index-1], 'rollout': rollout, 'update': update, 'checkpoint': saved}
        write(output/'result.json', report)
        write(checkpoint/'ready.json', {'round': index, **saved})
        write(root/'latest_checkpoint.json', {'round': index, 'directory': str(checkpoint), **saved})
        # Model snapshots remain available; only the newly superseded optimizer is removed.
        if previous is not None:(previous/'optimizer.pt').unlink()
        previous = checkpoint
        rounds.append(report)
        del transitions, bootstrap, advantages, returns
        gc.collect(); torch.cuda.empty_cache()
        publish(stage='training_round_complete', round=index, reward_sum=update['reward_sum'], optimizer_steps=update['optimizer_steps'])
    assert hash_parameters(frozen) == frozen_before
    write(root/'training_summary.json', {'rounds': rounds, 'frozen_sha256_before': frozen_before,
                                       'frozen_sha256_after': hash_parameters(frozen),
                                       'total_controls': sum(r['rollout']['steps'] for r in rounds),
                                       'total_optimizer_steps': sum(r['update']['optimizer_steps'] for r in rounds)})
    optimizer_to(optimizer, 'cpu')
    gc.collect(); torch.cuda.empty_cache()
    final = evaluate(model, plan, 'final', publish)
    report = compare(plan, baseline, final)
    assert hash_parameters(frozen) == frozen_before
    write(root/'comparison.json', report)
    publish(stage='comparison_complete', comparison=report)
    return report
