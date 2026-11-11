"""32-episode synchronous batches on two workers with 17 + 13 PPO steps."""
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import torch

from common import torch_model, write
from experiment import collect, compare, copy_evidence, enforce_storage_budget, evaluate, seed_all
from training import (actor_parameters, build_optimizer, hash_parameters, load_overlay,
                      optimizer_to, save_checkpoint, update_policy, load_warm_start)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def coordination(plan):
    return Path(plan['output']) / 'coordination'


def check_peer_failure(plan):
    for failed in coordination(plan).glob('failure_rank*.json'):
        raise RuntimeError('Peer failed: ' + failed.read_text()[:1500])


def wait_json(path, plan, timeout=None):
    end = time.monotonic() + (timeout or plan['compute_timeout_seconds'])
    while time.monotonic() < end:
        check_peer_failure(plan)
        if Path(path).is_file():
            return json.loads(Path(path).read_text())
        threading.Event().wait(1)
    raise TimeoutError('Coordination deadline: ' + str(path))


def trainable_hash(model):
    return hash_parameters((n, p) for n, p in model.named_parameters() if p.requires_grad)


def snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore(model, state, expected):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(state) == names
    outcome = model.load_state_dict(state, strict=False)
    assert not outcome.unexpected_keys and set(outcome.missing_keys) == set(model.state_dict()) - names
    assert trainable_hash(model) == expected


def recover_one(model, plan, round_index, episode_index):
    if round_index != 1 or episode_index not in plan['reuse']['round1_episode_indices']:
        return None
    receipt = plan['reuse']['round1_receipts'][episode_index]
    assert receipt['episode'] == episode_index + 1
    assert receipt['policy_hash'] == trainable_hash(model)
    path = Path(receipt['rollout_path'])
    assert sha256(path) == receipt['sha256']
    data = torch.load(path, map_location='cpu', weights_only=True)
    assert data['instance'] == plan['train_instances'][episode_index]
    assert data['seed'] == plan['train_seeds'][0][episode_index]
    max_value_error = 0.
    with torch.no_grad():
        for row in data['transitions']:
            actual = model.default_forward({k: v.to(model.device) for k, v in row['forward_inputs'].items()})
            torch.testing.assert_close(actual['logprobs'], row['prev_logprobs'].to(model.device), atol=1e-4, rtol=1e-4)
            max_value_error = max(max_value_error, float((actual['values'].flatten()-row['prev_values'].to(model.device).flatten()).abs().max()))
    assert max_value_error <= 1e-4
    target = Path(plan['output']) / 'training/round_01' / f'episode_{episode_index+1:02d}'
    copy_evidence(path.parent, target)
    write(target / 'recovery.json', {'source': str(path), 'sha256': receipt['sha256'],
          'reused_complete_episode': True, 'logprob_replay_verified': True,
          'max_value_error': max_value_error})
    return data, path, True


def collect_partition(model, plan, rank, round_index, policy_hash, publish):
    assert trainable_hash(model) == policy_hash
    for index in range(rank, plan['protocol']['episodes_per_round'], 2):
        check_peer_failure(plan)
        publish(stage='training_rollout', round=round_index, episode=index+1,
                instance=plan['train_instances'][index], seed=plan['train_seeds'][round_index-1][index])
        recovered = recover_one(model, plan, round_index, index)
        if recovered is None:
            data = collect(model, plan, round_index, index, publish)
            path = Path(plan['output']) / 'training' / f'round_{round_index:02d}' / f'episode_{index+1:02d}/rollout.pt'
            reused = False
        else:
            data, path, reused = recovered
        assert trainable_hash(model) == policy_hash
        write(coordination(plan) / f'round_{round_index:02d}_episode_{index+1:02d}.json',
              {'rank': rank, 'round': round_index, 'episode': index+1, 'policy_hash': policy_hash,
               'rollout_path': str(path), 'sha256': sha256(path), 'reused': reused,
               'instance': data['instance'], 'seed': data['seed'], 'status': data['status']})
        del data
    write(coordination(plan) / f'round_{round_index:02d}_rank{rank}_done.json', {'policy_hash': policy_hash})


def gather_episodes(plan, round_index, policy_hash):
    episodes = []
    for index in range(plan['protocol']['episodes_per_round']):
        receipt = wait_json(coordination(plan) / f'round_{round_index:02d}_episode_{index+1:02d}.json', plan)
        assert receipt['policy_hash'] == policy_hash
        path = Path(receipt['rollout_path'])
        assert sha256(path) == receipt['sha256']
        data = torch.load(path, map_location='cpu', weights_only=True)
        assert data['instance'] == plan['train_instances'][index]
        assert data['seed'] == plan['train_seeds'][round_index-1][index]
        episodes.append(data)
    for rank in (0, 1):
        assert wait_json(coordination(plan) / f'round_{round_index:02d}_rank{rank}_done.json', plan)['policy_hash'] == policy_hash
    return episodes


def publish_collection(model, plan, round_index, frozen_hash):
    directory = (Path(plan['reuse']['initial_policy_directory']) if round_index == 1 else
                 Path(plan['resume']['checkpoint_directory']) if round_index == 2 and plan.get('resume') else
                 Path(plan['output']) / 'checkpoints' / f'round_{round_index-1:02d}')
    manifest = {'round': round_index, 'directory': str(directory), 'trainable_sha256': trainable_hash(model),
                'frozen_sha256': frozen_hash, 'instances': plan['train_instances'],
                'seeds': plan['train_seeds'][round_index-1]}
    write(coordination(plan) / f'round_{round_index:02d}_policy.json', manifest)
    return manifest


def evaluate_partition(model, original_state, plan, rank, final_manifest, publish):
    root = Path(plan['output'])
    final_state = torch.load(Path(final_manifest['directory']) / 'trainable.pt', map_location='cpu', weights_only=True)
    assert hash_parameters(final_state.items()) == final_manifest['trainable_sha256']
    final_rows = []
    for instance, seed in plan['evaluation']['cases'][rank::2]:
        check_peer_failure(plan)
        case = f'i{instance}_s{seed}'
        single = {**plan, 'evaluation': {**plan['evaluation'], 'cases': [[instance, seed]]}}
        if not (root / 'baseline' / case / 'result.json').is_file():
            restore(model, original_state, plan['expected_original_trainable_sha256'])
            evaluate(model, single, 'baseline', publish, reuse_existing=True, write_summary=False)
        restore(model, final_state, final_manifest['trainable_sha256'])
        final_rows.extend(evaluate(model, single, 'final', publish, reuse_existing=True, write_summary=False))
    write(coordination(plan) / f'final_rank{rank}_done.json',
          {'policy_hash': final_manifest['trainable_sha256'], 'rank': rank, 'results': final_rows})
    del final_state


def run(plan, publish):
    rank = int(os.environ['B1K_RANK'])
    assert rank in (0, 1) and plan['gpu_count'] == 2
    assert plan['action_interface']['gripper_open_tolerance'] == 0
    assert plan['optimizer']['updates_per_round'] == [17, 13]
    seed_all(plan['model_seed'])
    model = torch_model(plan)
    assert trainable_hash(model) == plan['expected_original_trainable_sha256']
    original_state = snapshot(model)
    frozen_hash = hash_parameters((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    assert frozen_hash == plan['reuse']['frozen_hash']
    # Restore the completed first round; do not repeat sampling or its 17 updates.
    assert plan['resume']['completed_round'] == 1
    load_overlay(model, plan['resume']['checkpoint_directory'], plan['resume']['expected_trainable_sha256'])
    if rank == 0:
        write(Path(plan['output']) / 'initialization.json',
              {'original_trainable_hash': plan['expected_original_trainable_sha256'],
               'collection_hash': trainable_hash(model), 'resumed_after_round': 1,
               'completed_optimizer_steps_before_resume': 17, 'new_optimizer': False})
        source = Path(plan['resume']['source_output'])
        first = json.loads((source/'training/round_01/result.json').read_text())
        assert first['update']['optimizer_steps'] == 17
        assert sha256(source/'training/round_01/result.json') == plan['resume']['source_result_sha256']
        write(Path(plan['output'])/'training/round_01/result.json', first)
        for source_receipt in sorted((source/'coordination').glob('round_01_episode_*.json')):
            receipt = json.loads(source_receipt.read_text())
            receipt['reused_by_resume'] = True
            receipt['source_receipt'] = str(source_receipt)
            write(coordination(plan)/source_receipt.name, receipt)
        write(Path(plan['output'])/'training/round_01/recovery.json',
              {'source':str(source),'completed_episodes_reused':32,'completed_updates_reused':17})
        checks = plan['reuse']['baseline_checks']
        assert checks['weights_unchanged'] and checks['trainable_sha_before'] == plan['expected_original_trainable_sha256']
        for instance, seed in plan['reuse']['baseline_cases']:
            case = f'i{instance}_s{seed}'
            source = Path(plan['reuse']['baseline_root']) / case
            target = Path(plan['output']) / 'baseline' / case
            copy_evidence(source, target)
            write(target / 'recovery.json', {'source': str(source), 'original_interface': True,
                  'reused_complete_episode': True, 'original_checkpoint_hash': checks['trainable_sha_before']})
        write(coordination(plan) / 'baseline_reuse_ready.json', {'cases': plan['reuse']['baseline_cases']})
        optimizer = build_optimizer(model, plan)
        load_warm_start(model, optimizer, plan)
        steps = sorted({int(state['step']) for state in optimizer.state.values() if 'step' in state})
        assert steps and max(steps) == 17 and min(steps) > 0, 'First-round Adam state was not restored'
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    assert torch.isfinite(value).all(), 'Nonfinite restored optimizer state'
        write(Path(plan['output'])/'optimizer_restore.json',
              {'step_counters':steps,'state_entries':len(optimizer.state),
               'learning_rates':[group['lr'] for group in optimizer.param_groups],
               'trainable_sha256':trainable_hash(model)})
        rounds = [first]
        for index in range(2, 3):
            optimizer_to(optimizer, 'cpu'); gc.collect(); torch.cuda.empty_cache()
            manifest = publish_collection(model, plan, index, frozen_hash)
            collect_partition(model, plan, 0, index, manifest['trainable_sha256'], publish)
            publish(stage='waiting_rollout_barrier', round=index)
            episodes = gather_episodes(plan, index, manifest['trainable_sha256'])
            assert len(episodes) == 32 and trainable_hash(model) == manifest['trainable_sha256']
            publish(stage='training_update', round=index, chunks=sum(len(e['transitions']) for e in episodes))
            optimizer_to(optimizer, model.device)
            def publish_update(**fields):
                check_peer_failure(plan)
                publish(**fields)
            update, advantages, returns = update_policy(model, optimizer, episodes, plan, index, publish_update)
            optimizer_to(optimizer, 'cpu')
            # A KL stop must not silently change the promised comparison budget.
            assert update['optimizer_steps'] == plan['optimizer']['updates_per_round'][index-1], 'Frozen update budget not completed; keep evidence, do not change LR or KL guard'
            checkpoint = Path(plan['output']) / 'checkpoints' / f'round_{index:02d}'
            enforce_storage_budget(plan, plan['runtime_storage']['max_checkpoint_write_reservation_bytes'])
            saved = save_checkpoint(model, optimizer, checkpoint)
            report = {'round': index, 'rollouts': [{'instance': e['instance'], 'seed': e['seed'], **e['status']} for e in episodes],
                      'update': update, 'checkpoint': saved, 'collection_policy_unchanged': True}
            write(Path(plan['output']) / 'training' / f'round_{index:02d}' / 'result.json', report)
            write(checkpoint / 'ready.json', {'round': index, **saved})
            write(Path(plan['output']) / 'latest_checkpoint.json', {'round': index, 'directory': str(checkpoint), **saved})
            rounds.append(report); del episodes, advantages, returns
            gc.collect(); torch.cuda.empty_cache()
            publish(stage='training_round_complete', round=index, optimizer_steps=update['optimizer_steps'])
        assert sum(r['update']['optimizer_steps'] for r in rounds) == 30
        assert hash_parameters((n, p) for n, p in model.named_parameters() if not p.requires_grad) == frozen_hash
        write(Path(plan['output']) / 'training_summary.json',
              {'rounds': rounds, 'total_optimizer_steps': 30, 'episodes': 64,
               'frozen_sha256_before': frozen_hash, 'frozen_sha256_after': frozen_hash})
        final_manifest = {'directory': str(checkpoint), 'trainable_sha256': trainable_hash(model), 'frozen_sha256': frozen_hash}
        write(coordination(plan) / 'final_policy.json', final_manifest)
        del optimizer; gc.collect(); torch.cuda.empty_cache()
    else:
        for index in range(2, 3):
            publish(stage='waiting_collection_policy', round=index)
            manifest = wait_json(coordination(plan) / f'round_{index:02d}_policy.json', plan)
            assert manifest['frozen_sha256'] == frozen_hash
            assert manifest['instances'] == plan['train_instances'] and manifest['seeds'] == plan['train_seeds'][index-1]
            load_overlay(model, manifest['directory'], manifest['trainable_sha256'])
            collect_partition(model, plan, 1, index, manifest['trainable_sha256'], publish)
        publish(stage='waiting_final_policy')
        final_manifest = wait_json(coordination(plan) / 'final_policy.json', plan)
    wait_json(coordination(plan) / 'baseline_reuse_ready.json', plan)
    evaluate_partition(model, original_state, plan, rank, final_manifest, publish)
    if rank == 0:
        for peer in (0, 1):
            result = wait_json(coordination(plan) / f'final_rank{peer}_done.json', plan)
            assert result['policy_hash'] == final_manifest['trainable_sha256']
        baseline = []; final = []
        for instance, seed in plan['evaluation']['cases']:
            for phase, target in [('baseline', baseline), ('final', final)]:
                target.append(json.loads((Path(plan['output']) / phase / f'i{instance}_s{seed}/result.json').read_text()))
        report = compare(plan, baseline, final)
        for phase, rows, digest in [('baseline', baseline, plan['expected_original_trainable_sha256']),
                                    ('final', final, final_manifest['trainable_sha256'])]:
            write(Path(plan['output']) / phase / 'summary.json', rows)
            write(Path(plan['output']) / phase / 'checks.json', {'weights_unchanged': True,
                  'trainable_sha_before': digest, 'trainable_sha_after': digest,
                  'cases': plan['evaluation']['cases'], 'evaluation_policy': plan['eval_policy']})
        write(Path(plan['output']) / 'comparison.json', report)
        return report
    publish(stage='waiting_comparison')
    return wait_json(Path(plan['output']) / 'comparison.json', plan)
