"""Two independent GPU workers, synchronous rollouts and one unchanged PPO learner.

Shared AFS carries atomic manifests, never competing writers for one episode.
Both policies must match the collection manifest before executing controls.
"""
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
                      optimizer_to, save_checkpoint, update_policy, warmup_critic)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def coordination(plan):
    return Path(plan['output']) / 'coordination'


def wait_json(path, plan, timeout=None):
    deadline = time.monotonic() + (timeout or plan['compute_timeout_seconds'])
    while time.monotonic() < deadline:
        # A peer failure invalidates this batch, even when a partial result exists.
        for failed in coordination(plan).glob('failure_rank*.json'):
            raise RuntimeError('Peer failed: ' + failed.read_text()[:1500])
        if Path(path).is_file():
            return json.loads(Path(path).read_text())
        threading.Event().wait(.5)
    raise TimeoutError('Synchronization deadline: ' + str(path))


def trainable_hash(model):
    return hash_parameters((n, p) for n, p in model.named_parameters() if p.requires_grad)


def recover_one(model, plan, round_index, episode_index):
    relative = Path('training') / f'round_{round_index:02d}' / f'episode_{episode_index + 1:02d}'
    for source_root in plan['resume_sources']:
        source = Path(source_root) / relative
        rollout = source / 'rollout.pt'
        if not rollout.is_file() or not (source / 'result.json').is_file():
            continue
        expected = plan['migration']['completed_rollouts'].get(str(rollout))
        assert expected, 'Unfrozen completed episode cannot be silently accepted'
        assert rollout.stat().st_size == expected['bytes']
        assert sha256(rollout) == expected['sha256']
        episode = torch.load(rollout, map_location='cpu', weights_only=True)
        assert episode['instance'] == plan['train_instances'][episode_index]
        assert episode['seed'] == plan['train_seeds'][round_index - 1][episode_index]
        device = next(model.parameters()).device
        value_error = 0.
        with torch.no_grad():
            for row in episode['transitions']:
                actual = model.default_forward({k: v.to(device) for k, v in row['forward_inputs'].items()})
                torch.testing.assert_close(actual['logprobs'], row['prev_logprobs'].to(device), atol=1e-4, rtol=1e-4)
                value_error = max(value_error, float((actual['values'].reshape(-1) - row['prev_values'].to(device).reshape(-1)).abs().max()))
        assert value_error <= 1e-4, f'Recovered critic differs: {value_error}'
        destination = Path(plan['output']) / relative
        copy_evidence(source, destination)
        write(destination / 'recovery.json', {'source': str(source), 'rollout_path': str(rollout),
              'sha256': expected['sha256'], 'reused_complete_episode': True,
              'logprob_replay_verified': True, 'max_value_error': value_error})
        return episode, rollout, True
    return None


def collect_partition(model, plan, rank, round_index, policy_hash, publish):
    # Every episode resets Python/NumPy/Torch RNG to its original explicit seed.
    assert trainable_hash(model) == policy_hash
    for index in range(rank, plan['protocol']['episodes_per_round'], 2):
        publish(stage='training_rollout', round=round_index, episode=index + 1,
                instance=plan['train_instances'][index])
        recovered = recover_one(model, plan, round_index, index)
        if recovered is None:
            data = collect(model, plan, round_index, index, publish)
            path = Path(plan['output']) / 'training' / f'round_{round_index:02d}' / f'episode_{index + 1:02d}' / 'rollout.pt'
            reused = False
        else:
            data, path, reused = recovered
        assert trainable_hash(model) == policy_hash, 'Collection modified the policy'
        write(coordination(plan) / f'round_{round_index:02d}_episode_{index + 1:02d}.json',
              {'rank': rank, 'round': round_index, 'episode': index + 1, 'policy_hash': policy_hash,
               'rollout_path': str(path), 'sha256': sha256(path), 'reused': reused,
               'instance': data['instance'], 'seed': data['seed'], 'status': data['status']})
        del data
    write(coordination(plan) / f'round_{round_index:02d}_rank{rank}_done.json', {'policy_hash': policy_hash})


def gather_episodes(plan, round_index, policy_hash):
    episodes = []
    # Original episode ordering is retained; GAE, normalization and shuffle are unchanged.
    for index in range(plan['protocol']['episodes_per_round']):
        receipt = wait_json(coordination(plan) / f'round_{round_index:02d}_episode_{index + 1:02d}.json', plan)
        assert receipt['policy_hash'] == policy_hash
        path = Path(receipt['rollout_path'])
        assert sha256(path) == receipt['sha256']
        data = torch.load(path, map_location='cpu', weights_only=True)
        assert data['instance'] == plan['train_instances'][index]
        assert data['seed'] == plan['train_seeds'][round_index - 1][index]
        episodes.append(data)
    for rank in (0, 1):
        assert wait_json(coordination(plan) / f'round_{round_index:02d}_rank{rank}_done.json', plan)['policy_hash'] == policy_hash
    return episodes


def publish_collection(model, plan, round_index, frozen_hash):
    expected = trainable_hash(model)
    if round_index == 1:
        directory = Path(plan['output']) / 'collection_policy'
        directory.mkdir(exist_ok=False)
        enforce_storage_budget(plan, 2 * 1024**3)
        path = directory / 'trainable.pt'
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, path)
        verified = torch.load(path, map_location='cpu', weights_only=True)
        assert hash_parameters(verified.items()) == expected
        del verified
    else:
        directory = Path(plan['output']) / 'checkpoints' / f'round_{round_index - 1:02d}'
    manifest = {'round': round_index, 'directory': str(directory), 'trainable_sha256': expected,
                'frozen_sha256': frozen_hash, 'instances': plan['train_instances'],
                'seeds': plan['train_seeds'][round_index - 1]}
    write(coordination(plan) / f'round_{round_index:02d}_policy.json', manifest)
    return manifest


def final_partition(model, plan, rank, policy_hash, publish):
    assert trainable_hash(model) == policy_hash
    cases = plan['evaluation']['cases'][rank::2]
    partition = {**plan, 'evaluation': {**plan['evaluation'], 'cases': cases}}
    rows = evaluate(model, partition, 'final', publish, write_summary=False)
    assert trainable_hash(model) == policy_hash
    write(coordination(plan) / f'final_rank{rank}_done.json',
          {'policy_hash': policy_hash, 'rank': rank, 'cases': cases, 'results': rows})


def follower(plan, publish):
    seed_all(plan['model_seed'])
    model = torch_model(plan)
    frozen_hash = hash_parameters((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    for index in range(1, plan['rounds'] + 1):
        publish(stage='waiting_collection_policy', round=index)
        manifest = wait_json(coordination(plan) / f'round_{index:02d}_policy.json', plan)
        assert manifest['frozen_sha256'] == frozen_hash
        assert manifest['instances'] == plan['train_instances']
        assert manifest['seeds'] == plan['train_seeds'][index - 1]
        load_overlay(model, manifest['directory'], manifest['trainable_sha256'])
        collect_partition(model, plan, 1, index, manifest['trainable_sha256'], publish)
    publish(stage='waiting_final_policy')
    manifest = wait_json(coordination(plan) / 'final_policy.json', plan)
    assert manifest['frozen_sha256'] == frozen_hash
    load_overlay(model, manifest['directory'], manifest['trainable_sha256'])
    final_partition(model, plan, 1, manifest['trainable_sha256'], publish)
    publish(stage='waiting_comparison')
    return wait_json(Path(plan['output']) / 'comparison.json', plan)


def leader(plan, publish):
    root = Path(plan['output'])
    old = Path(plan['previous_run'])
    assert json.loads((old / 'baseline/checks.json').read_text())['weights_unchanged']
    copy_evidence(old / 'baseline', root / 'baseline')
    baseline = json.loads((root / 'baseline/summary.json').read_text())
    assert [(r['instance'], r['seed']) for r in baseline] == [tuple(x) for x in plan['evaluation']['cases']]
    # These completed old-checkpoint evaluations are evidence only, never executed again.
    copy_evidence(Path(plan['resume_sources'][0]) / 'previous_final', root / 'previous_final')
    prior = Path(plan['resume_sources'][0]) / 'previous_comparison.json'
    if prior.is_file():
        write(root / 'previous_comparison.json', json.loads(prior.read_text()))
    seed_all(plan['model_seed'])
    model = torch_model(plan)
    assert trainable_hash(model) == plan['expected_original_trainable_sha256']
    frozen_hash = hash_parameters((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    actor_hash = hash_parameters(actor_parameters(model))
    publish(stage='critic_warmup')
    warmup = warmup_critic(model, plan, publish)
    assert actor_hash == hash_parameters(actor_parameters(model))
    write(root / 'critic_warmup.json', warmup)
    optimizer = build_optimizer(model, plan)
    rounds = []
    for index in range(1, plan['rounds'] + 1):
        optimizer_to(optimizer, 'cpu')
        gc.collect()
        torch.cuda.empty_cache()
        manifest = publish_collection(model, plan, index, frozen_hash)
        collect_partition(model, plan, 0, index, manifest['trainable_sha256'], publish)
        publish(stage='waiting_rollout_barrier', round=index)
        episodes = gather_episodes(plan, index, manifest['trainable_sha256'])
        assert trainable_hash(model) == manifest['trainable_sha256']
        publish(stage='training_update', round=index, chunks=sum(len(e['transitions']) for e in episodes))
        optimizer_to(optimizer, model.device)
        update, advantages, returns = update_policy(model, optimizer, episodes, plan, index, publish)
        optimizer_to(optimizer, 'cpu')
        checkpoint = root / 'checkpoints' / f'round_{index:02d}'
        enforce_storage_budget(plan, plan['runtime_storage']['max_checkpoint_write_reservation_bytes'])
        saved = save_checkpoint(model, optimizer, checkpoint)
        report = {'round': index, 'rollouts': [{'instance': e['instance'], 'seed': e['seed'], **e['status']} for e in episodes],
                  'update': update, 'checkpoint': saved, 'collection_policy_unchanged': True}
        write(root / 'training' / f'round_{index:02d}' / 'result.json', report)
        write(checkpoint / 'ready.json', {'round': index, **saved})
        write(root / 'latest_checkpoint.json', {'round': index, 'directory': str(checkpoint), **saved})
        if index > 1:
            (root / 'checkpoints' / f'round_{index - 1:02d}' / 'optimizer.pt').unlink()
        rounds.append(report)
        del episodes, advantages, returns
        gc.collect()
        torch.cuda.empty_cache()
        publish(stage='training_round_complete', round=index, optimizer_steps=update['optimizer_steps'])
    assert hash_parameters((n, p) for n, p in model.named_parameters() if not p.requires_grad) == frozen_hash
    write(root / 'training_summary.json', {'rounds': rounds, 'frozen_sha256_before': frozen_hash,
          'frozen_sha256_after': frozen_hash, 'total_controls': sum(e['steps'] for r in rounds for e in r['rollouts']),
          'total_optimizer_steps': sum(r['update']['optimizer_steps'] for r in rounds)})
    final_hash = trainable_hash(model)
    write(coordination(plan) / 'final_policy.json', {'directory': str(checkpoint),
          'trainable_sha256': final_hash, 'frozen_sha256': frozen_hash})
    final_partition(model, plan, 0, final_hash, publish)
    for rank in (0, 1):
        receipt = wait_json(coordination(plan) / f'final_rank{rank}_done.json', plan)
        assert receipt['policy_hash'] == final_hash
    final = [json.loads((root / 'final' / f'i{i}_s{s}' / 'result.json').read_text())
             for i, s in plan['evaluation']['cases']]
    write(root / 'final/summary.json', final)
    write(root / 'final/checks.json', {'weights_unchanged': True, 'trainable_sha_before': final_hash,
          'trainable_sha_after': final_hash, 'cases': plan['evaluation']['cases'], 'evaluation_policy': plan['eval_policy']})
    report = compare(plan, baseline, final)
    write(root / 'comparison.json', report)
    publish(stage='comparison_complete', comparison=report)
    return report


def run(plan, publish):
    rank = int(os.environ['B1K_RANK'])
    assert rank in (0, 1) and plan['gpu_count'] == 2
    return leader(plan, publish) if rank == 0 else follower(plan, publish)
