import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import training as base
from parallel_training import Parallel, Checkpoints, buffers, parity_gate, train, warm_critic
from variant_update import prepare_rewards, update_policy


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.tensor([.1]))
        self.value_head = torch.nn.Linear(1, 1)
        self.frozen = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)

    def default_forward(self, inputs, compute_values=True):
        x = inputs['x']
        result = {'logprobs': (self.actor*x).reshape(1, 1, 1).expand(1, 1, 23)/23}
        if compute_values:
            result['values'] = self.value_head(x).reshape(1)
        return result

    def _observation_from_forward_inputs(self, inputs):
        return inputs

    def _build_prefix_cache_for_actor(self, observation):
        return observation['x'][:,None,:], torch.ones((1,1),dtype=torch.bool), None


def episode(model, length, success):
    x = torch.tensor([[1. if success else -1.]])
    previous = model.default_forward({'x': x})
    rows = []
    for i in range(length):
        rows.append({'forward_inputs': {'x': x}, 'prev_logprobs': previous['logprobs'].detach().clone(),
                     'prev_values': previous['values'].detach().clone(), 'terminal': i == length-1,
                     'reward': float(success and i == length-1), 'task_reward': float(success and i == length-1),
                     'goals_before': {'goals': {'a': False}}, 'goals_after': {'goals': {'a': bool(success)}}})
    return {'instance': 1, 'status': {'success': success}, 'transitions': rows, 'bootstrap': torch.zeros(1)}


class InterruptAtCommit(Exception):
    pass


def warm_worker(rank, root):
    torch.set_num_threads(1)
    root=Path(root)
    dist.init_process_group('gloo',init_method='file://'+str(root/'warm_rendezvous'),rank=rank,world_size=4)
    torch.manual_seed(0);model=Toy();episodes=[episode(model,9,True),episode(model,14,False)]
    plan={'optimizer':{'gamma':.99},'critic_warmup':{'max_samples_per_trajectory':4,
        'validation_stride':3,'learning_rate':.05,'epochs':10,'batch_size':4,'seed':7}}
    warm_critic(model,episodes,plan,Parallel('cpu'),lambda **_:None)
    if rank==0:
        torch.save({'head':model.value_head.state_dict(),
                    'values':torch.stack([x['prev_values'] for e in episodes for x in e['transitions']])},root/'warm_actual.pt')
    dist.destroy_process_group()


def worker(rank, world, root, variant, interrupt, learning_rate):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method='file://'+str(root/'rendezvous'), rank=rank, world_size=world)
    torch.manual_seed(0)
    model = Toy()
    episodes = [episode(model, 9, True), episode(model, 14, False)]
    plan = {'comparison_variant': variant, 'optimizer': {'actor_lr': learning_rate, 'value_lr': .0001,
        'weight_decay': 0., 'shuffle_seed': 5, 'updates_per_round': [2], 'gamma': 1. if variant == 'gamma1_ppo' else .99,
        'gae_lambda': .95, 'clip_ratio': .2, 'kl_target': .01}, 'shaping': {'gamma': 1., 'beta': 1.}}
    if variant != 'control':
        prepare_rewards(episodes, plan)
    parallel = Parallel('cpu')
    optimizer = base.build_optimizer(model, plan)
    rows, data = buffers(episodes, plan)
    gate = parity_gate(model, optimizer, rows, data, plan, parallel, sample_count=17)
    assert gate['passed'], gate
    manager = Checkpoints(root/'checkpoint', {'variant': variant}, parallel)

    def publish(**state):
        if interrupt and state.get('stage') == 'update_checkpoint_saved' and state['accepted_optimizer_steps'] == 1:
            raise InterruptAtCommit()

    try:
        result = train(model, optimizer, rows, data, plan, parallel, manager, publish)
    except InterruptAtCommit:
        directory, metadata = manager.latest()
        model = Toy()
        base.load_overlay(model, directory, metadata['trainable_sha256'])
        optimizer = base.build_optimizer(model, plan)
        optimizer.load_state_dict(torch.load(directory/'optimizer.pt', weights_only=True))
        data = torch.load(directory/'buffers.pt', weights_only=True)
        result = train(model, optimizer, rows, data, plan, parallel, manager, lambda **_: None, resume=metadata)
    if rank == 0:
        torch.save({'model': model.state_dict(), 'result': result, 'gate': gate}, root/'actual.pt')
    dist.destroy_process_group()


class DistributedTraining(unittest.TestCase):
    def test_distributed_critic_warmup_matches_serial(self):
        torch.manual_seed(0);model=Toy();episodes=[episode(model,9,True),episode(model,14,False)]
        plan={'optimizer':{'gamma':.99},'critic_warmup':{'max_samples_per_trajectory':4,
            'validation_stride':3,'learning_rate':.05,'epochs':10,'batch_size':4,'seed':7}}
        base.warm_modem_critic(model,episodes,plan,lambda **_:None)
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(warm_worker,args=(root,),nprocs=4,join=True)
            actual=torch.load(Path(root)/'warm_actual.pt',weights_only=True)
            for name,value in model.value_head.state_dict().items():
                torch.testing.assert_close(actual['head'][name],value,atol=1e-7,rtol=1e-7)
            torch.testing.assert_close(actual['values'],torch.stack([x['prev_values'] for e in episodes for x in e['transitions']]),atol=1e-7,rtol=1e-7)

    def compare(self, variant, interrupt=False, learning_rate=.001):
        torch.manual_seed(0)
        model = Toy()
        episodes = [episode(model, 9, True), episode(model, 14, False)]
        plan = {'comparison_variant': variant, 'optimizer': {'actor_lr': learning_rate, 'value_lr': .0001,
            'weight_decay': 0., 'shuffle_seed': 5, 'updates_per_round': [2],
            'gamma': 1. if variant == 'gamma1_ppo' else .99, 'gae_lambda': .95, 'clip_ratio': .2, 'kl_target': .01},
            'shaping': {'gamma': 1., 'beta': 1.}}
        if variant != 'control':
            prepare_rewards(episodes, plan)
        optimizer = base.build_optimizer(model, plan)
        expected = (base.update_policy(model, optimizer, episodes, plan)[0] if variant == 'control'
                    else update_policy(model, optimizer, episodes, plan))
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(worker, args=(4, root, variant, interrupt, learning_rate), nprocs=4, join=True)
            actual = torch.load(Path(root)/'actual.pt', weights_only=True)
            self.assertEqual(actual['result']['optimizer_steps'], expected['optimizer_steps'])
            self.assertEqual(actual['result']['early_stopped'], expected['early_stopped'])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(actual['model'][name], value, atol=1e-6, rtol=1e-5)
            self.assertTrue(actual['gate']['passed'])
            self.assertEqual(len(list((Path(root)/'checkpoint/resume').glob('step_*'))), 1)

    def test_all_objectives_match_serial_with_uneven_shards(self):
        for variant in ['control', 'gamma1_ppo', 'balanced_ppo', 'group_outcome']:
            with self.subTest(variant=variant):
                self.compare(variant)

    def test_resume_after_committed_step_matches_uninterrupted(self):
        self.compare('balanced_ppo', interrupt=True)

    def test_global_kl_rejection_rolls_back_all_ranks(self):
        self.compare('group_outcome', learning_rate=10.)

    def test_resume_after_final_directory_move_before_ready(self):
        model = Toy()
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as root:
            manager = Checkpoints(root, {'source': 'fixed'}, Parallel('cpu'))
            directory, _ = manager.save(model, optimizer, {'values': torch.ones(3)}, {'accepted': 2})
            final = Path(root)/'checkpoints/round_01'
            final.parent.mkdir()
            directory.rename(final)
            restored, metadata = manager.latest()
            self.assertEqual(restored, final)
            self.assertEqual(metadata['state']['accepted'], 2)

    def test_only_uncommitted_resume_snapshots_are_removed(self):
        model = Toy()
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as root:
            manager = Checkpoints(root, {'source': 'fixed'}, Parallel('cpu'))
            committed, _ = manager.save(model, optimizer, {}, {'accepted': 1})
            incomplete = Path(root)/'resume/step_02_incomplete'
            incomplete.mkdir()
            (incomplete/'partial').write_text('interrupted write')
            unrelated = Path(root)/'retained_result.json'
            unrelated.write_text('{}')
            manager.clean_uncommitted()
            self.assertTrue(committed.is_dir())
            self.assertFalse(incomplete.exists())
            self.assertTrue(unrelated.is_file())


if __name__ == '__main__':
    unittest.main()
