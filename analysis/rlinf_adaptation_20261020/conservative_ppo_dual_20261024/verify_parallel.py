"""Process-level synchronization, reuse, stale-policy and peer-failure checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

import parallel
from common import write


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.]))

    def default_forward(self, inputs):
        return {'logprobs': inputs['x'] * self.weight, 'values': inputs['x'] * self.weight}


def make_episode(index):
    x = torch.tensor([index + .5])
    row = {'forward_inputs': {'x': x}, 'prev_logprobs': x.clone(),
           'prev_values': x.clone(), 'reward': 1., 'terminal': True, 'done': True}
    return {'transitions': [row], 'bootstrap': torch.zeros(1),
            'instance': index + 1, 'seed': 2000 + index,
            'status': {'steps': 16, 'score': 1., 'success': True}}


def worker(plan_path, rank):
    plan = json.loads(Path(plan_path).read_text())
    def collect(model, plan, round_index, index, publish):
        target = Path(plan['output']) / 'training/round_01' / f'episode_{index + 1:02d}'
        target.mkdir(parents=True, exist_ok=False)
        write(target / 'called.json', {'rank': rank, 'start': time.time()})
        # A short delay makes overlap observable without real simulation.
        time.sleep(.3)
        episode = make_episode(index)
        torch.save(episode, target / 'rollout.pt')
        write(target / 'result.json', episode['status'])
        return episode
    parallel.collect = collect
    model = Model()
    parallel.collect_partition(model, plan, rank, 1, parallel.trainable_hash(model), lambda **kw: None)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / 'prior/training/round_01/episode_01'
        source.mkdir(parents=True)
        torch.save(make_episode(0), source / 'rollout.pt')
        write(source / 'result.json', make_episode(0)['status'])
        plan = {'output': str(root / 'result'), 'resume_sources': [str(root / 'prior')],
                'train_instances': [1, 2, 3, 4], 'train_seeds': [[2000, 2001, 2002, 2003]],
                'compute_timeout_seconds': 20, 'protocol': {'episodes_per_round': 4},
                'migration': {'completed_rollouts': {str(source / 'rollout.pt'):
                    {'bytes': (source / 'rollout.pt').stat().st_size, 'sha256': parallel.sha256(source / 'rollout.pt')}}}}
        write(root / 'plan.json', plan)
        children = [subprocess.Popen([sys.executable, __file__, '--worker', str(root / 'plan.json'), str(rank)]) for rank in (0, 1)]
        assert all(p.wait(timeout=30) == 0 for p in children)
        model = Model()
        rows = parallel.gather_episodes(plan, 1, parallel.trainable_hash(model))
        assert [r['instance'] for r in rows] == [1, 2, 3, 4]
        assert [r['seed'] for r in rows] == [2000, 2001, 2002, 2003]
        calls = list((root / 'result').glob('training/round_01/episode_*/called.json'))
        assert len(calls) == 3 and not any(p.parent.name == 'episode_01' for p in calls)
        assert not (root / 'result/training/round_01/episode_01/rollout.pt').exists(), 'Reused raw data must not be duplicated'
        for actual, expected in zip(rows, [make_episode(i) for i in range(4)]):
            torch.testing.assert_close(actual['transitions'][0]['prev_logprobs'], expected['transitions'][0]['prev_logprobs'], atol=0, rtol=0)
        model.weight.data.fill_(2.)
        try:
            parallel.collect_partition(model, plan, 0, 1, parallel.trainable_hash(Model()), lambda **kw: None)
            raise AssertionError('Stale policy was accepted')
        except AssertionError as e:
            if str(e) == 'Stale policy was accepted':raise
        wrong = copy.deepcopy(plan)
        wrong['output'] = str(root / 'bad_recovery')
        try:
            parallel.recover_one(model, wrong, 1, 0)
            raise AssertionError('Wrong-policy recovery was accepted')
        except AssertionError as e:
            if str(e) == 'Wrong-policy recovery was accepted':raise
        receipt = parallel.coordination(plan) / 'round_01_episode_04.json'
        original = json.loads(receipt.read_text())
        write(receipt, {**original, 'policy_hash': 'wrong'})
        try:
            parallel.gather_episodes(plan, 1, parallel.trainable_hash(Model()))
            raise AssertionError('Mixed-policy batch was accepted')
        except AssertionError as e:
            if str(e) == 'Mixed-policy batch was accepted':raise
        write(parallel.coordination(plan) / 'failure_rank1.json', {'error_type': 'injected'})
        before = time.monotonic()
        try:
            parallel.wait_json(root / 'missing.json', plan)
            raise AssertionError('Peer failure was ignored')
        except RuntimeError:pass
        assert time.monotonic() - before < 1.
    result = {'passed': True, 'checks': ['two independent processes', 'one owner per episode',
        'completed episode reused without resampling or raw-data copy', 'original ordering and samples',
        'stale policy rejected', 'wrong-policy recovery rejected', 'mixed-policy batch rejected', 'peer failure terminates wait']}
    Path(__file__).with_name('parallel_test_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':worker(sys.argv[2], int(sys.argv[3]))
    else:main()
