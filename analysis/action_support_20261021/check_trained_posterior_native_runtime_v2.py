"""Bounded deployment-interface check using completed CONTROL weights only.

No simulator, training, expert subtask input, or candidate performance estimate.
This is not the gated server entry point and does not enable a native rollout.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from vla_runtime import load_base
from vla_data import VlaPilotData, digest
from train_posterior_support import choose_eval
from posterior_support_native import DeployedPosteriorAdapter, PosteriorRawPolicy


HERE = Path(__file__).resolve().parent
ROOT = HERE / 'posterior_native_runtime_check_v2'


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        assert digest(HERE / name) == expected, name
    source = HERE / 'posterior_support_pilot_ada01/formal_ambient_aux'
    receipt = json.loads((source / 'completion.json').read_text())
    assert receipt['accepted'] and receipt['returncode'] == 0
    weights = source / 'checkpoint.pt'
    assert digest(weights) == plan['checkpoint_sha256']
    saved = torch.load(weights, map_location='cpu', weights_only=False)
    assert (saved['complete'] and saved['step'] == 1000 and saved['seed'] == 17
            and saved['method'] == 'ambient_aux')
    training = json.loads((HERE / 'posterior_support_pilot_ada01/plan.json').read_text())
    assert saved['plan_sha256'] == digest(HERE / 'posterior_support_pilot_ada01/plan.json')
    torch.set_num_threads(4)
    torch.manual_seed(17)
    base = load_base(training['checkpoint'], training['tokenizer'])
    adapter = DeployedPosteriorAdapter(base).cuda().eval()
    expected = {name for name, p in adapter.named_parameters() if p.requires_grad}
    assert set(saved['weights']) == expected
    loaded = adapter.load_state_dict(saved['weights'], strict=False)
    assert not loaded.unexpected_keys and not (set(loaded.missing_keys) & expected)
    del saved
    data = VlaPilotData(training['data_directory'], base)
    raw_path = data.directory / data.plan['raw_targets']['path']
    with np.load(raw_path, allow_pickle=False) as raw:
        states = raw['state'].copy()
    chosen = choose_eval(data, data.test_ids, 1)
    assert chosen.tolist() == plan['observation_indices']
    policy = PosteriorRawPolicy(adapter, 'ambient_aux', seed=0)
    rows, outputs = [], []
    for index in chosen:
        index = int(index)
        observation = {
            'observation/state': states[index],
            'observation/image_0': np.array(data.images[index, 0]),
            'observation/image_1': np.array(data.images[index, 1]),
            'observation/image_2': np.array(data.images[index, 2]),
            'prompt': data.plan['samples'][index]['prompt'],
        }
        # The deployed policy receives only current images, proprio, and task.
        policy.reset()
        first = policy.infer(observation)
        policy.reset()
        repeated = policy.infer(observation)
        assert first['subtask'] == repeated['subtask']
        assert first['subtask_source'] == repeated['subtask_source'] == 'generated'
        np.testing.assert_allclose(first['actions'], repeated['actions'], rtol=0, atol=2e-5)
        assert base.action_horizon == policy.metadata['model_horizon'] == 32
        assert base.action_chunk == policy.metadata['action_horizon'] == 16
        assert first['actions'].shape == (16, 23) and np.isfinite(first['actions']).all()
        env = dict(states=states[index:index+1],
                   main_images=np.array(data.images[index:index+1, 0]),
                   wrist_images=np.array(data.images[index:index+1, 1:]),
                   task_descriptions=[observation['prompt']])
        images, normalized, _ = base.checkpoint_io.prepare(env)
        np.testing.assert_allclose(normalized, data.normalized_state[index:index+1], rtol=0, atol=1e-5)
        for j, key in enumerate(images):
            np.testing.assert_array_equal(images[key][0], data.images[index, j])
        rows.append(dict(index=index, sample_id=data.plan['samples'][index]['sample_id'],
                         generated_subtask=first['subtask'],
                         reset_max_action_difference=float(np.abs(first['actions']-repeated['actions']).max()),
                         actions_shape=list(first['actions'].shape)))
        outputs.append(first['actions'])
    np.savez_compressed(ROOT / 'actions.npz', indices=chosen, actions=np.stack(outputs))
    result = dict(passed=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  source_method='ambient_aux', checkpoint_sha256=plan['checkpoint_sha256'],
                  plan_sha256=digest(ROOT / 'plan.json'),
                  actions_sha256=digest(ROOT / 'actions.npz'), trainable_tensor_count=len(expected),
                  observations=rows, generated_subtasks=True, finite_actions=True,
                  image_and_state_preparation_match=True, reset_reproducible=True,
                  new_training_steps=0, simulator_episodes=0,
                  scope='Actual completed control weights through deployment adapter and raw policy; cached observations only. Not a candidate or task-success evaluation.')
    (ROOT / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
