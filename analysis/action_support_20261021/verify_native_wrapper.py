"""Exercise the actual native receding-horizon wrapper without a simulator."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'BEHAVIOR-VLAs/openpi/src'),
               str(ROOT / 'BEHAVIOR-VLAs/openpi/packages/openpi-client/src')]
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from vla_native_policy import TorchRawPolicy, SeededNativeWrapper, SEED_FIELD


class CaptureAdapter(nn.Module):
    def __init__(self, mask_draws):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.base = SimpleNamespace(joint_tokenizer=SimpleNamespace(
            processor=SimpleNamespace(decode=lambda ids: 'generated subtask')))
        self.mask_draws = mask_draws
        self.calls = []
    def predict_actions(self, env, *, method, noise, rng, steps):
        self.calls.append({'env': env, 'noise': noise.clone()})
        torch.rand(self.mask_draws, generator=rng)
        return noise[:, :16, :23], {'subtask_lengths': torch.tensor([2]),
                                    'subtask_tokens': torch.tensor([[7, 1]])}


def sequence(mask_draws):
    adapter = CaptureAdapter(mask_draws)
    raw = TorchRawPolicy(adapter, 'standard')
    native = B1KPolicyWrapper(raw, 'b1k/R1Pro', 'task', 'receding_horizon', 16, 32)
    policy = SeededNativeWrapper(native, raw)
    obs = {'robot_r1::proprio': np.zeros(61, dtype=np.float32)}
    keys = ['robot_r1::robot_r1:zed_link:Camera:0::rgb',
            'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb',
            'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb']
    for channel, key in enumerate(keys):
        obs[key] = np.full((240, 240, 4), 40 + channel * 50, dtype=np.uint8)
    try:
        policy.act(obs)
        raise AssertionError('Missing first-step seed was accepted')
    except ValueError as error:
        assert 'explicit' in str(error)
    actions = torch.stack([policy.act({**obs, **({SEED_FIELD: 0} if i == 0 else {})}) for i in range(34)])
    assert len(adapter.calls) == 3 and policy.applied_seed == 0
    reference = torch.cat([x['noise'][0, :16, :23] for x in adapter.calls])[:34]
    torch.testing.assert_close(actions, reference, rtol=0, atol=0)
    env = adapter.calls[0]['env']
    assert env['states'].shape == (1, 61)
    assert env['main_images'].shape == (1, 224, 224, 3)
    assert env['wrist_images'].shape == (1, 2, 224, 224, 3)
    assert env['main_images'][0, 0, 0, 0] == 40
    assert env['wrist_images'][0, 0, 0, 0, 0] == 90
    assert env['wrist_images'][0, 1, 0, 0, 0] == 140
    assert native.last_subtasks == {0: 'generated subtask'}
    assert native.last_subtask_sources == {0: 'generated'}
    policy.reset()
    repeat = torch.stack([policy.act({**obs, **({SEED_FIELD: 0} if i == 0 else {})}) for i in range(34)])
    torch.testing.assert_close(repeat, actions, rtol=0, atol=0)
    return actions


def main():
    first = sequence(0)
    other = sequence(97)
    torch.testing.assert_close(first, other, rtol=0, atol=0)
    result = {'passed': True, 'native_wrapper': 'Unmodified B1KPolicyWrapper',
              'environment_steps': 34, 'model_calls_per_episode': 3,
              'camera_order_and_resize_verified': True,
              'exact_first16_execution_verified': True, 'reset_reproducibility_verified': True,
              'action_noise_independent_of_mask_rng_consumption': True,
              'scope': 'Real wrapper around a captured-action stub; no VLA quality or simulator claim'}
    (HERE / 'native_wrapper_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
