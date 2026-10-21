"""One-batch interface/gradient check on the real 1688977 checkpoint.

This is not a policy training run or an estimate of task performance.
"""
import hashlib
import json
import argparse
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'analysis/rlinf_adaptation_20261020/upstream'
sys.path[:0] = [str(ROOT / 'BEHAVIOR-VLAs/rlinf_adapter'), str(UPSTREAM)]
for name in ('rlinf.models', 'rlinf.models.embodiment'):
    module = types.ModuleType(name)
    module.__path__ = [str(UPSTREAM.joinpath(*name.split('.')))]
    sys.modules[name] = module

import numpy as np
import torch
from omegaconf import OmegaConf

from b1k_rlinf.factory import build_model
from b1k_rlinf.io import DELTA_INDICES, extract_state
from vla_support_adapter import SupportActionAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'vla_interface_v2.json')
    args = parser.parse_args()
    torch.set_num_threads(12)
    torch.manual_seed(42)
    start = time.monotonic()
    checkpoint = ROOT / 'analysis/rlinf_adaptation_20261020/checkpoint1688977_openpi'
    tokenizer = ROOT / 'analysis/retraining_20261015_comet_rft60_40/launch_bundle/inputs/paligemma_tokenizer.model'
    cache = ROOT / 'analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1'
    records = json.loads((cache / 'cache_manifest.json').read_text())['records']
    record = records[0]
    sample = cache / record['npz_path']
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == record['npz_sha256']
    cfg = OmegaConf.create({
        'model_path': str(checkpoint), 'pi05': True, 'precision': 'fp32',
        'action_dim': 23, 'num_action_chunks': 16, 'num_steps': 10,
        'openpi': {'model_action_dim': 32, 'action_horizon': 32, 'max_token_len': 512,
                   'train_expert_only': True, 'noise_method': 'flow_sde', 'noise_level': .4,
                   'ignore_last': True, 'joint_logprob': False, 'add_value_head': False,
                   'config_name': 'pi05_behavior'},
        'b1k_joint': {'enabled': True, 'norm_stats_path': str(checkpoint / 'norm_stats.json'),
                      'tokenizer_path': str(tokenizer), 'state_layout': 'checkpoint61'}})
    print('Loading original checkpoint in memory only', flush=True)
    base = build_model(cfg, torch.float32).eval()
    adapter = SupportActionAdapter(base).eval()
    with np.load(sample, allow_pickle=False) as z:
        assert not z['action_is_pad'].any(), 'This check requires a full valid action chunk'
        env = {'states': z['observation.state'][None].copy(),
               'main_images': z['observation.rgb.zed_link_camera_0'][None].copy(),
               'wrist_images': np.stack([z['observation.rgb.left_realsense_link_camera_0'],
                                         z['observation.rgb.right_realsense_link_camera_0']])[None].copy(),
               'task_descriptions': [str(z['prompt'].item())]}
        target = z['action'].astype(np.float64)
        subtask = str(z['subtask'].item())
    images, normalized, prompts = base.checkpoint_io.prepare(env)
    tokens = base.joint_tokenizer.processor.encode(subtask, add_eos=True)
    arrays = [base.joint_tokenizer.actions(prompts[0], normalized[0], tokens)]
    observation = base._token_observation(images, normalized, arrays)
    state = extract_state(env['states'], 'checkpoint61')[0]
    physical_mask = np.zeros(23)
    physical_mask[DELTA_INDICES] = 1
    stats = base.checkpoint_io.stats['actions']
    target = (target - state * physical_mask - stats['mean']) / (stats['std'] + 1e-6)
    target = torch.from_numpy(np.pad(target.astype(np.float32), ((0, 0), (0, 9)))[None])
    noise = torch.randn_like(target)
    t = torch.tensor([.5])
    print('Checking zero-initialized conditioning against original action expert', flush=True)
    with torch.no_grad():
        obs, _, valid, kv = adapter.prepare_context(observation)
        expected = base.velocity_from_suffix(base.run_suffix(obs, noise, t, kv, valid))
        actual = adapter.velocity(obs, noise, t, torch.ones(1, 23), valid, kv)
        parity = float((expected - actual).abs().max())
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    del kv, expected, actual, obs
    print('Computing projected training loss and real action-expert gradients', flush=True)
    with np.load(sample, allow_pickle=False) as z:
        raw_active = torch.from_numpy(np.ptp(z['action'], axis=0)[None] > 0)
    losses = adapter.loss(observation, target, noise=noise, time=t, raw_active=raw_active)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    gradient = {}
    for name, parameter in [('mask_head', adapter.mask_head[-1].weight),
                            ('mask_condition', adapter.mask_condition.weight),
                            ('action_output', base.action_out_proj.weight),
                            ('action_expert_attention', base.llm.layers[0].attn.q_proj[1].weight)]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        gradient[name] = float(parameter.grad.norm())
        assert gradient[name] > 0
    frozen = [p for p in base.parameters() if not p.requires_grad]
    assert frozen and all(p.grad is None for p in frozen)
    before = adapter.mask_condition.weight.detach().clone()
    # Single in-memory SGD step only; do not overwrite the source checkpoint.
    parameters = [p for p in adapter.parameters() if p.requires_grad and p.grad is not None]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    assert torch.isfinite(norm)
    torch.optim.SGD(parameters, lr=1e-5).step()
    changed = float((before - adapter.mask_condition.weight).abs().max())
    assert changed > 0
    adapter.zero_grad(set_to_none=True)
    print('Sampling without target activity labels', flush=True)
    actions, inferred_mask, probs = adapter.sample(observation, noise=noise, steps=10,
                                                  rng=torch.Generator().manual_seed(17))
    assert actions.shape == (1, 32, 32) and torch.isfinite(actions).all()
    variation = actions[..., :23].amax(1) - actions[..., :23].amin(1)
    flat_error = float(variation[inferred_mask == 0].abs().max()) if (inferred_mask == 0).any() else 0.
    assert flat_error < 1e-5
    decoded = base.decode_actions(actions, observation.state)
    assert decoded.shape == (1, 16, 23) and torch.isfinite(decoded).all()
    print('Checking complete deployment using generated subtask tokens', flush=True)
    deployed, details = adapter.predict_actions(env, noise=noise,
                                                rng=torch.Generator().manual_seed(17))
    assert deployed.shape == (1, 16, 23) and torch.isfinite(deployed).all()
    count = int(details['subtask_lengths'][0])
    subtask_ids = details['subtask_tokens'][0, :count].tolist()
    assert subtask_ids[-1] == base.joint_tokenizer.eos_id
    result = {'passed': True, 'scope': 'One real expert observation and one in-memory SGD step; no full training, deployment, simulator, or task-success evidence',
              'checkpoint': str(checkpoint), 'sample_sha256': record['npz_sha256'],
              'task_index': record['task_index'], 'conditioning': 'existing expert subtask for SFT interface check',
              'zero_conditioning_velocity_max_error': parity, 'gradient_norms': gradient,
              'preclip_global_gradient_norm': float(norm), 'frozen_backbone_has_no_gradients': True,
              'mask_condition_step_max_change': changed, 'generated_model_shape': list(actions.shape),
              'decoded_environment_shape': list(decoded.shape), 'predicted_flat_channels': int((inferred_mask == 0).sum()),
              'max_flat_channel_range': flat_error, 'elapsed_seconds': time.monotonic() - start,
              'new_parameters': sum(p.numel() for p in adapter.mask_head.parameters()) + adapter.mask_condition.weight.numel(),
              'full_deployment_generated_subtask': base.joint_tokenizer.processor.decode(subtask_ids),
              'full_deployment_used_expert_subtask': False,
              'full_deployment_shape': list(deployed.shape)}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
