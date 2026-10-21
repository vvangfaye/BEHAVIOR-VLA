"""Compare cached-prefix standard SFT with original joint-forward SFT."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from vla_runtime import ROOT, load_base
from vla_data import digest
from vla_support_adapter import SupportActionAdapter
from b1k_rlinf.io import DELTA_INDICES, extract_state


def main():
    torch.set_num_threads(12)
    torch.manual_seed(17)
    started = time.monotonic()
    here = Path(__file__).resolve().parent
    cache = ROOT / 'analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1'
    record = json.loads((cache / 'cache_manifest.json').read_text())['records'][0]
    sample = cache / record['npz_path']
    assert digest(sample) == record['npz_sha256']
    base = load_base(ROOT / 'analysis/rlinf_adaptation_20261020/checkpoint1688977_openpi',
                     ROOT / 'analysis/retraining_20261015_comet_rft60_40/launch_bundle/inputs/paligemma_tokenizer.model',
                     device='cpu', precision='fp32')
    with np.load(sample) as z:
        env = {'states': z['observation.state'][None].copy(),
               'main_images': z['observation.rgb.zed_link_camera_0'][None].copy(),
               'wrist_images': np.stack([z['observation.rgb.left_realsense_link_camera_0'],
                                         z['observation.rgb.right_realsense_link_camera_0']])[None].copy(),
               'task_descriptions': [str(z['prompt'].item())]}
        raw = z['action'].astype(np.float64)
        subtask = str(z['subtask'].item())
    images, state, prompts = base.checkpoint_io.prepare(env)
    label = base.joint_tokenizer.processor.encode(subtask, add_eos=True)
    obs = base._token_observation(images, state, [base.joint_tokenizer.actions(prompts[0], state[0], label)])
    physical = extract_state(env['states'], 'checkpoint61')[0]
    delta = np.zeros(23)
    delta[DELTA_INDICES] = 1
    stats = base.checkpoint_io.stats['actions']
    y = (raw - physical * delta - stats['mean']) / (stats['std'] + 1e-6)
    actions = torch.from_numpy(np.pad(y.astype(np.float32), ((0, 0), (0, 9)))[None])
    active = torch.from_numpy((np.ptp(raw, axis=0) > 0)[None])
    noise = torch.randn(actions.shape, generator=torch.Generator().manual_seed(762))
    timestep = torch.tensor([.3])
    selected = {'output': base.action_out_proj.weight,
                'expert_attention': base.llm.layers[0].attn.q_proj[1].weight}
    print('Original joint-forward loss and gradients', flush=True)
    # JointPi0RL exposes a batched autoregressive prefix mask while the
    # inherited original SFT function accepts a 1-D mask. For this B=1
    # comparison, removing the singleton batch axis is exactly equivalent;
    # keep the original dense forward/loss implementation otherwise intact.
    original_embed_prefix = base.embed_prefix
    def single_row_prefix(observation):
        tokens, valid, ar = original_embed_prefix(observation)
        assert ar.ndim == 2 and ar.shape[0] == 1
        return tokens, valid, ar[0]
    base.embed_prefix = single_row_prefix
    try:
        expected = base.compute_loss(obs, actions, train=False, noise=noise, time=timestep).mean()
    finally:
        base.embed_prefix = original_embed_prefix
    expected.backward()
    reference = {name: p.grad.detach().clone() for name, p in selected.items()}
    base.zero_grad(set_to_none=True)
    print('Cached-prefix standard baseline loss and gradients', flush=True)
    adapter = SupportActionAdapter(base).eval()
    actual = adapter.loss(obs, actions, noise=noise, time=timestep, mode='standard', raw_active=active)['loss']
    actual.backward()
    torch.testing.assert_close(actual, expected.detach(), atol=5e-6, rtol=5e-5)
    gradients = {}
    for name, parameter in selected.items():
        assert parameter.grad is not None and parameter.grad.norm() > 0
        torch.testing.assert_close(parameter.grad, reference[name], atol=5e-6, rtol=5e-4)
        gradients[name] = {'max_absolute_difference': float((parameter.grad - reference[name]).abs().max()),
                           'native_norm': float(reference[name].norm()), 'cached_norm': float(parameter.grad.norm())}
    assert all(p.grad is None for p in adapter.mask_head.parameters())
    assert adapter.mask_condition.weight.grad is None
    assert all(p.grad is None for p in base.parameters() if not p.requires_grad)
    result = {'passed': True, 'native_loss': float(expected.detach()), 'cached_standard_loss': float(actual.detach()),
              'absolute_loss_difference': float((actual - expected.detach()).abs().detach()),
              'gradient_checks': gradients, 'auxiliary_heads_have_no_standard_loss_gradients': True,
              'sample_sha256': record['npz_sha256'], 'elapsed_seconds': time.monotonic() - started,
              'reference_mask_adaptation': 'Remove singleton batch axis from joint prefix AR mask for inherited SFT function; same mask entries',
              'scope': 'One real checkpoint and one real expert observation, FP32; no optimizer update, B>1 or GPU equivalence claim'}
    (here / 'sft_equivalence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
