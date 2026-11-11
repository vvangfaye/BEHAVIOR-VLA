"""Verify the real optimizer/precision path before scheduling full VLA SFT."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from vla_runtime import load_base
from vla_support_adapter import SupportActionAdapter
from b1k_rlinf.io import DELTA_INDICES, extract_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    torch.set_num_threads(4)
    torch.manual_seed(17)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    start = time.monotonic()
    base = load_base(plan['checkpoint'], plan['tokenizer'], device='cuda', precision='bf16')
    adapter = SupportActionAdapter(base).cuda()
    with np.load(args.sample, allow_pickle=False) as z:
        assert not z['action_is_pad'].any()
        env = {'states': z['observation.state'][None].copy(),
               'main_images': z['observation.rgb.zed_link_camera_0'][None].copy(),
               'wrist_images': np.stack([z['observation.rgb.left_realsense_link_camera_0'],
                                         z['observation.rgb.right_realsense_link_camera_0']])[None].copy(),
               'task_descriptions': [str(z['prompt'].item())]}
        raw = z['action'].astype(np.float64)
        subtask = str(z['subtask'].item())
    images, normalized, prompts = base.checkpoint_io.prepare(env)
    tokens = base.joint_tokenizer.processor.encode(subtask, add_eos=True)
    obs = base._token_observation(images, normalized,
                                  [base.joint_tokenizer.actions(prompts[0], normalized[0], tokens)])
    active = torch.tensor(np.ptp(raw, axis=0)[None] > 0, device='cuda')
    mask = np.zeros(23)
    mask[DELTA_INDICES] = 1
    state = extract_state(env['states'], 'checkpoint61')[0]
    stats = base.checkpoint_io.stats['actions']
    y = (raw - state * mask - stats['mean']) / (stats['std'] + 1e-6)
    target = torch.tensor(np.pad(y.astype(np.float32), ((0, 0), (0, 9)))[None], device='cuda')
    expert = [p for p in base.parameters() if p.requires_grad]
    structure = [*adapter.mask_head.parameters(), *adapter.mask_condition.parameters()]
    assert all(p.dtype == torch.float32 for p in expert + structure)
    optimizer = torch.optim.AdamW([{'params': expert, 'lr': 1e-5},
                                   {'params': structure, 'lr': 1e-3}], weight_decay=.01)
    attention = base.llm.layers[0].attn.q_proj[1].weight
    before = attention.detach().clone()
    records = []
    for step in range(2):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for micro in range(8):
            noise = torch.randn_like(target)
            t = torch.distributions.Beta(1.5, 1.).sample((1,)).cuda() * .999 + .001
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = adapter.loss(obs, target, noise=noise, time=t, raw_active=active)
                loss = result['loss'] / 8
            assert torch.isfinite(loss)
            loss.backward()
            loss_sum += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(expert + structure, 1.)
        assert torch.isfinite(norm) and attention.grad is not None and attention.grad.norm() > 0
        assert all(p.grad is None for p in base.parameters() if not p.requires_grad)
        optimizer.step()
        torch.cuda.synchronize()
        records.append({'step': step + 1, 'loss': loss_sum, 'gradient_norm': float(norm),
                        'seconds': time.monotonic() - tick,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        print(json.dumps(records[-1]), flush=True)
    delta = float((attention - before).abs().max())
    assert delta > 0
    assert all(v.dtype == torch.float32 for state in optimizer.state.values()
               for k, v in state.items() if k in ('exp_avg', 'exp_avg_sq'))
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        action, details = adapter.predict_actions(env, rng=torch.Generator(device='cuda').manual_seed(17))
    assert action.shape == (1, 16, 23) and torch.isfinite(action).all()
    n = int(details['subtask_lengths'][0])
    result = {'passed': True, 'gpu': torch.cuda.get_device_name(0),
              'optimizer': 'AdamW with fp32 trainable weights and fp32 moments; bf16 autocast',
              'completed_updates': 2, 'global_batch': 8, 'micro_batch': 1,
              'action_expert_attention_max_update': delta, 'updates': records,
              'predicted_subtask': base.joint_tokenizer.processor.decode(details['subtask_tokens'][0, :n].tolist()),
              'output_shape': list(action.shape), 'elapsed_seconds': time.monotonic() - start,
              'scope': 'GPU training path and memory verification only; repeated single existing observation; no full-training or task-success claim'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
