"""Reusable on-policy update and checkpoint checks for the bounded PPO pilot."""
import hashlib
import math
from pathlib import Path

import torch


def hash_parameters(named):
    digest = hashlib.sha256()
    for name, parameter in named:
        digest.update(name.encode())
        raw = parameter.detach().contiguous().view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), 8 * 1024**2):
            digest.update(raw[start:start + 8 * 1024**2].cpu().numpy().tobytes())
    return digest.hexdigest()


def optimizer_to(optimizer, device):
    # Non-capturable AdamW keeps scalar step counters on the CPU.
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to('cpu' if key == 'step' else device)


def build_optimizer(model, plan):
    config = plan['optimizer']
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in named if not n.startswith('value_head.')], 'lr': config['actor_lr']},
        {'params': [p for n, p in named if n.startswith('value_head.')], 'lr': config['value_lr']},
    ], weight_decay=config['weight_decay'], eps=1e-8, betas=(0.9, 0.999))
    return optimizer


def load_warm_start(model, optimizer, plan):
    directory = Path(plan['warm_start']['directory'])
    overlay = torch.load(directory/'trainable.pt', map_location='cpu', weights_only=True)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(overlay) == names, 'Warm-start trainable parameter inventory changed'
    outcome = model.load_state_dict(overlay, strict=False)
    assert not outcome.unexpected_keys
    assert set(outcome.missing_keys) == set(model.state_dict()) - names
    del overlay
    actual = hash_parameters((n, p) for n, p in model.named_parameters() if p.requires_grad)
    assert actual == plan['warm_start']['expected_trainable_sha256']
    state = torch.load(directory/'optimizer.pt', map_location='cpu', weights_only=True)
    optimizer.load_state_dict(state)
    del state
    assert [group['lr'] for group in optimizer.param_groups] == [plan['optimizer']['actor_lr'], plan['optimizer']['value_lr']]
    assert all(group['weight_decay'] == plan['optimizer']['weight_decay'] for group in optimizer.param_groups)
    optimizer_to(optimizer, 'cpu')
    return actual


def update_policy(model, optimizer, transitions, bootstrap, plan):
    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
    from rlinf.algorithms.losses import compute_ppo_actor_critic_loss
    from rlinf.algorithms.utils import preprocess_loss_inputs

    assert len(transitions) >= 2
    device = next(model.parameters()).device
    config = plan['optimizer']
    values = torch.stack([row['prev_values'].reshape(1) for row in transitions] + [bootstrap])
    rewards = torch.tensor([[row['reward']] for row in transitions], dtype=torch.float32)
    dones = torch.tensor([[False]] + [[row['terminal']] for row in transitions], dtype=torch.bool)
    advantages, returns = compute_gae_advantages_and_returns(
        rewards=rewards, values=values, dones=dones, gamma=config['gamma'],
        gae_lambda=config['gae_lambda'], normalize_advantages=True,
    )
    replay_error = 0.0
    with torch.no_grad():
        for old in transitions:
            replay = {k: v.to(device) for k, v in old['forward_inputs'].items()}
            actual = model.default_forward(replay)['logprobs']
            expected = old['prev_logprobs'].to(device)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            replay_error = max(replay_error, float((actual-expected).abs().max()))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    metrics = []
    expert_gradient = value_gradient = False
    updates = 0
    accumulation = config['accumulate_chunks']
    for start in range(0, len(transitions), accumulation):
        indices = list(range(start, min(start + accumulation, len(transitions))))
        optimizer.zero_grad(set_to_none=True)
        for index in indices:
            old = transitions[index]
            replay = {k: v.to(device) for k, v in old['forward_inputs'].items()}
            output = model.default_forward(replay)
            fields = preprocess_loss_inputs(
                logprobs=output['logprobs'].float(), old_logprobs=old['prev_logprobs'].to(device).float(),
                advantages=advantages[index].to(device), values=output['values'].float(),
                prev_values=old['prev_values'].to(device).float(), returns=returns[index].to(device),
                logprob_type='chunk_level', reward_type='chunk_level', single_action_dim=23,
            )
            loss, details = compute_ppo_actor_critic_loss(
                **fields, clip_ratio_low=config['clip_ratio'], clip_ratio_high=config['clip_ratio'],
                value_clip=0.2, huber_delta=10.0,
            )
            assert torch.isfinite(loss), 'Non-finite PPO loss'
            (loss/len(indices)).backward()
            metrics.append({'chunk': index, 'loss': float(loss.detach()), **{
                k: float(v.detach().float().mean()) for k, v in details.items() if torch.is_tensor(v)
            }})
        assert all(p.grad is None for _, p in frozen)
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        assert torch.isfinite(norm), 'Non-finite PPO gradient'
        expert_gradient |= any(p.grad is not None and bool(p.grad.abs().max() > 0) for n, p in trainable if not n.startswith('value_head.'))
        value_gradient |= any(p.grad is not None and bool(p.grad.abs().max() > 0) for n, p in trainable if n.startswith('value_head.'))
        optimizer.step()
        updates += 1
    optimizer.zero_grad(set_to_none=True)
    assert expert_gradient and value_gradient
    return {'passed': True, 'chunks': len(transitions), 'optimizer_steps': updates,
            'pre_update_replay_error': replay_error, 'expert_gradient_nonzero': expert_gradient,
            'value_gradient_nonzero': value_gradient, 'reward_sum': float(rewards.sum()),
            'external_nonzero_reward_chunks': sum(row['reward'] != 0 for row in transitions),
            'metrics': metrics}, advantages, returns


def save_checkpoint(model, optimizer, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    overlay = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    expected = hash_parameters(overlay.items())
    torch.save(overlay, directory/'trainable.pt')
    del overlay
    recovered = torch.load(directory/'trainable.pt', map_location='cpu', weights_only=True)
    assert hash_parameters(recovered.items()) == expected
    del recovered
    torch.save(optimizer.state_dict(), directory/'optimizer.pt')
    return {'trainable_sha256': expected, 'round_trip': True,
            'model_bytes': (directory/'trainable.pt').stat().st_size,
            'optimizer_bytes': (directory/'optimizer.pt').stat().st_size}
