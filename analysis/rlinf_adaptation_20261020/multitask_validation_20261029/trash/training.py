"""Reusable on-policy update and checkpoint checks for the bounded PPO pilot."""
import hashlib
import math
import copy
from pathlib import Path

import torch


def actor_parameters(model):
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith('value_head.')]


def load_overlay(model, directory, expected):
    overlay = torch.load(Path(directory)/'trainable.pt', map_location='cpu', weights_only=True)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(overlay) == names
    assert hash_parameters(overlay.items()) == expected
    result = model.load_state_dict(overlay, strict=False)
    assert not result.unexpected_keys and set(result.missing_keys) == set(model.state_dict())-names
    assert hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad) == expected


def prefix_features(model, replay):
    with torch.no_grad():
        observation = model._observation_from_forward_inputs(replay)
        hidden, mask, _ = model._build_prefix_cache_for_actor(observation)
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return ((hidden*weights).sum(1)/weights.sum(1).clamp(min=1)).detach()


def discounted_returns(rows, gamma, bootstrap=0.):
    result = []
    value = float(bootstrap)
    for row in reversed(rows):
        value = float(row['reward']) + gamma*value*(not row['terminal'])
        result.append(value)
    return list(reversed(result))


def fit_critic_features(model, features, targets, config):
    """Warm up only the head; held-out chunks are reporting-only, never selection."""
    actor_hash = hash_parameters(actor_parameters(model))
    device = next(model.parameters()).device
    x = features.to(device)
    y = targets.to(device).float().reshape(-1)
    valid = torch.arange(len(y), device=device)%config['validation_stride'] == 0
    assert valid.any() and (~valid).any()
    train = (~valid).nonzero().flatten()
    optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=config['learning_rate'], weight_decay=.01)
    def mse(mask):
        with torch.no_grad():
            return float((model.value_head(x[mask]).flatten()-y[mask]).square().mean())
    before = {'train_mse':mse(~valid), 'validation_mse':mse(valid)}
    generator = torch.Generator().manual_seed(config['seed'])
    history=[]
    for epoch in range(config['epochs']):
        order = train[torch.randperm(len(train), generator=generator).to(device)]
        for indices in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=(model.value_head(x[indices]).flatten()-y[indices]).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.value_head.parameters(),1.)
            optimizer.step()
        if epoch in (0, config['epochs']-1):
            history.append({'epoch':epoch+1,'train_mse':mse(~valid),'validation_mse':mse(valid)})
    after={'train_mse':mse(~valid),'validation_mse':mse(valid)}
    assert actor_hash == hash_parameters(actor_parameters(model)), 'Critic warmup changed actor'
    assert all(math.isfinite(v) for v in after.values())
    assert after['train_mse'] < before['train_mse'], 'Critic did not learn its warmup target'
    optimizer.zero_grad(set_to_none=True)
    return {'before':before,'after':after,'samples':len(y),'train_samples':int((~valid).sum()),
            'validation_samples':int(valid.sum()),'history':history,'actor_unchanged':True,
            'target_min':float(y.min()),'target_max':float(y.max()),
            'limitation':'Same-trajectory held-out chunks; not an independent calibration benchmark.'}


def batch_advantages(episodes, config):
    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
    all_rows=[]; raw=[]; targets=[]
    for episode in episodes:
        rows=episode['transitions']; bootstrap=episode['bootstrap']
        values=torch.stack([row['prev_values'].reshape(1) for row in rows]+[bootstrap.reshape(1)])
        rewards=torch.tensor([[row['reward']] for row in rows],dtype=torch.float32)
        dones=torch.tensor([[False]]+[[row['terminal']] for row in rows],dtype=torch.bool)
        adv,ret=compute_gae_advantages_and_returns(rewards=rewards, values=values,dones=dones,
            gamma=config['gamma'],gae_lambda=config['gae_lambda'],normalize_advantages=False)
        all_rows.extend(rows);raw.append(adv);targets.append(ret)
    raw=torch.cat(raw).flatten(); returns=torch.cat(targets).flatten()
    advantages=(raw-raw.mean())/(raw.std()+1e-5)
    assert len(all_rows)==len(advantages) and torch.isfinite(advantages).all()
    return all_rows,advantages,returns,raw


def cpu_clone(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_clone(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_clone(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_clone(v) for v in value)
    return copy.deepcopy(value)


@torch.no_grad()
def batch_kl(model, rows):
    device=next(model.parameters()).device
    total=0.; max_abs=0.
    for row in rows:
        replay={k:v.to(device) for k,v in row['forward_inputs'].items()}
        logp=model.default_forward(replay,compute_values=False)['logprobs'].float()
        old=row['prev_logprobs'].to(device).float()
        log_ratio=(logp-old).reshape(logp.shape[0],-1).sum(-1)
        # Sample estimator is nonnegative, unlike an individual old-new logp.
        estimate=torch.expm1(log_ratio)-log_ratio
        if not torch.isfinite(estimate).all():return {'mean':float('inf'),'max_abs_log_ratio':float('inf')}
        total+=float(estimate.mean());max_abs=max(max_abs,float(log_ratio.abs().max()))
    return {'mean':total/len(rows),'max_abs_log_ratio':max_abs}


def partition_updates(order, budget):
    """Use every sampled chunk exactly once with a fixed optimizer-step budget."""
    if budget <= 0 or len(order) < budget:
        raise ValueError('Not enough sampled chunks for the frozen update budget')
    if len(set(order)) != len(order):
        raise ValueError('Sampling order contains duplicate chunks')
    size, extra = divmod(len(order), budget)
    groups = []; start = 0
    for index in range(budget):
        end = start + size + int(index < extra)
        groups.append(order[start:end]); start = end
    assert start == len(order) and [x for group in groups for x in group] == order
    return groups


def update_policy(model, optimizer, episodes, plan, round_index=1, publish=None):
    from rlinf.algorithms.losses import compute_ppo_actor_critic_loss
    from rlinf.algorithms.utils import preprocess_loss_inputs
    rows,advantages,returns,raw=batch_advantages(episodes,plan['optimizer'])
    device=next(model.parameters()).device; config=plan['optimizer']
    replay_error=0.
    with torch.no_grad():
        for row in rows:
            replay={k:v.to(device) for k,v in row['forward_inputs'].items()}
            actual=model.default_forward(replay)['logprobs']
            expected=row['prev_logprobs'].to(device)
            torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
            replay_error=max(replay_error,float((actual-expected).abs().max()))
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    generator=torch.Generator().manual_seed(config['shuffle_seed']+round_index)
    order=torch.randperm(len(rows),generator=generator).tolist()
    metrics=[];steps=[];accepted=0;expert_grad=False;value_grad=False;stopped=False
    budget = config.get('updates_per_round')
    if budget is not None:
        assert 1 <= round_index <= len(budget)
        groups = partition_updates(order, int(budget[round_index-1]))
    else:
        groups = [order[start:start+config['accumulate_chunks']]
                  for start in range(0,len(rows),config['accumulate_chunks'])]
    for indices in groups:
        prior={n:p.detach().cpu().clone() for n,p in named}
        prior_optimizer=cpu_clone(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        group=[]
        for index in indices:
            row=rows[index];replay={k:v.to(device) for k,v in row['forward_inputs'].items()}
            output=model.default_forward(replay)
            fields=preprocess_loss_inputs(logprobs=output['logprobs'].float(),
                old_logprobs=row['prev_logprobs'].to(device).float(),advantages=advantages[index].to(device),
                values=output['values'].float(),prev_values=row['prev_values'].to(device).float(),
                returns=returns[index].to(device),logprob_type='chunk_level',reward_type='chunk_level',single_action_dim=23)
            loss,details=compute_ppo_actor_critic_loss(**fields,clip_ratio_low=config['clip_ratio'],
                clip_ratio_high=config['clip_ratio'],value_clip=.2,huber_delta=10.)
            assert torch.isfinite(loss)
            (loss/len(indices)).backward()
            group.append({'chunk':index,'loss':float(loss.detach()),**{
                k:float(v.detach().float().mean()) for k,v in details.items() if torch.is_tensor(v)}})
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_([p for n,p in named],1.)
        assert torch.isfinite(norm)
        expert_grad|=any(p.grad is not None and bool(p.grad.abs().max()>0) for n,p in named if not n.startswith('value_head.'))
        value_grad|=any(p.grad is not None and bool(p.grad.abs().max()>0) for n,p in named if n.startswith('value_head.'))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        kl=batch_kl(model,rows)
        rejected=not math.isfinite(kl['mean']) or kl['mean']>config['kl_target']
        if rejected:
            with torch.no_grad():
                for name,parameter in named:parameter.copy_(prior[name])
            optimizer.load_state_dict(prior_optimizer);optimizer_to(optimizer,device)
            stopped=True
        else:accepted+=1
        step={'candidate':len(steps)+1,'accepted':not rejected,
              'candidate_kl':kl['mean'] if math.isfinite(kl['mean']) else 'nonfinite',
              'gradient_norm_before_clip':float(norm),'accepted_optimizer_steps':accepted}
        steps.append(step);metrics.extend(group)
        del prior,prior_optimizer
        if publish:publish(stage='training_update',round=round_index,**step)
        if rejected:break
    final_kl=batch_kl(model,rows)
    assert final_kl['mean']<=config['kl_target']+1e-6
    return {'passed':True,'episodes':len(episodes),'chunks':len(rows),'optimizer_steps':accepted,
        'attempted_optimizer_steps':len(steps),'early_stopped':stopped,'final_batch_kl':final_kl,
        'pre_update_replay_error':replay_error,'expert_gradient_nonzero':expert_grad,
        'value_gradient_nonzero':value_grad,'reward_sum':sum(row['reward'] for row in rows),
        'advantage_raw_mean':float(raw.mean()),'advantage_raw_std':float(raw.std()),
        'order':order,'update_group_sizes':[len(group) for group in groups],
        'requested_optimizer_steps':len(groups),'update_steps':steps,'metrics':metrics},advantages,returns


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


def warm_modem_critic(model, episodes, plan, publish):
    """Fit the fresh critic on this task; then refresh GAE baselines, not logp."""
    config = plan['critic_warmup']
    features, targets = [], []
    device = next(model.parameters()).device
    actor_before = hash_parameters(actor_parameters(model))
    for i, episode in enumerate(episodes):
        rows = episode['transitions']
        assert rows[-1]['terminal'] and float(episode['bootstrap']) == 0.0
        values = discounted_returns(rows, plan['optimizer']['gamma'])
        count = min(len(rows), config['max_samples_per_trajectory'])
        indices = torch.linspace(0, len(rows)-1, count).round().long().unique().tolist()
        for j in indices:
            replay = {k:v.to(device) for k,v in rows[j]['forward_inputs'].items()}
            features.append(prefix_features(model, replay).cpu())
            targets.append(values[j])
        publish(stage='critic_feature_extraction', completed_training_episodes=i+1,
                feature_samples=len(targets))
    report = fit_critic_features(model, torch.cat(features), torch.tensor(targets), config)
    assert actor_before == hash_parameters(actor_parameters(model))
    # The critic does not affect action generation. Logprobs/chains stay frozen
    # at collection; only the value baseline changes after critic-only fitting.
    with torch.no_grad():
        for i, episode in enumerate(episodes):
            for row in episode['transitions']:
                replay = {k:v.to(device) for k,v in row['forward_inputs'].items()}
                row['prev_values'] = model.value_head(prefix_features(model,replay)).reshape(1).cpu()
            publish(stage='critic_baseline_refresh', completed_training_episodes=i+1)
    return {**report, 'actor_unchanged':True, 'training_only':True,
            'source_episodes':len(episodes), 'success_and_timeout_used':True,
            'new_values_used_before_gae':True, 'collection_logprobs_unchanged':True}
