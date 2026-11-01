"""Three explicit alternatives on the same frozen on-policy rollout batch."""
from collections import defaultdict
import hashlib
import json
import math
import time
import torch
import training as base
from goals import shaping_bonus


def prepare_rewards(episodes,plan):
    variant=plan['comparison_variant']
    if variant=='gamma1_ppo':
        assert plan['optimizer']['gamma']==plan['shaping']['gamma']==1.0
        for episode in episodes:
            for row in episode['transitions']:
                row['reward']=float(row['task_reward'])+shaping_bonus(
                    row['goals_before'],row['goals_after'],row['terminal'],plan['shaping'])
    else:
        assert variant in ('balanced_ppo','group_outcome')
    return episodes


def episode_weights(episodes):
    lengths=[len(x['transitions']) for x in episodes]
    if not lengths or min(lengths)<=0:raise ValueError('Empty episode')
    n=sum(lengths);count=len(lengths)
    return torch.tensor([n/(count*length) for length in lengths for _ in range(length)],dtype=torch.float32)


def group_advantages(episodes):
    groups=defaultdict(list)
    for index,episode in enumerate(episodes):groups[int(episode['instance'])].append(index)
    values=torch.zeros(len(episodes),dtype=torch.float32);mixed=0
    for group in groups.values():
        if len(group)!=2:raise ValueError('This fixed batch requires two rollouts per instance')
        outcomes=torch.tensor([float(episodes[i]['status']['success']) for i in group])
        std=outcomes.std(unbiased=False)
        if std>0:
            mixed+=1
            values[group]=(outcomes-outcomes.mean())/std
    if mixed==0:raise ValueError('No within-instance outcome contrast')
    rows=[r for e in episodes for r in e['transitions']]
    adv=torch.cat([values[i].expand(len(e['transitions'])) for i,e in enumerate(episodes)])
    return rows,adv,{'groups':len(groups),'mixed_outcome_groups':mixed,'episode_advantages':values.tolist()}


def update_policy(model,optimizer,episodes,plan,round_index=1,publish=None):
    from rlinf.algorithms.losses import compute_ppo_actor_critic_loss,compute_grpo_actor_loss_fn
    from rlinf.algorithms.utils import preprocess_loss_inputs
    variant=plan['comparison_variant'];config=plan['optimizer'];started=time.time()
    if variant=='group_outcome':
        rows,advantages,group_stats=group_advantages(episodes)
        returns=raw=None
    else:
        rows,advantages,returns,raw=base.batch_advantages(episodes,config);group_stats=None
    weights=episode_weights(episodes) if variant in ('balanced_ppo','group_outcome') else torch.ones(len(rows))
    device=next(model.parameters()).device;replay_error=0.
    with torch.no_grad():
        for i,row in enumerate(rows):
            replay={k:v.to(device) for k,v in row['forward_inputs'].items()}
            actual=model.default_forward(replay,compute_values=False)['logprobs']
            expected=row['prev_logprobs'].to(device)
            torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
            replay_error=max(replay_error,float((actual-expected).abs().max()))
            if publish and ((i+1)%256==0 or i+1==len(rows)):
                publish(stage='replay_validation',checked_chunks=i+1,total_chunks=len(rows))
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    generator=torch.Generator().manual_seed(config['shuffle_seed']+round_index)
    order=torch.randperm(len(rows),generator=generator).tolist()
    groups=base.partition_updates(order,int(config['updates_per_round'][round_index-1]))
    steps=[];accepted=0;stopped=False;actor_gradient=False;critic_gradient=False

    def measured_kl(candidate):
        total=0.;largest=0.
        with torch.no_grad():
            for i,row in enumerate(rows):
                replay={k:v.to(device) for k,v in row['forward_inputs'].items()}
                now=model.default_forward(replay,compute_values=False)['logprobs'].float()
                old=row['prev_logprobs'].to(device).float()
                ratio=(now-old).reshape(now.shape[0],-1).sum(-1)
                estimate=torch.expm1(ratio)-ratio
                if not torch.isfinite(estimate).all():return {'mean':float('inf'),'max_abs_log_ratio':float('inf')}
                total+=float(estimate.mean());largest=max(largest,float(ratio.abs().max()))
                if publish and ((i+1)%512==0 or i+1==len(rows)):
                    publish(stage='kl_validation',candidate=candidate,checked_chunks=i+1,
                            total_chunks=len(rows),accepted_optimizer_steps=accepted)
        return {'mean':total/len(rows),'max_abs_log_ratio':largest}

    for candidate,indices in enumerate(groups,1):
        prior={n:p.detach().cpu().clone() for n,p in named};old_optimizer=base.cpu_clone(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for j,index in enumerate(indices,1):
            row=rows[index];replay={k:v.to(device) for k,v in row['forward_inputs'].items()}
            output=model.default_forward(replay,compute_values=variant!='group_outcome')
            args={'logprobs':output['logprobs'].float(),'old_logprobs':row['prev_logprobs'].to(device).float(),
                'advantages':advantages[index].to(device),'logprob_type':'chunk_level',
                'reward_type':'chunk_level','single_action_dim':23}
            if variant!='group_outcome':
                args.update(values=output['values'].float(),prev_values=row['prev_values'].to(device).float(),
                            returns=returns[index].to(device))
            fields=preprocess_loss_inputs(**args)
            loss,_=(compute_grpo_actor_loss_fn(**fields,clip_ratio_low=config['clip_ratio'],clip_ratio_high=config['clip_ratio'])
                if variant=='group_outcome' else compute_ppo_actor_critic_loss(**fields,
                    clip_ratio_low=config['clip_ratio'],clip_ratio_high=config['clip_ratio'],value_clip=.2,huber_delta=10.))
            loss=loss*weights[index].to(device)
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            (loss/len(indices)).backward();loss_sum+=float(loss.detach())
            if publish and (j%128==0 or j==len(indices)):
                publish(stage='gradient_accumulation',candidate=candidate,completed_chunks=j,
                    candidate_chunks=len(indices),accepted_optimizer_steps=accepted)
        assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in named],1.)
        assert torch.isfinite(norm)
        actor_gradient|=any(p.grad is not None and bool(p.grad.abs().max()>0) for n,p in named if not n.startswith('value_head.'))
        critic_gradient|=any(p.grad is not None and bool(p.grad.abs().max()>0) for n,p in named if n.startswith('value_head.'))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        kl=measured_kl(candidate);rejected=not math.isfinite(kl['mean']) or kl['mean']>config['kl_target']
        if rejected:
            with torch.no_grad():
                for n,p in named:p.copy_(prior[n])
            optimizer.load_state_dict(old_optimizer);base.optimizer_to(optimizer,device);stopped=True
        else:accepted+=1
        step={'candidate':candidate,'accepted':not rejected,'candidate_kl':kl['mean'] if math.isfinite(kl['mean']) else 'nonfinite',
            'accepted_optimizer_steps':accepted,'mean_weighted_loss':loss_sum/len(indices),
            'gradient_norm_before_clip':float(norm),'elapsed_seconds':time.time()-started}
        steps.append(step);del prior,old_optimizer
        if publish:publish(stage='training_update',**step)
        if rejected:break
    final_kl=measured_kl(len(steps));assert final_kl['mean']<=config['kl_target']+1e-6
    if variant=='group_outcome':assert not critic_gradient
    return {'passed':True,'variant':variant,'episodes':len(episodes),'chunks':len(rows),
        'requested_optimizer_steps':len(groups),'optimizer_steps':accepted,'attempted_optimizer_steps':len(steps),
        'early_stopped':stopped,'update_steps':steps,'final_batch_kl':final_kl,
        'pre_update_replay_error':replay_error,'actor_gradient_nonzero':actor_gradient,
        'critic_gradient_nonzero':critic_gradient,'group_statistics':group_stats,
        'trajectory_weighting':variant in ('balanced_ppo','group_outcome'),'elapsed_seconds':time.time()-started}
