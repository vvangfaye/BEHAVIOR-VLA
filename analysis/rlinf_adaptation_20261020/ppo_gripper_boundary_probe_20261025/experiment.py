"""Recover only missing cases, then independent conservative PPO from original weights."""
import gc
import fcntl
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from common import Simulator, SimulatorInitializationError, env_observation, label_text, torch_model, write
from training import (actor_parameters, build_optimizer, hash_parameters, load_overlay,
                      optimizer_to, save_checkpoint, update_policy, warmup_critic)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def enforce_storage_budget(plan, reserve=0):
    root=Path(plan['output'])
    with (root/'storage_budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        used=0
        for path in root.rglob('*'):
            try:
                if path.is_file() and not path.is_symlink():used+=path.stat().st_blocks*512
            except FileNotFoundError:
                pass  # Atomic JSON publication can remove the temporary name.
        limit=plan['runtime_storage']['max_persistent_bytes']
        assert used+reserve<=limit, f'Persistent result budget exceeded: {used}+{reserve}>{limit}'
        write(root/'storage_usage.json',{'allocated_bytes':used,'reserved_next_write_bytes':reserve,'limit_bytes':limit})


def start_simulator(plan, output, instance, mode, max_steps, publish):
    output=Path(output)
    assert not output.exists(), 'Existing attempt must not be overwritten'
    for attempt in range(1, plan['protocol']['max_initialization_attempts_per_incomplete_case']+1):
        try:
            seed_all(plan['seed'])
            return Simulator(plan,output,instance,mode,max_steps)
        except SimulatorInitializationError as error:
            archive=output.with_name(output.name+f'_initialization_failure_{attempt:02d}')
            output.rename(archive)
            write(archive/'initialization_failure.json',{'attempt':attempt,'error':str(error),'no_controls_executed':True})
            publish(initialization_retry={'instance':instance,'attempt':attempt,'error':str(error)})
            if attempt == plan['protocol']['max_initialization_attempts_per_incomplete_case']:raise


def evaluate(model, plan, phase, publish, reuse_existing=False, write_summary=True):
    root=Path(plan['output'])/phase
    before=hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
    rows=[]
    for instance,seed in plan['evaluation']['cases']:
        output=root/f'i{instance}_s{seed}'
        if reuse_existing and (output/'result.json').is_file():
            result=json.loads((output/'result.json').read_text())
            assert (result['instance'],result['seed']) == (instance,seed)
            rows.append(result)
            continue
        assert not output.exists(), 'Incomplete existing controls cannot be silently retried'
        enforce_storage_budget(plan,256*1024**2)
        seed_all(seed)
        publish(stage=phase,instance=instance,seed=seed,completed_cases=len(rows))
        sim=start_simulator({**plan,'seed':seed},output,instance,'public_test',plan['eval_steps'],publish)
        rng=np.random.default_rng(seed);actions_log=[]
        try:
            while sim.current['status']['steps']<plan['eval_steps']:
                raw=sim.current['observation']
                noise=torch.from_numpy(rng.standard_normal((1,32,32)).astype(np.float32)).to(model.device)
                with torch.no_grad():
                    actions,result=model.predict_action_batch(env_observation(raw),mode='eval',noise=noise)
                text=label_text(model,result);action=actions[0].cpu().numpy()
                assert action.shape==(16,23) and np.isfinite(action).all()
                remaining=plan['eval_steps']-sim.current['status']['steps']
                actions_log.append({'subtask':text,'actions':action[:remaining].tolist()})
                sim.step(action[:remaining],text)
                write(output/'progress.json',sim.current['status'])
                if sim.current['done']:break
            row={'instance':instance,'seed':seed,**sim.current['status']}
            write(output/'result.json',row);write(output/'actions.json',actions_log);rows.append(row)
        finally:sim.close()
        publish(stage=phase,instance=instance,seed=seed,completed_cases=len(rows),last_result=rows[-1])
    after=hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
    assert after==before
    if write_summary:
        write(root/'summary.json',rows)
        write(root/'checks.json',{'weights_unchanged':True,'trainable_sha_before':before,'trainable_sha_after':after,
              'cases':plan['evaluation']['cases'],'evaluation_policy':plan['eval_policy']})
    return rows


def collect(model,plan,round_index,episode_index,publish):
    enforce_storage_budget(plan,256*1024**2)
    instance=plan['train_instances'][episode_index]
    seed=plan['train_seeds'][round_index-1][episode_index]
    seed_all(seed)
    output=Path(plan['output'])/'training'/f'round_{round_index:02d}'/f'episode_{episode_index+1:02d}'
    sim=start_simulator({**plan,'seed':seed},output/'rollout',instance,'train',plan['train_steps_per_round'],publish)
    transitions=[]
    try:
        while sim.current['status']['steps']<plan['train_steps_per_round']:
            with torch.no_grad():
                actions,result=model.predict_action_batch(env_observation(sim.current['observation']),mode='train')
            previous={k:{n:t.detach().cpu() for n,t in v.items()} if isinstance(v,dict) else v.detach().cpu()
                      for k,v in result.items() if k!='model_actions'}
            remaining=plan['train_steps_per_round']-sim.current['status']['steps']
            current=sim.step(actions[0].cpu().numpy()[:remaining],label_text(model,result))
            previous.update(reward=float(sum(current['rewards']))*10.,terminal=current['terminal'],done=current['done'])
            transitions.append(previous)
            write(output/'progress.json',{'stage':'rollout','round':round_index,'episode':episode_index+1,
                  'instance':instance,'chunks':len(transitions),**current['status']})
            if current['done']:break
        if transitions[-1]['terminal']:bootstrap=torch.zeros(1)
        else:
            with torch.no_grad():_,end=model.predict_action_batch(env_observation(sim.current['observation']),mode='train')
            bootstrap=end['prev_values'].reshape(1).cpu()
        final_status=sim.current['status']
    finally:sim.close()
    episode={'transitions':transitions,'bootstrap':bootstrap,'status':final_status,'instance':instance,'seed':seed}
    torch.save(episode,output/'rollout.pt')
    write(output/'result.json',{'instance':instance,'seed':seed,**final_status})
    return episode


def compare(plan, baseline, final):
    assert len(baseline) == len(final) == len(plan['evaluation']['cases'])
    pairs = []
    for left, right in zip(baseline, final):
        assert (left['instance'], left['seed']) == (right['instance'], right['seed'])
        case = f"i{left['instance']}_s{left['seed']}"
        a = json.loads((Path(plan['output'])/'baseline'/case/'initial.json').read_text())
        b = json.loads((Path(plan['output'])/'final'/case/'initial.json').read_text())
        assert a['physical_state'].keys() == b['physical_state'].keys()
        error = max(float(np.max(np.abs(np.asarray(a['physical_state'][k][part])-np.asarray(b['physical_state'][k][part]))))
                    for k in a['physical_state'] for part in (0,1))
        assert all(a[key] == b[key] for key in ('physics_dt','rendering_dt','observation_dt'))
        pairs.append({'instance': left['instance'], 'seed': left['seed'], 'baseline': left, 'final': right,
                      'initial_max_error': error, 'score_delta': right['score']-left['score'],
                      'success_delta': int(right['success'])-int(left['success'])})
    def aggregate(rows):
        return {'episodes': len(rows), 'mean_score': sum(r['score'] for r in rows)/len(rows),
                'successes': sum(r['success'] for r in rows), 'success_rate': sum(r['success'] for r in rows)/len(rows)}
    before, after = aggregate(baseline), aggregate(final)
    valid = all(p['initial_max_error'] <= plan['evaluation']['paired_initial_pose_tolerance'] for p in pairs)
    ds, dp = after['mean_score']-before['mean_score'], after['success_rate']-before['success_rate']
    if not valid: conclusion = 'Initial-state pairing check failed; do not attribute the difference to PPO.'
    elif ds > 0 and dp > 0: conclusion = 'Improvement observed on this fixed six-case pilot only; broader reliability remains untested.'
    elif ds == 0 and dp == 0: conclusion = 'No improvement observed on the fixed evaluation panel.'
    elif ds < 0 or dp < 0: conclusion = 'At least one aggregate metric regressed on the fixed evaluation panel.'
    else: conclusion = 'Only one aggregate metric improved; evidence is mixed.'
    return {'comparison_valid': valid, 'baseline': before, 'final': after, 'mean_score_delta': ds,
            'success_rate_delta': dp, 'success_to_failure': sum(p['success_delta'] < 0 for p in pairs),
            'failure_to_success': sum(p['success_delta'] > 0 for p in pairs), 'pairs': pairs,
            'conclusion': conclusion, 'scope': 'Two fixed instances, three paired seeds, one training run; no checkpoint selection.'}



def copy_evidence(source,target):
    target.mkdir(parents=True,exist_ok=False)
    records=[]
    for path in source.rglob('*'):
        if path.is_file() and path.suffix=='.json' and not any(x in path.parts for x in ['cache','appdata','config','warp']):
            relative=path.relative_to(source); dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest);records.append(str(relative))
    return records


def recover_episode(model, plan, round_index, episode_index):
    source=Path(plan['resume_run'])/'training'/f'round_{round_index:02d}'/f'episode_{episode_index+1:02d}'
    path=source/'rollout.pt'
    if not path.is_file() or not (source/'result.json').is_file():
        return None
    # No previous PPO update is allowed in this specific recovery. A completed
    # episode is reused only after checking its actual sampled policy likelihood.
    assert not list((Path(plan['resume_run'])/'training').glob('round_*/result.json'))
    episode=torch.load(path,map_location='cpu',weights_only=True)
    assert episode['instance']==plan['train_instances'][episode_index]
    assert episode['seed']==plan['train_seeds'][round_index-1][episode_index]
    device=next(model.parameters()).device
    value_error=0.
    with torch.no_grad():
        for row in episode['transitions']:
            replay={k:v.to(device) for k,v in row['forward_inputs'].items()}
            actual=model.default_forward(replay)
            torch.testing.assert_close(actual['logprobs'],row['prev_logprobs'].to(device),atol=1e-4,rtol=1e-4)
            value_error=max(value_error,float((actual['values'].reshape(-1)-row['prev_values'].to(device).reshape(-1)).abs().max()))
    # The warmup is deterministically repeated from the original head. Reject
    # stale critic values rather than mix two baselines inside the batch.
    assert value_error<=1e-4, f'Recovered critic differs: {value_error}'
    target=Path(plan['output'])/'training'/f'round_{round_index:02d}'/f'episode_{episode_index+1:02d}'
    copy_evidence(source,target)
    shutil.copyfile(path,target/'rollout.pt')
    write(target/'recovery.json',{'source':str(source),'reused_complete_episode':True,
        'logprob_replay_verified':True,'max_value_error':value_error})
    return episode


def run(plan,publish):
    root=Path(plan['output']);old=Path(plan['previous_run'])
    assert json.loads((old/'latest_checkpoint.json').read_text())['round']==5
    assert json.loads((old/'baseline/checks.json').read_text())['weights_unchanged']
    baseline=json.loads((old/'baseline/summary.json').read_text())
    assert [(x['instance'],x['seed']) for x in baseline] == [tuple(x) for x in plan['evaluation']['cases']]
    copy_evidence(old/'baseline',root/'baseline')
    copied=[]
    for instance,seed in plan['recovery']['preserve_completed']:
        case=f'i{instance}_s{seed}'
        copied.extend(copy_evidence(Path(plan['resume_run'])/'previous_final'/case,root/'previous_final'/case))
    seed_all(plan['model_seed']);model=torch_model(plan)
    original_hash=hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
    assert original_hash==plan['expected_original_trainable_sha256']
    frozen_hash=hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    load_overlay(model,old/'checkpoints/round_05',plan['recovery']['trainable_sha256'])
    publish(stage='previous_final',completed_cases=3)
    previous_final=evaluate(model,plan,'previous_final',publish,reuse_existing=True)
    compare_plan={**plan,'output':str(root/'previous_comparison')}
    # compare() uses fixed baseline/final paths; copy only initial JSON for its check.
    for phase,source in [('baseline',root/'baseline'),('final',root/'previous_final')]:
        for instance,seed in plan['evaluation']['cases']:
            case=f'i{instance}_s{seed}';dest=Path(compare_plan['output'])/phase/case
            dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/case/'initial.json',dest/'initial.json')
    previous_report=compare(compare_plan,baseline,previous_final)
    write(root/'previous_comparison.json',previous_report)
    assert previous_report['comparison_valid']
    publish(stage='previous_comparison_complete',previous_comparison=previous_report)
    del model;gc.collect();torch.cuda.empty_cache()
    seed_all(plan['model_seed']);model=torch_model(plan)
    assert hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)==original_hash
    actor_before=hash_parameters(actor_parameters(model))
    publish(stage='critic_warmup')
    warmup=warmup_critic(model,plan,publish)
    write(root/'critic_warmup.json',warmup)
    assert hash_parameters(actor_parameters(model))==actor_before
    optimizer=build_optimizer(model,plan);rounds=[]
    for index in range(1,plan['rounds']+1):
        optimizer_to(optimizer,'cpu');gc.collect();torch.cuda.empty_cache();episodes=[]
        before=hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
        for episode_index in range(plan['protocol']['episodes_per_round']):
            publish(stage='training_rollout',round=index,episode=episode_index+1,instance=plan['train_instances'][episode_index])
            recovered=recover_episode(model,plan,index,episode_index)
            if recovered is not None:
                episodes.append(recovered)
                publish(stage='training_rollout',round=index,episode=episode_index+1,reused_complete_episode=True)
            else:
                episodes.append(collect(model,plan,index,episode_index,publish))
        assert hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)==before
        publish(stage='training_update',round=index,chunks=sum(len(x['transitions']) for x in episodes))
        optimizer_to(optimizer,model.device)
        update,advantages,returns=update_policy(model,optimizer,episodes,plan,index,publish)
        optimizer_to(optimizer,'cpu')
        checkpoint=root/'checkpoints'/f'round_{index:02d}'
        enforce_storage_budget(plan,plan['runtime_storage']['max_checkpoint_write_reservation_bytes'])
        saved=save_checkpoint(model,optimizer,checkpoint)
        report={'round':index,'rollouts':[{'instance':x['instance'],'seed':x['seed'],**x['status']} for x in episodes],
                'update':update,'checkpoint':saved,'collection_policy_unchanged':True}
        write(root/'training'/f'round_{index:02d}'/'result.json',report)
        write(checkpoint/'ready.json',{'round':index,**saved});write(root/'latest_checkpoint.json',{'round':index,'directory':str(checkpoint),**saved})
        if index>1:(root/'checkpoints'/f'round_{index-1:02d}'/'optimizer.pt').unlink()
        rounds.append(report);del episodes,advantages,returns;gc.collect();torch.cuda.empty_cache()
        publish(stage='training_round_complete',round=index,optimizer_steps=update['optimizer_steps'])
    assert hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen_hash
    write(root/'training_summary.json',{'rounds':rounds,'frozen_sha256_before':frozen_hash,'frozen_sha256_after':frozen_hash,
        'total_controls':sum(x['steps'] for r in rounds for x in r['rollouts']),
        'total_optimizer_steps':sum(x['update']['optimizer_steps'] for x in rounds)})
    final=evaluate(model,plan,'final',publish)
    report=compare(plan,baseline,final)
    write(root/'comparison.json',report);publish(stage='comparison_complete',comparison=report)
    return report
