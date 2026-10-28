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
        roots = plan['runtime_storage'].get('shared_budget_roots', [str(root)])
        for storage_root in dict.fromkeys(roots):
            for path in Path(storage_root).rglob('*'):
                try:
                    if path.is_file() and not path.is_symlink():used+=path.stat().st_blocks*512
                except FileNotFoundError:
                    pass
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
            task_reward = float(sum(current['rewards']))*10.
            mid_reward = float(sum(current.get('mid_rewards', [])))
            previous.update(reward=task_reward+mid_reward, task_reward=task_reward,
                            mid_reward=mid_reward, terminal=current['terminal'],done=current['done'])
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
    import hashlib
    episode['mid_reward_contract_sha256']=hashlib.sha256(json.dumps(plan['mid_reward'],sort_keys=True).encode()).hexdigest()
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
        assert np.allclose(a['proprio'], b['proprio'], atol=1e-4, rtol=0), 'Proprio pairing failed'
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
    elif ds > 0 and dp > 0: conclusion = 'Improvement observed on this fixed 30-case panel only; broader reliability remains untested.'
    elif ds == 0 and dp == 0: conclusion = 'No improvement observed on the fixed evaluation panel.'
    elif ds < 0 or dp < 0: conclusion = 'At least one aggregate metric regressed on the fixed evaluation panel.'
    else: conclusion = 'Only one aggregate metric improved; evidence is mixed.'
    return {'comparison_valid': valid, 'baseline': before, 'final': after, 'mean_score_delta': ds,
            'success_rate_delta': dp, 'success_to_failure': sum(p['success_delta'] < 0 for p in pairs),
            'failure_to_success': sum(p['success_delta'] > 0 for p in pairs), 'pairs': pairs,
            'conclusion': conclusion, 'scope': 'Ten fixed instances, three paired seeds, one training run; no checkpoint selection.'}



def copy_evidence(source,target):
    target.mkdir(parents=True,exist_ok=False)
    records=[]
    for path in source.rglob('*'):
        if path.is_file() and path.suffix=='.json' and not any(x in path.parts for x in ['cache','appdata','config','warp']):
            relative=path.relative_to(source); dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest);records.append(str(relative))
    return records


