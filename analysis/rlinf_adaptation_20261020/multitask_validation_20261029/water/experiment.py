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
from training import hash_parameters


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
        sim=start_simulator({**plan,'seed':seed},output,instance,plan.get('evaluation_mode','public_test'),plan['eval_steps'],publish)
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


def collect(model, plan, round_index, episode_index, publish):
    from goals import shaping_bonus
    import hashlib
    enforce_storage_budget(plan, 256*1024**2)
    instance = plan['train_instances'][episode_index]
    seed = plan['train_seeds'][round_index-1][episode_index]
    seed_all(seed)
    output = Path(plan['output'])/'training'/f'round_{round_index:02d}'/f'episode_{episode_index+1:02d}'
    sim = start_simulator({**plan, 'seed': seed, 'collecting_training': True}, output/'rollout',
                          instance, 'train', plan['train_steps_per_round'], publish)
    transitions = []
    try:
        while sim.current['status']['steps'] < plan['train_steps_per_round']:
            before = sim.current['physical_goals']
            with torch.no_grad():
                actions, result = model.predict_action_batch(env_observation(sim.current['observation']), mode='train')
            previous = {k: {n:t.detach().cpu() for n,t in v.items()} if isinstance(v,dict) else v.detach().cpu()
                        for k,v in result.items() if k!='model_actions'}
            remaining = plan['train_steps_per_round']-sim.current['status']['steps']
            current = sim.step(actions[0].cpu().numpy()[:remaining], label_text(model,result))
            task_reward = float(sum(current['rewards']))*plan['shaping']['task_reward_multiplier']
            shaped = shaping_bonus(before, current['physical_goals'], current['done'], plan['shaping'])
            # This experiment optimizes success within a fixed time limit: both
            # true success and time-limit failure end that finite-horizon return.
            previous.update(reward=task_reward+shaped, task_reward=task_reward, shaping_reward=shaped,
                            terminal=bool(current['done']), done=bool(current['done']),
                            environment_terminal=bool(current['terminal']), truncated=bool(current['truncated']),
                            goals_before=before, goals_after=current['physical_goals'])
            transitions.append(previous)
            write(output/'progress.json', {'round':round_index, 'episode':episode_index+1,
                  'instance':instance, 'chunks':len(transitions), **current['status']})
            if current['done']:
                break
        assert transitions[-1]['terminal'], 'Incomplete trajectory must not enter PPO'
        final_status = sim.current['status']
    finally:
        sim.close()
    contract = hashlib.sha256(json.dumps({'goals':plan['goal_contract'],'shaping':plan['shaping']},sort_keys=True).encode()).hexdigest()
    episode = {'transitions':transitions, 'bootstrap':torch.zeros(1), 'status':final_status,
               'instance':instance, 'seed':seed, 'reward_contract_sha256':contract}
    torch.save(episode, output/'rollout.pt')
    write(output/'result.json', {'instance':instance, 'seed':seed, **final_status})
    return episode

