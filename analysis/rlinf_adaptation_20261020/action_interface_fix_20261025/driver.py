"""Paired frozen-checkpoint evaluation using the shared execution interface."""
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
import urllib.request

import numpy as np
import torch

from b1k_rlinf.action_interface import ExecutionActionInterface
from common import ExecutionStopped, torch_model, write
from experiment import evaluate, seed_all
from training import hash_parameters, load_overlay


def upload(url, data):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, data=data, method='PUT'), timeout=90):pass


def verify_raw_policy(model, predict, obs, noise):
    """Same full model and noise: execution mapping cannot alter PPO density."""
    interface=model.execution_interface
    python_state=random.getstate()
    cpu_state=torch.get_rng_state()
    gpu_state=torch.cuda.get_rng_state_all()
    try:
        model.execution_interface=ExecutionActionInterface()
        random.seed(937)
        raw, before=predict(obs, mode='train', noise=noise.clone(),
                            rng=torch.Generator(device=model.device).manual_seed(829))
        model.execution_interface=interface
        random.seed(937)
        executed, after=predict(obs, mode='train', noise=noise.clone(),
                                rng=torch.Generator(device=model.device).manual_seed(829))
        for key in ('model_actions', 'prev_logprobs'):
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        torch.testing.assert_close(before['forward_inputs']['action'], after['forward_inputs']['action'], atol=0, rtol=0)
        torch.testing.assert_close(executed, interface(raw), atol=0, rtol=0)
        with torch.no_grad():
            replay=model.default_forward(after['forward_inputs'], compute_values=False)['logprobs']
        torch.testing.assert_close(replay, after['prev_logprobs'], atol=1e-4, rtol=1e-4)
        return {'passed':True,'raw_chain_unchanged':True,'stored_raw_action_unchanged':True,
                'logprob_unchanged':True,'max_replay_error':float((replay-after['prev_logprobs']).abs().max()),
                'execution_matches_interface':True}
    finally:
        model.execution_interface=interface
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(gpu_state)


def main():
    plan=json.loads(Path(__file__).with_name('plan.json').read_text())
    root=Path(plan['output']);root.mkdir(parents=True, exist_ok=False);write(root/'plan.json',plan)
    state={'state':'running','stage':'loading','started_at':time.time(),'instance':plan['validation']['instance']}
    def publish(**fields):
        state.update(fields, updated_at=time.time());write(root/'status.json',state)
        upload(os.environ['B1K_STATUS_PUT'],json.dumps(state).encode())
    def deadline(*args):raise ExecutionStopped('Interface validation budget exhausted')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline)
    signal.alarm(plan['compute_timeout_seconds'])
    phases={};traces={};gates={};phase='baseline'
    try:
        seed_all(plan['model_seed']);model=torch_model(plan)
        assert model.execution_interface.gripper_open_tolerance == .001
        write(root/'action_interface.json',model.execution_interface.contract())
        original_predict=model.predict_action_batch
        def predict(obs, mode='eval', **kwargs):
            if phase not in gates:
                gates[phase]=verify_raw_policy(model, original_predict, obs, kwargs['noise'])
                write(root/'policy_density_checks.json',gates)
            executed,result=original_predict(obs,mode=mode,**kwargs)
            raw=result['policy_actions']
            torch.testing.assert_close(executed,model.execution_interface(raw),atol=0,rtol=0)
            keep=[i for i in range(23) if i not in (14,22)]
            torch.testing.assert_close(executed[...,keep],raw[...,keep],atol=0,rtol=0)
            traces.setdefault(phase,[]).append({'raw_gripper':raw[0,:,[14,22]].cpu().tolist(),
                'executed_gripper':executed[0,:,[14,22]].cpu().tolist(),
                'changed_commands':int((executed!=raw).sum())})
            return executed,result
        model.predict_action_batch=predict
        base_hash=hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
        assert base_hash==plan['expected_original_trainable_sha256']
        frozen=hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        publish(stage='baseline')
        phases['baseline']=evaluate(model,plan,'baseline',publish)
        write(root/'baseline/interface_trace.json',traces['baseline'])
        final=plan['audit']['checkpoints'][-1]
        load_overlay(model,final['directory'],final['hash'])
        phase='ppo'
        publish(stage='ppo')
        phases['ppo']=evaluate(model,plan,'ppo',publish)
        write(root/'ppo/interface_trace.json',traces['ppo'])
        assert frozen==hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        case=f"i{plan['validation']['instance']}_s0"
        left=json.loads((root/'baseline'/case/'initial.json').read_text())
        right=json.loads((root/'ppo'/case/'initial.json').read_text())
        assert left['physical_state'].keys()==right['physical_state'].keys()
        error=max(float(abs(np.asarray(left['physical_state'][k][i])-np.asarray(right['physical_state'][k][i])).max())
                  for k in left['physical_state'] for i in (0,1))
        assert error<=1e-4
        assert np.allclose(left['proprio'],right['proprio'],rtol=0,atol=1e-4)
        assert all(left[k]==right[k] for k in ['physics_dt','rendering_dt','observation_dt'])
        report={'passed':True,'instance':plan['validation']['instance'],'seed':0,'baseline':phases['baseline'],
            'ppo':phases['ppo'],'same_interface':model.execution_interface.contract(),'same_worker':True,
            'initial_max_error':error,'density_checks':gates,'base_hash':base_hash,'ppo_hash':final['hash'],
            'modified_gripper_commands':{k:sum(x['changed_commands'] for x in v) for k,v in traces.items()},
            'no_weight_updates':True,'frozen_parameters_unchanged':True}
        write(root/'report.json',report);upload(os.environ['B1K_RESULT_PUT'],json.dumps(report).encode())
        publish(state='complete',stage='complete')
    except BaseException as error:
        report={'passed':False,'instance':plan['validation']['instance'],'error_type':type(error).__name__,
                'error':str(error),'completed_phases':phases,'traceback':traceback.format_exc()}
        write(root/'failure.json',report);upload(os.environ['B1K_RESULT_PUT'],json.dumps(report).encode())
        publish(state='failed',stage='failed',error_type=type(error).__name__)
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
