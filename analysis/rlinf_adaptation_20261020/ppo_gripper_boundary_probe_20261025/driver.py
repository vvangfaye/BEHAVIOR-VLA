"""A single labelled intervention, never mixed into the original evaluation."""
import json
import os
from pathlib import Path
import signal
import time
import traceback
import urllib.request

import numpy as np

from common import ExecutionStopped, Simulator, torch_model, write
from experiment import evaluate, seed_all
from training import hash_parameters, load_overlay


def snap_open(actions):
    output=np.array(actions,copy=True)
    counts={}
    for index in (14,22):
        mask=(output[:,index]>=.999)&(output[:,index]<1.)
        counts[index]=int(mask.sum())
        output[mask,index]=1.
    return output,counts


def upload(url,data):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url,data=data,method='PUT'),timeout=90):pass


def main():
    plan=json.loads(Path(__file__).with_name('plan.json').read_text())
    root=Path(plan['output']);root.mkdir(parents=True,exist_ok=False);write(root/'plan.json',plan)
    state={'state':'running','stage':'loading','started_at':time.time()}
    def publish(**fields):
        state.update(fields,updated_at=time.time());write(root/'status.json',state)
        upload(os.environ['B1K_STATUS_PUT'],json.dumps(state).encode())
    def deadline(*args):raise ExecutionStopped('Gripper intervention timeout')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline)
    signal.alarm(plan['compute_timeout_seconds'])
    trace=[];original_step=Simulator.step
    try:
        seed_all(plan['model_seed']);model=torch_model(plan)
        final=plan['audit']['checkpoints'][-1]
        load_overlay(model,final['directory'],final['hash'])
        def step(sim,actions,subtask):
            corrected,counts=snap_open(actions)
            trace.append({'start_step':sim.current['status']['steps'],'changed':counts,
                          'before_gripper':np.asarray(actions)[:,[14,22]].tolist(),
                          'after_gripper':corrected[:,[14,22]].tolist()})
            return original_step(sim,corrected,subtask)
        Simulator.step=step
        publish(stage='intervention_evaluation')
        result=evaluate(model,plan,'intervention',publish)
        assert hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)==final['hash']
        initial=json.loads((root/'intervention/i301_s0/initial.json').read_text())
        report={'passed':True,'evaluation':result,'initial_physical_sha256':initial['physical_sha256'],'intervention':plan['diagnostic_control']['intervention'],
                'corrected_commands':sum(sum(x['changed'].values()) for x in trace),
                'raw_trace':trace,'weights_unchanged':True,
                'limitation':'One diagnostic intervention, not an unmodified-policy evaluation or validated fix.'}
        write(root/'audit.json',report);upload(os.environ['B1K_RESULT_PUT'],json.dumps(report).encode())
        publish(state='complete',stage='complete')
    except BaseException as error:
        report={'passed':False,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}
        write(root/'failure.json',report);upload(os.environ['B1K_RESULT_PUT'],json.dumps(report).encode())
        publish(state='failed',stage='failed',error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0);Simulator.step=original_step


if __name__=='__main__':main()
