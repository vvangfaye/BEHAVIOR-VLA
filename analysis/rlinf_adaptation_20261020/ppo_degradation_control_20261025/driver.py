"""Fresh paired baseline and PPO evaluation in one worker; never updates weights."""
import json
import os
from pathlib import Path
import signal
import time
import traceback
import urllib.request

import numpy as np

from common import ExecutionStopped, torch_model, write
from experiment import evaluate, seed_all
from training import hash_parameters, load_overlay


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
    def deadline(*args):raise ExecutionStopped('Control experiment timeout')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline)
    signal.alarm(plan['compute_timeout_seconds'])
    try:
        seed_all(plan['model_seed']);model=torch_model(plan)
        current=hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
        assert current==plan['expected_original_trainable_sha256']
        frozen=hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        publish(stage='fresh_baseline')
        baseline=evaluate(model,plan,'baseline',publish)
        final=plan['audit']['checkpoints'][-1]
        load_overlay(model,final['directory'],final['hash'])
        publish(stage='fresh_final')
        after=evaluate(model,plan,'final',publish)
        assert frozen==hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        a=json.loads((root/'baseline/i301_s0/initial.json').read_text());b=json.loads((root/'final/i301_s0/initial.json').read_text())
        assert a['physical_state'].keys()==b['physical_state'].keys()
        error=max(float(abs(np.asarray(a['physical_state'][k][i])-np.asarray(b['physical_state'][k][i])).max())
                  for k in a['physical_state'] for i in (0,1))
        assert error<=1e-4
        assert all(a[k]==b[k] for k in ['physics_dt','rendering_dt','observation_dt'])
        report={'passed':True,'baseline':baseline,'after_ppo':after,'initial_max_error':error,
                'same_worker':True,'frozen_parameters_unchanged':True,'no_training':True,
                'baseline_hash':current,'final_hash':final['hash']}
        write(root/'audit.json',report);upload(os.environ['B1K_RESULT_PUT'],json.dumps(report).encode())
        publish(state='complete',stage='complete')
    except BaseException as error:
        report={'passed':False,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}
        write(root/'failure.json',report);upload(os.environ['B1K_RESULT_PUT'],json.dumps(report).encode())
        publish(state='failed',stage='failed',error_type=type(error).__name__)
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
