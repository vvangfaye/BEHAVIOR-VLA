"""Local, offline real-VLA replay test; no environment or physical actions."""
from pathlib import Path
import copy
import hashlib
import json
import random
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'native_input_screen_20261020'))
import probe as native
from execution_state_capture import capture,restore

def run():
    import jax
    import numpy as np
    import torch
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    source_plan=json.loads((HERE.parent/'native_input_screen_20261020/plan.json').read_text())
    checkpoint=Path(source_plan['checkpoints']['baseline1688977'])
    rows=native.inputs()
    selected=rows[:2]
    with native.flow.offline_assets():
        print('Loading existing baseline checkpoint on CPU',flush=True)
        loaded=native.make_policy(checkpoint)
        policy=loaded['policy']
        if policy._frame_memory is not None or policy._is_pytorch_model:
            raise ValueError('This protocol requires the existing JAX policy without frame memory')
        wrapper=B1KPolicyWrapper(policy,'b1k/R1Pro',selected[0]['observation']['prompt'],
            'receding_horizon',16,32,obs_size=(224,224))
        # Reuse already verified, camera-processed native observations. This test
        # does not cover the physical simulator or camera preprocessing itself.
        wrapper.process_input=lambda obs:[copy.deepcopy(selected[obs['_cache_index']]['observation'])]
        def observation(index):
            return {wrapper.robot.name+'::proprio':selected[index]['observation']['observation/state'].copy(),
                    '_cache_index':index,'prompt':selected[index]['observation']['prompt']}
        random.seed(20261020);np.random.seed(20261020);torch.manual_seed(20261020)
        policy._rng=jax.random.key(20261020)
        print('First native inference',flush=True)
        first=wrapper.act(observation(0)).numpy().copy()
        identity='baseline1688977:'+str(checkpoint)
        saved=capture(wrapper,identity)
        sequence=[0]*15+[1]*5
        def continue_policy(label):
            actions=[];subtasks=[];sources=[]
            for step,index in enumerate(sequence):
                action=wrapper.act(observation(index)).numpy().copy()
                if action.shape!=(23,) or not np.isfinite(action).all():
                    raise ValueError('Unexpected native action')
                actions.append(action)
                subtasks.append(copy.deepcopy(wrapper.last_subtasks))
                sources.append(copy.deepcopy(wrapper.last_subtask_sources))
                if step in (0,14,15,19):print(json.dumps({'pass':label,'step':step,'subtask':subtasks[-1]}),flush=True)
            return np.stack(actions),subtasks,sources,np.asarray(jax.random.key_data(policy._rng)).tolist()
        reference=continue_policy('reference')
        restore(wrapper,saved,identity)
        repeated=continue_policy('restored')
        max_difference=float(np.max(np.abs(reference[0]-repeated[0])))
        labels_equal=reference[1:]==repeated[1:]
        result={
            'kind':'real_vla_in_process_branch_parity',
            'checkpoint':str(checkpoint),'backend':jax.default_backend(),
            'model_config':loaded['config'],'normalization_sha256':loaded['norm_sha256'],
            'input_sha256':[row['input_sha256'] for row in selected],
            'wrapper_source_sha256':hashlib.sha256(Path(sys.modules[B1KPolicyWrapper.__module__].__file__).read_bytes()).hexdigest(),
            'capture_source_sha256':hashlib.sha256((HERE/'execution_state_capture.py').read_bytes()).hexdigest(),
            'model_calls':3,'control_outputs_compared':len(sequence),'action_horizon':16,
            'max_absolute_action_difference':max_difference,
            'action_arrays_exactly_equal':bool(np.array_equal(reference[0],repeated[0])),
            'subtasks_sources_and_final_rng_equal':labels_equal,
            'passed':bool(max_difference<=1e-6 and labels_equal),
            'simulation_calls':0,'training':False,'network_downloads':False,
            'scope':'Real JAX VLA and actual action-buffer wrapper on a prescribed sequence of cached, camera-processed observations. Includes replay across one replan boundary. No simulator, new physical observations, contact restoration or historical trajectory fidelity is validated.',
        }
        np.savez_compressed(HERE/'neural_branch_parity_actions.npz',first=first,reference=reference[0],repeated=repeated[0])
        return result

if __name__=='__main__':
    started=time.time()
    destination=HERE/'neural_branch_parity_result.json'
    if destination.exists():raise RuntimeError('Existing result must be inspected, not overwritten')
    try:
        result=run();result['status']='complete'
    except BaseException as error:
        result={'status':'failed','error_type':type(error).__name__}
        raise
    finally:
        result['elapsed_seconds']=time.time()-started
        destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['model_config','checkpoint']}),flush=True)
