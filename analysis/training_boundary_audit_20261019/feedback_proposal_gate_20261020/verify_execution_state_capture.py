"""Exercise real wrapper code with stochastic stub outputs across a replan."""
import ast
from collections import deque
import hashlib
import json
import logging
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch
from execution_state_capture import capture, restore

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]/'BEHAVIOR-VLAs/openpi/src/openpi/shared/eval_b1k_wrapper.py'

def main():
    tree = ast.parse(SOURCE.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'B1KPolicyWrapper')
    ns = {'np': np, 'torch': torch, 'BasePolicy': object,
          'logger': logging.getLogger('branch_state_test'),
          'ROBOT_REGISTRY': {'stub': SimpleNamespace(name='robot', action=[], observations={})}}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(SOURCE),'exec'),ns)
    class Stub:
        metadata = {}
        _frame_memory = None
        _interval, _temperature, _vote_window, _vote_threshold = 1, 0.0, 3, 2
        _is_pytorch_model = True
        _external_memory_observations, _pending_memory_ms = False, 0.0
        def reset(self):
            self._cached_tokens = np.array([1,2],dtype=np.int32)
            self._cached_text, self._last_prompt = 'pick object', 'same task'
            self._calls_since_generation = self._request_index = self._next_vote_id = 0
            self._vote_history = deque(maxlen=3)
        def __init__(self): self.reset()
        def infer(self, obs):
            self._request_index += 1
            value = random.random() + np.random.rand() + torch.rand(()).item()
            self._vote_history.append(('pick object', self._cached_tokens.copy()))
            return {'actions':np.arange(4*23,dtype=np.float32).reshape(4,23)+value,
                    'subtask':self._cached_text,'subtask_source':'generated'}
    wrapper = ns['B1KPolicyWrapper'](Stub(),'stub','same task','receding_horizon',4,4)
    wrapper.process_input = lambda obs:[{}]
    obs={'robot::proprio':np.zeros(61,dtype=np.float32)}
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    for _ in range(2): wrapper.act(obs)
    state = capture(wrapper,'frozen-checkpoint')
    reference = [wrapper.act(obs).numpy().copy() for _ in range(8)]
    restore(wrapper,state,'frozen-checkpoint')
    replay = [wrapper.act(obs).numpy().copy() for _ in range(8)]
    assert all(np.array_equal(a,b) for a,b in zip(reference,replay))
    rejected=[]
    for change in ['checkpoint','configuration','frame_memory']:
        try:
            if change=='configuration': wrapper.action_horizon=8
            if change=='frame_memory': wrapper.policy._frame_memory=object()
            restore(wrapper,state,'wrong' if change=='checkpoint' else 'frozen-checkpoint')
        except ValueError: rejected.append(change)
        else: raise AssertionError('Invalid restore was accepted')
        finally:
            wrapper.action_horizon=4
            wrapper.policy._frame_memory=None
    result={'matched_actions':len(reference),'crosses_replan_boundaries':True,
            'python_numpy_torch_rng_restored':True,'rejected_mismatches':rejected,
            'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'real_neural_policy_parity_verified':False,'simulator_parity_verified':False,
            'model_or_simulator_calls':0}
    (HERE/'execution_state_capture_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__': main()
