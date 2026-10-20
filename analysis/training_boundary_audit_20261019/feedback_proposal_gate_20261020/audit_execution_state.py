"""Execute the archived/current B1K wrapper with a deterministic stub policy.

Checks a branching-protocol confound, not robot or neural-policy performance.
No source modifications, networking, neural inference or simulation.
"""
import ast
import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import tarfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT/'BEHAVIOR-VLAs/openpi/src/openpi/shared/eval_b1k_wrapper.py'
ARCHIVE = HERE.parent/'pilot_cloud_bundle.tar.gz'
MEMBER = 'project/BEHAVIOR-VLAs/openpi/src/openpi/shared/eval_b1k_wrapper.py'

def main():
    source = SOURCE.read_bytes()
    with tarfile.open(ARCHIVE) as archive:
        assert archive.extractfile(MEMBER).read() == source
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'B1KPolicyWrapper')
    namespace = {
        'np': np, 'torch': torch, 'BasePolicy': object,
        'logger': logging.getLogger('execution_state_audit'),
        'ROBOT_REGISTRY': {'stub': SimpleNamespace(name='robot', action=[], observations={})},
    }
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(SOURCE),'exec'),namespace)
    Wrapper = namespace['B1KPolicyWrapper']

    class StubPolicy:
        metadata = {}
        def __init__(self): self.calls = 0
        def infer(self, observation):
            values = 10*self.calls + np.arange(4,dtype=np.float32)
            self.calls += 1
            return {'actions': np.repeat(values[:,None],23,axis=1)}
        def reset(self): self.calls = 0

    wrapped = Wrapper(StubPolicy(),'stub','same task','receding_horizon',4,4)
    # Bypass only camera preprocessing; execute the actual buffering and reset methods.
    wrapped.process_input = lambda obs: [{'same_observation': True}]
    observation = {'robot::proprio': np.zeros(61,dtype=np.float32)}
    first = [float(wrapped.act(observation)[0]) for _ in range(2)]
    saved = copy.deepcopy(wrapped)
    continued = float(wrapped.act(observation)[0])
    reset_branch = copy.deepcopy(saved)
    reset_branch.reset()
    reset_action = float(reset_branch.act(observation)[0])
    restored_branch = copy.deepcopy(saved)
    restored_action = float(restored_branch.act(observation)[0])
    replan_branch = copy.deepcopy(saved)
    replan_branch._clear_action_buffers()
    forced_replan_action = float(replan_branch.act(observation)[0])
    assert first == [0.0,1.0]
    assert continued == restored_action == 2.0
    assert reset_action == 0.0 and forced_replan_action == 10.0

    result = {
        'kind':'actual_execution_wrapper_test_with_stub_policy',
        'wrapper_sha256':hashlib.sha256(source).hexdigest(),
        'matches_archived_pilot_source':True,
        'observation_identical_in_all_calls':True,
        'first_two_actions':first,
        'continue_cached_action':continued,
        'restore_complete_stub_execution_state_action':restored_action,
        'reset_policy_and_buffers_action':reset_action,
        'clear_only_action_buffers_and_force_replan_action':forced_replan_action,
        'proved':'Identical observations do not ensure identical action execution when branch initialization changes queue phase or forces replanning.',
        'limitations':[
            'The policy is a deterministic stub; no finding about actual VLA failure or success rates.',
            'Deepcopy is used only on the stub for this demonstration, not proposed for production neural models.',
            'The deployed runtime configuration must be checked separately; source equality alone does not prove every historical evaluation used this mode.'
        ],
        'requirements_for_physical_pilot':[
            'Match and log action-replan phase in nominal and disturbed branches.',
            'If the experiment deliberately starts with fresh policy state, reset both branches identically and state the changed estimand.',
            'For native-history continuation, capture server-side action queues, cached subtask tokens and voting state, enabled frame memory, and sampler RNG, plus client-side queues.',
            'Verify nominal restore-and-continue actions and processed inputs before interpreting disturbance effects.'
        ],
        'new_neural_model_calls':0,'new_simulator_calls':0,'new_cloud_jobs':0,
    }
    HERE.joinpath('audit_execution_state.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['matches_archived_pilot_source','continue_cached_action','restore_complete_stub_execution_state_action','reset_policy_and_buffers_action','clear_only_action_buffers_and_force_replan_action','new_neural_model_calls']}))

if __name__=='__main__':
    main()
