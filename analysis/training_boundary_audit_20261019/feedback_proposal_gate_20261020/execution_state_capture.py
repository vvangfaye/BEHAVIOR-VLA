"""In-process state capture for the inspected B1K/SubtaskPolicy stack.

Does not copy model parameters, serialize arbitrary objects, restore the simulator,
or claim end-to-end parity. Enabled frame memory is deliberately unsupported until
its own state contract is implemented and verified.
"""
import copy
import random

import numpy as np
import torch

WRAPPER_FIELDS = (
    'batch_size', 'action_buffer', 'sequence_indices', 'sequence_lengths',
    'num_active_sequences', 'step_counter', '_last_prompt', '_last_subtask',
    'last_subtasks', 'last_subtask_sources', 'last_subtask_votes', '_memory_control_step',
)
POLICY_FIELDS = (
    '_cached_tokens', '_cached_text', '_last_prompt', '_calls_since_generation',
    '_request_index', '_next_vote_id', '_vote_history',
    '_external_memory_observations', '_pending_memory_ms',
)
WRAPPER_CONFIG = ('control_mode', 'action_horizon', 'max_len', 'obs_size',
                  'provided_subtask_as_prompt', 'temporal_ensemble_max')
POLICY_CONFIG = ('_interval', '_temperature', '_vote_window', '_vote_threshold',
                 '_is_pytorch_model')

def _configuration(wrapper):
    policy = wrapper.policy
    if getattr(policy, '_frame_memory', 'missing') is not None:
        raise ValueError('Enabled or unknown frame-memory state is unsupported')
    for obj, fields in [(wrapper, WRAPPER_FIELDS + WRAPPER_CONFIG),
                        (policy, POLICY_FIELDS + POLICY_CONFIG)]:
        if any(not hasattr(obj, field) for field in fields):
            raise ValueError('Policy stack differs from the inspected state contract')
    return {
        'wrapper': {key: copy.deepcopy(getattr(wrapper, key)) for key in WRAPPER_CONFIG},
        'policy': {key: copy.deepcopy(getattr(policy, key)) for key in POLICY_CONFIG},
    }

def capture(wrapper, checkpoint_identity):
    if not isinstance(checkpoint_identity, str) or not checkpoint_identity:
        raise ValueError('Explicit checkpoint identity is required')
    policy = wrapper.policy
    return {
        'kind': 'b1k_subtask_policy_in_process_state_v1',
        'checkpoint_identity': checkpoint_identity,
        'policy_object_id': id(policy),
        'configuration': _configuration(wrapper),
        'wrapper': {key: copy.deepcopy(getattr(wrapper, key)) for key in WRAPPER_FIELDS},
        'policy': {key: copy.deepcopy(getattr(policy, key)) for key in POLICY_FIELDS},
        'jax_rng_present': hasattr(policy, '_rng'),
        'jax_rng': copy.deepcopy(getattr(policy, '_rng', None)),
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state().clone(),
        'cuda_rng': [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }

def restore(wrapper, state, checkpoint_identity):
    if (state.get('kind') != 'b1k_subtask_policy_in_process_state_v1'
            or checkpoint_identity != state['checkpoint_identity']
            or id(wrapper.policy) != state['policy_object_id']
            or _configuration(wrapper) != state['configuration']):
        raise ValueError('State belongs to another checkpoint, object or configuration')
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state['cuda_rng']) != expected_cuda:
        raise ValueError('CUDA device set changed')
    # Check everything above before mutating any state.
    for key, value in state['wrapper'].items():
        setattr(wrapper, key, copy.deepcopy(value))
    for key, value in state['policy'].items():
        setattr(wrapper.policy, key, copy.deepcopy(value))
    if state['jax_rng_present']:
        wrapper.policy._rng = copy.deepcopy(state['jax_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].clone())
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all([value.clone() for value in state['cuda_rng']])
