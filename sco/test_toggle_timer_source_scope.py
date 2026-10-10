"""Compare the inventory prefix offsets with actual OmniGibson serializers."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
from omnigibson.prims.entity_prim import EntityPrim
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.object_states.toggle import ToggledOn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recorded_native_state import uuid
from audit_toggle_timer_source_scope import decode_toggle_prefix


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('joints', [0, 2])
def test_prefix_matches_actual_root_entity_and_toggle_serializers(dynamic, joints):
    root = {'pos': th.tensor([1., 2., 3.]), 'ori': th.tensor([0., 0., 0., 1.])}
    if dynamic:
        root.update(lin_vel=th.zeros(3), ang_vel=th.zeros(3))
    state = {'is_asleep': False, 'root_link': root, 'non_kin': {
        'ToggledOn': {'value': True, 'hand_in_marker_steps': 3.0}, 'Temperature': {'temperature': 23.0}}}
    if joints:
        state.update(joint_pos=th.arange(joints, dtype=th.float32), joint_vel=th.zeros(joints))
    link = object.__new__(RigidDynamicPrim if dynamic else XFormPrim)
    body = SimpleNamespace(n_joints=joints, root_link=link)
    packet = th.cat([EntityPrim.serialize(body, state), ToggledOn.serialize(None, state['non_kin']['ToggledOn']), th.tensor([23.0])])
    row = np.concatenate([np.asarray([0, 0, 0, 0, 0, 0, 1, uuid('object_registry'), 1, uuid('oven')], dtype=np.float32), packet.numpy()])
    cache = {'sha256': 'a' * 64, 'states': np.stack([row, row]), 'state_sizes': np.asarray([len(row)] * 2),
             'scene_file': {'objects_info': {'init_info': {'oven': {'args': {'name': 'oven'}}}},
                            'state': {'registry': {'object_registry': {'oven': state}}}}}
    result = decode_toggle_prefix(cache, 'oven', 1)
    assert result['timer'] == 3.0 and result['value'] is True and result['last_written_frame'] == 1
    cache['states'][1, 8] = 0
    cache['state_sizes'][1] = 9
    assert decode_toggle_prefix(cache, 'oven', 1)['last_written_frame'] == 0
    cache['states'][0, 8] = 0
    cache['state_sizes'][0] = 9
    initial = decode_toggle_prefix(cache, 'oven', 1)
    assert initial['source_kind'] == 'recorded_initial_scene_without_prior_sparse_update'
    assert initial['timer'] == 3.0 and initial['value'] is True and initial['last_written_frame'] is None
    state['non_kin'] = dict(reversed(list(state['non_kin'].items())))
    with pytest.raises(ValueError, match='first recorded'):
        decode_toggle_prefix(cache, 'oven', 1)
