"""Regression for missing legacy assisted-grasp state, using native robot code."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch as th

import eval_b1k_subtask_samples as runner
from test_materialized_state import SOURCE, real_class
from test_subtask_sample_eval import _sample, _write_hdf5


class RobotBase:
    def deserialize(self, state):
        return {}, 0
    def _load_state(self, state):
        pass


NativeRobot = real_class(
    SOURCE / 'robots/robot.py', 'Robot', ['deserialize', '_load_state'],
    base=RobotBase, _AG_MAGIC=1e8 + 123456, _AG_STATE_SIZE=17,
)


def native_robot():
    robot = NativeRobot()
    robot._controllers = {}
    robot.is_manipulation = True
    robot.grasping_mode = 'assisted'
    robot.arm_names = ['left', 'right']
    robot._ag_obj_constraint_params = {'left': {'old': 1}, 'right': None}
    robot.released = []
    robot.created = []
    robot.release_grasp_immediately = lambda arm: robot.released.append(arm)
    robot._create_assisted_grasp_joint = lambda arm, params: robot.created.append((arm, params))
    return robot


def test_native_legacy_deserializer_clears_existing_grasp_instead_of_preserving_unknown_state():
    robot = native_robot()
    state, consumed = robot.deserialize(th.tensor([], dtype=th.float32))
    assert consumed == 0 and state['ag_obj_constraint_params'] == {}
    robot._load_state(state)
    assert robot.released == ['left'] and robot.created == []


def test_native_serialized_grasp_is_recreated_with_actual_target_and_joint_parameters():
    robot = native_robot()
    obj = SimpleNamespace(name='teddy_bear', links={'base_link': object()})
    robot.scene = SimpleNamespace(object_registry=lambda key, value: obj if (key, value) in [('uuid', 123), ('name', obj.name)] else None)
    payload = th.tensor([1e8+123456, 1, 123, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], dtype=th.float32)
    state, consumed = robot.deserialize(payload)
    assert consumed == 19 and state['ag_obj_constraint_params']['right']['target_obj'] == obj.name
    robot._load_state(state)
    assert robot.released == ['left']
    arm, params = robot.created[0]
    assert arm == 'right' and params['target_obj'] is obj and params['joint_type'] == 'FixedJoint'


def legacy_file(tmp_path):
    sample = _sample()
    path = runner.raw_hdf5_path(tmp_path, sample)
    _write_hdf5(path)
    with h5py.File(path, 'r+') as file:
        config = json.loads(file['data'].attrs['config'])
        config['robots'] = [{'name': 'robot', 'grasping_mode': 'assisted'}]
        file['data'].attrs['config'] = json.dumps(config)
        scene = json.loads(file['data'].attrs['scene_file'])
        scene['state'] = {'registry': {'object_registry': {'robot': {'controllers': {}, 'ag_obj_constraint_params': {'left': {}, 'right': {}}}}}}
        file['data'].attrs['scene_file'] = json.dumps(scene)
    return path, sample


@pytest.mark.parametrize("start", [0, 5])
def test_raw_cache_blocks_exact_scoring_but_allows_explicit_read_only_diagnostic(tmp_path, start):
    path, sample = legacy_file(tmp_path)
    sample = {**sample, "start": start, "segment_frames": sample["end"] - start}
    with pytest.raises(runner.InvalidSampleState, match='no serialized grasp constraints'):
        runner.load_episode_cache(path, [sample])
    cache = runner.load_episode_cache(path, [sample], allow_unverified_grasp_state=True)
    audit = cache['grasp_serialization_audit']
    assert audit['legacy_missing_assisted_grasp_state']
    assert not audit['runtime_grasp_restoration_verified']


def test_diagnostic_cache_cannot_enter_scoring_materialization(tmp_path, monkeypatch):
    path, sample = legacy_file(tmp_path)
    cache = runner.load_episode_cache(path, [sample], allow_unverified_grasp_state=True)
    monkeypatch.setitem(__import__('sys').modules, 'omnigibson', SimpleNamespace())
    monkeypatch.setitem(__import__('sys').modules, 'omnigibson.macros', SimpleNamespace(gm=object()))
    monkeypatch.setattr(runner, '_validate_recorded_episode', lambda *args: None)
    with pytest.raises(runner.InvalidSampleState, match='cannot enter exact scoring'):
        runner.materialize_episode_start_states(object(), cache, [sample])


def test_magic_in_padding_does_not_establish_recorded_grasp_information():
    cache = {'recorded_config': {'robots': [{'name': 'robot', 'grasping_mode': 'assisted'}]},
             'scene_file': {'state': {'registry': {'object_registry': {'robot': {'controllers': {}}}}}},
             'states': np.array([[0, 0, 1e8+123456]], dtype=np.float32), 'state_sizes': np.array([2])}
    assert runner.recorded_grasp_serialization_audit(cache)['legacy_missing_assisted_grasp_state']
