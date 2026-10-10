"""Exercise real registry/USD serializers on CPU without launching Isaac Sim."""
import ast
from collections import defaultdict
import copy
from contextlib import nullcontext
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_b1k_subtask_samples as runner

SOURCE = Path(__file__).resolve().parents[1] / 'BEHAVIOR-1K/OmniGibson/omnigibson'


def real_class(path, name, methods=None, base=None, **scope):
    tree = ast.parse(path.read_text())
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name))
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        assert {n.name for n in node.body} == set(methods)
        node.bases = [ast.Name(id='TestBase', ctx=ast.Load())]
        scope['TestBase'] = base
        node.keywords = []
    namespace = {'th': th, 'log': logging.getLogger(__name__), **scope}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]


Serializable = real_class(SOURCE / 'utils/python_utils.py', 'Serializable')


class ScalarState(Serializable):
    stateful = True
    def __init__(self, value): self.value = th.tensor([value], dtype=th.float32)
    def _dump_state(self): return {'value': self.value}
    def _load_state(self, state): self.value = state['value'].clone()
    def serialize(self, state): return state['value'].reshape(-1)
    def deserialize(self, state): return {'value': state[:1]}, 1


class Legacy(ScalarState): pass
class Extra(ScalarState): pass


class ObjectBase(Serializable):
    def _dump_state(self): return {'root_link': {'pos': self.position}}
    def _load_state(self, state): self.position = state['root_link']['pos'].clone()
    def serialize(self, state): return state['root_link']['pos'].reshape(-1)
    def deserialize(self, state): return {'root_link': {'pos': state[:1]}}, 1
    def clear_states_cache(self): pass


USD = real_class(SOURCE / 'objects/usd_object.py', 'USDObject',
    ['_dump_state', '_load_state', 'load_non_kin_state', 'serialize', '_deserialize_non_kin_states', 'deserialize'],
    base=ObjectBase, get_state_name=lambda cls: cls.__name__, REGISTERED_OBJECT_STATES={'Legacy':Legacy, 'Extra':Extra})


def make_object(name, uuid):
    obj = USD()
    obj.name, obj.uuid, obj.position = name, uuid, th.tensor([3.0])
    obj._states = {Legacy: Legacy(9), Extra: Extra(5)}
    obj._recorded_non_kin_state_names = {'Legacy'}
    obj._recorded_non_kin_state_order = ('Legacy',)
    return obj


class Lookup(Serializable):
    def __init__(self, objects):
        self.objects = objects
        self.hash_key = 'uuid'
        self._dump_filter = lambda obj: obj.name == 'apple'
        self._load_filter = lambda obj: True
    def __call__(self, key, value):
        return next((obj for obj in self.objects if getattr(obj, key) == value), None)


Registry = real_class(SOURCE / 'utils/registry_utils.py', 'SerializableRegistry',
    ['_dump_state', '_load_state', 'serialize', 'deserialize', 'set_dump_filter'], base=Lookup,
    USDObject=USD, m=SimpleNamespace(STABILIZE_SKIPPED_OBJECTS=False))


class Simulator:
    def __init__(self, registry):
        self.registry = registry
        self.scenes = [SimpleNamespace(object_registry=registry)]
        self.loads, self.renders, self.mutate_input = [], 0, False
    def dump_state(self, serialized=False):
        state = self.registry.dump_state(serialized=False)
        if serialized: return self.registry.serialize(state)
        return {0: {'registry': {'object_registry': state}}}
    def load_state(self, state, serialized=False):
        self.loads.append(serialized)
        if serialized:
            self.registry.load_state(state, serialized=True)
        else:
            self.registry.load_state(state[0]['registry']['object_registry'], serialized=False)
            if self.mutate_input: state[0]['registry']['object_registry'].clear()
    def update_handles(self): pass
    def render(self): self.renders += 1


@pytest.fixture
def world(monkeypatch):
    registry = Registry([make_object('apple',101), make_object('table',202)])
    sim = Simulator(registry)
    gm = SimpleNamespace(ENABLE_TRANSITION_RULES=True, unlocked=nullcontext)
    monkeypatch.setitem(sys.modules, 'omnigibson', SimpleNamespace(sim=sim))
    monkeypatch.setitem(sys.modules, 'omnigibson.macros', SimpleNamespace(gm=gm))
    monkeypatch.setattr(runner, '_restore_episode_baseline', lambda *args: None)
    monkeypatch.setattr(runner, '_reset_evaluator_rollout', lambda *args: None)
    return sim, registry, gm


def record_row(registry):
    """A real serializer produces a sparse row with only the recorded state."""
    states = {obj.name: obj.dump_state(serialized=False) for obj in registry.objects}
    for state in states.values(): state['non_kin'].pop('Extra')
    return registry.serialize(states)


def cached(sim):
    return {'state':runner._dump_full_simulator_state(sim.scenes[0],sim),
            'state_format':runner.FULL_SNAPSHOT_FORMAT, 'transition_steps':[],
            'restoration':{'full_snapshot':True}}


def test_real_serializers_reproduce_uuid_offset_bug_and_named_snapshot_roundtrips(world):
    sim, registry, _ = world
    previous = registry._dump_filter
    registry.set_dump_filter(lambda obj: True)
    flat = sim.dump_state(serialized=True)
    # Runtime wrote Legacy + Extra, while demo-aligned reader consumes Legacy only.
    with pytest.raises(AssertionError, match='hash_key uuid: 5'):
        sim.load_state(flat, serialized=True)
    registry.set_dump_filter(previous)
    snapshot = cached(sim)
    assert registry._dump_filter is previous
    assert set(snapshot['state'][0]['registry']['object_registry']) == {'apple','table'}
    for obj in registry.objects:
        obj.position.fill_(40);obj._states[Legacy].value.fill_(80);obj._states[Extra].value.fill_(90)
    runner.restore_materialized_sample_state(object(), {}, snapshot)
    for obj in registry.objects:
        assert obj.position.item()==3 and obj._states[Legacy].value.item()==9 and obj._states[Extra].value.item()==5
        assert obj._recorded_non_kin_state_order == ('Legacy',)
    assert sim.loads[-1] is False and sim.renders == 3


def test_raw_recorded_flat_rows_still_use_recording_schema(world):
    sim, registry, _ = world
    row = record_row(registry)
    for obj in registry.objects: obj._states[Legacy].value.fill_(0)
    sim.load_state(row, serialized=True)
    assert all(obj._states[Legacy].value.item()==9 for obj in registry.objects)
    assert all(obj._states[Extra].value.item()==5 for obj in registry.objects)


def test_initial_environment_constructs_missing_recorded_states_before_loading(monkeypatch):
    """Exercise the evaluator path and native constructor-schema helper together."""
    path = SOURCE / 'envs/data_wrapper.py'
    tree = ast.parse(path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_add_recorded_non_kin_states_to_scene_file')
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), namespace)
    native_helper = namespace[helper.name]
    wrapper = SimpleNamespace(_add_recorded_non_kin_states_to_scene_file=native_helper)
    monkeypatch.setitem(sys.modules, 'omnigibson.envs.data_wrapper', wrapper)
    scene = {'objects_info': {'init_info': {'charcoal_grill': {'args': {'name': 'charcoal_grill'}}}},
             'state': {'registry': {'object_registry': {
                 'charcoal_grill': {'non_kin': {'Legacy': {'value': [9]}}}}}}}
    original = copy.deepcopy(scene)
    cache = {'scene_file': scene, 'recorded_config': {
        'scene': {}, 'robots': [{'name': 'recorded_robot'}], 'task': {}}}
    monkeypatch.setattr(runner, 'complete_recorded_scene_file',
                        lambda *args: (copy.deepcopy(scene), Path('recorded.json'), 0))
    monkeypatch.setattr(runner, '_configure_recorded_robot',
                        lambda robot, *args: (copy.deepcopy(robot), {}))

    def environment(configs):
        args = configs['scene']['scene_file']['objects_info']['init_info']['charcoal_grill']['args']
        obj = make_object('charcoal_grill', 101)
        # Current taxonomy supplies Extra but has dropped the recorded Legacy state.
        obj._states = {Extra: Extra(5)}
        for name in args.get('recorded_non_kin_state_names', []):
            obj._states[{'Legacy': Legacy}[name]] = Legacy(0)
        state, consumed = obj.deserialize(th.tensor([3.0, 9.0]))
        assert consumed == 2 and state['non_kin']['Legacy']['value'].item() == 9
        assert args['recorded_non_kin_state_names'] == ['Legacy']
        return obj

    gm = SimpleNamespace(DATA_PATH='recorded-assets', ENABLE_TRANSITION_RULES=False,
                         unlocked=nullcontext)
    module = SimpleNamespace(DISABLED_TRANSITION_RULES=[], load_human_stats=lambda task: {},
                             load_available_tasks=lambda: {'recorded_task': [{}]})
    monkeypatch.setitem(sys.modules, 'omnigibson', SimpleNamespace(Environment=environment))
    monkeypatch.setitem(sys.modules, 'omnigibson.macros', SimpleNamespace(gm=gm))
    monkeypatch.setitem(sys.modules, 'omnigibson.eval', SimpleNamespace(evaluator=module))
    monkeypatch.setitem(sys.modules, 'hydra.utils', SimpleNamespace(instantiate=lambda wrapper, env: env))

    class Evaluator:
        def __init__(self, cfg):
            self.cfg, self.robot_eval_config = cfg, {}
            self.env = self.load_env('wrapper')
        def _build_robot_config(self, **kwargs): return {'name': 'policy_robot'}
        def configure_robot_cameras(self, *args): pass

    cfg = SimpleNamespace(task=SimpleNamespace(name='recorded_task'))
    sample = {'task': 'recorded_task'}
    # Omitting construction-time schema reproduces the strict real deserializer failure.
    wrapper._add_recorded_non_kin_states_to_scene_file = lambda **kwargs: None
    with pytest.raises(ValueError, match='could not construct'):
        runner.create_sample_evaluator(SimpleNamespace(Evaluator=Evaluator), cfg, cache, sample, 256)
    wrapper._add_recorded_non_kin_states_to_scene_file = native_helper
    evaluator = runner.create_sample_evaluator(SimpleNamespace(Evaluator=Evaluator), cfg, cache, sample, 256)
    assert evaluator.env.name == 'charcoal_grill'
    assert scene == original and 'recorded_non_kin_state_names' not in original['objects_info']['init_info']['charcoal_grill']['args']


def test_transition_object_keeps_flat_recording_aligned_before_next_uuid(world, monkeypatch):
    sim, registry, _ = world
    path = SOURCE / 'envs/data_wrapper.py'
    tree = ast.parse(path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_recorded_non_kin_state_name_union')
    scope = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), scope)
    monkeypatch.setitem(sys.modules, 'omnigibson.envs.data_wrapper', SimpleNamespace(
        _is_system_particle_template_info=lambda *args: False,
        _is_system_particle_template_name=lambda *args: False,
        _recorded_non_kin_state_name_union=scope[helper.name]))
    sliced = make_object('slice',303)
    sliced._recorded_non_kin_state_names = None
    sliced._recorded_non_kin_state_order = None
    sliced.set_position = lambda pos: None
    registry.objects.insert(0, sliced)
    row = record_row(registry)
    # A fresh transition object accepts Extra and consumes the next object's UUID.
    with pytest.raises(AssertionError, match='hash_key uuid: 3'):
        sim.load_state(row, serialized=True)
    registry.objects.remove(sliced)
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.python_utils', SimpleNamespace(
        create_object_from_init_info=lambda info: sliced))
    def add_object(obj):
        assert obj._recorded_non_kin_state_names == {'Legacy'}
        registry.objects.insert(0,obj)
    sim.scenes[0].add_object = add_object
    sim.step = lambda: None
    recorded_scene = {'state': {'registry': {'object_registry': {
        'apple': {'non_kin': {'Legacy': {}}}}}}}
    runner._apply_transition(sim.scenes[0],
        {'objects': {'add': [{'name':'slice'}]}}, recorded_scene)
    sim.load_state(row, serialized=True)
    assert sliced._states[Legacy].value.item() == 9
    assert sliced._states[Extra].value.item() == 5
    assert all(obj.position.item() == 3 for obj in registry.objects)


def test_snapshot_is_detached_from_live_tensors_and_from_mutating_loaders(world):
    sim, registry, _ = world
    snapshot = cached(sim)
    registry.objects[0]._states[Legacy].value.fill_(100)
    original = snapshot['state'][0]['registry']['object_registry']['apple']['non_kin']['Legacy']['value']
    assert original.item()==9
    sim.mutate_input = True
    for _ in range(2):
        runner.restore_materialized_sample_state(object(), {}, snapshot)
        assert registry.objects[0]._states[Legacy].value.item()==9
        assert set(snapshot['state'][0]['registry']['object_registry'])=={'apple','table'}


@pytest.mark.parametrize('change', ['missing','extra','scene','malformed'])
def test_full_snapshot_cannot_silently_skip_missing_or_extra_objects(world, change):
    sim, registry, _ = world
    snapshot = cached(sim)
    objects = snapshot['state'][0]['registry']['object_registry']
    if change=='missing': objects['absent'] = objects.pop('table')
    elif change=='extra': registry.objects.append(make_object('unexpected',303))
    elif change=='scene': snapshot['state'][1] = snapshot['state'].pop(0)
    else: snapshot['state'][0] = {}
    with pytest.raises(runner.InvalidSampleState):
        runner.restore_materialized_sample_state(object(), {}, snapshot)
    assert not sim.loads and sim.renders==0


def test_topology_transitions_are_recreated_before_named_state_restore(world, monkeypatch):
    sim, registry, _ = world
    registry.objects.append(make_object('slice',303))
    snapshot = cached(sim)
    snapshot['transition_steps'] = [1]
    registry.objects.pop()
    events=[]
    scene_file = {'state': {'object_registry': {'apple': {'non_kin': {'Legacy': {}}}}}}
    def apply(scene, transition, recorded_scene_file):
        assert recorded_scene_file is scene_file
        events.append(transition);registry.objects.append(make_object('slice',303))
    monkeypatch.setattr(runner, '_apply_transition', apply)
    evaluator = SimpleNamespace(env=SimpleNamespace(scene=sim.scenes[0]))
    runner.restore_materialized_sample_state(evaluator,
        {'transition_by_step':{1:{'add':'slice'}}, 'scene_file':scene_file}, snapshot)
    assert events==[{'add':'slice'}] and sim.loads==[False]


def test_native_recipe_cache_drops_removed_parent_before_model_steps(world):
    sim, registry, gm = world
    parent, container = registry.objects
    path = SOURCE / 'transition_rules.py'

    class OnTop: pass
    class ContainedParticles: pass

    def on_top(obj):
        if obj not in registry.objects:
            raise KeyError('/' + obj.name)
        return False

    for obj in registry.objects:
        obj.fixed_base = False
        obj.states = {OnTop: SimpleNamespace(get_value=lambda other, obj=obj: on_top(obj))}
    container.states[ContainedParticles] = SimpleNamespace(link=SimpleNamespace(
        check_points_in_volume=lambda positions: th.zeros(len(positions), dtype=th.bool)))

    class CategoryRegistry:
        def get_dict(self, key):
            assert key == 'category'
            return {'food': [obj for obj in registry.objects if obj is not container],
                    'container': [container]}

    class Scene:
        object_registry = CategoryRegistry()
        @property
        def objects(self): return registry.objects

    scene = Scene()
    sim.scenes[0].transition_rule_api = None

    class RuleBase:
        ENABLED = True
        candidate_filters = {'container': None}
        expect_restored = False
        def get_object_candidates(self, objects):
            assert objects is registry.objects
            return {'container': [container]}
        def refresh(self, object_candidates):
            if self.expect_restored:
                assert gm.ENABLE_TRANSITION_RULES is False
                assert sim.loads[-1] is False and sim.renders == 3
            self.object_candidates = object_candidates

    Recipe = real_class(path, 'RecipeRule', ['refresh', '_compute_container_info'],
        base=RuleBase, defaultdict=defaultdict, OnTop=OnTop, ContainedParticles=ContainedParticles)
    API = real_class(path, 'TransitionRuleAPI',
        ['get_rule_candidates', 'refresh_all_rules', 'refresh_rules'], base=object)
    rule = Recipe()
    rule.scene, rule._recipes = scene, {}
    api = API()
    api.scene, api.all_rules, api.active_rules = scene, {rule}, set()
    api.refresh_all_rules()
    assert parent in rule._objects
    # Simulate the recorded slicing topology, while rules are disabled.
    registry.objects.remove(parent)
    sliced = make_object('half_apple_0', 303)
    sliced.fixed_base = False
    sliced.states = {OnTop: SimpleNamespace(get_value=lambda other: on_top(sliced))}
    registry.objects.append(sliced)
    positions = {'obj_positions': th.zeros((2, 3))}
    with pytest.raises(KeyError, match='/apple'):
        rule._compute_container_info({'container': [container]}, container, positions)
    sim.scenes[0].transition_rule_api = api
    rule.expect_restored = True
    runner.restore_materialized_sample_state(object(), {}, cached(sim))
    assert gm.ENABLE_TRANSITION_RULES is True and api.active_rules == {rule}
    assert parent not in rule._objects and sliced in rule._objects
    assert set(rule._objects_to_idx) == set(registry.objects)
    result = rule._compute_container_info({'container': [container]}, container, positions)
    assert not result['in_volume'].any()


def test_old_flat_snapshot_is_rejected_before_baseline_reset(world, monkeypatch):
    sim, _, _ = world
    monkeypatch.setattr(runner, '_restore_episode_baseline', lambda *args: pytest.fail('Do not reset for invalid snapshots'))
    with pytest.raises(runner.InvalidSampleState, match='unsupported state format'):
        runner.restore_materialized_sample_state(object(), {}, {'state':th.ones(3),'transition_steps':[]})
    assert not sim.loads


def test_recording_replay_materializes_named_snapshots_for_each_start(world, monkeypatch):
    sim, registry, _ = world
    monkeypatch.setattr(runner, '_validate_recorded_episode', lambda *args: None)
    rows=[]
    for value in [1,2,3]:
        for obj in registry.objects: obj._states[Legacy].value.fill_(value)
        rows.append(record_row(registry))
    samples=[{'sample_id':'first','start':0,'task':'fixture','episode_index':0}, {'sample_id':'last','start':2,'task':'fixture','episode_index':0}]
    cache={'states':th.stack(rows).numpy(), 'state_sizes':[len(rows[0])]*3,
           'transition_by_step':{}, 'transitions':{}, 'sample_evidence':{'first':{},'last':{}}}
    materialized = runner.materialize_episode_start_states(SimpleNamespace(env=SimpleNamespace(scene=sim.scenes[0])),cache,samples)
    assert sim.loads==[True,True,True]
    for sample_id,value in [('first',1),('last',3)]:
        item=materialized[sample_id]
        assert item['state_format']==runner.FULL_SNAPSHOT_FORMAT
        assert item['restoration']['full_snapshot_format']==runner.FULL_SNAPSHOT_FORMAT
        runner.restore_materialized_sample_state(object(),cache,item)
        assert all(obj._states[Legacy].value.item()==value for obj in registry.objects)


def test_capture_failure_restores_dump_filter(world, monkeypatch):
    sim, registry, _ = world
    previous=registry._dump_filter
    monkeypatch.setattr(sim,'dump_state',lambda **kwargs: (_ for _ in ()).throw(RuntimeError('capture failure')))
    with pytest.raises(RuntimeError,match='capture failure'):
        runner._dump_full_simulator_state(sim.scenes[0],sim)
    assert registry._dump_filter is previous
