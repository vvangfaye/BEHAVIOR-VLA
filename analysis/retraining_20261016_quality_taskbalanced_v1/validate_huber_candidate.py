"""CPU JAX checks for robust residuals and the actual Pi0 loss integration."""
import ast
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'expert_huber_candidate/bundle'
SOURCE = ROOT / 'expert_control/formal_stage/bundle'


def function(path, name, owner=None, namespace=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body if owner is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = copy.deepcopy(next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node]
    scope = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), scope)
    return scope[name]


def main():
    global BUNDLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'expert_huber_candidate')
    candidate = parser.parse_args().candidate.resolve()
    BUNDLE = candidate / 'bundle'
    contract = json.loads((BUNDLE / 'inputs/formal_run_contract.json').read_text())['action_loss']
    if contract['kind'] != 'scaled_huber':
        raise ValueError('Expected a robust-loss candidate')
    delta_value = float(contract['delta'])
    if jax.default_backend() != 'cpu':
        raise ValueError('This validation must not allocate a GPU')
    path = BUNDLE / 'openpi/src/openpi/shared/action_loss.py'
    spec = importlib.util.spec_from_file_location('candidate_action_loss', path)
    loss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loss)
    values = jnp.array([-585.684, -delta_value, -1., 0., 1., delta_value, 298.869], dtype=jnp.float32)
    assert np.array_equal(np.asarray(loss.squared_or_huber(values)), np.asarray(jnp.square(values)))
    small = jnp.linspace(-delta_value, delta_value, 1001)
    np.testing.assert_array_equal(loss.squared_or_huber(small, delta_value), jnp.square(small))
    objective = jax.jit(lambda x: loss.squared_or_huber(x, delta_value))
    actual = np.asarray(objective(values))
    expected = np.where(np.abs(np.asarray(values)) <= delta_value, np.asarray(values)**2,
                        2*delta_value*np.abs(np.asarray(values))-delta_value**2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(loss.squared_or_huber(x, delta_value))))(values)
    np.testing.assert_allclose(gradient, 2*jnp.clip(values, -delta_value, delta_value), rtol=1e-6)
    for delta in (0, -1, float('inf'), float('nan'), True):
        try:
            loss.validate_delta(delta)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid delta accepted')

    model_path = BUNDLE / 'openpi/src/openpi/models/pi0.py'
    namespace = {'jax': jax, 'jnp': jnp, '_action_loss': loss,
                 '_model': SimpleNamespace(preprocess_observation=lambda rng, obs, train: obs)}
    namespace['make_attn_mask'] = function(model_path, 'make_attn_mask', namespace=namespace)
    compute = function(model_path, 'compute_loss_with_metrics', 'Pi0', namespace)
    shape, key = (2, 32, 32), jax.random.key(19)
    actions = jnp.zeros(shape)
    residual = jnp.linspace(-4, 4, math.prod(shape)).reshape(shape)
    residual = residual.at[0, 0, 6].set(585.684).at[1, 1, 6].set(-298.869)
    noise_key = jax.random.split(key, 3)[1]
    velocity = jax.random.normal(noise_key, shape) + residual
    observation = SimpleNamespace(token_ar_mask=jnp.ones((2, 2), dtype=bool),
        token_loss_mask=jnp.ones((2, 2), dtype=bool), has_subtask_target=jnp.array([True, False]),
        source_kind_id=jnp.array([0, 3]))

    class Stub:
        train_subtask_prediction = True
        subtask_action_conditioning = 'task_and_subtask'
        action_horizon = 32
        subtask_loss_weight = 1.0
        def __init__(self, delta):
            self.action_loss_huber_delta = delta
            self.PaliGemma = SimpleNamespace(llm=lambda *a, **kw: ((jnp.zeros((2, 2, 1)), jnp.zeros((2, 32, 1))), None))
        def embed_prefix(self, *a, **kw):
            return jnp.zeros((2, 2, 1)), jnp.ones((2, 2), bool), jnp.zeros((2, 2), bool)
        def embed_suffix(self, *a):
            return jnp.zeros((2, 32, 1)), jnp.ones((2, 32), bool), jnp.ones((32,), bool), None
        def embed_memory(self, *a): return {}
        def action_out_proj(self, *a): return velocity
        def _compute_subtask_metrics(self, *a):
            return jnp.array([.2, 0.]), jnp.array([1., 0.]), jnp.array([3, 0])

    results = {}
    for delta in (None, delta_value):
        stub = Stub(delta)
        total, metrics = jax.jit(lambda x: compute(stub, key, observation, x, train=True))(actions)
        expected_action = jnp.mean(loss.squared_or_huber(residual, delta), axis=-1)
        np.testing.assert_allclose(total, expected_action + jnp.array([.4, 0.])[:, None], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(metrics['action_loss'], jnp.mean(expected_action), rtol=1e-5)
        np.testing.assert_allclose(metrics['subtask_loss'], .2, rtol=1e-5)
        assert float(metrics['source_count/replay']) == float(metrics['source_count/full_episode']) == 1
        if delta is not None:
            np.testing.assert_allclose(metrics['action_mse'], jnp.mean(residual**2), rtol=1e-5)
            for code, kind in [(0, 'replay'), (1, 'full_episode')]:
                np.testing.assert_allclose(metrics['action_mse/'+kind], jnp.mean(residual[code]**2), rtol=1e-5)
        results[str(delta)] = float(metrics['action_loss'])

    config_path = BUNDLE / 'openpi/src/openpi/models/pi0_config.py'
    post = function(config_path, '__post_init__', 'Pi0Config', {'math': math, '_action_loss': loss})
    settings = dict(framesamp_modul=False, subtask_action_conditioning='task_and_subtask', max_token_len=512,
        discrete_state_input=True, train_subtask_prediction=True, pi05=True, subtask_loss_weight=1.,
        max_subtask_len=64, subtask_loss_chunk_size=8, pytorch_compile_mode=None)
    post(SimpleNamespace(**settings, action_loss_huber_delta=delta_value))
    for delta in (0, -1, float('inf'), float('nan')):
        try: post(SimpleNamespace(**settings, action_loss_huber_delta=delta))
        except ValueError: pass
        else: raise AssertionError('Configuration admitted an invalid loss')

    trainer = ast.parse((BUNDLE / 'openpi/scripts/b1k/train_b1k.py').read_text())
    assignment = next(n for n in ast.walk(trainer) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'loss_delta' for t in n.targets))
    guard = next(n for n in ast.walk(trainer) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                 and any(isinstance(x, ast.Constant) and x.value == 'Action loss differs from the frozen formal contract' for x in ast.walk(n)))
    code = compile(ast.fix_missing_locations(ast.Module(body=[assignment, guard], type_ignores=[])), '<actual trainer loss guard>', 'exec')
    for delta, contract, allowed in [(delta_value, {'action_loss': loss.contract(delta_value)}, True), (delta_value, {}, False),
                                     (None, {}, True), (None, {'action_loss': loss.contract(delta_value)}, False),
                                     (delta_value, {'action_loss': loss.contract(delta_value + 1)}, False)]:
        scope = {'config': SimpleNamespace(model=SimpleNamespace(action_loss_huber_delta=delta)),
                 'data_loader': SimpleNamespace(formal_run_contract=contract), '_action_loss': loss}
        try: exec(code, scope)
        except ValueError:
            if allowed: raise
        else:
            if not allowed: raise AssertionError('Trainer admitted a mismatched loss contract')

    old_model = SOURCE / 'openpi/src/openpi/models/pi0.py'
    def node(path, name):
        tree = ast.parse(path.read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Pi0')
        return copy.deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name))
    init = node(model_path, '__init__')
    init.body = [n for n in init.body if not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == 'action_loss_huber_delta')]
    assert ast.dump(init, include_attributes=False) == ast.dump(node(old_model, '__init__'), include_attributes=False)
    assert ast.dump(node(model_path, 'sample_actions'), include_attributes=False) == ast.dump(node(old_model, 'sample_actions'), include_attributes=False)
    result = {'passed': True, 'delta': delta_value, 'jax_version': jax.__version__, 'backend': jax.default_backend(),
        'mse_default_exact': True, 'small_error_loss_unchanged': True, 'gradient_bound_verified': True,
        'actual_pi0_loss_method_jit_with_stub_backbone': True, 'language_loss_and_source_counts_unchanged': True,
        'raw_mse_metrics_verified': True, 'formal_loss_contract_guard_verified': True,
        'parameter_constructors_and_sampling_method_unchanged': True,
        'full_model_forward_or_gpu_training_tested': False, 'example_objectives': results,
        'loss_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cloud_jobs_created': 0}
    (candidate / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
