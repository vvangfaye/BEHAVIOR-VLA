"""One cached observation; CPU-only support check for a specified estimator.

This is not a policy-performance experiment. No parameters are modified.
"""
import ast
import copy
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from types import MethodType, SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
PROJECT = AUDIT.parents[1]
os.sched_setaffinity(0, list(range(16)))
sys.path.insert(0, str(AUDIT/'flow_endpoint_screen_20261020'))
import flow_endpoint_screen as flow
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.shared import nnx_utils

report = {'status':'running', 'stage':'loading', 'backend':jax.default_backend(),
          'training':False, 'simulation':False, 'uploads':False,
          'scope':'Derivative support on one teacher-conditioned cached observation; no performance or full-rank claim.'}
started = time.monotonic()


def save(stage, **fields):
    report.update(stage=stage, elapsed_seconds=time.monotonic()-started, **fields)
    (HERE/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'stage':stage, 'elapsed_seconds':report['elapsed_seconds']}), flush=True)


def static_scan_version(model):
    original = model.sample_actions.__func__
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    function.name = 'sample_actions_static_scan'
    changes = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'while_loop':
                assert len(node.targets)==1
                old_target = node.targets[0]
                node.targets = [ast.Tuple(elts=[old_target,ast.Name(id='_unused_scan_outputs',ctx=ast.Store())],ctx=ast.Store())]
                node.value = ast.parse('jax.lax.scan(lambda carry, unused: (step(carry), None), (noise, 1.0), None, length=num_steps)',mode='eval').body
                changes += 1
    assert changes==1
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, '<local-static-scan-autodiff-check>', 'exec'), namespace)
    transformed = ast.unparse(tree)
    (HERE/'transformed_sampler.txt').write_text(transformed+'\n')
    report['sampler_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    report['transformed_sampler_sha256'] = hashlib.sha256(transformed.encode()).hexdigest()
    return MethodType(namespace[function.name], model)


def main():
    cache = flow.cached.VerifiedRawCache(AUDIT/'pilot_three_tasks_cache_v1/cache_manifest.json')
    samples, episodes = flow.selected_samples(cache,1,'prefix')
    spec = json.loads((AUDIT/'native_expert_label_screen_20261020/plan.json').read_text())['runs']['baseline1688977']
    checkpoint = Path(spec['checkpoint'])
    captured = {}
    original = policy_config.create_trained_policy
    def create(cfg, ckpt, **kwargs):
        policy = original(cfg, ckpt, sample_kwargs={'num_steps':10}, **kwargs)
        captured['policy'] = policy
        return policy
    args = SimpleNamespace(pi05_ckpt=str(checkpoint),subtask_conditioning='provided',
        max_token_len=512,horizon=32,dataset=str(cache.paths['dataset_root']),memory_cache_dir=None,
        repo_id=cache.manifest['repo_id'],subtask_prompts=cache.paths['subtask_prompts'],
        norm_asset_id='b1k_full100_joint_subtask_v1_drop_episode_s1',action_noise_seed=None)
    with flow.offline_assets():
        with mock.patch.object(policy_config,'create_trained_policy',side_effect=create):
            _, get_sample = flow.ev.build_pi05(args,episodes,sample_provider=cache)
        policy = captured['policy']
        item = get_sample(samples[0])
        prepared = policy._pad(policy._tokenizer_transform(policy._observation_transform(copy.deepcopy(item['policy_input']))))
        observation = model_lib.Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None],prepared))
        model = policy._model
        native = nnx_utils.module_jit(model.sample_actions,static_argnames=('num_steps',))
        scanned = nnx_utils.module_jit(static_scan_version(model),static_argnames=('num_steps',))
        key = jax.random.key(20261020)
        noise = jax.random.normal(key,(1,model.action_horizon,model.action_dim))
        report.update(checkpoint=str(checkpoint),noise_shape=list(noise.shape),
            observation_hash=flow.ev.input_content_sha256(observation.to_dict()),
            policy_input_hash=item['policy_input_sha256'],
            episode_index=samples[0].episode_index,frame_index=samples[0].frame_index,
            conditioning='provided expert subtask')
        save('native_forward')
        a = native(key,observation,num_steps=10,noise=noise)
        a.block_until_ready()
        save('static_scan_forward')
        b = scanned(key,observation,num_steps=10,noise=noise)
        b.block_until_ready()
        left,right = np.asarray(a),np.asarray(b)
        parity = {'max_abs_difference':float(np.max(np.abs(left-right))),
                  'exact_equal':bool(np.array_equal(left,right)),
                  'allclose_rtol1e5_atol1e6':bool(np.allclose(left,right,rtol=1e-5,atol=1e-6))}
        save('parity_checked',parity=parity)
        if not parity['allclose_rtol1e5_atol1e6']:
            save('stopped_on_parity_mismatch',status='inconclusive')
            return
        f = lambda n:scanned(key,observation,num_steps=10,noise=n)[:,:16,:23].reshape(-1)
        direction=jax.random.normal(jax.random.fold_in(key,1),noise.shape)
        cotangent=jax.random.normal(jax.random.fold_in(key,2),(368,))
        save('jvp_compiling')
        t=time.monotonic()
        out,jv=jax.jvp(f,(noise,),(direction,))
        jv.block_until_ready()
        save('jvp_complete',jvp={'seconds':time.monotonic()-t,'norm':float(jnp.linalg.norm(jv)),
                                'finite':bool(jnp.isfinite(jv).all())})
        t=time.monotonic()
        save('vjp_compiling')
        _,pullback=jax.vjp(f,noise)
        jtw=pullback(cotangent)[0]
        jtw.block_until_ready()
        lhs=float(jnp.vdot(jv,cotangent));rhs=float(jnp.vdot(direction,jtw))
        save('vjp_complete',vjp={'seconds':time.monotonic()-t,'norm':float(jnp.linalg.norm(jtw)),
            'finite':bool(jnp.isfinite(jtw).all()),'adjoint_lhs':lhs,'adjoint_rhs':rhs,
            'adjoint_relative_error':abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)})
        save('hvp_compiling')
        t=time.monotonic()
        grad = jax.grad(lambda n:jnp.vdot(f(n),cotangent))
        _,hvp=jax.jvp(grad,(noise,),(direction,))
        hvp.block_until_ready()
        save('finished',status='complete',hvp={'seconds':time.monotonic()-t,
            'norm':float(jnp.linalg.norm(hvp)),'finite':bool(jnp.isfinite(hvp).all())})


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        save('exception',status='failed',exception_type=type(exc).__name__,exception=str(exc))
        raise
