"""Full reverse-mode noise Jacobian at one already-cached expert observation.

Measures numerical conditioning, not on-policy outcomes or estimator variance.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from unittest import mock

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent
os.sched_setaffinity(0,list(range(16)))
spec=importlib.util.spec_from_file_location('native_ad_support',AUDIT/'native_flow_autodiff_probe_20261020/probe.py')
support=importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
# The imported helper writes transformed source; direct it to this new run only.
support.HERE=HERE
jax,jnp,np,flow=support.jax,support.jnp,support.np,support.flow
from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from scipy.sparse.linalg import cg

report={'status':'running','stage':'loading','training':False,'simulation':False,'uploads':False,
        'scope':'One teacher-conditioned observation, reverse-mode numerical Jacobian of normalized prefix actions; not population rank, VLA performance, or policy-gradient variance.'}
started=time.monotonic()


def save(stage,**fields):
    report.update(stage=stage,elapsed_seconds=time.monotonic()-started,**fields)
    (HERE/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'stage':stage,'elapsed_seconds':report['elapsed_seconds'],
                      'rows':report.get('rows_computed',0)}),flush=True)


def main():
    cache=flow.cached.VerifiedRawCache(AUDIT/'pilot_three_tasks_cache_v1/cache_manifest.json')
    samples,episodes=flow.selected_samples(cache,1,'prefix')
    checkpoint=json.loads((AUDIT/'native_expert_label_screen_20261020/plan.json').read_text())['runs']['baseline1688977']['checkpoint']
    original=policy_config.create_trained_policy
    captured={}
    def create(cfg,ckpt,**kwargs):
        p=original(cfg,ckpt,sample_kwargs={'num_steps':10},**kwargs)
        captured['policy']=p
        return p
    args=SimpleNamespace(pi05_ckpt=checkpoint,subtask_conditioning='provided',max_token_len=512,
        horizon=32,dataset=str(cache.paths['dataset_root']),memory_cache_dir=None,
        repo_id=cache.manifest['repo_id'],subtask_prompts=cache.paths['subtask_prompts'],
        norm_asset_id='b1k_full100_joint_subtask_v1_drop_episode_s1',action_noise_seed=None)
    with flow.offline_assets():
        with mock.patch.object(policy_config,'create_trained_policy',side_effect=create):
            _,get_sample=flow.ev.build_pi05(args,episodes,sample_provider=cache)
        policy=captured['policy']
        item=get_sample(samples[0])
        prepared=policy._pad(policy._tokenizer_transform(policy._observation_transform(copy.deepcopy(item['policy_input']))))
        observation=model_lib.Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None],prepared))
        model=policy._model
        scanned=nnx_utils.module_jit(support.static_scan_version(model),static_argnames=('num_steps',))
        key=jax.random.key(20261020)
        noise=jax.random.normal(key,(1,model.action_horizon,model.action_dim))
        f=lambda n:scanned(key,observation,num_steps=10,noise=n)[:,:16,:23].reshape(-1)
        report.update(checkpoint=checkpoint,backend=jax.default_backend(),
            observation_hash=flow.ev.input_content_sha256(observation.to_dict()),
            policy_input_hash=item['policy_input_sha256'],
            sampler_source_sha256=support.report['sampler_source_sha256'],
            transformed_sampler_sha256=support.report['transformed_sampler_sha256'],
            parameter_storage='bfloat16',jacobian_arithmetic='native autodiff then host float64 SVD',
            episode_index=samples[0].episode_index,frame_index=samples[0].frame_index)
        previous=json.loads((AUDIT/'native_flow_autodiff_probe_20261020/result.json').read_text())
        for field in ['observation_hash','policy_input_hash','sampler_source_sha256','transformed_sampler_sha256']:
            assert report[field]==previous[field]
        assert previous['parity']['exact_equal']
        save('linearize_reverse')
        output,pullback=jax.vjp(f,noise)
        output.block_until_ready()
        np.save(HERE/'normalized_prefix.npy',np.asarray(output))
        m=output.size
        block_size=16
        apply=jax.jit(jax.vmap(lambda u:pullback(u)[0]))
        rows=[]
        batch_seconds=[]
        for start in range(0,m,block_size):
            idx=jnp.arange(start,start+block_size)
            basis=jax.nn.one_hot(idx,m,dtype=jnp.float32)
            t=time.monotonic()
            block=apply(basis)
            block.block_until_ready()
            batch_seconds.append(time.monotonic()-t)
            valid=min(block_size,m-start)
            rows.append(np.asarray(block[:valid]).reshape(valid,-1))
            save('jacobian_rows',rows_computed=start+valid,last_batch_seconds=batch_seconds[-1])
        jac=np.concatenate(rows,axis=0).astype(np.float64)
        assert jac.shape==(368,1024) and np.isfinite(jac).all()
        np.save(HERE/'reverse_jacobian.npy',jac)
        save('svd',jacobian_shape=list(jac.shape),batch_seconds=batch_seconds)
        singular=np.linalg.svd(jac,compute_uv=False)
        np.save(HERE/'singular_values.npy',singular)
        largest,smallest=float(singular[0]),float(singular[-1])
        spectrum={'largest':largest,'smallest':smallest,
                  'condition_number':largest/smallest if smallest>0 else None,
                  'quantiles':{str(q):float(np.quantile(singular,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]},
                  'ranks_at_relative_threshold':{str(t):int(np.sum(singular>t*largest)) for t in [.01,.001,.0001,.00001,.000001]}}
        # This is host algebra for the measured Jacobian, not native matrix-free timing.
        c=jac@jac.T
        rng=np.random.default_rng(20261020)
        solves=[]
        for _ in range(3):
            rhs=rng.normal(size=m)
            count=[0]
            def step(_): count[0]+=1
            t=time.monotonic()
            solution,info=cg(c,rhs,rtol=1e-8,atol=0,maxiter=2000,callback=step)
            solves.append({'iterations':count[0],'info':int(info),
                'relative_residual':float(np.linalg.norm(c@solution-rhs)/np.linalg.norm(rhs)),
                'lift_amplification':float(np.linalg.norm(jac.T@solution)/np.linalg.norm(rhs)),
                'host_solve_seconds':time.monotonic()-t})
        save('finished',status='complete',spectrum=spectrum,normal_equation_solves=solves,
             peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
             limitations=['One cached expert state and one noise; no across-state inference.',
                'Bfloat16 automatic derivatives have finite precision; this is not a certified lower bound on singular values.',
                'Host CG timing excludes repeated native JVP/VJP and cannot estimate training throughput.',
                'No third-solve derivative or parameter-gradient variance measured.'])


if __name__=='__main__':
    try: main()
    except Exception as exc:
        save('exception',status='failed',exception_type=type(exc).__name__,exception=str(exc))
        raise
