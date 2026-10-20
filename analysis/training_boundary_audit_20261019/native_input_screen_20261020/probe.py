"""Compare fixed-label and freshly generated-label actions on six cached states.

These are observations visited by the parent policy, not expert states. Recorded
actions and generated labels are not treated as correct supervision. No claim
of historical RNG replay, post-terminal state, or closed-loop outcome is made.
"""
import argparse
from collections import Counter
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'flow_endpoint_screen_20261020'))
import flow_endpoint_screen as flow


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    from openpi.configs.robots import ROBOT_REGISTRY
    root=HERE.parent/'failure_receipts_20261020'
    manifest_path=root/'selected_frames_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    provenance=json.loads((root/'selected_frames_v1/range_read_provenance.json').read_text())
    assert provenance['manifest_sha256']==digest(manifest_path)
    collection_path=PROJECT/'BEHAVIOR-VLAs/eval_results/sco_full_episode_collection_20261014_final1000_round3/round-3/plan.json'
    collection=json.loads(collection_path.read_text())
    assert collection['checkpoint_identity']['checkpoint_step']==1688977
    rows=[]
    for entry,proof in zip(manifest['episodes'],provenance['episodes'],strict=True):
        assert entry['id']==proof['id']
        receipt_path=root/entry['receipt_file']
        assert digest(receipt_path)==entry['receipt_file_sha256']
        receipt=json.loads(receipt_path.read_text())
        artifact=next(a for a in proof['artifacts'] if a['file'].endswith('/frames.npz'))
        path=root/'selected_frames_v1'/artifact['file']
        assert digest(path)==artifact['sha256']
        with np.load(path,allow_pickle=False) as data:
            assert data['frame_index'].tolist()==entry['frames']
            for name,expected in proof['provenance']['selected_array_sha256'].items():
                assert hashlib.sha256(data[name].tobytes()).hexdigest()==expected
            for index,frame in enumerate(entry['frames']):
                past=[e for e in receipt['result']['subtask_events'] if e['step']<=frame]
                assert past
                label=past[-1]['subtask']
                assert isinstance(label,str) and label.strip()
                obs={'observation/state':data['observation.state'][index].copy(),
                     'prompt':collection['task_prompts'][receipt['task']]}
                assert obs['observation/state'].ndim==1 and obs['observation/state'].dtype==np.float32
                for camera,name in enumerate(ROBOT_REGISTRY['b1k/R1Pro'].observations):
                    # The live policy uses image_0, image_1, image_2 in this order.
                    key=ROBOT_REGISTRY['b1k/R1Pro'].observations[name].dataset_key
                    obs[f'observation/{name}']=data[key][index].copy()
                    assert obs[f'observation/{name}'].shape==(224,224,3)
                assert len([k for k in obs if k.startswith('observation/image_')])==3
                rows.append({'id':entry['id'],'task':receipt['task'],'instance':receipt['instance_id'],
                             'frame':frame,'recorded_label':label,'observation':obs,
                             'input_sha256':flow.ev.input_content_sha256(obs),
                             'frame_archive_sha256_verified':False,'npz_sha256':artifact['sha256']})
    assert len(rows)==6
    return rows


def make_policy(checkpoint):
    from openpi.policies import policy_config
    cache=flow.cached.VerifiedRawCache(HERE.parent/'pilot_three_tasks_cache_v1/cache_manifest.json')
    captured={};original=policy_config.create_trained_policy
    def create(cfg,path,**kwargs):
        assert not captured
        p=original(cfg,path,sample_kwargs={'num_steps':10},subtask_interval=1,
                   subtask_temperature=0.0,**kwargs)
        captured.update(policy=p,config=dataclasses.asdict(dataclasses.replace(
            cfg.model,subtask_action_conditioning=p._model.subtask_action_conditioning)))
        return p
    args=SimpleNamespace(pi05_ckpt=str(checkpoint),subtask_conditioning='provided',max_token_len=512,
         horizon=32,dataset=str(cache.paths['dataset_root']),memory_cache_dir=None,
         repo_id=cache.manifest['repo_id'],subtask_prompts=cache.paths['subtask_prompts'],
         norm_asset_id='b1k_full100_joint_subtask_v1_drop_episode_s1',action_noise_seed=None)
    with mock.patch.object(policy_config,'create_trained_policy',side_effect=create):
        flow.ev.build_pi05(args,flow.ev.read_episode_ids(cache.paths['episodes_file']),
                           sample_provider=lambda _:None)
    captured['norm_sha256']=digest(args._norm_stats_path)
    return captured


def run(checkpoint):
    import jax
    from flax import nnx
    rows=inputs()
    with flow.offline_assets():
        loaded=make_policy(checkpoint)
        policy=loaded['policy']; output=[];repeat=None
        for row in rows:
            root=jax.random.fold_in(jax.random.fold_in(jax.random.key(20261020),row['instance']),row['frame'])
            noise=np.asarray(jax.random.normal(root,(32,32)))
            for mode in ('recorded_label','fresh_generated_label'):
                observation=copy.deepcopy(row['observation'])
                if mode=='recorded_label':observation['subtask']=row['recorded_label']
                policy.reset();policy._rng=jax.random.key(0)
                before=flow.ev.input_content_sha256(observation)
                result=policy.infer(observation,noise=noise.copy())
                actions=np.asarray(result['actions'])
                assert actions.shape==(32,23) and np.isfinite(actions).all()
                assert flow.ev.input_content_sha256(observation)==before
                assert result['subtask_source']==('provided' if mode=='recorded_label' else 'generated')
                entry={k:v for k,v in row.items() if k!='observation'}
                entry.update(mode=mode,model_label=result['subtask'],subtask_source=result['subtask_source'],
                             noise_sha256=flow.ev.input_content_sha256(noise),actions=flow.numeric(actions),
                             actions_sha256=flow.ev.input_content_sha256(actions))
                if repeat is None and mode=='fresh_generated_label':
                    policy.reset();policy._rng=jax.random.key(0)
                    again=policy.infer(copy.deepcopy(observation),noise=noise.copy())
                    repeat={'label_equal':again['subtask']==result['subtask'],
                            'max_action_difference':float(np.max(np.abs(np.asarray(again['actions'])-actions)))}
                    assert repeat['label_equal'] and repeat['max_action_difference']<1e-6
                output.append(entry)
                print(json.dumps({k:entry[k] for k in ('id','frame','mode','model_label')}),flush=True)
        types=Counter(str(p.dtype) for p in jax.tree.leaves(nnx.state(policy._model,nnx.Param)))
        assert set(types)=={'bfloat16'}
        return {'kind':'cached_native_observation_label_action_screen','status':'complete',
                'checkpoint':str(checkpoint),'backend':jax.default_backend(),'model_config':loaded['config'],
                'normalization_sha256':loaded['norm_sha256'],'parameter_dtype_counts':dict(types),
                'same_model_repeat':repeat,'samples':output,
                'interpretation':'Six selected parent-policy states in two different instances of one task; '
                  'one common action-noise draw per state. Recorded labels/actions are not correctness labels. '
                  'No historical noise replay, accuracy score, feedback-stability proof or closed-loop inference.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.preflight:
        rows=inputs();print(json.dumps([{k:v for k,v in r.items() if k!='observation'} for r in rows],indent=2))
    else:
        assert args.checkpoint and args.out and not args.out.exists()
        result=run(args.checkpoint.resolve())
        with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
