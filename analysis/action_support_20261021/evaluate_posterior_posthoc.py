"""Replay a completed control, then apply the same analytic mixer at inference.

No training and no simulator. Separate artifact from the frozen two-arm pilot.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from vla_runtime import load_base
from vla_data import VlaPilotData,digest
from posterior_support_adapter import PosteriorSupportAdapter
from train_posterior_support import evaluate,atomic_json


class TracedAdapter(PosteriorSupportAdapter):
    def sample(self,observation,*,noise,steps=10,rng=None,mode='posterior'):
        self.noise_hashes.append(hashlib.sha256(noise.detach().float().cpu().numpy().tobytes()).hexdigest())
        return super().sample(observation,noise=noise,steps=steps,rng=rng,mode=mode)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    here=Path(__file__).resolve().parent
    for name,expected in plan['source_sha256'].items():
        if digest(here/name)!=expected:raise ValueError('Frozen source changed: '+name)
    original=Path(plan['source_directory'])
    completion=json.loads((original/'completion.json').read_text())
    if not completion['accepted'] or completion['returncode']!=0:
        raise ValueError('Control training is not accepted')
    weights=original/'checkpoint.pt'
    if digest(weights)!=plan['checkpoint_sha256']:
        raise ValueError('Control weights changed')
    saved=torch.load(weights,map_location='cpu',weights_only=False)
    if not(saved['complete'] and saved['step']==1000 and saved['method']=='ambient_aux'
           and saved['seed']==17 and saved['plan_sha256']==plan['training_plan_sha256']):
        raise ValueError('Unexpected control checkpoint')
    torch.set_num_threads(4);torch.manual_seed(17)
    base=load_base(plan['base_checkpoint'],plan['tokenizer'],device='cuda',precision='bf16')
    adapter=TracedAdapter(base).cuda().eval()
    expected={name for name,p in adapter.named_parameters() if p.requires_grad}
    if set(saved['weights'])!=expected:raise ValueError('Trainable state schema differs')
    loaded=adapter.load_state_dict(saved['weights'],strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys)&expected:
        raise ValueError('Incomplete control state')
    del saved
    data=VlaPilotData(plan['data_directory'],base)
    with np.load(original/'test.npz',allow_pickle=False) as z:
        reference={k:z[k].copy() for k in ['index','episode','frame','target','samples']}
    output=Path(plan['output_directory'])
    autocast=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    adapter.noise_hashes=[]
    replay=evaluate(adapter,data,reference['index'],'ambient_aux',output/'replay.json',
                    draws=4,steps=10,batch=1,autocast=autocast)
    normal_noise=adapter.noise_hashes.copy()
    with np.load(output/'replay.npz',allow_pickle=False) as z:
        for key in ['index','episode','frame','target']:
            np.testing.assert_array_equal(z[key],reference[key])
        max_difference=float(np.abs(z['samples']-reference['samples']).max())
    if max_difference>2e-5:
        raise ValueError('Control replay does not match the completed original: '+str(max_difference))
    adapter.noise_hashes=[]
    posthoc=evaluate(adapter,data,reference['index'],'posterior',output/'posthoc.json',
                     draws=4,steps=10,batch=1,autocast=autocast)
    if adapter.noise_hashes!=normal_noise:raise ValueError('Gaussian noise streams differ')
    with np.load(output/'posthoc.npz',allow_pickle=False) as z:
        assert np.isfinite(z['samples']).all()
        for key in ['index','episode','frame','target']:
            np.testing.assert_array_equal(z[key],reference[key])
    atomic_json(output/'noise_hashes.json',dict(replay=normal_noise,posthoc=adapter.noise_hashes))
    result=dict(complete=True,observed_at=datetime.now(timezone.utc).isoformat(),
                plan_sha256=digest(args.plan),source_method='ambient_aux',
                source_checkpoint_sha256=plan['checkpoint_sha256'],
                source_training_plan_sha256=plan['training_plan_sha256'],
                replay_max_abs_difference=max_difference,noise_streams_equal=True,
                replay=replay,posthoc=posthoc,
                scope='Identical trained control with and without inference-only posterior mixing; offline expert-subtask observations only.')
    atomic_json(output/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
