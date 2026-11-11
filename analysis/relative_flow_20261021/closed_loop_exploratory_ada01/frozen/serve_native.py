"""Serve completed weights for the separate exploratory task-outcome study.

This does not reinterpret or bypass the failed offline promotion decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from native_policy import DeployedAdapter,RawPolicy
from vla_runtime import load_base
from vla_native_policy import SeededNativeWrapper,acknowledged_server_class


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['base-checkpoint','tokenizer','trained-weights','native-code','experiment-plan']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=['extra_fm','relative_difference'],required=True)
    p.add_argument('--task',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--server-token',required=True);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    plan=json.loads(args.experiment_plan.read_text())
    if plan['protocol']['study_type']!='exploratory_native_after_offline_failure':
        raise ValueError('Separate task-outcome study plan required')
    reference=plan['trained_checkpoints'][args.method]
    if args.trained_weights.resolve()!=Path(reference['path']).resolve() or sha(args.trained_weights)!=reference['sha256']:
        raise ValueError('Weights differ from the frozen task-outcome study')
    if (args.base_checkpoint.resolve()!=Path(plan['checkpoint']).resolve()
            or args.tokenizer.resolve()!=Path(plan['tokenizer']).resolve()):
        raise ValueError('Base checkpoint or tokenizer mismatch')
    if not 1000<=args.port<65536 or args.seed!=0:raise ValueError('Unexpected port or seed')
    saved=torch.load(args.trained_weights,map_location='cpu',weights_only=False)
    if not(saved['complete'] and saved['step']==1000 and saved['seed']==17
           and saved['method']==args.method and saved['plan_sha256']==plan['training_plan_sha256']):
        raise ValueError('Incomplete or mismatched trained checkpoint')
    torch.set_num_threads(4)
    base=load_base(args.base_checkpoint,args.tokenizer)
    adapter=DeployedAdapter(base).cuda().eval()
    expected={n for n,p in adapter.named_parameters() if p.requires_grad}
    if set(saved['weights'])!=expected:raise ValueError('Parameter schema differs')
    loaded=adapter.load_state_dict(saved['weights'],strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys)&expected:raise ValueError('Incomplete parameter load')
    del saved
    sys.path.insert(0,str(args.native_code/'openpi/src'))
    sys.path.insert(0,str(args.native_code/'openpi/packages/openpi-client/src'))
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.serving import websocket_b1k_server
    from openpi.configs.tasks import TASK_REGISTRY
    bucket,task=args.task.split('/',1)
    raw=RawPolicy(adapter,args.method,seed=0)
    native=B1KPolicyWrapper(raw,'b1k/R1Pro',TASK_REGISTRY[bucket][task],
                            'receding_horizon',action_horizon=16,max_len=32)
    wrapper=SeededNativeWrapper(native,raw)
    server=acknowledged_server_class(websocket_b1k_server)(wrapper,host='127.0.0.1',port=args.port,
        server_token=args.server_token,metadata={**raw.metadata,'trained':True,
            'experimental_method':args.method,'study_type':plan['protocol']['study_type']})
    server.serve_forever()


if __name__=='__main__':main()
