"""Native server, gated on the independently scored fixed offline pilot.

Preparing this entry point does not submit or launch any simulator run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from vla_runtime import load_base
from posterior_support_native import DeployedPosteriorAdapter,PosteriorRawPolicy
from vla_native_policy import SeededNativeWrapper,acknowledged_server_class


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def verify_gate(pilot,method):
    if method not in ('ambient_aux','posterior'):
        raise ValueError('Unknown method')
    report=json.loads((pilot/'reanalysis.json').read_text())
    if report.get('ready') is not True or report.get('screen_passed') is not True:
        raise ValueError('The frozen offline gate has not passed')
    plan=pilot/'plan.json'
    if digest(plan)!=report['plan_sha256']:
        raise ValueError('Training plan does not match scored pilot')
    if digest(pilot/'analysis_plan.json')!=report['analysis_plan_sha256']:
        raise ValueError('Analysis plan changed after scoring')
    folder=pilot/('formal_'+method)
    sources=report['source_hashes'][method]
    if set(sources)!={'test.npz','test.json','result.json','completion.json'}:
        raise ValueError('Incomplete scored source manifest')
    for name,expected in sources.items():
        if digest(folder/name)!=expected:
            raise ValueError('Scored source changed: '+name)
    contribution=json.loads((pilot/'training_contribution_result.json').read_text())
    if any(contribution.get(key) is not True for key in
           ('ready','primary_gate_passed','posthoc_gate_passed','training_contribution_gate_passed')):
        raise ValueError('The training-versus-posthoc gate has not passed')
    if contribution['source_primary_report_sha256']!=digest(pilot/'reanalysis.json'):
        raise ValueError('Training contribution refers to a different primary analysis')
    if contribution['gate_plan_sha256']!=digest(pilot/'training_contribution_gate_plan.json'):
        raise ValueError('Training contribution gate plan changed after scoring')
    return folder/'checkpoint.pt',report['plan_sha256']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--method',choices=['ambient_aux','posterior'],required=True)
    p.add_argument('--native-code',type=Path,required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--server-token',required=True)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    if not 1000<=args.port<65536:raise ValueError('Use a nonprivileged port')
    weights,plan_hash=verify_gate(args.pilot_root,args.method)
    training_plan=json.loads((args.pilot_root/'plan.json').read_text())
    if args.tokenizer.resolve()!=Path(training_plan['tokenizer']).resolve():
        raise ValueError('Tokenizer differs from the frozen training plan')
    if args.base_checkpoint.resolve()!=Path(training_plan['checkpoint']).resolve():
        raise ValueError('Base checkpoint differs from the frozen training plan')
    saved=torch.load(weights,map_location='cpu',weights_only=False)
    if not(saved['complete'] and saved['step']==1000 and saved['seed']==17
           and saved['method']==args.method and saved['plan_sha256']==plan_hash):
        raise ValueError('Checkpoint does not match the completed fixed pilot')
    if Path(saved['base_checkpoint']).resolve()!=args.base_checkpoint.resolve():
        raise ValueError('Base checkpoint differs from training')
    torch.set_num_threads(4)
    base=load_base(args.base_checkpoint,args.tokenizer,device='cuda',precision='bf16')
    adapter=DeployedPosteriorAdapter(base).cuda().eval()
    expected={n for n,p in adapter.named_parameters() if p.requires_grad}
    if set(saved['weights'])!=expected:raise ValueError('Trainable parameter schema differs')
    result=adapter.load_state_dict(saved['weights'],strict=False)
    if result.unexpected_keys or set(result.missing_keys)&expected:
        raise ValueError('Incomplete trainable state')
    del saved
    sys.path.insert(0,str(args.native_code/'openpi/src'))
    sys.path.insert(0,str(args.native_code/'openpi/packages/openpi-client/src'))
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.serving import websocket_b1k_server
    from openpi.configs.tasks import TASK_REGISTRY
    bucket,task=args.task.split('/',1)
    raw=PosteriorRawPolicy(adapter,args.method,seed=args.seed)
    wrapped=B1KPolicyWrapper(raw,'b1k/R1Pro',TASK_REGISTRY[bucket][task],
                             'receding_horizon',action_horizon=16,max_len=32)
    wrapped=SeededNativeWrapper(wrapped,raw)
    server=acknowledged_server_class(websocket_b1k_server)(
        wrapped,host='127.0.0.1',port=args.port,server_token=args.server_token,
        metadata={**raw.metadata,'trained':True,'experimental_method':args.method,
                  'training_plan_sha256':plan_hash})
    server.serve_forever()


if __name__=='__main__':main()
