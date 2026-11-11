"""Finite phase worker: collect, update once, or evaluate. No GPU waits on peers."""
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import tarfile
import time
import traceback
import urllib.request

import torch

from common import ExecutionStopped, torch_model, write
from eval_utils import paired_report
from experiment import collect, evaluate, seed_all, enforce_storage_budget
from training import actor_parameters, build_optimizer, hash_parameters, load_overlay, save_checkpoint, update_policy


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def upload(url, value):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url,data=value,method='PUT'),timeout=30):pass


def policy_hash(model):
    return hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)


def main():
    plan=json.loads(Path(__file__).with_name('plan.json').read_text())
    phase=os.environ['B1K_PHASE'];rank=int(os.environ['B1K_RANK']);attempt=int(os.environ['B1K_ATTEMPT'])
    root=Path(plan['output']);root.mkdir(parents=True,exist_ok=True)
    task=f'{phase}_r{rank}_a{attempt}'
    state={'state':'running','phase':phase,'rank':rank,'attempt':attempt,'task':task,
           'stage':'loading','started_at':time.time(),'new_optimizer_steps':0}
    status_file=root/'worker_status'/f'{task}.json'
    def publish(**fields):
        state.update(fields,updated_at=time.time());write(status_file,state)
        try:upload(os.environ['B1K_STATUS_PUT'],json.dumps(state).encode())
        except Exception as error:print('Status upload: '+type(error).__name__,flush=True)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    def stopped(*_):raise ExecutionStopped('Phase budget exhausted or stopped; preserve results')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGALRM,stopped)
    signal.alarm(plan['phase_limits_seconds'][phase]-600)
    success=False
    try:
        seed_all(plan['model_seed']);model=torch_model(plan)
        assert policy_hash(model)==plan['expected_original_trainable_sha256']
        frozen=hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        assert frozen==plan['reuse']['frozen_hash']
        actor_before=hash_parameters(actor_parameters(model))
        if phase in ('sample','train_eval'):
            load_overlay(model,plan['reuse']['initial_policy_directory'],plan['reuse']['initial_policy_hash'])
            assert hash_parameters(actor_parameters(model))==actor_before
        write(root/'identities'/f'{task}.json',{'actor_unchanged_at_start':True,'policy_hash':policy_hash(model),'frozen_hash':frozen})
        contract=hashlib.sha256(json.dumps(plan['mid_reward'],sort_keys=True).encode()).hexdigest()
        if phase=='sample':
            for index in range(rank,32,2):
                receipt_path=root/'training_receipts'/f'episode_{index+1:02d}.json'
                path=root/'training/round_01'/f'episode_{index+1:02d}/rollout.pt'
                if receipt_path.exists():
                    receipt=json.loads(receipt_path.read_text());assert sha(path)==receipt['sha256']
                    assert receipt['policy_hash']==policy_hash(model) and receipt['contract']==contract
                    publish(stage='reuse_completed',episode=index+1)
                    continue
                data=None
                if path.exists():
                    try:
                        candidate=torch.load(path,map_location='cpu',weights_only=True)
                        assert candidate['mid_reward_contract_sha256']==contract
                        assert candidate['instance']==plan['train_instances'][index]
                        assert candidate['seed']==plan['train_seeds'][0][index]
                        assert candidate['status']==json.loads((path.parent/'result.json').read_text()) or candidate['status']=={k:v for k,v in json.loads((path.parent/'result.json').read_text()).items() if k not in ('instance','seed')}
                        data=candidate
                    except Exception:data=None
                if data is None:
                    if path.parent.exists():
                        archive=root/'incomplete_attempts'/f'episode_{index+1:02d}_before_{task}'
                        archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();path.parent.rename(archive)
                    publish(stage='training_rollout',episode=index+1,instance=plan['train_instances'][index],seed=plan['train_seeds'][0][index])
                    data=collect(model,plan,1,index,publish)
                assert data['mid_reward_contract_sha256']==contract
                assert abs(sum(x['reward']-x['task_reward']-x['mid_reward'] for x in data['transitions']))<1e-4
                mid=sum(x['mid_reward'] for x in data['transitions']);assert 0<=mid<=1.5+1e-6
                summary=json.loads((path.parent/'rollout/reward_summary.json').read_text())
                assert abs(summary['total_mid_reward']-mid)<1e-5
                if data['status']['success']:
                    assert summary['initially_satisfied']['finger'] or any(e['milestone']=='finger' for e in summary['events']), 'Successful radio toggle without verified finger milestone'
                assert policy_hash(model)==plan['reuse']['initial_policy_hash']
                receipt={'episode':index+1,'instance':data['instance'],'seed':data['seed'],'status':data['status'],
                         'policy_hash':policy_hash(model),'contract':contract,'path':str(path),'sha256':sha(path),
                         'mid_reward':mid,'reward_summary':summary,'source_task':task}
                write(receipt_path,receipt)
                rows=[json.loads(p.read_text()) for p in sorted((root/'training_receipts').glob('episode_*.json'))]
                publish(stage='training_rollout_complete',episode=index+1,collected=len(rows),
                        successes=sum(r['status']['success'] for r in rows),mid_reward_events=sum(len(r['reward_summary']['events']) for r in rows))
                del data;gc.collect()
            own=[json.loads((root/'training_receipts'/f'episode_{i+1:02d}.json').read_text()) for i in range(rank,32,2)]
            publish(state='complete',stage='collection_complete',collection=own)
            success=True
        else:
            checkpoint=root/'checkpoints/round_01'
            if phase=='train_eval':
                receipts=[json.loads((root/'training_receipts'/f'episode_{i+1:02d}.json').read_text()) for i in range(32)]
                assert all(r['contract']==contract and r['policy_hash']==policy_hash(model) for r in receipts)
                assert sum(r['mid_reward'] for r in receipts)>0, 'No effective middle rewards; do not train a mislabeled control'
                assert any(any(e['milestone']=='finger' for e in r['reward_summary']['events']) for r in receipts), 'No verified finger milestone in the collection'
                if (checkpoint/'ready.json').exists():
                    ready=json.loads((checkpoint/'ready.json').read_text());assert ready['optimizer_steps']==17
                    load_overlay(model,str(checkpoint),ready['trainable_sha256'])
                else:
                    if checkpoint.exists():
                        archive=root/'incomplete_attempts'/f'checkpoint_before_{task}'
                        archive.parent.mkdir(parents=True,exist_ok=True);checkpoint.rename(archive)
                    episodes=[]
                    for i,r in enumerate(receipts):
                        assert sha(r['path'])==r['sha256']
                        data=torch.load(r['path'],map_location='cpu',weights_only=True)
                        assert data['instance']==plan['train_instances'][i] and data['seed']==plan['train_seeds'][0][i]
                        episodes.append(data)
                    optimizer=build_optimizer(model,plan)
                    publish(stage='training_update',training_episodes=32,mid_reward_sum=sum(r['mid_reward'] for r in receipts))
                    update,advantages,returns=update_policy(model,optimizer,episodes,plan,1,publish)
                    write(root/'training_updates'/task/'result.json',update)
                    assert update['optimizer_steps']==17, 'KL stop: fixed budget not completed; do not change learning rate'
                    enforce_storage_budget(plan,6*1024**3)
                    saved=save_checkpoint(model,optimizer,checkpoint)
                    ready={'directory':str(checkpoint),'round':1,'optimizer_steps':17,'contract':contract,**saved}
                    write(checkpoint/'ready.json',ready)
                    del optimizer,episodes,advantages,returns;gc.collect();torch.cuda.empty_cache()
                write(root/'training_complete.json',{'episodes':32,'optimizer_steps':17,'checkpoint':ready,
                      'successes':sum(r['status']['success'] for r in receipts),'mid_reward_sum':sum(r['mid_reward'] for r in receipts)})
                # Read-only baseline evidence is copied once before evaluating either shard.
                for case,source in plan['baseline_sources'].items():
                    for name,digest in source['files'].items():
                        data=(Path(source['directory'])/name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
                        target=root/'baseline'/case/name;target.parent.mkdir(parents=True,exist_ok=True)
                        if target.exists():assert target.read_bytes()==data
                        else:target.write_bytes(data)
                write(root/'baseline_ready.json',{'cases':30})
                publish(stage='checkpoint_ready',checkpoint=ready,new_optimizer_steps=17,baseline_ready=True)
            else:
                assert (root/'baseline_ready.json').is_file()
                ready=json.loads((checkpoint/'ready.json').read_text())
                assert ready['optimizer_steps']==17 and ready['contract']==contract
                load_overlay(model,str(checkpoint),ready['trainable_sha256'])
            assert policy_hash(model)==ready['trainable_sha256']
            cases=plan['evaluation']['cases'][rank::2]
            for instance,seed in cases:
                single={**plan,'evaluation':{**plan['evaluation'],'cases':[[instance,seed]]}}
                case=root/'final'/f'i{instance}_s{seed}'
                if case.exists() and not all((case/name).is_file() for name in ('result.json','initial.json','spec.json','actions.json')):
                    archive=root/'incomplete_attempts'/f'i{instance}_s{seed}_before_{task}'
                    archive.parent.mkdir(parents=True,exist_ok=True);case.rename(archive)
                evaluate(model,single,'final',publish,reuse_existing=True,write_summary=False)
                report=paired_report(root,cases,plan['evaluation']['paired_initial_pose_tolerance'])
                write(root/f'comparison_rank{rank}.json',report)
                publish(stage='evaluation',comparison=report,checkpoint=ready)
            assert hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen
            write(root/f'checks_rank{rank}.json',{'frozen_parameters_unchanged':True,'evaluations_preserve_weights':True,'checkpoint':ready})
            publish(state='complete',stage='evaluation_complete',comparison=report)
            success=True
    except BaseException as error:
        write(root/'failures'/f'{task}.json',{'error_type':type(error).__name__,'traceback':traceback.format_exc()})
        publish(state='failed',stage='failed',error_type=type(error).__name__,error=str(error)[:400])
    finally:
        signal.alarm(0)
        data=io.BytesIO()
        with tarfile.open(fileobj=data,mode='w:gz') as archive:
            for path in sorted(root.rglob('*.json')):
                archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
        try:upload(os.environ['B1K_RESULT_PUT'],data.getvalue())
        except Exception as error:print('Result upload: '+type(error).__name__,flush=True)
    raise SystemExit(0 if success else 1)


if __name__=='__main__':main()
