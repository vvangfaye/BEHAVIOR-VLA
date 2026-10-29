"""Finite baseline, on-policy collection, PPO update, and final evaluation workers."""
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

from common import ExecutionStopped, torch_model, write, reuse_baseline_case
from eval_utils import paired_report
from experiment import collect, enforce_storage_budget, evaluate, seed_all
from training import (actor_parameters, build_optimizer, hash_parameters, load_overlay,
                      optimizer_to, save_checkpoint, update_policy, warm_modem_critic)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def policy_hash(model):
    return hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)


def upload(url, data):
    request = urllib.request.Request(url, data=data, method='PUT')
    with urllib.request.urlopen(request, timeout=60) as response:
        assert response.status in (200,201,204)


def main():
    plan = json.loads(Path(__file__).with_name('plan.json').read_text())
    phase = os.environ['B1K_PHASE']; rank = int(os.environ['B1K_RANK'])
    attempt = int(os.environ['B1K_ATTEMPT']); task = f"{plan['worker_prefix']}{phase}_r{rank}_a{attempt}"
    root = Path(plan['output']); root.mkdir(parents=True, exist_ok=True)
    state = {'state':'running','phase':phase,'rank':rank,'attempt':attempt,'task':task,
             'started_at':time.time(),'stage':'loading','accepted_optimizer_steps':0}
    def publish(**fields):
        state.update(fields, updated_at=time.time())
        write(root/'worker_status'/f'{task}.json', state)
        try:
            upload(os.environ['B1K_STATUS_PUT'], json.dumps(state, allow_nan=False).encode())
        except Exception as error:
            print('Status upload: '+type(error).__name__, flush=True)
        print(json.dumps(state,ensure_ascii=False,allow_nan=False),flush=True)
    def stopped(*_):
        raise ExecutionStopped('Stopped or phase deadline; preserve completed evidence')
    signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGALRM,stopped)
    success = False
    try:
        assert phase in plan['phase_order'] and rank in ((0,) if phase.startswith('train') else range(plan['parallel_workers']))
        signal.alarm(plan['phase_limits_seconds'][phase]-300)
        publish(stage='loading')
        seed_all(plan['model_seed']); model = torch_model(plan)
        assert policy_hash(model) == plan['source_identity']['cold_trainable_hash']
        frozen = hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        assert frozen == plan['source_identity']['frozen_hash']
        cold_actor = hash_parameters(actor_parameters(model))
        contract = hashlib.sha256(json.dumps({'goals':plan['goal_contract'],'shaping':plan['shaping']},sort_keys=True).encode()).hexdigest()
        round_index = int(phase[-1]) if phase.startswith(('sample','train')) else None
        if phase in ('sample2','train2','final'):
            index = 2 if phase=='final' else 1
            prior = root/'checkpoints'/f'round_{index:02d}'
            ready = json.loads((prior/'ready.json').read_text())
            assert ready['round'] == index and ready['reward_contract_sha256'] == contract
            load_overlay(model, str(prior), ready['trainable_sha256'])
        write(root/'identities'/f'{task}.json',{'cold_actor_hash':cold_actor,'loaded_policy_hash':policy_hash(model),
                                             'frozen_hash':frozen,'reward_contract':contract})
        # The model is on GPU; release temporary CPU allocations from checkpoint
        # loading/hash validation before the simulator's scene-loading peak.
        gc.collect();torch.cuda.empty_cache()
        try:
            import ctypes
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError,OSError):
            pass
        if phase.startswith('sample'):
            collection_hash = policy_hash(model); receipts = []
            for index in range(rank,32,plan['parallel_workers']):
                receipt_path = root/'training_receipts'/f'round_{round_index:02d}'/f'episode_{index+1:02d}.json'
                output = root/'training'/f'round_{round_index:02d}'/f'episode_{index+1:02d}'
                path = output/'rollout.pt'
                if receipt_path.is_file():
                    receipt = json.loads(receipt_path.read_text())
                    assert receipt['policy_hash']==collection_hash and receipt['contract']==contract
                    assert sha(path)==receipt['sha256']
                    receipts.append(receipt); continue
                # A complete trajectory that survived a preemption before its
                # receipt can be recovered. Never replace a completed receipt.
                data = None
                if path.is_file() and (output/'result.json').is_file():
                    candidate = torch.load(path,map_location='cpu',weights_only=True)
                    assert candidate['reward_contract_sha256']==contract
                    assert candidate['instance']==plan['train_instances'][index]
                    assert candidate['seed']==plan['train_seeds'][round_index-1][index]
                    assert candidate['transitions'][-1]['terminal']
                    data = candidate
                if data is None:
                    if output.exists():
                        archive = root/'incomplete_attempts'/f'{phase}_episode_{index+1:02d}_before_{task}'
                        archive.parent.mkdir(parents=True,exist_ok=True); assert not archive.exists(); output.rename(archive)
                    publish(stage='training_rollout',round=round_index,episode=index+1,
                            instance=plan['train_instances'][index],collected_by_worker=len(receipts))
                    data = collect(model,plan,round_index,index,publish)
                assert all(abs(r['reward']-r['task_reward']-r['shaping_reward'])<1e-5 for r in data['transitions'])
                assert policy_hash(model)==collection_hash
                summary = json.loads((output/'rollout/goal_summary.json').read_text())
                receipt = {'round':round_index,'episode':index+1,'instance':data['instance'],'seed':data['seed'],
                           'status':data['status'],'policy_hash':collection_hash,'contract':contract,
                           'path':str(path),'sha256':sha(path),'goal_summary':summary,
                           'task_reward':sum(r['task_reward'] for r in data['transitions']),
                           'shaping_reward':sum(r['shaping_reward'] for r in data['transitions']),
                           'chunks':len(data['transitions']),'source_task':task}
                write(receipt_path,receipt); receipts.append(receipt)
                publish(stage='training_rollout_complete',round=round_index,collected_by_worker=len(receipts),
                        successes_by_worker=sum(r['status']['success'] for r in receipts),last_result=data['status'])
                del data; gc.collect()
            publish(state='complete',stage='collection_complete',round=round_index,
                    collected_by_worker=len(receipts),successes_by_worker=sum(r['status']['success'] for r in receipts))
        elif phase.startswith('train'):
            collection_hash = policy_hash(model)
            checkpoint = root/'checkpoints'/f'round_{round_index:02d}'
            if (checkpoint/'ready.json').is_file():
                ready = json.loads((checkpoint/'ready.json').read_text())
                assert ready['round']==round_index and ready['reward_contract_sha256']==contract
                assert ready['optimizer_steps']==17
                load_overlay(model,str(checkpoint),ready['trainable_sha256'])
            else:
                if checkpoint.exists():
                    archive = root/'incomplete_attempts'/f'checkpoint_{round_index}_before_{task}'
                    archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();checkpoint.rename(archive)
                receipts = [json.loads((root/'training_receipts'/f'round_{round_index:02d}'/f'episode_{i+1:02d}.json').read_text()) for i in range(32)]
                assert all(r['policy_hash']==collection_hash and r['contract']==contract for r in receipts)
                episodes = []
                for r in receipts:
                    assert sha(r['path'])==r['sha256']
                    episode = torch.load(r['path'],map_location='cpu',weights_only=True)
                    assert episode['instance']==r['instance'] and episode['seed']==r['seed']
                    assert episode['reward_contract_sha256']==contract
                    episodes.append(episode)
                assert any(r['task_reward']>0 for r in receipts), 'No successful task reward in 32 rollouts; stop instead of training without a success signal'
                if round_index==1:
                    warm = warm_modem_critic(model,episodes,plan,publish)
                    write(root/'critic_warmup.json',warm)
                optimizer = build_optimizer(model,plan)
                if round_index==2:
                    previous = root/'checkpoints/round_01/optimizer.pt'
                    optimizer.load_state_dict(torch.load(previous,map_location='cpu',weights_only=True))
                    assert [g['lr'] for g in optimizer.param_groups]==[plan['optimizer']['actor_lr'],plan['optimizer']['value_lr']]
                    optimizer_to(optimizer,next(model.parameters()).device)
                publish(stage='training_update',round=round_index,training_episodes=32)
                report, advantages, returns = update_policy(model,optimizer,episodes,plan,round_index,publish)
                write(root/'training_updates'/phase/'result.json',report)
                assert report['optimizer_steps']==17, 'Fixed update budget not reached; preserve KL stop and do not retune'
                enforce_storage_budget(plan,6*1024**3)
                saved = save_checkpoint(model,optimizer,checkpoint)
                ready = {**saved,'directory':str(checkpoint),'round':round_index,'optimizer_steps':17,
                         'cumulative_optimizer_steps':17*round_index,'reward_contract_sha256':contract,
                         'collection_policy_hash':collection_hash}
                write(checkpoint/'ready.json',ready)
                del optimizer,episodes,advantages,returns;gc.collect();torch.cuda.empty_cache()
            publish(state='complete',stage='training_complete',round=round_index,checkpoint=ready,
                    accepted_optimizer_steps=17,cumulative_optimizer_steps=17*round_index)
        else:
            # Both original and final policies use the same ODE, execution
            # interface, seeds and case-specific train/public instance loader.
            cases = plan.get('evaluation_runtime_order',plan['evaluation']['cases'])[rank::plan['parallel_workers']]
            completed = 0
            for instance,seed in cases:
                output = root/phase/f'i{instance}_s{seed}'
                required = ('result.json','initial.json','spec.json','actions.json','goal_summary.json')
                if output.exists() and not all((output/name).is_file() for name in required):
                    archive = root/'incomplete_attempts'/f'{phase}_i{instance}_s{seed}_before_{task}'
                    archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();output.rename(archive)
                if phase=='baseline':
                    reuse_baseline_case(plan,f'i{instance}_s{seed}',output)
                single = {**plan,'evaluation':{**plan['evaluation'],'cases':[[instance,seed]]},
                          'evaluation_mode':'train' if instance<301 else 'public_test','collecting_training':False}
                rows = evaluate(model,single,phase,publish,reuse_existing=True,write_summary=False)
                spec = json.loads((output/'spec.json').read_text())
                assert spec['mode']==single['evaluation_mode'] and not spec['shaping_enabled']
                completed += 1
                details = {'completed_evaluations':completed,'expected_evaluations':len(cases),'last_result':rows[-1]}
                if phase=='final':
                    details['training_panel'] = paired_report(root,plan['evaluation']['training_cases'])
                    details['heldout_panel'] = paired_report(root,plan['evaluation']['heldout_cases'])
                    write(root/f'comparison_rank{rank}.json',details)
                publish(stage='evaluation',**details)
            publish(state='complete',stage='evaluation_complete',completed_evaluations=completed)
        assert hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen
        write(root/'checks'/f'{task}.json',{'frozen_unchanged':True,'policy_hash':policy_hash(model),
                                          'phase':phase,'rank':rank,'completed':True})
        success = True
    except BaseException as error:
        write(root/'failures'/f'{task}.json',{'error_type':type(error).__name__,'traceback':traceback.format_exc()})
        publish(state='failed',stage='failed',error_type=type(error).__name__,error=str(error)[:400])
    finally:
        signal.alarm(0)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            for path in sorted(root.rglob('*.json')):
                relative=path.relative_to(root)
                if relative.parts[0] in ('baseline','final'):
                    case=root/relative.parts[0]/relative.parts[1]
                    if not all((case/name).is_file() for name in ('result.json','actions.json','goal_summary.json')):
                        continue
                try:
                    body=path.read_bytes()
                    json.loads(body)
                except (FileNotFoundError,json.JSONDecodeError):
                    continue
                # Snapshot bytes first so another worker's atomic status update
                # cannot change a tar member's size between stat and read.
                member=tarfile.TarInfo(str(relative));member.size=len(body)
                archive.addfile(member,io.BytesIO(body))
        try:
            upload(os.environ['B1K_RESULT_PUT'],stream.getvalue())
        except Exception as error:
            print('Result upload: '+type(error).__name__,flush=True)
    raise SystemExit(0 if success else 1)


if __name__=='__main__':
    main()
