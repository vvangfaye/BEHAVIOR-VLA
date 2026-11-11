"""Train one alternative, or evaluate one fixed case, in an isolated directory."""
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
from common import ExecutionStopped,torch_model,write
from experiment import evaluate,enforce_storage_budget,seed_all
from training import actor_parameters,build_optimizer,hash_parameters,load_overlay,save_checkpoint,warm_modem_critic
from variant_update import prepare_rewards,update_policy


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def upload(url,data):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url,data=data,method='PUT'),timeout=60) as response:
        assert response.status in (200,201,204)


def main():
    plan=json.loads(Path(__file__).with_name('plan.json').read_text())
    phase=os.environ['B1K_PHASE'];attempt=int(os.environ['B1K_ATTEMPT']);variant=plan['comparison_variant']
    root=Path(plan['output']);root.mkdir(parents=True,exist_ok=True)
    state={'state':'running','phase':phase,'variant':variant,'started_at':time.time(),'accepted_optimizer_steps':0}
    def publish(**fields):
        state.update(fields,updated_at=time.time());write(root/'status.json',state)
        try:upload(os.environ['B1K_STATUS_PUT'],json.dumps(state,allow_nan=False).encode())
        except Exception as error:print('Status upload: '+type(error).__name__,flush=True)
        print(json.dumps(state,allow_nan=False),flush=True)
    def stop(*_):raise ExecutionStopped('Interrupted; preserve complete artifacts')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    signal.alarm(plan['phase_limits_seconds'][phase]-300)
    success=False
    try:
        publish(stage='loading');seed_all(plan['model_seed']);model=torch_model(plan)
        assert hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)==plan['source_identity']['cold_trainable_hash']
        frozen=hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        assert frozen==plan['source_identity']['frozen_hash']
        gc.collect();torch.cuda.empty_cache()
        if phase=='train1':
            checkpoint=root/'checkpoints/round_01'
            if (checkpoint/'ready.json').exists():
                ready=json.loads((checkpoint/'ready.json').read_text())
                assert ready['variant']==variant and ready['source_reward_contract_sha256']==plan['source_reward_contract_sha256']
                load_overlay(model,checkpoint,ready['trainable_sha256'])
            else:
                if checkpoint.exists():
                    old=root/'incomplete_attempts'/f'checkpoint_before_attempt_{attempt}'
                    old.parent.mkdir(exist_ok=True);assert not old.exists();checkpoint.rename(old)
                episodes=[]
                receipts=plan['source_rollout_receipts'];assert len(receipts)==32
                for item in receipts:
                    source=Path(item['path'])
                    assert source.is_relative_to(plan['source_rollout_root'])
                    assert sha(source)==item['sha256']
                    episode=torch.load(source,map_location='cpu',weights_only=True)
                    assert episode['reward_contract_sha256']==plan['source_reward_contract_sha256']==item['contract']
                    assert episode['instance']==item['instance'] and episode['seed']==item['seed']
                    assert episode['transitions'][-1]['terminal'] and len(episode['transitions'])==item['chunks']
                    episodes.append(episode)
                    publish(stage='loading_rollouts',loaded_episodes=len(episodes),training_episodes=32)
                prepare_rewards(episodes,plan)
                if variant!='group_outcome':
                    warm=warm_modem_critic(model,episodes,plan,publish);write(root/'critic_warmup.json',warm)
                else:
                    write(root/'critic_warmup.json',{'skipped':True,'reason':'critic-free group outcome objective'})
                optimizer=build_optimizer(model,plan)
                report=update_policy(model,optimizer,episodes,plan,publish=publish)
                write(root/'training_updates/train1/result.json',report)
                enforce_storage_budget(plan,6*1024**3)
                saved=save_checkpoint(model,optimizer,checkpoint)
                ready={**saved,'directory':str(checkpoint),'round':1,'variant':variant,
                    'optimizer_steps':report['optimizer_steps'],'requested_optimizer_steps':17,
                    'source_reward_contract_sha256':plan['source_reward_contract_sha256'],
                    'target_reward_contract_sha256':hashlib.sha256(json.dumps({'goals':plan['goal_contract'],
                        'shaping':plan['shaping']},sort_keys=True).encode()).hexdigest()}
                write(checkpoint/'ready.json',ready)
                del episodes,optimizer;gc.collect();torch.cuda.empty_cache()
            publish(state='complete',stage='training_complete',checkpoint=ready,
                    accepted_optimizer_steps=ready['optimizer_steps'])
        elif phase=='final':
            checkpoint=Path(plan['comparison_checkpoint']);ready=json.loads((checkpoint/'ready.json').read_text())
            load_overlay(model,checkpoint,ready['trainable_sha256'])
            case=plan['evaluation']['cases'][0];output=root/'final'/f'i{case[0]}_s{case[1]}'
            required=['result.json','initial.json','spec.json','actions.json','goal_summary.json']
            if output.exists() and not all((output/n).is_file() for n in required):
                old=root/'incomplete_attempts'/f'case_before_attempt_{attempt}'
                old.parent.mkdir(exist_ok=True);assert not old.exists();output.rename(old)
            rows=evaluate(model,plan,'final',publish,reuse_existing=True)
            assert len(rows)==1
            publish(state='complete',stage='evaluation_complete',completed_evaluations=1,
                    last_result=rows[0],checkpoint_steps=ready['optimizer_steps'])
        else:raise ValueError('Unexpected comparison phase')
        assert hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen
        write(root/'checks.json',{'frozen_unchanged':True,'variant':variant,'phase':phase})
        success=True
    except BaseException as error:
        write(root/'failure.json',{'type':type(error).__name__,'traceback':traceback.format_exc()})
        publish(state='failed',stage='failed',error_type=type(error).__name__,error=str(error)[:400])
    finally:
        signal.alarm(0);buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for path in sorted(root.rglob('*.json')):
                try:body=path.read_bytes();json.loads(body)
                except (OSError,ValueError):continue
                info=tarfile.TarInfo(str(path.relative_to(root)));info.size=len(body)
                archive.addfile(info,io.BytesIO(body))
        try:upload(os.environ['B1K_RESULT_PUT'],buffer.getvalue())
        except Exception as error:print('Result upload: '+type(error).__name__,flush=True)
    raise SystemExit(0 if success else 1)


if __name__=='__main__':main()
