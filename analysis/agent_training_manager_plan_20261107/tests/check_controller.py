"""Exercise the real PPO loop on a small Gaussian policy and its value head."""
import contextlib
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace

import torch
from omegaconf import OmegaConf
from rlinf.b1k_tools.ppo_update_controller import UpdateController
from rlinf.utils.utils import get_loss_agg_func
from rlinf.b1k_tools.ppo_control_runtime import digest
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
import rlinf.workers.actor.embodied_fsdp_actor_worker as modified

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('original_actor', '/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/rlinf/workers/actor/embodied_fsdp_actor_worker.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 1)
        self.value_head = torch.nn.Linear(2, 1)

    def forward(self, forward_inputs, **kwargs):
        features = forward_inputs['features']
        mean = self.actor(features).unsqueeze(-1).expand(-1, 2, 1)
        logprobs = -0.5 * (forward_inputs['actions'] - mean).square()
        return {'logprobs': logprobs, 'values': self.value_head(features)}


def make_worker(module, directory, policy=None, forced=None):
    torch.manual_seed(71)
    model = Policy()
    config = OmegaConf.create({
        'algorithm': {'adv_type':'gae','loss_type':'actor_critic','logprob_type':'chunk_level',
            'reward_type':'chunk_level','entropy_type':'token_level','update_epoch':4,
            'gamma':0.99,'gae_lambda':0.95,'group_size':1,'clip_ratio_high':0.2,
            'clip_ratio_low':0.2,'value_clip':0.2,'huber_delta':10.,'entropy_bonus':0.,
            'length_normalization':True},
        'actor': {'seed':42,'global_batch_size':4,'micro_batch_size':2,
            'model':{'model_type':'openpi','num_action_chunks':2,'action_dim':1}},
        'runner':{'task_type':'embodied'},
        'env':{'train':{'max_episode_steps':6,'total_num_envs':4,'rollout_epoch':1,
                         'task_stats_dir':str(directory)}}})
    if policy:
        config.algorithm.ppo_control = {'enable':True,'policy':policy,
            'out_dir':str(directory),'rule_kl_threshold':0.01,'kl_stop':100.,
            'verify_actor_identity':True,'forced_mode':forced or 'standard'}
    optimizer = torch.optim.AdamW([
        {'params':model.actor.parameters(),'lr':0.001},
        {'params':model.value_head.parameters(),'lr':0.01}],weight_decay=.01)
    fake = SimpleNamespace(cfg=config,model=model,optimizer=optimizer,optimizer_steps=0,
        critic_warmup_steps=0,device=torch.device('cpu'),_rank=0,_world_size=1,
        gradient_accumulation=2,amp_context=contextlib.nullcontext(),
        is_weight_offloaded=False,is_optimizer_offloaded=False,
        enable_sft_co_train=False,_group_success_metrics={},version=0,
        grad_scaler=torch.amp.GradScaler('cpu',enabled=False))
    fake.loss_agg_func = get_loss_agg_func("token-mean")
    fake.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    fake.torch_platform = SimpleNamespace(empty_cache=lambda:None,current_device=lambda:'cpu')
    fake._strategy = SimpleNamespace(clip_grad_norm_=lambda model:torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    fake.before_micro_batch = lambda *args,**kwargs:contextlib.nullcontext()
    fake._actor_prefix_store_for_training = lambda:None
    fake._trajectory_length_metrics = lambda:{}
    fake._am_read_new = lambda *args:[{'task':i,'s':i%2} for i in range(4)]
    fake.optimizer_step = MethodType(FSDPModelManager.optimizer_step,fake)
    fake.train_micro_batch = MethodType(inspect.unwrap(module.EmbodiedFSDPActor.train_micro_batch),fake)
    fake.compute_advantages_and_returns = MethodType(inspect.unwrap(module.EmbodiedFSDPActor.compute_advantages_and_returns),fake)
    fake.run_training = MethodType(inspect.unwrap(module.EmbodiedFSDPActor.run_training),fake)
    return fake


def collect(worker, iteration):
    generator=torch.Generator().manual_seed(100+iteration)
    features=torch.randn(3,4,2,generator=generator)
    actions=torch.randn(3,4,2,1,generator=generator)
    with torch.no_grad():
        output=worker.model({'features':features.flatten(0,1),'actions':actions.flatten(0,1)})
    values=torch.cat([output['values'].reshape(3,4,1),torch.zeros(1,4,1)])
    dones=torch.zeros(4,4,2,dtype=torch.bool)
    dones[-1]=True
    rewards=torch.zeros(3,4,2)
    rewards[-1,:, -1]=torch.tensor([0.,1.,0.,1.])
    worker.rollout_batch={'forward_inputs':{'features':features,'actions':actions},
        'prev_logprobs':output['logprobs'].reshape(3,4,2,1), 'prev_values':values,
        'rewards':rewards,'dones':dones,'loss_mask':torch.ones(3,4,1,dtype=torch.bool),
        'loss_mask_sum':torch.full((3,4,1),6.)}
    worker.version=iteration-1
    worker.compute_advantages_and_returns()


def prepare_module(module):
    module.Worker=SimpleNamespace(torch_platform=SimpleNamespace(current_device=lambda:'cpu'))
    module.clear_memory=lambda:None
    module.all_reduce_dict=lambda values,**kwargs:values


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temporary:
        temporary=Path(temporary)
        torch.distributed.init_process_group('gloo',init_method='file://'+str(temporary/'group'),rank=0,world_size=1)
        prepare_module(original)
        prepare_module(modified)
        baseline=make_worker(original,temporary/'base')
        standard=make_worker(modified,temporary/'standard','fixed')
        for iteration in range(1,4):
            for worker in [baseline,standard]:
                collect(worker,iteration)
                worker.run_training()
            assert digest(baseline.model.parameters())==digest(standard.model.parameters())
            assert digest(baseline.model.parameters(),baseline.optimizer)==digest(standard.model.parameters(),standard.optimizer)
        modes={}
        for mode in ['short_update','critic_first']:
            worker=make_worker(modified,temporary/mode,'forced',mode)
            for iteration in range(1,4):
                collect(worker,iteration)
                metrics=worker.run_training()
            modes[mode]={k:v for k,v in metrics.items() if k.startswith('pc/')}
        assert modes['short_update']['pc/actor_steps']==6
        assert modes['critic_first']['pc/actor_steps']==9
        assert modes['critic_first']['pc/critic_only_steps']==3
        config={'policy':'agent'}
        controller=UpdateController(config)
        controller.history=[{},{}]
        assert controller.choose(3,{},lambda *args:{'mode':'invalid'})['mode']=='standard'
        restored=UpdateController(config,controller.state_dict())
        assert restored.state_dict()==controller.state_dict()
        torch.distributed.destroy_process_group()
    report={'standard_matches_original_parameters_and_optimizer':True,
            'critic_actor_and_optimizer_identity_checks_passed':True,
            'mode_steps':modes,'invalid_reply_fallback':True,'state_round_trip':True}
    (ROOT/'cpu_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
