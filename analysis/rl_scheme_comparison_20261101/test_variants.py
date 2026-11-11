import copy
import unittest
import torch
from variant_update import episode_weights,group_advantages,prepare_rewards,update_policy
from training import build_optimizer,hash_parameters


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.actor=torch.nn.Parameter(torch.tensor(0.1))
        self.value_head=torch.nn.Linear(1,1)
        self.frozen=torch.nn.Parameter(torch.tensor(3.),requires_grad=False)
    @property
    def device(self):return self.actor.device
    def default_forward(self,inputs,compute_values=True):
        x=inputs['x'];out={'logprobs':(self.actor*x).reshape(1,1,1).expand(1,1,23)/23}
        if compute_values:out['values']=self.value_head(x).reshape(1)
        return out


def episode(model,length,success,instance=1):
    x=torch.tensor([[1. if success else -1.]])
    old=model.default_forward({'x':x})
    rows=[]
    for t in range(length):
        rows.append({'forward_inputs':{'x':x},'prev_logprobs':old['logprobs'].detach().clone(),
            'prev_values':old['values'].detach().clone(),'terminal':t==length-1,
            'reward':float(success and t==length-1),'task_reward':float(success and t==length-1),
            'goals_before':{'goals':{'a':True,'b':False}},'goals_after':{'goals':{'a':True,'b':True}}})
    return {'instance':instance,'status':{'success':success},'transitions':rows,'bootstrap':torch.zeros(1)}


class Variants(unittest.TestCase):
    def test_equal_total_weight_per_episode(self):
        w=episode_weights([{'transitions':[None]},{'transitions':[None]*3}])
        self.assertAlmostEqual(float(w[:1].sum()),float(w[1:].sum()),places=6)
        self.assertAlmostEqual(float(w.mean()),1.,places=6)
    def test_group_relative_contrast(self):
        m=ToyPolicy();rows,a,stats=group_advantages([episode(m,2,True),episode(m,4,False)])
        self.assertTrue(torch.equal(a,torch.tensor([1.,1.,-1.,-1.,-1.,-1.])))
        self.assertEqual(stats['mixed_outcome_groups'],1)
    def test_no_group_signal_rejected(self):
        m=ToyPolicy()
        with self.assertRaises(ValueError):group_advantages([episode(m,2,True),episode(m,3,True)])
    def test_gamma_one_shaping_telescopes(self):
        m=ToyPolicy();e=episode(m,2,True)
        e['transitions'][1]['goals_before']={'goals':{'a':True,'b':True}}
        e['transitions'][1]['task_reward']=10.
        p={'comparison_variant':'gamma1_ppo','optimizer':{'gamma':1.},'shaping':{'gamma':1.,'beta':1.}}
        prepare_rewards([e],p)
        self.assertAlmostEqual(sum(x['reward'] for x in e['transitions']),9.)
    def test_critic_free_update_changes_only_actor(self):
        torch.manual_seed(0);m=ToyPolicy();episodes=[episode(m,8,True),episode(m,32,False)]
        plan={'comparison_variant':'group_outcome','optimizer':{'shuffle_seed':5,'updates_per_round':[2],
            'clip_ratio':.2,'kl_target':.01,'actor_lr':.001,'value_lr':.0001,'weight_decay':0.}}
        before=hash_parameters(m.value_head.named_parameters());actor=float(m.actor.detach())
        result=update_policy(m,build_optimizer(m,plan),episodes,plan)
        self.assertEqual(result['optimizer_steps'],2)
        self.assertGreater(float(m.actor.detach()),actor)
        self.assertEqual(hash_parameters(m.value_head.named_parameters()),before)
        self.assertFalse(result['critic_gradient_nonzero'])
        self.assertLessEqual(result['final_batch_kl']['mean'],.01)
    def test_both_ppo_variants_update_actor_and_critic(self):
        for variant,gamma in [('gamma1_ppo',1.),('balanced_ppo',.99)]:
            torch.manual_seed(0);m=ToyPolicy();episodes=[episode(m,12,True),episode(m,28,False)]
            plan={'comparison_variant':variant,'optimizer':{'shuffle_seed':5,'updates_per_round':[2],
                'clip_ratio':.2,'kl_target':.01,'actor_lr':.001,'value_lr':.0001,'weight_decay':0.,
                'gamma':gamma,'gae_lambda':.95},'shaping':{'gamma':gamma,'beta':1.}}
            prepare_rewards(episodes,plan)
            result=update_policy(m,build_optimizer(m,plan),episodes,plan)
            self.assertEqual(result['optimizer_steps'],2)
            self.assertTrue(result['actor_gradient_nonzero'])
            self.assertTrue(result['critic_gradient_nonzero'])
    def test_kl_rejection_restores_parameters(self):
        torch.manual_seed(0);m=ToyPolicy();episodes=[episode(m,8,True),episode(m,32,False)]
        before=hash_parameters((n,p.reshape(-1)) for n,p in m.named_parameters())
        plan={'comparison_variant':'group_outcome','optimizer':{'shuffle_seed':5,'updates_per_round':[2],
            'clip_ratio':.2,'kl_target':.01,'actor_lr':10.,'value_lr':.0001,'weight_decay':0.}}
        result=update_policy(m,build_optimizer(m,plan),episodes,plan)
        self.assertTrue(result['early_stopped']);self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(before,hash_parameters((n,p.reshape(-1)) for n,p in m.named_parameters()))


if __name__=='__main__':unittest.main()
