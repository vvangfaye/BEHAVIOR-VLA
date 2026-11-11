"""Native VLA velocity field; all four arms share this parameterization."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'action_support_20261021'))
import vla_runtime
import torch
from torch import nn
from rlinf.models.embodiment.openpi.modules import model as model_utils
from rlinf.models.embodiment.openpi.pi0 import make_attn_mask


def bridge(a,b,time,noise):
    return (1-time[:,None,None])*(a+b)/2+time[:,None,None]*noise/(2**.5)


def difference_loss(left,right,a,b,time,physical_dim=23):
    a,b=a[...,:physical_dim].float(),b[...,:physical_dim].float()
    squared_distance=(a-b).square().sum((1,2))
    weight=torch.exp(-(1-time).square()*squared_distance/(4*time.square()))
    residual=time[:,None,None]*(left[...,:physical_dim].float()-right[...,:physical_dim].float())-(b-a)
    return .5*(weight*residual.square().mean((1,2))).mean()


class NativeAdapter(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base

    def prepare(self,observation):
        observation=model_utils.preprocess_observation(observation,train=False)
        observation=model_utils._observation_to_dtype(observation,self.base.embed_dtype)
        with torch.no_grad():
            _,valid,cache=self.base.build_prefix_cache(observation)
        return observation,valid,cache

    def velocity(self,prepared,noisy,time):
        observation,valid,cache=prepared
        tokens,suffix_valid,ar,cond=self.base.embed_suffix(observation,noisy,time)
        suffix_attention=make_attn_mask(suffix_valid,ar)
        prefix_attention=valid[:,None,:].expand(-1,tokens.shape[1],-1)
        attention=torch.cat((prefix_attention,suffix_attention),-1)
        positions=valid.sum(-1)[:,None]+suffix_valid.int().cumsum(-1)-1
        output,_=self.base.llm([None,tokens],positions=positions,mask=attention,
                               kv_cache=cache,adarms_cond=[None,cond])
        return self.base.velocity_from_suffix(output[1][:,-self.base.action_horizon:]).float()

    @torch.no_grad()
    def sample(self,observation,noise,steps):
        prepared=self.prepare(observation)
        actions=noise.float()
        for k in range(steps):
            time=torch.full((len(actions),),1-k/steps,device=actions.device)
            actions=actions-self.velocity(prepared,actions,time)/steps
        return actions
