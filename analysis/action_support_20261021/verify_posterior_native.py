"""Real wrapper + captured-action stub: wire semantics, not model quality."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

import vla_runtime  # register pinned model imports
from posterior_support_native import PosteriorRawPolicy,DeployedPosteriorAdapter
from verify_native_wrapper import CaptureAdapter,B1KPolicyWrapper
from vla_native_policy import SeededNativeWrapper,SEED_FIELD


def sequence(method):
    adapter=CaptureAdapter(0)
    raw=PosteriorRawPolicy(adapter,method,seed=13)
    native=B1KPolicyWrapper(raw,'b1k/R1Pro','task','receding_horizon',16,32)
    wrapper=SeededNativeWrapper(native,raw)
    obs={'robot_r1::proprio':np.zeros(61,dtype=np.float32)}
    for i,name in enumerate(['zed_link','left_realsense_link','right_realsense_link']):
        obs[f'robot_r1::robot_r1:{name}:Camera:0::rgb']=np.full((240,240,4),40+50*i,dtype=np.uint8)
    try:wrapper.act(obs)
    except ValueError as e:assert 'explicit' in str(e)
    else:raise AssertionError('First action accepted without explicit seed')
    actions=torch.stack([wrapper.act({**obs,**({SEED_FIELD:13} if i==0 else {})}) for i in range(34)])
    assert len(adapter.calls)==3
    expected=torch.cat([c['noise'][0,:16,:23] for c in adapter.calls])[:34]
    torch.testing.assert_close(actions,expected,rtol=0,atol=0)
    first=adapter.calls[0]['env']
    assert first['states'].shape==(1,61)
    assert first['main_images'][0,0,0,0]==40
    assert first['wrist_images'][0,0,0,0,0]==90
    assert first['wrist_images'][0,1,0,0,0]==140
    wrapper.reset()
    repeat=torch.stack([wrapper.act({**obs,**({SEED_FIELD:13} if i==0 else {})}) for i in range(34)])
    torch.testing.assert_close(repeat,actions,rtol=0,atol=0)
    try:raw.infer({'subtask':'provided'})
    except ValueError as e:assert 'generated' in str(e)
    else:raise AssertionError('Accepted provided expert subtask')
    assert native.last_subtasks=={0:'generated subtask'}
    assert raw.metadata['provided_activity_labels'] is False
    return actions


def generation_binding():
    calls=[]
    def prefix(prompt,state):
        calls.append('prefix');return [3]
    def action_tokens(prompt,state,label):
        assert label.tolist()==[7,1]
        calls.append('action_tokens_from_generated_subtask');return [3,7,1]
    def token_observation(images,state,tokens):
        calls.append(('token_observation',tokens))
        return SimpleNamespace(state=torch.zeros(1,61))
    def sample(obs,*,noise,steps,rng,mode):
        assert calls[-1]==('token_observation',[[3,7,1]])
        assert steps==10 and mode=='posterior'
        return noise,torch.ones(1,23),torch.full((1,23),.5)
    base=SimpleNamespace(
        checkpoint_io=SimpleNamespace(prepare=lambda env:({},np.zeros((1,61)),['task'])),
        joint_tokenizer=SimpleNamespace(prefix=prefix,actions=action_tokens),
        _token_observation=token_observation,
        generate_subtask=lambda obs:(torch.tensor([[7,1]]),torch.tensor([2])),
        action_horizon=32,action_dim=32,
        decode_actions=lambda a,state:a[...,:23])
    obj=SimpleNamespace(base=base,sample=sample)
    noise=torch.ones(1,32,32)
    actions,info=DeployedPosteriorAdapter.predict_actions(obj,{},method='posterior',noise=noise,steps=10)
    assert actions.shape==(1,32,23) and torch.equal(actions,noise[...,:23])
    assert info['subtask_tokens'].tolist()==[[7,1]]
    assert calls==['prefix',('token_observation',[[3]]),
                   'action_tokens_from_generated_subtask',('token_observation',[[3,7,1]])]


def main():
    generation_binding()
    a,b=sequence('ambient_aux'),sequence('posterior')
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    result=dict(passed=True,methods=['ambient_aux','posterior'],same_action_noise=True,
                exact_prefix16=True,camera_order=True,explicit_seed_required=True,
                reset_reproducible=True,supplied_subtask_rejected=True,
                generated_subtask_action_conditioning_binding=True,
                scope='Unmodified native wrapper and capture stub; no trained VLA inference or simulation.')
    Path(__file__).with_name('posterior_native_wrapper_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
