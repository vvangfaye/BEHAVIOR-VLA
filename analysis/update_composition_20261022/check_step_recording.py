"""Regression: use the real scalar adapter, with a tiny vector env underneath."""
import importlib.util
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'simulation_bridge'))
from step_recording import install_step_recorder

source=ROOT/'BEHAVIOR-VLAs/BEHAVIOR-1K/OmniGibson/omnigibson/eval/single_env.py'
spec=importlib.util.spec_from_file_location('real_scalar_adapter',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class VectorEnvironment:
    num_envs=1
    def __init__(self):self.calls=0
    def step(self,action):
        self.calls+=1
        assert action.shape==(1,23)
        return [{'sensor':1}],[.25],[False],[False],[{'valid':True}]


def main():
    wrapped=VectorEnvironment();env=module.SingleEnvironmentAdapter(wrapped)
    original=env.step
    def old_recorder(action):return original(action)
    env.step=old_recorder
    try:env.step(torch.zeros(23))
    except RecursionError:pass
    else:raise AssertionError('Old proxy assignment no longer reproduces the failure')
    wrapped=VectorEnvironment();env=module.SingleEnvironmentAdapter(wrapped);record={}
    install_step_recorder(env,record)
    result=env.step(torch.zeros(23))
    assert wrapped.calls==1 and result[1]==.25 and record=={'reward':.25,'info':{'valid':True}}
    assert 'step' not in wrapped.__dict__
    print('PASS: reproduced original recursion; fixed recorder calls the actual vector environment exactly once.')


if __name__=='__main__':main()
