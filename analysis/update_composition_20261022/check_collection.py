"""Check real boundary switching and termination without launching a simulator."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

# Collector only needs native imports for unrelated training functions.
# Substitute no implementations of collection or environment transitions.
native=types.ModuleType('native');native.NativeObjective=object
sys.modules['native']=native
from closed_loop import collect


class Environment:
    def __init__(self,terminal_step):
        self.end=terminal_step;self.reset_calls=0
        self.current=dict(observation={'id':0},status={'steps':0,'success':False,'score':0.},done=False)
    def step(self,actions,text):
        previous=self.current['status']['steps']
        following=min(previous+len(actions),self.end)
        self.current=dict(observation={'id':following},status={'steps':following,
            'success':following==self.end,'score':following/self.end},done=following==self.end)


seen=[]
def step(updated,observation):
    seen.append((updated,observation['id']))
    return np.full((16,23),float(updated)),str(updated),{}

env=Environment(100)
episode=collect(env,step,max_steps=64,cut_step=17,before_is_updated=False)
assert seen==[(False,0),(False,16),(True,32),(True,48)]
assert episode.cut_executed==32 and episode.control_steps==64 and not episode.success
assert env.reset_calls==0
seen.clear()
episode=collect(Environment(20),step,max_steps=64,cut_step=40,before_is_updated=True)
assert seen==[(True,0),(True,16)]
assert episode.cut_executed is None and episode.success
assert episode.decisions[-1]['executed']==4
seen.clear()
episode=collect(Environment(100),step,max_steps=19,cut_step=0,before_is_updated=False)
assert seen==[(True,0),(True,16)] and episode.control_steps==19
assert episode.decisions[-1]['executed']==3
result={'passed':True,'scope':'Collector switching/termination contract only; artificial environment, no performance claim'}
Path(__file__).with_name('collection_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
