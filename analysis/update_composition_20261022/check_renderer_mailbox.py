"""Transport contract only: no real simulation or model benefit is tested."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from renderer_mailbox import Client, RemoteSimulator, encode, decode
import renderer_service


def main():
    raw={'images':[np.arange(36,dtype=np.uint8).reshape(3,4,3)],
         'state':np.arange(61,dtype=np.float32),'prompt':'test task'}
    copied=decode(encode(raw))
    np.testing.assert_array_equal(copied['images'][0],raw['images'][0])
    np.testing.assert_array_equal(copied['state'],raw['state'])
    assert copied['state'].dtype==np.float32
    try:encode(np.array([object()],dtype=object))
    except ValueError:pass
    else:raise AssertionError('Object serialization accepted')
    calls=[]
    class Env:
        current=dict(observation=raw,status=dict(steps=0,success=False,score=0.))
        def step(self,actions,subtask):
            calls.append(('step',actions.copy(),subtask))
            self.current=dict(observation=raw,status=dict(steps=len(actions),success=False,score=.25))
            return self.current
    @contextmanager
    def environment(plan,output,instance,mode,max_steps):
        calls.append(('open',plan['seed'],instance,mode,max_steps))
        try:yield Env()
        finally:calls.append(('close',))
    for prewarm in (False,True):
        calls.clear()
        errors=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=dict(directory=str(root/'messages'),session='test',idle_timeout_seconds=5,
                        episode_output_root=str(root/'episodes'),prewarm_first=prewarm)
            plan=dict(environment={'renderer_mailbox':config},max_seconds=10,max_training_episodes=1,
                      iterations=[dict(task='test',instances=[1],environment_seeds=[7],max_steps=20)])
            renderer_service.simulator=environment
            def worker():
                try:renderer_service.serve(plan,root/'server_output')
                except Exception as error:errors.append(error)
            thread=threading.Thread(target=worker)
            thread.start()
            if prewarm:
                deadline=time.monotonic()+3
                while not (root/'messages/status.json').exists() or json.loads((root/'messages/status.json').read_text())['stage']!='prewarmed':
                    if time.monotonic()>deadline:raise TimeoutError('Prewarm did not complete')
                    time.sleep(.02)
                assert len(calls)==1 and calls[0][0]=='open'
            client=Client(config['directory'],config['session'])
            env=RemoteSimulator(client,dict(task='test',seed=7,startup_timeout_seconds=1),
                                root/'episodes'/'iteration_000_episode_0',1,'train',20)
            actions=np.ones((3,23),dtype=np.float32)
            assert env.step(actions,'generated')['status']['steps']==3
            env.close()
            client.request(dict(op='shutdown'),timeout=2)
            thread.join(timeout=3)
            assert not thread.is_alive() and not errors,errors
            assert calls[0]==('open',7,1,'train',20) and calls[-1]==('close',)
            np.testing.assert_array_equal(calls[1][1],actions)
            assert calls[1][2]=='generated'
            assert not list((root/'messages').glob('*.npz'))
            result=json.loads((root/'server_output/result.json').read_text())
            assert result['complete'] and result['rendered_episodes']==1
    print(json.dumps(dict(passed=True,scope='Lossless arrays, exact actions and lifecycle; prewarmed first episode reused without a reset; synthetic environment only')))


if __name__=='__main__':main()
