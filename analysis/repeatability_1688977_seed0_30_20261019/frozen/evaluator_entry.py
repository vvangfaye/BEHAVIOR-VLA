"""Native evaluator + explicit seed0 ACK + read-only initial state/action trace."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from omnigibson.eval import eval as native
from seed_protocol import SeededClient, PROTOCOL

native.DEFAULT_EVAL_SEED=0
Original=native.Evaluator


def plain(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value).tolist()


def digest_obs(value):
    if isinstance(value,dict):return {str(k):digest_obs(v) for k,v in sorted(value.items())}
    if isinstance(value,str):return value
    if value is None:return None
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    a=np.asarray(value)
    if a.dtype.hasobject:return str(value)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


class Evaluator(Original):
    def reset(self,*args,**kwargs):
        result=super().reset(*args,**kwargs)
        policy=self.policy.wrapped if isinstance(self.policy,SeededClient) else self.policy
        self.policy=SeededClient(policy,dict(protocol=PROTOCOL,policy_seed=0))
        if not kwargs.get('prepare_for_task_instance',False):
            root=Path(os.environ['REPEATABILITY_OUTPUT'])
            objects={}
            for obj in sorted(self.env.scene.objects,key=lambda o:o.name):
                position,orientation=obj.get_position_orientation()
                item=dict(position=plain(position),orientation=plain(orientation))
                if getattr(obj,'n_joints',0)>0:
                    item['joint_positions']=plain(obj.get_joint_positions())
                    item['joint_velocities']=plain(obj.get_joint_velocities())
                objects[obj.name]=item
            physics_raw=json.dumps(objects,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            state=dict(objects=objects,physics_sha256=hashlib.sha256(physics_raw).hexdigest(),
                       observation=digest_obs(self.obs),before_first_policy_action=True)
            (root/'initial_state.json').write_text(json.dumps(state,sort_keys=True,allow_nan=False)+'\n')
            old=getattr(self,'_action_stream',None)
            if old is not None:old.close()
            self._action_stream=(root/'actions.f32').open('wb')
            self._action_hash=hashlib.sha256();self._action_count=0;self._action_dim=None
        return result

    def step(self,*args,**kwargs):
        terminated,truncated=super().step(*args,**kwargs)
        root=Path(os.environ['REPEATABILITY_OUTPUT'])
        action=self.robot_action
        if hasattr(action,'detach'):action=action.detach().cpu().numpy()
        action=np.asarray(action,dtype='<f4').reshape(-1)
        if self._action_dim is None:self._action_dim=len(action)
        if len(action)!=self._action_dim:raise ValueError('Action dimension changed')
        raw=action.tobytes();self._action_stream.write(raw);self._action_hash.update(raw);self._action_count+=1
        if terminated or truncated:
            self._action_stream.close()
            proof=self.policy.proof()
            with Path(os.environ['PAIRED_SEED_PROOF']).open('x') as stream:
                json.dump(dict(proof,environment_seed=0),stream)
            (root/'action_trace.json').write_text(json.dumps(dict(path=str(root/'actions.f32'),
                dtype='<f4',steps=self._action_count,action_dim=self._action_dim,
                sha256=self._action_hash.hexdigest()))+'\n')
        return terminated,truncated


native.Evaluator=Evaluator
native.main()
