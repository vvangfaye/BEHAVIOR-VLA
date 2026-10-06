"""Record executed control actions paired with preceding onboard observations."""
from __future__ import annotations
import json
from pathlib import Path
import h5py
import numpy as np

CAMERAS = {
 'observation.rgb.zed_link_camera_0':'robot_r1::robot_r1:zed_link:Camera:0::rgb',
 'observation.rgb.left_realsense_link_camera_0':'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb',
 'observation.rgb.right_realsense_link_camera_0':'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb',
}


def array(value):
    if hasattr(value, 'detach'):
        value=value.detach().cpu().numpy()
    return np.array(value,copy=True)


def capture_observation(obs):
    frame = {'observation.state':array(obs['robot_r1::proprio']).reshape(-1).astype(np.float32)}
    if frame['observation.state'].size < 57 or not np.isfinite(frame['observation.state']).all():
        raise ValueError('R1Pro proprioception is incomplete or nonfinite')
    for key, source in CAMERAS.items():
        image = array(obs[source])
        if image.ndim==4 and image.shape[0]==1:
            image=image[0]
        image=image[...,:3]
        if image.ndim!=3 or image.shape[-1]!=3 or image.dtype != np.uint8:
            raise ValueError(f'Invalid onboard RGB camera: {source}')
        frame[key]=image
    return frame


class TrajectoryWriter:
    """One independent attempt per file; failed trials never become BC labels."""
    def __init__(self,path,metadata):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=h5py.File(self.path,'x')
        self.file.attrs['schema_version']=1
        self.file.attrs['metadata']=json.dumps(metadata,sort_keys=True)
        self.file.attrs['complete']=False
        self.count=0
        self.hasher=__import__('hashlib').sha256()
        self.closed=False

    def record(self,frame,action,terminated,truncated):
        action=array(action).reshape(-1).astype(np.float32)
        if action.shape!=(23,) or not np.isfinite(action).all():
            raise ValueError('Executed action must be a finite 23-dimensional R1Pro control')
        values={**frame,'action':action,'next.terminated':np.array([terminated],dtype=bool),
                'next.truncated':np.array([truncated],dtype=bool)}
        for key,value in values.items():
            if not np.isfinite(value).all():raise ValueError(f'Nonfinite trajectory field {key}')
            if key not in self.file:
                self.file.create_dataset(key,shape=(0,*value.shape),maxshape=(None,*value.shape),
                                         dtype=value.dtype,chunks=(1,*value.shape),compression='gzip')
            ds=self.file[key]
            if ds.shape[1:]!=value.shape:raise ValueError('Observation/action dimensions changed during a trial')
            ds.resize(self.count+1,axis=0);ds[self.count]=value
            self.hasher.update(key.encode());self.hasher.update(value.tobytes())
        self.count+=1

    def finish(self,result):
        if self.count!=result['steps']:
            raise ValueError('Trajectory action count differs from evaluator control steps')
        self.file.attrs['result']=json.dumps(result,sort_keys=True)
        self.file.attrs['control_steps']=self.count
        self.file.attrs['content_sha256']=self.hasher.hexdigest()
        self.file.attrs['complete']=True
        self.file.flush();self.file.close();self.closed=True

    def close(self):
        if not self.closed:
            self.file.close();self.closed=True


def inspect_trajectory(path,*,require_success=True):
    with h5py.File(path,'r') as f:
        if f.attrs.get('schema_version')!=1 or not f.attrs.get('complete'):
            raise ValueError('Unfinished or unknown trajectory')
        metadata=json.loads(f.attrs['metadata']);result=json.loads(f.attrs['result'])
        n=int(f.attrs['control_steps'])
        if n<=0 or n!=result['steps']:
            raise ValueError('Trajectory has no aligned executed actions')
        if require_success and (result.get('success') is not True or result.get('metric_eligible') is not True):
            raise ValueError('Only eligible, successful subtask trajectories can be BC data')
        if result.get('conditioning',{}).get('source')!='provided' or result.get('subtask')!=metadata['sample']['subtask']:
            raise ValueError('Trajectory conditioning differs from its fixed annotated subtask')
        if metadata.get('conditioning')!='task_and_subtask':raise ValueError('Wrong training conditioning')
        if type(metadata.get('fps')) is not int or metadata['fps'] <= 0:
            raise ValueError('Invalid trajectory control frequency')
        if f['observation.state'].ndim != 2 or not 57 <= f['observation.state'].shape[1] <= 512:
            raise ValueError('Invalid R1Pro proprioception shape')
        for key in CAMERAS:
            if key not in f or f[key].ndim != 4 or f[key].shape[-1] != 3 or f[key].dtype != np.uint8 or any(not 1 <= dim <= 224 for dim in f[key].shape[1:3]):
                raise ValueError('Invalid RGB trajectory camera contract')
        required=['observation.state','action',*CAMERAS,'next.terminated','next.truncated']
        if any(k not in f or len(f[k])!=n for k in required) or f['action'].shape!=(n,23):
            raise ValueError('Incomplete trajectory arrays')
        hash_state=__import__('hashlib').sha256()
        # Same field insertion order used by record().
        order=['observation.state',*CAMERAS,'action','next.terminated','next.truncated']
        for i in range(n):
            for key in order:
                value=f[key][i]
                if not np.isfinite(value).all():raise ValueError('Trajectory has nonfinite observations/actions')
                hash_state.update(key.encode());hash_state.update(value.tobytes())
        if hash_state.hexdigest()!=f.attrs['content_sha256']:
            raise ValueError('Trajectory arrays changed after collection')
        return {'metadata':metadata,'result':result,'steps':n,'content_sha256':hash_state.hexdigest(),
                'state_shape':list(f['observation.state'].shape[1:]),
                'camera_shapes':{k:list(f[k].shape[1:]) for k in CAMERAS}}
