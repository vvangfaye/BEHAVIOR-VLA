"""Bounded read-only range access to frozen TAR/HDF5 archives; no job submission."""
from collections import OrderedDict
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import textwrap

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path[:0] = ['/private/tmp/b1k-high-exposure-audit-deps',
    '/private/tmp/behavior-eval100-20261015-env/lib/python3.12/site-packages',
    str(REPO/'BEHAVIOR-VLAs/sco')]
import h5py
import numpy as np
from PIL import Image, ImageDraw
import video_eval_adapter


class ObjectFile(io.RawIOBase):
    def __init__(self,client,key):
        self.client,self.key=client,key
        h=client.head_object(Bucket='260010165',Key=key)
        self.size,self.etag=h['ContentLength'],h['ETag']
        self.pos=0;self.requests=0;self.downloaded=0
        self.cache=OrderedDict()
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos+offset if whence==1 else self.size+offset)
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        key=(self.pos,n)
        if key not in self.cache:
            if n>8*1024**2:raise ValueError('Unexpected large HDF5 read')
            r=self.client.get_object(Bucket='260010165',Key=self.key,
                Range=f'bytes={self.pos}-{self.pos+n-1}',IfMatch=self.etag)
            b=r['Body']
            try:raw=b.read()
            finally:b.close()
            if len(raw)!=n:raise ValueError('Short range read')
            self.requests+=1;self.downloaded+=len(raw)
            self.cache[key]=raw
            while len(self.cache)>2048:self.cache.popitem(last=False)
        raw=self.cache[key];self.cache.move_to_end(key);self.pos+=n
        return raw
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)


class MemberFile(io.RawIOBase):
    def __init__(self,source,offset,size):self.source,self.offset,self.size,self.pos=source,offset,size,0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos+offset if whence==1 else self.size+offset)
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        self.source.seek(self.offset+self.pos);data=self.source.read(n);self.pos+=len(data);return data
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))


def review(index,stride=30,probe=False):
    entry=json.loads((ROOT/'review_plan.json').read_text())['episodes'][index]
    row,selected=entry['exposure'],entry['selection']
    dest=ROOT/row['root_id'];dest.mkdir(exist_ok=True)
    if (dest/'review.json').exists() and not probe:
        return json.loads((dest/'review.json').read_text())
    started=time.time();obj=ObjectFile(video_eval_adapter.aoss_client(),selected['source_archive'])
    with tarfile.open(fileobj=obj,mode='r:') as archive:
        member=archive.getmember('episode.hdf5')
        view=MemberFile(obj,member.offset_data,member.size)
        with h5py.File(view,'r') as f:
            metadata=json.loads(f.attrs['metadata']);result=json.loads(f.attrs['result'])
            segments=json.loads(f.attrs['segments']);n=int(f.attrs['control_steps'])
            assert metadata['task']==row['task'] and metadata['instance_id']==int(row['instance_id'])
            assert f.attrs['content_sha256']==selected['content_sha256']
            assert n==int(row['frames']) and result['success'] and result['terminated'] and not result['truncated']
            if probe:
                ids=np.array([0,n//4,n//2,3*n//4,n-1],dtype=np.int64)
            else:ids=np.unique(np.r_[np.arange(0,n,stride),n-1]).astype(np.int64)
            # One-second samples locate persistent low-motion intervals; rapid grasps can be missed.
            action=np.stack([f['action'][int(i)] for i in ids]);state=np.stack([f['observation.state'][int(i)] for i in ids])
            np.savez_compressed(dest/('probe.npz' if probe else 'sampled_arrays.npz'),frame=ids,action=action,state=state)
            labels=[]
            for i in ids:
                matching=[s for s in segments if s['start']<=i<s['end']]
                labels.append(matching[0]['subtask'] if matching else '')
            physical=np.c_[state[:,53:57],state[:,3:10],state[:,28:35],state[:,24:26].sum(1),state[:,49:51].sum(1)]
            arm_delta=np.max(np.abs(np.diff(physical[:,:18],axis=0)),axis=1)
            grip_delta=np.max(np.abs(np.diff(physical[:,18:],axis=0)),axis=1)
            base_speed=np.max(np.abs(state[:,:3]),axis=1)
            # Values are explicit diagnostic thresholds, not an acceptance decision.
            low=(arm_delta<.02)&(grip_delta<.005)&(np.maximum(base_speed[:-1],base_speed[1:])<.02)
            low_runs=[dict(start_frame=int(ids[a]),end_frame=int(ids[b]),seconds=(int(ids[b])-int(ids[a]))/30)
                for a,b in runs(low) if int(ids[b])-int(ids[a])>=300]
            switches=[]
            for hand,column in [('left',18),('right',19)]:
                width=physical[:,column]
                # Absolute width threshold only identifies opening/closing cycles; it cannot establish grasp success.
                binary=width>.03
                changes=np.flatnonzero(binary[1:]!=binary[:-1])+1
                switches.append(dict(hand=hand,threshold_m=.03,one_hz_state_changes=len(changes),
                    change_frames=ids[changes].tolist(),min_width=float(width.min()),max_width=float(width.max())))
            long_segments=sorted(segments,key=lambda s:s['end']-s['start'],reverse=True)[:5]
            raw=dict(root_id=row['root_id'],task=row['task'],instance_id=int(row['instance_id']),
                actual_training_draws=int(row['actual_draws']),frames=n,
                source=dict(bucket='260010165',key=selected['source_archive'],etag=obj.etag,bytes=obj.size,
                    expected_content_sha256=selected['content_sha256'],hdf5_attribute_matches_frozen_index=True,
                    full_content_rehash_performed=False),
                outcome={k:result.get(k) for k in ['success','terminated','truncated','q_score','steps']},
                stride_frames=stride,sampled_frames=len(ids),finite_actions=bool(np.isfinite(action).all()),
                finite_states=bool(np.isfinite(state).all()),low_motion_thresholds=dict(joint_max_rad=.02,gripper_max_m=.005,base_velocity_max=.02),
                low_motion_runs=sorted(low_runs,key=lambda s:s['seconds'],reverse=True),
                sampled_low_motion_transition_fraction=float(low.mean()),gripper_switches=switches,
                longest_recorded_label_segments=long_segments,all_segments=segments,
                sampled_labels=labels,
                caveat='Sparse1Hz samples; low motion and gripper cycles do not prove task stagnation or failed grasp. No object pose/contact/per-step goal predicates available in this archive.',
                downloaded_bytes=obj.downloaded,range_requests=obj.requests,elapsed_seconds=time.time()-started,
                new_training_jobs=0)
            target=dest/('probe.json' if probe else 'review.json')
            target.write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
    return raw


def visuals(index):
    entry=json.loads((ROOT/'review_plan.json').read_text())['episodes'][index]
    row,selected=entry['exposure'],entry['selection'];dest=ROOT/row['root_id']
    raw=json.loads((dest/'review.json').read_text())
    if (dest/'visual_samples.json').exists():return json.loads((dest/'visual_samples.json').read_text())
    merged=[]
    for s in raw['all_segments']:
        if merged and merged[-1]['subtask']==s['subtask'] and merged[-1]['end']==s['start']:
            merged[-1]['end']=s['end']
        else:merged.append(dict(s))
    windows=[]
    for s in sorted(merged,key=lambda s:s['end']-s['start'],reverse=True):
        if s['subtask']:
            windows.append(dict(start_frame=s['start'],end_frame=s['end']-1,label=s['subtask'],reason='long unchanged-label interval'))
        if len(windows)==2:break
    if raw['low_motion_runs']:
        low=raw['low_motion_runs'][0]
        windows[-1]=dict(start_frame=low['start_frame'],end_frame=low['end_frame'],
            label=' | '.join(sorted({s['subtask'] for s in merged if s['start']<=low['end_frame'] and s['end']>low['start_frame']})),
            reason='longest1Hz low-motion candidate')
    obj=ObjectFile(video_eval_adapter.aoss_client(),selected['source_archive'])
    with tarfile.open(fileobj=obj,mode='r:') as archive:
        m=archive.getmember('episode.hdf5')
        with h5py.File(MemberFile(obj,m.offset_data,m.size),'r') as f:
            cameras=['observation.rgb.zed_link_camera_0','observation.rgb.left_realsense_link_camera_0','observation.rgb.right_realsense_link_camera_0']
            for number,w in enumerate(windows):
                ids=np.linspace(w['start_frame'],w['end_frame'],6).astype(int).tolist();w['frames']=ids
                canvas=Image.new('RGB',(1344,830),'white');draw=ImageDraw.Draw(canvas)
                draw.text((8,5),f"{row['task']} / instance{row['instance_id']} / {row['actual_draws']} training draws",fill='black')
                draw.text((8,25),f"{w['reason']}: {w['start_frame']/30:.1f}-{w['end_frame']/30:.1f}s",fill='black')
                for j,line in enumerate(textwrap.wrap(w['label'],160)):
                    draw.text((8,45+j*15),line,fill='black')
                for col,frame in enumerate(ids):
                    draw.text((col*224+4,90),f'frame{frame} / {frame/30:.1f}s',fill='black')
                    for cam,key in enumerate(cameras):
                        pixels=f[key][frame];assert pixels.ndim==3 and pixels.shape[-1]==3
                        tile=Image.fromarray(pixels).resize((224,224))
                        canvas.paste(tile,(col*224,110+cam*234))
                        draw.text((col*224+3,110+cam*234),['HEAD','LEFT','RIGHT'][cam],fill='red')
                name=f'window_{number:02d}.jpg';canvas.save(dest/name,quality=92)
                w['contact_sheet']=name
    result=dict(root_id=row['root_id'],task=row['task'],windows=windows,
        downloaded_bytes=obj.downloaded,range_requests=obj.requests,
        scope='2 selected intervals x6 temporal samples x3 cameras; not a complete continuous video review')
    (dest/'visual_samples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def dense(index,start_seconds,end_seconds):
    entry=json.loads((ROOT/'review_plan.json').read_text())['episodes'][index]
    row,selected=entry['exposure'],entry['selection'];dest=ROOT/row['root_id']
    obj=ObjectFile(video_eval_adapter.aoss_client(),selected['source_archive'])
    with tarfile.open(fileobj=obj,mode='r:') as archive:
        m=archive.getmember('episode.hdf5')
        with h5py.File(MemberFile(obj,m.offset_data,m.size),'r') as f:
            ids=np.arange(int(start_seconds*30),int(end_seconds*30)+1,3)
            action=np.stack([f['action'][int(i)] for i in ids]);state=np.stack([f['observation.state'][int(i)] for i in ids])
            name=f'dense_{start_seconds}_{end_seconds}'
            np.savez_compressed(dest/(name+'.npz'),frame=ids,action=action,state=state)
            moments=np.linspace(ids[0],ids[-1],7).astype(int).tolist()
            canvas=Image.new('RGB',(7*224,830),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,10),f"{row['task']} instance{row['instance_id']} / dense follow-up {start_seconds}-{end_seconds}s",fill='black')
            for col,frame in enumerate(moments):
                draw.text((col*224+4,70),f'{frame/30:.1f}s / frame{frame}',fill='black')
                for cam,key in enumerate(['observation.rgb.zed_link_camera_0','observation.rgb.left_realsense_link_camera_0','observation.rgb.right_realsense_link_camera_0']):
                    canvas.paste(Image.fromarray(f[key][frame]).resize((224,224)),(col*224,110+234*cam))
            canvas.save(dest/(name+'.jpg'),quality=92)
            result=dict(task=row['task'],start_seconds=start_seconds,end_seconds=end_seconds,
                action_state_hz=10,samples=len(ids),camera_frames=moments,
                downloaded_bytes=obj.downloaded,range_requests=obj.requests,
                gripper_width_min=[float(state[:,24:26].sum(1).min()),float(state[:,49:51].sum(1).min())],
                gripper_width_max=[float(state[:,24:26].sum(1).max()),float(state[:,49:51].sum(1).max())],
                arm_qpos_range_max=float(np.ptp(state[:,list(range(3,10))+list(range(28,35))],axis=0).max()))
            (dest/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,default=0);parser.add_argument('--probe',action='store_true')
    parser.add_argument('--all',action='store_true');parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--visual',action='store_true')
    parser.add_argument('--dense-start',type=int);parser.add_argument('--dense-end',type=int)
    args=parser.parse_args()
    if args.all:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(visuals if args.visual else review,i):i for i in range(12)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    r=future.result();print(json.dumps({k:r[k] for k in ['task','sampled_frames','downloaded_bytes','range_requests','elapsed_seconds','low_motion_runs'] if k in r},ensure_ascii=False),flush=True)
                except Exception as error:
                    print(json.dumps({'index':futures[future],'error_type':type(error).__name__,'error':str(error)[:300]}),flush=True)
    else:
        r=dense(args.index,args.dense_start,args.dense_end) if args.dense_start is not None else (visuals(args.index) if args.visual else review(args.index,probe=args.probe))
        print(json.dumps({k:r[k] for k in ['task','sampled_frames','downloaded_bytes','range_requests','elapsed_seconds','low_motion_runs','longest_recorded_label_segments','windows','samples','gripper_width_min','gripper_width_max','arm_qpos_range_max'] if k in r},ensure_ascii=False),flush=True)
