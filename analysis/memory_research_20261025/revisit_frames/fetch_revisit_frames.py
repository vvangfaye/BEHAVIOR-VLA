from pathlib import Path
import json,sys,traceback
import numpy as np,pyarrow.parquet as pq,av
sys.path.insert(0,'analysis/memory_research_20261025/witness_frames')
from fetch_frames import RangeReader
base=Path('analysis/memory_research_20261025/revisit_frames');base.mkdir(exist_ok=True)
cache=Path('analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1/samples')
cached=[]
for p in cache.glob('*.npz'):
 with np.load(p,allow_pickle=False) as x:
  if int(x['episode_index'])==19018:cached.append({'file':str(p),'frame_index':int(x['frame_index'])})
m=pq.read_table('analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1/dataset/meta/episodes/chunk-095/file-000.parquet',filters=[('episode_index','=',19018)]).to_pylist()[0]
frames=[3101,5037,6551,10639];budget=20_000_000;total=0;sources=[]
for short,key in [('head','zed_link_camera_0'),('left','left_realsense_link_camera_0'),('right','right_realsense_link_camera_0')]:
 v='observation.rgb.'+key;f=int(m['videos/'+v+'/file_index']);chunk=int(m['videos/'+v+'/chunk_index']);offset=float(m['videos/'+v+'/from_timestamp']);rel=f'videos/{v}/chunk-{chunk:03d}/file-{f:03d}.mp4';url='https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/resolve/main/'+rel
 source={'camera':short,'url':url,'offset_seconds':offset,'frames':[]};reader=None
 try:
  reader=RangeReader(url,budget=budget-total,block=128*1024)
  container=av.open(reader);stream=container.streams.video[0]
  for t in frames:
   target=offset+t/30;container.seek(int(target/stream.time_base),stream=stream,backward=True)
   for fr in container.decode(stream):
    actual=float(fr.pts*stream.time_base)
    if actual+1e-5>=target:
     imagepath=base/f'{short}_{t}.png';fr.to_image().save(imagepath)
     source['frames'].append({'episode_frame':t,'requested_pts':target,'decoded_pts':actual,'local_png':str(imagepath)});break
   else:raise RuntimeError('No decoded target frame')
  container.close()
 except Exception as exc:source['error']=str(exc);traceback.print_exc()
 finally:
  if reader is not None:source['downloaded_bytes']=reader.used;total+=reader.used;source['video_bytes']=reader.length
  sources.append(source)
  receipt={'episode_index':19018,'raw_episode_id':950310,'fps':30,'budget_bytes':budget,'downloaded_bytes_total':total,'existing_npz_frames':cached,'requested_episode_frames':frames,'sources':sources,'scope':'Read only selected frames from existing public video; no rendering, simulation or new human collection.'}
  (base/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({'camera':short,'count':len(source['frames']),'bytes':source.get('downloaded_bytes'),'total_bytes':total,'error':source.get('error')}),flush=True)
 if total>=budget:break
