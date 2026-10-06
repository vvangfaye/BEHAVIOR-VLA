#!/usr/bin/env python3
"""Stage explicit original replay episodes via OSS, respecting shared file indices."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import pyarrow.parquet as pq
from subtask_trajectory import CAMERAS
from subtask_sample_panel import file_sha256
from strip_depth_from_info import main as strip_depth


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def relative_path(value):
    path=PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in value or '\n' in value or '\r' in value:
        raise ValueError('Replay file path escapes dataset root')
    return path.as_posix()


def metadata_inventory(root):
    root=Path(root)
    files=sorted((root/'meta').rglob('*'))
    inventory=[]
    for path in files:
        if path.is_symlink():raise ValueError('Source metadata must not contain symlinks')
        if path.is_file():
            inventory.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=file_sha256(path)))
    if not inventory or not (root/'meta/info.json').is_file():raise ValueError('Missing source metadata')
    return inventory


def required_files(root,episodes,held_out):
    if not episodes or any(type(i) is not int or i<0 for i in episodes) or len(set(episodes))!=len(episodes):
        raise ValueError('Replay needs explicit unique nonnegative episode IDs')
    if not isinstance(held_out,list) or any(not isinstance(p,list) or len(p)!=2 or not isinstance(p[0],str) or type(p[1]) is not int for p in held_out):
        raise ValueError('Held-out groups must be [task_name, integer_instance] pairs')
    reserved={tuple(p) for p in held_out}
    info=json.loads((Path(root)/'meta/info.json').read_text())
    if info.get('codebase_version')!='v3.0':raise ValueError('Replay staging requires LeRobot v3 metadata')
    for key in CAMERAS:
        if info.get('features',{}).get(key,{}).get('dtype')!='video':raise ValueError('Missing RGB video feature')
    task_names={}
    tasks_path=Path(root)/'meta/tasks.jsonl'
    if tasks_path.is_file():
        for line in tasks_path.read_text().splitlines():
            row=json.loads(line)
            if row.get('task_name'):task_names[row['task_index']]=row['task_name']
    records={}
    for path in sorted((Path(root)/'meta/episodes').glob('chunk-*/*.parquet')):
        for row in pq.read_table(path).to_pylist():
            i=row['episode_index']
            if i in records:raise ValueError('Duplicate source episode metadata')
            records[i]=row
    wanted=set();selected=[]
    for i in episodes:
        if i not in records:raise ValueError('Unknown replay episode ID')
        row=records[i];task=task_names.get(row['task_index'],row.get('task_name',row['tasks'][0]));instance=row['task_instance_id']
        if (task,instance) in reserved:raise ValueError('Replay contains held-out evaluation instance')
        if len(row['tasks'])!=1 or type(instance) is not int or type(row['length']) is not int or row['length']<=0:
            raise ValueError('Ambiguous original episode metadata')
        def format_path(template,chunk,file,video=None):
            if type(chunk) is not int or chunk<0 or type(file) is not int or file<0:raise ValueError('Invalid shared file index')
            return relative_path(template.format(chunk_index=chunk,file_index=file,video_key=video))
        wanted.add(format_path(info['data_path'],row['data/chunk_index'],row['data/file_index']))
        for key in CAMERAS:
            a=row[f'videos/{key}/from_timestamp'];b=row[f'videos/{key}/to_timestamp']
            if not 0<=a<b:raise ValueError('Invalid episode video timestamps')
            wanted.add(format_path(info['video_path'],row[f'videos/{key}/chunk_index'],row[f'videos/{key}/file_index'],key))
        selected.append(dict(episode_index=i,task_name=task,task_instance_id=instance,length=row['length']))
    return sorted(wanted),selected


def freeze(root,source,episodes,held_out,bucket):
    from nebula.submit_oss import object_key
    prefix=object_key(source,bucket.bucket_name)
    paths,selected=required_files(root,episodes,held_out)
    objects=[]
    for path in paths:
        head=bucket.head_object(prefix+'/'+path)
        if head.content_length<=0 or not head.etag:raise ValueError('Missing or empty replay object')
        objects.append(dict(path=path,bytes=head.content_length,etag=head.etag))
    metadata=metadata_inventory(root)
    plan=dict(schema_version=1,kind='original_replay_staging',source='oss://'+bucket.bucket_name+'/'+prefix,
              episodes=selected,held_out_instances=held_out,objects=objects,metadata=metadata,
              rgb_features=sorted(CAMERAS),download_bytes=sum(o['bytes'] for o in objects),
              metadata_bytes=sum(o['bytes'] for o in metadata),original_replay_frames=sum(r['length'] for r in selected))
    plan['fingerprint']=digest(plan)
    return plan


def validate(plan,root):
    if plan.get('schema_version')!=1 or plan.get('kind')!='original_replay_staging':raise ValueError('Unknown replay plan')
    if plan.get('fingerprint')!=digest({k:v for k,v in plan.items() if k!='fingerprint'}):raise ValueError('Replay plan fingerprint differs')
    if metadata_inventory(root)!=plan['metadata']:raise ValueError('Original metadata changed after replay selection')
    paths,episodes=required_files(root,[r['episode_index'] for r in plan['episodes']],plan['held_out_instances'])
    if episodes!=plan['episodes'] or paths!=[o['path'] for o in plan['objects']] or plan['rgb_features']!=sorted(CAMERAS):
        raise ValueError('Replay file selection differs from source episode indices')
    if any(type(o['bytes']) is not int or o['bytes']<=0 or not isinstance(o['etag'],str) or not o['etag'] for o in plan['objects']):
        raise ValueError('Invalid frozen OSS object identity')
    if plan['download_bytes']!=sum(o['bytes'] for o in plan['objects']) or plan['metadata_bytes']!=sum(o['bytes'] for o in plan['metadata']):
        raise ValueError('Replay storage budget differs from object inventory')


def stage(plan,metadata_root,output,bucket,*,workers=4,reserve_bytes=8*1024**3):
    validate(plan,metadata_root)
    from nebula.submit_oss import object_key
    prefix=object_key(plan['source'],bucket.bucket_name)
    output=Path(output)
    if output.exists():raise ValueError('Replay staging requires a fresh target')
    if not 1<=workers<=8 or reserve_bytes<0:raise ValueError('Invalid staging resource limit')
    available=shutil.disk_usage(output.parent).free
    needed=plan['download_bytes']+plan['metadata_bytes']+reserve_bytes
    if available<needed:raise ValueError('Insufficient local disk for selected replay and reserve')
    output.mkdir()
    # All metadata stays intact so selected global IDs and shared timestamps resolve.
    # Depth removal affects only this owned working copy; the frozen source is untouched.
    for item in plan['metadata']:
        target=output/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(metadata_root)/item['path'],target)
        if file_sha256(target)!=item['sha256']:raise ValueError('Metadata copy checksum differs')
    strip_depth(str(output))
    def copy_object(item):
        key=prefix+'/'+item['path']
        head=bucket.head_object(key)
        if head.content_length!=item['bytes'] or head.etag!=item['etag']:raise ValueError('OSS replay object changed after selection')
        target=output/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_suffix(target.suffix+'.partial');sha=hashlib.sha256();count=0
        response=bucket.get_object(key,headers={'If-Match':item['etag']})
        try:
            if response.etag!=item['etag']:raise ValueError('OSS replay download identity differs')
            with partial.open('xb') as stream:
                while block:=response.read(8*1024*1024):
                    stream.write(block);sha.update(block);count+=len(block)
        finally:
            response.resp.response.close()
        if count!=item['bytes']:raise ValueError('Incomplete replay download')
        partial.rename(target)
        return {**item,'sha256':sha.hexdigest()}
    with ThreadPoolExecutor(max_workers=workers) as pool:receipts=list(pool.map(copy_object,plan['objects']))
    (output/'selected_episodes.txt').write_text(''.join(str(r['episode_index'])+'\n' for r in plan['episodes']))
    report=dict(schema_version=1,kind='original_replay_staged',plan_fingerprint=plan['fingerprint'],
                source=plan['source'],episodes=plan['episodes'],objects=receipts,
                bytes_downloaded=sum(o['bytes'] for o in receipts),depth_features_removed=True,
                reader_verified=False,available_disk_bytes_before=available,required_disk_bytes=needed)
    (output/'replay_staging_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze')
    for name in ('metadata-root','episodes','held-out-instances','output'):f.add_argument('--'+name,type=Path,required=True)
    f.add_argument('--source',required=True)
    s=commands.add_parser('stage')
    for name in ('metadata-root','plan','output'):s.add_argument('--'+name,type=Path,required=True)
    s.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from nebula.submit_oss import OSSClient
    try:
        bucket=OSSClient().bucket
        if args.command=='freeze':
            if args.output.exists():raise ValueError('Replay selection needs a fresh plan path')
            plan=freeze(args.metadata_root,args.source,[int(x) for x in args.episodes.read_text().split()],json.loads(args.held_out_instances.read_text()),bucket)
            args.output.write_text(json.dumps(plan,indent=2)+'\n')
            print(json.dumps({k:plan[k] for k in ('fingerprint','download_bytes','original_replay_frames')}))
        else:
            report=stage(json.loads(args.plan.read_text()),args.metadata_root,args.output,bucket,workers=args.workers)
            print(json.dumps({k:report[k] for k in ('kind','plan_fingerprint','bytes_downloaded','reader_verified')}))
    except ValueError as error:
        print(str(error),file=sys.stderr);return 1
    except Exception as error:
        # OSS exceptions can contain signed request information: print only type.
        print('Replay staging failed ('+type(error).__name__+'); no automatic retry.',file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
