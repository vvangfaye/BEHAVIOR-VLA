"""Stage the accepted four-source bundle on a Nebula worker; never train.

The default reuses original expert data through the existing OSS mount and
copies only the new datasets. --copy-expert explicitly stages the frozen expert
object list onto local disk after checking its full space requirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sco'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_retraining_data import safe_extract, save, sha, verify_dataset


def mounted_object(mount, target):
    if target['bucket'] != 'traffic-model': raise ValueError('Unexpected dataset bucket')
    relative=PurePosixPath(target['key'])
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe object key')
    return Path(mount)/str(relative)


def verified_copy(source, destination, target):
    if destination.exists():
        if destination.stat().st_size==target['bytes'] and sha(destination)==target['sha256']:return
        raise ValueError('Existing staged object has different bytes')
    if source.stat().st_size != target['bytes']: raise ValueError('Mounted object size differs')
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_suffix(destination.suffix+'.partial')
    shutil.copyfile(source,partial)
    if sha(partial)!=target['sha256']: raise ValueError('Mounted object SHA256 differs')
    partial.replace(destination)


def stage(manifest_path, mount, output, *, copy_expert=False):
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('data_ready') is not True: raise ValueError('Four-source data has not passed acceptance')
    if manifest.get('training_allowed') is not False: raise ValueError('This program accepts data-only manifests')
    stamp=sha(manifest_path)
    if output.exists():
        marker=output/'_STAGED_DATA.json'
        if marker.exists() and json.loads(marker.read_text())['manifest_sha256']==stamp:return json.loads(marker.read_text())
        raise ValueError('Use a fresh staging directory; existing data is preserved')
    new_bytes=sum(r['target']['bytes'] for r in manifest['archives'])
    # Archive plus extracted dataset plus safety margin; expert copy is separate.
    minimum=new_bytes*3+16*1024**3
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<minimum: raise ValueError('Insufficient local staging space')
    temp=Path(tempfile.mkdtemp(prefix=output.name+'.partial-',dir=output.parent))
    cache=temp/'archives';cache.mkdir()
    roots={};indexes={}
    expert_target=manifest['original_expert']['target']
    expert_archive=cache/'expert.tar.gz'
    verified_copy(mounted_object(mount,expert_target),expert_archive,expert_target)
    expert_unpack=temp/'expert_metadata';safe_extract(expert_archive,expert_unpack)
    expert=expert_unpack/'expert_original'
    source=json.loads((expert_unpack/'source_objects.json').read_text())
    if copy_expert:
        count=sum(r['bytes'] for r in source['objects'] if not r['relative_path'].startswith('meta/'))
        if shutil.disk_usage(temp).free<count+minimum: raise ValueError('Insufficient disk for full original expert copy')
        for r in source['objects']:
            if r['relative_path'].startswith('meta/'):continue
            origin=mounted_object(mount,{'bucket':source['bucket'],'key':r['key']})
            if origin.stat().st_size!=r['bytes']:raise ValueError('Original mounted object size differs')
            target=expert/r['relative_path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,target)
    else:
        original=Path(mount)/source['prefix']
        for name in ('data','videos'):
            if not (original/name).is_dir():raise ValueError('Original expert mount is absent')
            (expert/name).symlink_to(original/name,target_is_directory=True)
    roots['expert_original']=str((output/'expert_metadata/expert_original').resolve())
    for kind in ('replay','recovered'):
        indexes[kind]=str(output/'expert_metadata/expert_original'/(kind+'_index.json'))
    for receipt in manifest['archives']:
        root_id=receipt['root_id'];target=receipt['target']
        packed=cache/(root_id+'.tar')
        verified_copy(mounted_object(mount,target),packed,target)
        unpack=temp/'datasets'/root_id
        safe_extract(packed,unpack)
        reports=list(unpack.rglob('export_report.json'))
        if len(reports)!=1:raise ValueError('Ambiguous exported dataset')
        dataset=reports[0].parent;report=json.loads(reports[0].read_text())
        result=verify_dataset(dataset,report['repo_id'],permit_empty_labels=receipt['kind']=='full_episode')
        if result['episodes']!=receipt['acceptance']['episodes'] or result['frames']!=receipt['acceptance']['frames']:
            raise ValueError('Worker reader counts differ from accepted source')
        relative=dataset.relative_to(temp)
        roots[root_id]=str(output/relative)
        idx=receipt['index_target'];local=temp/'indexes'/(root_id+'.json')
        verified_copy(mounted_object(mount,idx),local,idx)
        indexes[root_id]=str(output/'indexes'/(root_id+'.json'))
        packed.unlink()
    expert_archive.unlink()
    marker={'kind':'b1k_four_source_worker_staging','manifest_sha256':stamp,
        'source_fingerprint':manifest['source_fingerprint'],'roots':roots,'indexes':indexes,
        'expert_storage':'local_copy' if copy_expert else 'existing_oss_mount',
        'new_datasets_verified_by_real_reader':True,'training_executed':False,'training_allowed':False}
    save(temp/'roots.json',roots);save(temp/'indexes.json',indexes)
    save(temp/'_STAGED_DATA.json',marker)
    temp.rename(output)
    return marker


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--oss-mount',type=Path,default=Path('/data/oss_bucket_0'));p.add_argument('--output',type=Path,required=True);p.add_argument('--copy-expert',action='store_true')
    a=p.parse_args();print(json.dumps(stage(a.manifest,a.oss_mount,a.output,copy_expert=a.copy_expert),indent=2))
