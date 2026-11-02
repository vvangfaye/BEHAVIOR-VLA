"""Validate released map files and differential-test pure sampling functions."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import re
import sys
import typing
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'site'))
import h5py
import torch
from audit_evidence import check_file, write, OLD, MT, E20, AUTHOR


def pure_functions(path, names=None, extra=None):
    """Execute real NumPy function bodies without importing optional render libraries."""
    tree=ast.parse(path.read_text())
    selected=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))
              and (names is None or x.name in names)]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+selected,type_ignores=[])
    namespace={**vars(typing),'np':np,'re':re,'Path':Path,**(extra or {})}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace


def main():
    files=[]
    files += [(E20/'assets'/x['destination'],x) for x in json.loads((E20/'map_manifest.json').read_text())['items']]
    old=json.loads((OLD/'download_manifest.json').read_text())['items']
    files += [(OLD/'assets'/x['destination'],x) for x in old if x['destination'].startswith('maps/map_models/') or x['destination'].startswith('maps/robot/')]
    small=json.loads((MT/'small_manifest.json').read_text())['items']
    files += [(MT/'assets'/x['destination'],x) for x in small if x['destination'].startswith('maps/')]
    checked=[check_file(path,item) for path,item in files]
    assert all(x['passed'] for x in checked)
    summaries=[]
    data={}
    for task,base in [(21,MT),(22,E20),(26,MT)]:
        paths=sorted((base/'assets/maps/exported_neural_points'/f'task-{task:04d}/eval').glob('scene_*.hdf5'))
        assert len(paths)==20
        for path in paths:
            with h5py.File(path) as f:
                xyz=f['initial_points'][:];feat=f['initial_features'][:];ids=f['initial_instance_ids'][:]
                names=json.loads(f.attrs['instance_id_to_name'])
            assert xyz.shape==(len(ids),3) and feat.shape==(len(ids),64)
            assert np.isfinite(xyz).all() and np.isfinite(feat).all() and np.any(feat)
            unmapped=set(map(str,np.unique(ids)))-set(names)
            # Task26 author exports contain the uint16 unlabelled sentinel.
            # It has no task category and is excluded by the author sampler.
            assert unmapped<= {'65535'}, (path,unmapped)
            summaries.append(dict(task_id=task,instance=int(path.stem.split('_')[1]),points=len(ids),feature_dim=64,finite=True,unlabelled_sentinel_points=int((ids==65535).sum())))
            if task not in data:data[task]=(xyz,feat,ids,names)
    source=AUTHOR/'src/serf_b1k'
    run=MT/'runtime/repo/src/serf_b1k'
    category=pure_functions(source/'mapping/utils/category_utils.py')['extract_category_from_name']
    namespaces=[]
    for root in [source,run]:
        ns=pure_functions(root/'utils/point_sampling.py',extra={'extract_category_from_name':category})
        ns.update(pure_functions(root/'utils/point_item_ops.py',extra=ns))
        namespaces.append(ns)
    settings={21:(('dice','teddy_bear','toy_train','board_game'),('bed','bookcase','desk')),
              22:(('gym_shoe','sandal'),('hall_tree',)),
              26:(('bow','swiss_cheese','pillar_candle','butter_cookie','wicker_basket'),('bottom_cabinet','coffee_table'))}
    parity=[]
    for task,(xyz,feat,ids,names) in data.items():
        base=OLD if task==22 else MT
        model=base/'assets/maps/map_models'/f'task-{task:04d}/neural_points/robot'
        sampler=np.load(model/'sampler.npz',allow_pickle=False)
        feature_checkpoint=torch.load(model/'robot_neural_points.pt',map_location='cpu',weights_only=False)
        rfeat=feature_checkpoint['features'].detach().numpy().astype(np.float32)
        # Pure payload sampler treats robot positions opaquely; FK is audited separately.
        key=next(k for k in sampler.files if k in ['local_points_homo','local_points'])
        robot_xyz=sampler[key][:,:3].astype(np.float32)
        assert len(robot_xyz)==len(rfeat) and rfeat.shape[1]==64
        results=[]
        for ns in namespaces:
            np.random.seed(0)
            results.append(ns['sample_robot_map_payload'](total_n=25000,env_instance_ids=ids,
                env_xyz=xyz,env_feat=feat,robot_xyz=robot_xyz,robot_feat=rfeat,id_to_name=names,
                keep_all_categories=settings[task][0],budget_categories=settings[task][1]))
        for k in ['xyz','feat']:np.testing.assert_array_equal(results[0][0][k],results[1][0][k])
        np.testing.assert_array_equal(results[0][1],results[1][1])
        parity.append(dict(task_id=task,input_points=25000,robot_points=int(results[0][1].sum()),
                           feature_dim=64,author_runtime_sampling_bitwise_equal=True))
    write('maps.json',dict(files=checked,all_hashes_passed=True,scene_maps=summaries,
                           differential_sampling=parity,scope='Released HDF5 and robot assets; real NumPy sampling bodies. No simulator/FK execution or neural forward equivalence claimed.'))
    print(json.dumps({'files_verified':len(checked),'maps_verified':len(summaries),'sampling':parity}))


if __name__=='__main__':main()
