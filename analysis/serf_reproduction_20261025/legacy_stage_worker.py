import hashlib,json,os,shutil,subprocess,tarfile,tempfile,traceback,urllib.request
from pathlib import Path


def run(config):
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=False)
    scratch=Path(tempfile.mkdtemp(prefix='serf-legacy-stage-'))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result={'root':str(root),'stage':str(root),'phase':'download','passed':False}
    def publish():
        data=json.dumps(result,indent=2).encode();(root/'stage_result.json').write_bytes(data)
        with opener.open(urllib.request.Request(config['put'],data=data,method='PUT'),timeout=90):pass
    try:
        archive=scratch/'assets.tar.gz';h=hashlib.sha256();size=0
        with opener.open(config['get'],timeout=120) as r,archive.open('wb') as f:
            for b in iter(lambda:r.read(8*1024**2),b''):size+=len(b);h.update(b);f.write(b)
        assert size==config['bytes'] and h.hexdigest()==config['sha256']
        result['phase']='extract';publish()
        with tarfile.open(archive,'r:gz') as tar:
            for item in tar:
                rel=Path(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
                dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(item) as src,dest.open('xb') as f:shutil.copyfileobj(src,f)
        manifest=json.loads((root/'legacy_manifest.json').read_text())
        for rel,item in manifest.items():
            p=root/rel;assert p.stat().st_size==item['bytes'];h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            assert h.hexdigest()==item['sha256'],rel
        old=root.parent/'prepare02';runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
        (root/'data/omnigibson.key').symlink_to(runtime/'omnigibson-data/omnigibson.key')
        (root/'data/2025-challenge-task-instances').symlink_to(old/'assets/instances',target_is_directory=True)
        maps=root/'assets/maps';maps.mkdir(parents=True)
        for name in ['exported_neural_points','map_models']:(maps/name).symlink_to(old/'assets/maps'/name,target_is_directory=True)
        (maps/'robot').symlink_to(root/'data/omnigibson-robot-assets/models/r1pro',target_is_directory=True)
        result.update(phase='import',verified_files=len(manifest),asset_version=(root/'data/behavior-1k-assets/VERSION').read_text().strip())
        probe=(old/'source/import_probe.py').read_text().replace("result['robot_sampler_loaded']=True", "result['robot_sampler_loaded']=True\n        xyz, _ = sampler.get_points(np.zeros(28))\n        assert xyz.shape == (12579, 3) and np.isfinite(xyz).all()\n        result['robot_fk_points_verified']=list(xyz.shape)")
        path=root/'import_probe.py';path.write_text(probe)
        env=dict(os.environ);site=root.parent/'dev01/site'
        library=json.loads((runtime/'library_snapshots/behavior-libxml-repair3-20261008-011710/_READY').read_text())['library_path']
        env.update(OMNIGIBSON_DATA_PATH=str(root/'data'),OMNIGIBSON_DISABLE_TORCH_COMPILE='1',JAX_PLATFORMS='cpu',
            PYTHONPATH=str(site),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            SERF_MAP_ROOT=str(maps),LD_LIBRARY_PATH=':'.join([library,str(runtime/'graphics-runtime/usr/lib/x86_64-linux-gnu'),str(runtime/'envs/b1k-eval/lib')]))
        p=subprocess.run([str(runtime/'envs/b1k-eval/bin/python'),str(path),str(old/'source/repo'),'sim'],env=env,capture_output=True,text=True,timeout=300)
        (root/'import.log').write_text(p.stdout+p.stderr)
        lines=[x.split('=',1)[1] for x in p.stdout.splitlines() if x.startswith('SERF_PROBE_JSON=')]
        result['probes']={'sim':json.loads(lines[-1]) if lines else {'passed':False,'tail':(p.stdout+p.stderr)[-7000:]}}
        result['passed']=result['probes']['sim']['passed'];result['phase']='complete'
    except BaseException:result['error']=traceback.format_exc()
    publish();print(json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)
