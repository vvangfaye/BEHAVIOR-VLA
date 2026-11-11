import ast,hashlib,io,json,os,subprocess,tarfile,urllib.request,zipfile
from pathlib import Path


def run(config):
    root=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/dev01')
    stage=root.parent/'prepare02';repo=stage/'source/repo';site=root/'site';site.mkdir(exist_ok=True)
    prior=json.loads((root/'result.json').read_text())
    assert set(prior['probes'])=={'sim','policy'},'Wait for the original probes before changing dependencies'
    scene=repo/'BEHAVIOR-1K/OmniGibson/omnigibson/scenes/scene_base.py'
    scene.write_text(scene.read_text().replace('from omnigibson.utils.constants import ROBOT_CATEGORY, STRUCTURAL_DOOR_CATEGORIES','from omnigibson.utils.constants import STRUCTURAL_DOOR_CATEGORIES'))
    for filename,funcname in [('point_sampling.py','random_sampling'),('point_normalization.py','normalize_pc'),('point_item_ops.py','normalize_sampled_point_payload')]:
        path=repo/'src/serf_b1k/utils'/filename
        text=path.read_text()
        if '\nimport open3d as o3d\n' in text:
            text=text.replace('import open3d as o3d\n','')
            node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==funcname)
            lines=text.splitlines(True);lines.insert(node.body[0].lineno-1,'    import open3d as o3d\n')
            path.write_text(''.join(lines))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(config['wheel_get'],timeout=60) as r:data=r.read()
    assert hashlib.sha256(data).hexdigest()==config['wheel_sha256']
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for item in tar:
            assert item.isfile() and item.name.endswith('.whl')
            with zipfile.ZipFile(io.BytesIO(tar.extractfile(item).read())) as z:
                for name in z.namelist():
                    path=Path(name);assert not path.is_absolute() and '..' not in path.parts
                z.extractall(site)
    env=dict(os.environ)
    env.update(OMNIGIBSON_DATA_PATH=str(stage/'data'),OMNIGIBSON_DISABLE_TORCH_COMPILE='1',JAX_PLATFORMS='cpu',
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
               SERF_MAP_ROOT=str(stage/'assets/maps'),PYTHONNOUSERSITE='1',TMPDIR='/tmp',PYTHONPATH=str(site))
    runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
    library=json.loads((runtime/'library_snapshots/behavior-libxml-repair3-20261008-011710/_READY').read_text())['library_path']
    env['LD_LIBRARY_PATH']=':'.join([library,str(runtime/'graphics-runtime/usr/lib/x86_64-linux-gnu'),str(runtime/'envs/b1k-eval/lib'),env.get('LD_LIBRARY_PATH','')])
    result={'root':str(root),'stage':str(stage),'site_path':str(site),'probes':{},'iteration':config['iteration']}
    for mode,environment in [('sim','b1k-eval'),('policy','b1k-pi05')]:
        p=subprocess.run([str(runtime/'envs'/environment/'bin/python'),str(stage/'source/import_probe.py'),str(repo),mode],
                         env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
        (root/f'{mode}_{config["iteration"]}.log').write_text(p.stdout)
        lines=[s.split('=',1)[1] for s in p.stdout.splitlines() if s.startswith('SERF_PROBE_JSON=')]
        result['probes'][mode]=json.loads(lines[-1]) if lines else {'passed':False,'returncode':p.returncode,'tail':p.stdout[-6000:]}
        (root/f'iteration_{config["iteration"]}.json').write_text(json.dumps(result,indent=2))
    result['passed']=all(x['passed'] for x in result['probes'].values())
    payload=json.dumps(result,indent=2).encode();(root/'result.json').write_bytes(payload)
    with opener.open(urllib.request.Request(config['put'],data=payload,method='PUT'),timeout=90):pass
    print(json.dumps(result),flush=True)
