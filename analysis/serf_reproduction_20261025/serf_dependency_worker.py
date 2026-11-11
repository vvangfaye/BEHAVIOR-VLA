"""Check the actual SERF construction path without downloading model weights."""
import hashlib, io, json, os, shutil, subprocess, tarfile, tempfile, traceback, urllib.request
from pathlib import Path


def run(config):
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=False)
    local=Path(tempfile.mkdtemp(prefix='serf-deps02-'))
    result={'root':str(root),'source_sha256':config['source_sha256'],'passed':False}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
    stage=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/legacy01')
    site=root/'site';shutil.copytree(Path(config.get('base_dependency_site',root.parent/'policydeps01/site')),site)
    env=dict(os.environ);env.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMNIGIBSON_DATA_PATH=str(stage/'data'))
    ready=json.loads((runtime/'library_snapshots/behavior-libxml-repair3-20261008-011710/_READY').read_text())
    env['LD_LIBRARY_PATH']=ready['library_path']+':'+str(runtime/'envs/b1k-eval/lib')+':'+env.get('LD_LIBRARY_PATH','')
    try:
        names=['trimesh','lxml','networkx','six']
        # Existing compatible simulation packages supply only missing policy dependencies.
        inspect="import importlib.util,importlib.metadata,json;out={}\nfor name in "+repr(names)+":\n s=importlib.util.find_spec(name)\n out[name]={'origin':s.origin if s else None}\n if s:\n  d=importlib.metadata.distribution(name);out[name].update(version=d.version,metadata_path=str(d._path))\nprint(json.dumps(out))"
        inspect_env=dict(env);inspect_env['PYTHONPATH']=str(site)
        installed=json.loads(subprocess.check_output([str(runtime/'envs/b1k-pi05/bin/python'),'-c',inspect],env=inspect_env,text=True))
        available=json.loads(subprocess.check_output([str(runtime/'envs/b1k-eval/bin/python'),'-c',inspect],env=env,text=True))
        result['dependencies']={}
        for name in names:
            if installed[name]['origin']:
                result['dependencies'][name]={'reused_policy_version':installed[name]['version']};continue
            item=available[name];assert item['origin'],name
            src=Path(item['origin'])
            if src.name=='__init__.py':shutil.copytree(src.parent,site/name)
            else:shutil.copy2(src,site/src.name)
            metadata=Path(item['metadata_path']);shutil.copytree(metadata,site/metadata.name)
            result['dependencies'][name]={'copied_from_simulation':str(src),'version':item['version']}
        import threading
        for attempt in range(4):
            try:
                with opener.open(config['source_get'],timeout=90) as r:data=r.read()
                break
            except (OSError,urllib.error.URLError):
                if attempt==3:raise
                threading.Event().wait(2)
        assert hashlib.sha256(data).hexdigest()==config['source_sha256']
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for member in tar:
                rel=Path(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
                dest=local/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(tar.extractfile(member).read())
        repo=local/'repo';(repo/'BEHAVIOR-1K/datasets').mkdir(parents=True)
        (repo/'BEHAVIOR-1K/datasets/omnigibson-robot-assets').symlink_to(stage/'data/omnigibson-robot-assets',target_is_directory=True)
        env['PYTHONPATH']=str(site)+':'+str(root.parent/'dev01/site')
        probe="import sys,types,json\nfrom pathlib import Path\nrepo=Path("+repr(str(repo))+")\nsys.path[:0]=[str(repo/'src'),str(repo/'external/openpi/src'),str(repo/'external/openpi/packages/openpi-client/src')]\nog=types.ModuleType('omnigibson');og.__path__=[str(repo/'BEHAVIOR-1K/OmniGibson/omnigibson')];sys.modules['omnigibson']=og\n"
        probe+="""import jax,jax.numpy as jnp,numpy as np
from flax import nnx
from serf_b1k.models.robot_fk_jax import RobotFKJax
from serf_b1k.training.config import get_config
fk=RobotFKJax(repo/'BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf')
outputs=fk(jnp.zeros((1,256),dtype=jnp.float32))
assert all(x.shape==(1,3) and bool(jnp.isfinite(x).all()) for x in outputs)
print('ROBOT_FK_FORWARD_PASSED',flush=True)
cfg=get_config('pi_serf_behavior_b1k_fast--4d_env_robot_feat_map--50t_lora--task-0022')
model=nnx.eval_shape(cfg.model.create,jax.random.key(0))
print('SERF_MODEL_CONSTRUCTION_PASSED',len(jax.tree.leaves(nnx.state(model))),flush=True)
"""
        (root/'probe.py').write_text(probe)
        p=subprocess.run([str(runtime/'envs/b1k-pi05/bin/python'),'-u',str(root/'probe.py')],cwd=repo,env=env,
            capture_output=True,text=True,timeout=900)
        (root/'probe.log').write_text(p.stdout+p.stderr)
        result.update(site=str(site),returncode=p.returncode,passed=p.returncode==0 and 'SERF_MODEL_CONSTRUCTION_PASSED' in p.stdout,
                      stdout=p.stdout[-5000:],stderr=p.stderr[-7000:])
    except BaseException:result['error']=traceback.format_exc()
    finally:
        data=json.dumps(result,indent=2).encode();(root/'result.json').write_bytes(data)
        with opener.open(urllib.request.Request(config['result_put'],data=data,method='PUT'),timeout=60):pass
        print(json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)
