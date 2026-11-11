"""Stage a pinned small dependency and check the full serving import path on CPU."""
import base64, hashlib, importlib, io, json, os, subprocess, tarfile, tempfile, time, traceback, urllib.request, zipfile
from pathlib import Path


def run(config):
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=False)
    local=Path(tempfile.mkdtemp(prefix='serf-policy-deps-'))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result={'root':str(root),'source_sha256':config['source_sha256'],'passed':False}
    try:
        wheel=base64.b64decode(config['wheel_base64'])
        assert hashlib.sha256(wheel).hexdigest()==config['wheel_sha256']
        site=root/'site';site.mkdir()
        with zipfile.ZipFile(io.BytesIO(wheel)) as z:
            for name in z.namelist():
                path=Path(name)
                assert not path.is_absolute() and '..' not in path.parts
                if name.endswith('/'):continue
                dest=site/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
        with opener.open(config['source_get'],timeout=90) as r:data=r.read()
        assert hashlib.sha256(data).hexdigest()==config['source_sha256']
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for m in t:
                path=Path(m.name);assert m.isfile() and not path.is_absolute() and '..' not in path.parts
                dest=local/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(t.extractfile(m).read())
        repo=local/'repo'
        modules=['omnigibson.learning.utils.network_utils','b1k.training.config','serf_b1k.training.config',
                 'b1k.policies.policy_config','serf_b1k.policies.policy_config',
                 'b1k.shared.eval_b1k_wrapper','serf_b1k.shared.eval_b1k_wrapper']
        probe="import sys,types,importlib,json\nfrom pathlib import Path\nrepo=Path("+repr(str(repo))+ ")\nsys.path[:0]=[str(repo/'src'),str(repo/'external/openpi/src'),str(repo/'external/openpi/packages/openpi-client/src')]\nog=types.ModuleType('omnigibson');og.__path__=[str(repo/'BEHAVIOR-1K/OmniGibson/omnigibson')];sys.modules['omnigibson']=og\n"
        probe+="modules="+repr(modules)+"\nfor name in modules:importlib.import_module(name);print('IMPORTED='+name,flush=True)\nfrom omnigibson.learning.utils.network_utils import WebsocketPolicyServer\nprint('SERVING_IMPORTS_PASSED',flush=True)\n"
        (root/'probe.py').write_text(probe)
        runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
        ready=json.loads((runtime/'library_snapshots/behavior-libxml-repair3-20261008-011710/_READY').read_text())
        env=dict(os.environ);env.update(JAX_PLATFORMS='cpu',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
            OMNIGIBSON_DATA_PATH='/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/legacy01/data',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONPATH=str(site)+':/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/dev01/site',
            LD_LIBRARY_PATH=ready['library_path']+':'+str(runtime/'envs/b1k-eval/lib')+':'+env.get('LD_LIBRARY_PATH',''))
        p=subprocess.run([str(runtime/'envs/b1k-pi05/bin/python'),str(root/'probe.py')],env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
        (root/'probe.log').write_text(p.stdout)
        result.update(passed=p.returncode==0 and 'SERVING_IMPORTS_PASSED' in p.stdout,
                      returncode=p.returncode,tail=p.stdout[-6000:],site=str(site),wheel_sha256=config['wheel_sha256'])
    except BaseException:result['error']=traceback.format_exc()
    finally:
        data=json.dumps(result,indent=2).encode();(root/'result.json').write_bytes(data)
        with opener.open(urllib.request.Request(config['result_put'],data=data,method='PUT'),timeout=90):pass
        print(json.dumps(result),flush=True)
    until=time.monotonic()+600
    while not result['passed'] and time.monotonic()<until and not (root/'DONE').exists():time.sleep(5)
    if not result['passed']:raise SystemExit(1)
