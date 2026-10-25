"""Bounded CPU staging and import checks; only writes the new SERF directory."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import traceback
import urllib.request


def run(config):
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=False)
    runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result={'root':str(root),'phase':'download','passed':False}

    def publish():
        data=json.dumps(result,indent=2).encode()
        (root/'stage_result.json').write_bytes(data)
        with opener.open(urllib.request.Request(config['result_put'],data=data,method='PUT'),timeout=90):pass

    try:
        for name,item in config['archives'].items():
            target=root/(name+'.tar.gz')
            h=hashlib.sha256();count=0
            with opener.open(item['url'],timeout=90) as response,target.open('wb') as out:
                for data in iter(lambda:response.read(8*1024**2),b''):
                    count+=len(data);assert count<=item['size'];out.write(data);h.update(data)
            assert count==item['size'] and h.hexdigest()==item['sha256']
            dest=root/name;dest.mkdir()
            with tarfile.open(target,'r:gz') as tar:
                for member in tar:
                    p=Path(member.name)
                    assert not p.is_absolute() and '..' not in p.parts and member.isfile()
                    path=dest/p;path.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as source,path.open('xb') as out:shutil.copyfileobj(source,out)
        repo=root/'source/repo';assets=root/'assets'
        data_root=root/'data';data_root.mkdir()
        for name in ('behavior-1k-assets','omnigibson-robot-assets','omnigibson.key'):
            (data_root/name).symlink_to(runtime/'omnigibson-data'/name)
        (data_root/'2025-challenge-task-instances').symlink_to(assets/'instances',target_is_directory=True)
        robot=runtime/'omnigibson-data/omnigibson-robot-assets/models/r1pro'
        assert (robot/'urdf/r1pro.urdf').is_file()
        (assets/'maps/robot').symlink_to(robot,target_is_directory=True)
        dest=repo/'BEHAVIOR-1K/datasets';dest.mkdir(exist_ok=True)
        (dest/'omnigibson-robot-assets').symlink_to(runtime/'omnigibson-data/omnigibson-robot-assets',target_is_directory=True)
        env=dict(os.environ)
        env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                   OMNIGIBSON_DATA_PATH=str(data_root),OMNIGIBSON_DISABLE_TORCH_COMPILE='1',JAX_PLATFORMS='cpu',
                   SERF_MAP_ROOT=str(assets/'maps'),TMPDIR='/tmp',XDG_CACHE_HOME='/tmp/serf-stage-cache')
        result.update(phase='import_checks',probes={});publish()
        for mode,environment in [('policy','b1k-pi05'),('sim','b1k-eval')]:
            process=subprocess.run([str(runtime/'envs'/environment/'bin/python'),str(root/'source/import_probe.py'),str(repo),mode],
                capture_output=True,text=True,env=env,timeout=300)
            text=process.stdout+'\n'+process.stderr
            (root/f'import_{mode}.log').write_text(text)
            lines=[line[len('SERF_PROBE_JSON='):] for line in process.stdout.splitlines() if line.startswith('SERF_PROBE_JSON=')]
            result['probes'][mode]=json.loads(lines[-1]) if lines else {'passed':False,'exit_code':process.returncode,'tail':text[-5000:]}
            publish()
        result.update(phase='complete',passed=all(x['passed'] for x in result['probes'].values()))
    except BaseException:
        result['error']=re.sub(r'https?://\S+','[URL omitted]',traceback.format_exc())
    publish()
    print(json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)
