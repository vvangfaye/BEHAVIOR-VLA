"""CPU-only development window for bounded import repair, without GPU waiting."""
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def run(config):
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=False)
    stage=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/prepare02')
    repo=stage/'source/repo'
    path=repo/'BEHAVIOR-1K/OmniGibson/omnigibson/utils/python_utils.py'
    text=path.read_text()
    if '\nimport os\n' not in text:text=text.replace('\nimport inspect\n','\nimport os\nimport inspect\n')
    path.write_text(text)
    path=repo/'external/openpi/src/openpi/models_pytorch/gemma_pytorch.py'
    path.write_text(path.read_text().replace('import pytest','from transformers.cache_utils import Cache').replace('pytest.Cache','Cache'))
    env=dict(os.environ)
    env.update(OMNIGIBSON_DATA_PATH=str(stage/'data'),OMNIGIBSON_DISABLE_TORCH_COMPILE='1',JAX_PLATFORMS='cpu',
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
               SERF_MAP_ROOT=str(stage/'assets/maps'),PYTHONNOUSERSITE='1',TMPDIR='/tmp')
    runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
    result={'root':str(root),'stage':str(stage),'probes':{}}
    for mode,environment in [('sim','b1k-eval'),('policy','b1k-pi05')]:
        p=subprocess.run([str(runtime/'envs'/environment/'bin/python'),str(stage/'source/import_probe.py'),str(repo),mode],
                         env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
        (root/f'{mode}.log').write_text(p.stdout)
        lines=[s.split('=',1)[1] for s in p.stdout.splitlines() if s.startswith('SERF_PROBE_JSON=')]
        result['probes'][mode]=json.loads(lines[-1]) if lines else {'passed':False,'returncode':p.returncode,'tail':p.stdout[-5000:]}
        payload=json.dumps(result,indent=2).encode();(root/'result.json').write_bytes(payload)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(config['put'],data=payload,method='PUT'),timeout=90):pass
    print(json.dumps(result),flush=True)
    # A finite CPU-only window permits follow-up import repairs on this exact
    # isolated source tree without re-submitting jobs for every missing module.
    until=time.monotonic()+900
    while time.monotonic()<until and not (root/'DONE').exists():time.sleep(5)
