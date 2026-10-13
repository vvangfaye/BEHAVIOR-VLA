"""Bounded download bootstrap with persistent diagnostics before importing data libraries."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import tarfile
import time
import traceback
import urllib.request
import urllib.error


def run(payload):
    root=Path(payload['root']);root.mkdir(parents=True,exist_ok=False)
    o=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    def status(phase,**fields):
        data={'phase':phase,'time':time.time(),'training_executed':False,**fields}
        (root/'bootstrap_status.json').write_text(json.dumps(data,indent=2))
        request=urllib.request.Request(payload['control']['status_put'],data=json.dumps(data).encode(),method='PUT')
        for attempt in range(3):
            try:
                with o.open(request,timeout=30):pass
                return
            except (urllib.error.URLError,OSError):
                if attempt==2:return
                time.sleep(1+attempt)
    try:
        status('fetching_direct_transfer_code')
        for attempt in range(3):
            try:
                with o.open(payload['code_get'],timeout=60) as r:data=r.read()
                break
            except urllib.error.HTTPError:raise
            except (urllib.error.URLError,OSError):
                if attempt==2:raise
                time.sleep(1+attempt)
        (root/'code.tar.gz').write_bytes(data)
        assert hashlib.sha256(data).hexdigest()==payload['code_sha256'],'Transfer code hash differs'
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for m in t.getmembers():assert '/' not in m.name and m.isfile()
            t.extractall(root/'code')
        (root/'private_control.json').write_text(json.dumps(payload['control']));(root/'private_control.json').chmod(0o600)
        base='/mnt/afs/260010165/behavior_1k/retraining_20261013/parallel-1013-124351/shard-0'
        python='/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',LD_LIBRARY_PATH=str(Path(python).parent.parent/'lib'),PYTHONPATH=':'.join([str(root/'code')]+[base+'/'+s for s in ['deps','code/scripts','code/openpi/src','lerobot/src']]))
        status('direct_transfer_runtime_starting')
        with (root/'worker.log').open('w') as log:
            result=subprocess.run([python,str(root/'code/direct_oss_worker.py'),str(root)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            text=re.sub(r'https?://[^\s]+','<URL>',(root/'worker.log').read_text(errors='replace')[-6000:])
            failure=root/'failure.json'
            if failure.exists():text+='\n'+failure.read_text()[-8000:]
            raise RuntimeError('Direct worker exit '+str(result.returncode)+': '+text)
    except BaseException as e:
        text=re.sub(r'https?://[^\s]+','<URL>',''.join(traceback.format_exception(type(e),e,e.__traceback__)))
        (root/'bootstrap_failure.json').write_text(text)
        status('bootstrap_needs_diagnostic',error_type=type(e).__name__,error_message=text[-8000:])
        raise SystemExit(1)
