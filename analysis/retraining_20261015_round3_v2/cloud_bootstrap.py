"""CPU-only data export, reusing the previously verified AFS reader runtime."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
import traceback
import urllib.request


def run(payload):
    root=Path(payload['root'])
    assert str(root).startswith('/mnt/afs/260010165/behavior_1k/retraining_20261015_round3_v2/')
    root.mkdir(parents=True,exist_ok=False)
    cache=Path('/var/tmp/behavior-round3-preparation')/payload['name']
    cache.mkdir(parents=True,exist_ok=False)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def publish(phase,**extra):
        value=dict(phase=phase,observed_at=time.time(),training_executed=False,gpu_computation_enabled=False,**extra)
        (root/'bootstrap_status.json').write_text(json.dumps(value,indent=2)+'\n')
        raw=json.dumps(value).encode()
        for attempt in range(4):
            try:
                with opener.open(urllib.request.Request(payload['status_put'],data=raw,method='PUT'),timeout=30):pass
                return
            except OSError:
                if attempt==3:return
                time.sleep(2**attempt)
    try:
        publish('fetching_small_frozen_manifests')
        for asset in payload['assets']:
            with opener.open(asset['get'],timeout=60) as response:raw=response.read(asset['bytes']+1)
            assert len(raw)==asset['bytes'] and hashlib.sha256(raw).hexdigest()==asset['sha256']
            path=cache/asset['name'];path.write_bytes(raw);path.chmod(0o600)
            if asset['name']=='code.tar.gz':
                with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                    for member in tar.getmembers():
                        rel=Path(member.name)
                        assert not rel.is_absolute() and '..' not in rel.parts and member.isfile()
                    tar.extractall(cache/'code',filter='data')
        base=Path('/mnt/afs/260010165/behavior_1k/retraining_20261013/parallel-1013-124351/shard-0')
        python='/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python'
        assert Path(python).is_file() and (base/'deps').is_dir() and (base/'lerobot/src').is_dir()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONNOUSERSITE='1',
                 PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(Path(python).parent.parent/'lib'),
                 PYTHONPATH=':'.join([str(base/'deps'),str(cache/'code/sco'),str(cache/'code/scripts'),
                                    str(cache/'code/openpi/src'),str(base/'lerobot/src')]))
        result=subprocess.run([python,'-c','import torch,h5py,av,pyarrow;from lerobot.datasets import LeRobotDataset;assert not torch.cuda.is_available();print("reader_runtime_ok")'],
                              env=env,capture_output=True,text=True,timeout=120)
        assert result.returncode==0,'Existing CPU data-reader runtime cannot import'
        publish('reader_runtime_verified',assigned_sources=payload['source_count'])
        with (root/'worker.log').open('x') as log:
            result=subprocess.run([python,str(cache/'code/sco/build_retraining_parallel.py'),str(cache),str(root/'output')],
                                  env=env,stdout=log,stderr=subprocess.STDOUT,timeout=23*3600)
        if result.returncode:
            raise RuntimeError('CPU export failed; preserve detailed AFS failure record')
    except BaseException as error:
        message=re.sub(r'https?://[^\s"\'<>]+','<URL>',''.join(traceback.format_exception(type(error),error,error.__traceback__)))
        (root/'bootstrap_failure.json').write_text(message)
        publish('bootstrap_needs_diagnostic',error_type=type(error).__name__,error_message=message[-4000:])
        raise SystemExit(1)
