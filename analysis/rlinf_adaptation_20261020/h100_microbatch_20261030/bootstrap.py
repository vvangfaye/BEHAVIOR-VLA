"""Restore selected immutable inputs to ephemeral disk, then run one GPU job."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import threading
import traceback
import urllib.request


def run(config):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for key in list(os.environ):
        if key.lower() in ('http_proxy','https_proxy','all_proxy'):
            os.environ.pop(key)
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=False)
    log_fd = os.open(root/'bootstrap.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    scratch = Path(tempfile.mkdtemp(prefix='b1k-h100-microbatch-'))
    stage = 'bootstrap'
    active = {}
    active_lock = threading.Lock()
    stopped = threading.Event()

    def diagnostic():
        info = {'updated_at':time.time(), 'stage':stage, 'scratch':str(scratch),
                'scratch_free':shutil.disk_usage(scratch).free,
                'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        for name in ('memory.current','memory.max','memory.events'):
            path = Path('/sys/fs/cgroup')/name
            if path.exists():
                info[name] = path.read_text()
        with active_lock:
            info['active_transfers'] = dict(active)
        return info

    def heartbeat():
        while not stopped.wait(10):
            (root/'heartbeat.json').write_text(json.dumps(diagnostic(),indent=2)+'\n')

    def interrupted(number, frame):
        info = diagnostic()
        info.update(state='interrupted', signal=number)
        (root/'interrupted.json').write_text(json.dumps(info,indent=2)+'\n')
        os._exit(128+number)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    threading.Thread(target=heartbeat, daemon=True).start()

    def upload(name, body):
        with urllib.request.urlopen(urllib.request.Request(config[name+'_put'], data=body, method='PUT'), timeout=30):
            pass

    def status(**fields):
        body = json.dumps({'state':'running','stage':stage,'updated_at':time.time(),**fields}).encode()
        (root/'bootstrap_status.json').write_bytes(body)
        print(body.decode(), flush=True)
        try:
            upload('status', body)
        except Exception:
            pass

    try:
        if 'bundle_inline' in config:
            body = base64.b64decode(config['bundle_inline'])
        else:
            with urllib.request.urlopen(config['bundle_url'], timeout=120) as source:
                body = source.read(2*1024**2+1)
            assert len(body)<=2*1024**2, 'Bounded code bundle exceeded size limit'
        assert hashlib.sha256(body).hexdigest() == config['bundle_sha256']
        with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as archive:
            for item in archive:
                path = Path(item.name)
                assert not path.is_absolute() and '..' not in path.parts and item.isfile()
                target = root/path
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as source, target.open('xb') as dest:
                    shutil.copyfileobj(source, dest)
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'], text=True).strip()
        assert len(gpu.splitlines()) == 1 and ('H100' in gpu or 'N6LS' in gpu.upper()), gpu
        status(gpu=gpu)
        if shutil.disk_usage(scratch).free < config['input_bytes'] + 12*1024**3:
            raise RuntimeError('Insufficient ephemeral storage for bounded input restoration')
        stage = 'restore_existing_inputs'
        status(scratch_free=shutil.disk_usage(scratch).free)

        def download(item):
            dest = scratch/item['relative']
            dest.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(3):
                digest = hashlib.sha256()
                count = 0
                try:
                    with active_lock:
                        active[item['relative']] = {'bytes':0,'attempt':attempt+1,'started_at':time.time()}
                    with urllib.request.urlopen(item['url'], timeout=120) as source, dest.open('wb') as target:
                        while True:
                            block = source.read(8*1024**2)
                            if not block:
                                break
                            digest.update(block)
                            target.write(block)
                            count += len(block)
                            with active_lock:
                                active[item['relative']]['bytes'] = count
                    assert count == item['size'] and digest.hexdigest() == item['sha256'], item['relative']
                    print(json.dumps({'restored':item['relative'],'bytes':count}), flush=True)
                    with active_lock:
                        active.pop(item['relative'], None)
                    return {'relative':item['relative'],'size':count,'sha256':digest.hexdigest()}
                except Exception:
                    if attempt==2:
                        raise RuntimeError('Input restoration failed: '+item['relative']) from None
                    # A newly started pod can briefly precede working DNS.
                    # Bounded retry; no model or rollout computation has begun.
                    stopped.wait(2**(attempt+1))
        with ThreadPoolExecutor(max_workers=4) as pool:
            restored = list(pool.map(download,config['inputs']))
        (root/'inputs_verified.json').write_text(json.dumps(restored,indent=2)+'\n')
        stage = 'benchmark'
        base = Path('/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01')
        runtime = Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
        code = '/mnt/afs/260010165/behavior_1k/code_snapshots/b1k-video1688977-20261013'
        env = dict(os.environ)
        env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONHASHSEED='0',
            PYTHONPATH=':'.join([str(base/'site'),str(root/'run'),str(base/'adapter'),str(base/'upstream'),
                                 code+'/openpi/src',code+'/openpi/packages/openpi-client/src']),
            OPENPI_DATA_HOME=str(runtime/'openpi-cache'),RLINF_ROOT=str(base/'upstream'),
            JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
            B1K_BENCH_ROOT=str(root),B1K_BENCH_INPUTS=str(scratch),B1K_STATUS_PUT=config['status_put'])
        status(gpu=gpu,restored_bytes=sum(x['size'] for x in restored),input_sha256_verified=True)
        with (root/'driver.log').open('x') as log:
            command=[str(runtime/'envs/b1k-pi05/bin/python'),'-u',str(root/'run/benchmark.py')]
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=config['max_seconds']-1800)
        stage = 'complete' if result.returncode == 0 else 'failed'
        if result.returncode:
            lines=(root/'driver.log').read_text(errors='replace').splitlines()[-35:]
            print('\n'.join(re.sub(r'https?://[^\s]+','[URL omitted]',x) for x in lines),flush=True)
        status(returncode=result.returncode,gpu=gpu)
        return result.returncode
    except BaseException as error:
        stage = 'failed'
        report={'state':'failed','error_type':type(error).__name__,
                'traceback':re.sub(r'https?://[^\s]+','[URL omitted]',traceback.format_exc())}
        (root/'bootstrap_error.json').write_text(json.dumps(report,indent=2)+'\n')
        status(**report)
        return 1
    finally:
        stopped.set()
        # Only bounded metadata/logs are persisted; restored tensors stay ephemeral.
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for path in root.rglob('*'):
                if path.is_file() and path.suffix in ('.json','.log') and path.stat().st_size<32*1024**2:
                    archive.add(path,arcname=str(path.relative_to(root)))
        try:
            upload('result',buffer.getvalue())
        except Exception:
            pass
        shutil.rmtree(scratch,ignore_errors=True)
