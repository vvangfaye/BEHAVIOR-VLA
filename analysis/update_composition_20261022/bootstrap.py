"""Stage only the immutable code bundle; all model/data inputs already on AFS."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request


def run(config):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, scope='Full checkpoint meta-gradient plumbing only; no environment')
    def put(url, data):
        request = urllib.request.Request(url, data=json.dumps(data).encode(), method='PUT')
        with opener.open(request, timeout=60) as response:
            assert response.status in (200,201,204)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    code = 1
    gpu_description = None
    try:
        if config.get('record_gpu'):
            gpu_description=subprocess.check_output(
                ['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                text=True,timeout=20).strip()
        # Retry only this immutable GET. Never retry model execution or job creation.
        for attempt in range(4):
            try:
                with opener.open(config['bundle_url'], timeout=45) as response:
                    blob = response.read(1024*1024+1)
                break
            except urllib.error.URLError:
                if attempt==3:raise
                time.sleep(2**attempt)
        if len(blob)>1024*1024 or hashlib.sha256(blob).hexdigest()!=config['bundle_sha256']:
            raise ValueError('Invalid code bundle')
        archive = root/'code.tar.gz'
        archive.write_bytes(blob)
        with tarfile.open(archive) as tar:
            for member in tar:
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe bundle member')
                target=root/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source, target.open('xb') as out:
                    shutil.copyfileobj(source,out)
        plan=json.loads((root/'plan.json').read_text())
        for name,expected in plan['source_sha256'].items():
            if digest(root/name)!=expected:
                raise ValueError('Code integrity failure: '+name)
        for relative,expected in plan['upstream_sha256'].items():
            if digest(Path(config['upstream'])/relative)!=expected:
                raise ValueError('Upstream integrity failure: '+relative)
        env=dict(os.environ)
        env.pop('PYTHONHOME',None)
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                   RLINF_ROOT=config['upstream'], B1K_ADAPTER_ROOT=str(root/'adapter'),
                   PYTHONPATH=':'.join([str(root),config['site'],str(root/'adapter'),config['upstream']]))
        proc=subprocess.Popen([config['python'],'-u',str(root/'full_checkpoint.py'),
                '--plan',str(root/'plan.json'),'--output',str(root/'output')],
                cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
        timer=threading.Timer(3300,proc.terminate)
        timer.daemon=True;timer.start()
        try:
            with (root/'worker.log').open('x') as log:
                for line in proc.stdout:
                    log.write(line);log.flush()
                    try:
                        event=json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event,dict) and 'stage' in event:
                        print(json.dumps(event),flush=True)
                        try:put(config['progress_put'],event)
                        except Exception:pass
            code=proc.wait(timeout=60)
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill();proc.wait()
        if (root/'output/result.json').exists():
            result=json.loads((root/'output/result.json').read_text())
    except Exception as error:
        result['error_type']=type(error).__name__
        result['error']=str(error)[:500] if not isinstance(error,urllib.error.URLError) else 'network failure'
        if isinstance(error,urllib.error.URLError):
            result['network_reason_type']=type(error.reason).__name__
            result['network_reason']=str(error.reason)[:200].split('?')[0]
            result['http_status']=getattr(error,'code',None)
    result['worker_returncode']=code
    if gpu_description is not None:result['gpu_description']=gpu_description
    result['log_tail']=(root/'worker.log').read_text()[-16000:] if (root/'worker.log').exists() else ''
    put(config['result_put'],result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('log_tail','modules')}),flush=True)
    raise SystemExit(code if code else (0 if result.get('passed') else 1))
