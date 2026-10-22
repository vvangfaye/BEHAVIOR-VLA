"""Bounded worker for one fixed training arm, using existing simulation assets."""
import hashlib
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request


def run(config):
    def interrupted(*_):raise InterruptedError('Worker time limit or termination requested')
    signal.signal(signal.SIGTERM,interrupted)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=False)
    def put(url,value):
        data=json.dumps(value).encode()
        with opener.open(urllib.request.Request(url,data=data,method='PUT'),timeout=45) as response:
            assert response.status in (200,201,204)
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
        return h.hexdigest()
    result={'complete':False,'method':config['method'],'training_episodes':0};returncode=1
    try:
        if shutil.disk_usage(root).free<40*1024**3:raise RuntimeError('Less than 40 GiB free')
        archive=root/'code_data.tar.gz'
        for attempt in range(4):
            try:
                count=0;h=hashlib.sha256()
                with opener.open(config['bundle_url'],timeout=60) as response,archive.open('wb') as output:
                    for chunk in iter(lambda:response.read(1024**2),b''):
                        count+=len(chunk)
                        if count>config['bundle_bytes']:raise ValueError('Oversized bundle')
                        h.update(chunk);output.write(chunk)
                if count!=config['bundle_bytes'] or h.hexdigest()!=config['bundle_sha256']:
                    raise ValueError('Bundle checksum mismatch')
                break
            except urllib.error.URLError:
                if attempt==3:raise
                time.sleep(2**attempt)
        with tarfile.open(archive) as tar:
            for member in tar:
                name=Path(member.name)
                if not member.isfile() or name.is_absolute() or '..' in name.parts:
                    raise ValueError('Unsafe archive member')
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,target.open('xb') as output:shutil.copyfileobj(source,output)
        plan=json.loads((root/'plan.json').read_text())
        for name,expected in plan['source_sha256'].items():
            if digest(root/name)!=expected:raise ValueError('Source mismatch: '+name)
        for name,expected in plan['upstream_sha256'].items():
            if digest(Path(config['upstream'])/name)!=expected:raise ValueError('Upstream mismatch: '+name)
        admission=json.loads((root/'admission.json').read_text())
        if not admission.get('passed') or not admission.get('adaptation_effective'):
            raise ValueError('Effective adaptation prerequisite failed')
        for relative,expected in plan['original_source_sha256'].items():
            if digest(Path(plan['environment']['code'])/relative)!=expected:
                raise ValueError('Original evaluator source changed')
        env=dict(os.environ);env.pop('PYTHONHOME',None)
        runtime=Path(plan['environment']['runtime'])
        repair=Path(plan['environment']['supplemental_library'])
        if digest(repair/'_READY')!=plan['environment']['supplemental_library_sha256']:
            raise ValueError('Graphics library receipt changed')
        ready=json.loads((repair/'_READY').read_text())
        for name,expected in ready['libraries_sha256'].items():
            if digest(Path(ready['library_path'])/name)!=expected:raise ValueError('Graphics library changed')
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip().splitlines()
        if len(gpu)!=1:raise ValueError('Expected the declared single GPU')
        graphics=config.get('role')=='renderer' or not config.get('remote_renderer',False)
        if graphics:
            driver=gpu[0].split(',')[1].strip()
            libraries=[p for directory in ('/usr/local/nvidia/lib64','/usr/lib/x86_64-linux-gnu','/usr/lib64','/lib/x86_64-linux-gnu')
                       for p in Path(directory).glob('libEGL_nvidia.so.'+driver)]
            if not libraries:raise ValueError('NVIDIA graphics library unavailable')
            vk=root/'vk.json';egl=root/'egl.json'
            vk.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(libraries[0]),'api_version':'1.3.242'}}))
            egl.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(libraries[0])}}))
        env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
            PYTHONUNBUFFERED='1',PYTHONHASHSEED=str(plan['seed']),
            RLINF_ROOT=config['upstream'],B1K_ADAPTER_ROOT=str(root/'adapter'),
            PYTHONPATH=':'.join([str(root),config['site'],str(root/'adapter'),config['upstream']]),
            OPENPI_DATA_HOME=str(runtime/'openpi-cache'),OMNIGIBSON_DATA_PATH=str(runtime/'omnigibson-data'),
            OMNIGIBSON_HEADLESS='1',OMNI_KIT_ACCEPT_EULA='YES',OMNIGIBSON_DISABLE_TORCH_COMPILE='1',
            LD_LIBRARY_PATH=':'.join([ready['library_path'],str(runtime/'graphics-runtime/usr/lib/x86_64-linux-gnu'),
                                      str(runtime/'envs/b1k-eval/lib'),env.get('LD_LIBRARY_PATH','')]))
        if graphics:env.update(VK_ICD_FILENAMES=str(vk),__EGL_VENDOR_LIBRARY_FILENAMES=str(egl))
        entry='renderer_service.py' if config.get('role')=='renderer' else 'train.py'
        with (root/'worker.log').open('x') as log:
            worker=subprocess.Popen([config['python'],'-u',str(root/entry),'--plan',str(root/'plan.json'),
                '--output',str(root/'output')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
            deadline=time.monotonic()+84000
            try:
                while worker.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Training process exceeded its wall-time bound')
                    try:
                        if (root/'output/progress.json').exists():
                            progress=json.loads((root/'output/progress.json').read_text())
                            if config.get('role')=='renderer':
                                episodes=sorted(Path(plan['environment']['renderer_mailbox']['episode_output_root']).glob('iteration_*_episode_*'))
                                if episodes and (episodes[-1]/'simulator.log').exists():
                                    path=episodes[-1]/'simulator.log'
                                    with path.open('rb') as stream:
                                        stream.seek(max(0,path.stat().st_size-6000))
                                        progress['simulator_log_tail']=stream.read(6000).decode(errors='replace')
                            put(config['progress_put'],progress)
                            print(json.dumps(dict(role=config.get('role','trainer'),
                                stage=progress.get('stage'),elapsed_seconds=time.monotonic()-(deadline-84000))),flush=True)
                    except Exception:pass
                    try:worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                returncode=worker.returncode
            finally:
                try:os.killpg(worker.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=10)
                try:os.killpg(worker.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        if (root/'output/result.json').exists():result=json.loads((root/'output/result.json').read_text())
        if result.get('complete') and config.get('role')!='renderer':
            checkpoints=sorted((root/'output').glob('adapter_*.pt'))
            if not checkpoints:raise ValueError('Completed run has no adapter checkpoint')
            checkpoint=checkpoints[-1]
            with opener.open(urllib.request.Request(config['checkpoint_put'],data=checkpoint.read_bytes(),method='PUT'),timeout=60) as response:
                assert response.status in (200,201,204)
            result['checkpoint_sha256']=digest(checkpoint);result['checkpoint_bytes']=checkpoint.stat().st_size
    except Exception as error:
        result.update(error_type=type(error).__name__)
        if isinstance(error,urllib.error.URLError):result['network_reason_type']=type(error.reason).__name__
        else:result['error']=str(error)[:600]
    result.update(worker_returncode=returncode,method=config['method'],role=config.get('role','trainer'))
    if 'gpu' in locals():result['gpu_description']=gpu
    result['log_tail']=(root/'worker.log').read_text()[-16000:] if (root/'worker.log').exists() else ''
    put(config['result_put'],result)
    print(json.dumps({k:v for k,v in result.items() if k!='log_tail'}),flush=True)
    raise SystemExit(0 if result.get('complete') and returncode==0 else 1)
