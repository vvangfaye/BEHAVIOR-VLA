"""Run the tracking gate or the fixed twenty-instance evaluation in isolation."""
import hashlib
import concurrent.futures
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import tarfile
import tempfile
import traceback
import time
import urllib.request


def restore_completed_archives(archives, output, opener):
    import io
    completed={}
    for archive_item in archives:
        with opener.open(archive_item['url'],timeout=180) as response:
            data=response.read()
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            for member in archive:
                relative=Path(member.name)
                assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
                destination=output/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as stream:content=stream.read()
                if destination.exists():
                    assert destination.read_bytes()==content,'Resumed initial state differs'
                else:destination.write_bytes(content)
        for result_file in output.glob('*/result.json'):
            value=json.loads(result_file.read_text())
            if value.get('complete'):completed[result_file.parent.name]=value
    return completed


def materialize_data(base, local, bundle, expected_sha, opener):
    """Create a private file-symlink overlay; shared assets are never modified."""
    data_root=local/'data'
    data_root.mkdir()
    for directory,dirs,files in os.walk(base,followlinks=True):
        relative=Path(directory).relative_to(base)
        dest=data_root/relative;dest.mkdir(parents=True,exist_ok=True)
        for name in files:
            source=Path(directory)/name
            if source.is_file():(dest/name).symlink_to(source.resolve())
    archive=local/'extra_data.tar.gz'
    h=hashlib.sha256()
    with opener.open(bundle,timeout=180) as response,archive.open('wb') as output:
        for block in iter(lambda:response.read(8*1024**2),b''):output.write(block);h.update(block)
    assert h.hexdigest()==expected_sha
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            relative=Path(member.name)
            assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
            destination=data_root/relative
            with tf.extractfile(member) as stream:content=stream.read()
            if destination.exists():
                assert hashlib.sha256(destination.read_bytes()).digest()==hashlib.sha256(content).digest(),str(relative)
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(content)
    return data_root


def run(config):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    local=Path(tempfile.mkdtemp(prefix='serf-init-'))
    raw_opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    class StorageOpener:
        def open(self,request,timeout=None):
            for attempt in range(3):
                try:return raw_opener.open(request,timeout=timeout)
                except urllib.error.URLError:
                    if attempt==2:raise
                    time.sleep(2**attempt)
    opener=StorageOpener()
    report={'phase':'source','complete':False,'conditions':{},'source_sha256':config['source_sha256']}
    last_heartbeat=0.0

    def event(name,**fields):
        with (output/'events.jsonl').open('a') as f:
            f.write(json.dumps({'time':time.time(),'event':name,**fields})+'\n')
            f.flush()

    def terminate(signum,frame):
        report['termination_signal']=signum
        event('termination_signal',signal=signum)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)

    def publish():
        data=json.dumps(report,indent=2).encode()
        temp=output/'status.json.tmp';temp.write_bytes(data);temp.replace(output/'status.json')
        with opener.open(urllib.request.Request(config['result_put'],data=data,method='PUT'),timeout=90):pass

    def heartbeat(process=None,dest=None,force=False):
        nonlocal last_heartbeat
        if not force and time.monotonic()-last_heartbeat<30:return
        last_heartbeat=time.monotonic()
        state={'time':time.time()}
        for name in ['memory.current','memory.peak','memory.max','memory.events']:
            path=Path('/sys/fs/cgroup')/name
            if path.is_file():state[name]=path.read_text().strip()
        if process is not None:
            state.update(pid=process.pid,returncode=process.poll())
            path=Path(f'/proc/{process.pid}/status')
            if path.exists():
                state['process_memory']=[line for line in path.read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:','VmSize:'))]
        if dest is not None:
            for name in ['server.log','evaluation.log','progress.json']:
                path=dest/name
                if path.is_file():
                    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-5000));state[name]=f.read().decode(errors='replace')
        report['heartbeat']=state
        publish()

    try:
        stage=Path(config['stage'])
        report['phase']='prepare_private_data';publish()
        data_root=materialize_data(stage/'data',local,config['data_bundle_get'],config['data_bundle_sha256'],opener)
        preflight=json.loads(Path(config.get('checks_file',str(stage/'stage_result.json'))).read_text())
        assert preflight['probes']['sim']['passed'] is True
        for item in config.get('instance_states',[]):
            state=data_root/'2025-challenge-task-instances'/item['relative']
            assert state.is_file() and state.stat().st_size==item['size'], 'Missing instance state: '+str(item['instance'])
            assert hashlib.sha256(state.read_bytes()).hexdigest()==item['sha256'], 'Instance state hash differs: '+str(item['instance'])
        if config.get('instance_states'):
            report['instance_states_verified']=len(config['instance_states'])
            publish()
        for item in config.get('background_assets',[]):
            asset=stage/'data/omnigibson-robot-assets/models/background'/item['file']
            assert asset.stat().st_size==item['size']
            assert hashlib.sha256(asset.read_bytes()).hexdigest()==item['sha256']
        archive=local/'source.tar.gz'
        with opener.open(config['source_get'],timeout=90) as r,archive.open('wb') as f:shutil.copyfileobj(r,f)
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==config['source_sha256']
        with tarfile.open(archive) as tar:
            for member in tar:
                path=Path(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                dest=local/path;dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,dest.open('xb') as f:shutil.copyfileobj(src,f)
        repo=local/'repo'
        cotracker=local/'scaled_offline.pth'
        with opener.open(config['cotracker_get'],timeout=90) as r,cotracker.open('wb') as f:shutil.copyfileobj(r,f)
        assert hashlib.sha256(cotracker.read_bytes()).hexdigest()==config['cotracker_sha256']
        if config.get('maps'):
            maps_root=local/'maps';maps_dir=maps_root/f"exported_neural_points/task-{config['task_id']:04d}/eval"
            maps_dir.mkdir(parents=True)
            for item in config['map_model_assets']:
                target=maps_root/item['relative'];target.parent.mkdir(parents=True,exist_ok=True)
                with opener.open(item['url'],timeout=120) as response:content=response.read()
                assert len(content)==item['size'] and hashlib.sha256(content).hexdigest()==item['sha256']
                target.write_bytes(content)
            (maps_root/'robot').symlink_to(stage/'assets/maps/robot',target_is_directory=True)
            def fetch_map(item):
                dest=maps_dir/f"scene_{item['instance']}.hdf5"
                h=hashlib.sha256()
                with opener.open(item['url'],timeout=120) as r,dest.open('xb') as f:
                    for block in iter(lambda:r.read(8*1024**2),b''):f.write(block);h.update(block)
                assert dest.stat().st_size==item['size'] and h.hexdigest()==item['sha256']
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch_map,config['maps']))
            config['maps_root']=str(maps_root)
            report['maps_verified']=len(config['maps']);publish()
        runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
        (repo/'BEHAVIOR-1K/datasets').mkdir(exist_ok=True)
        (repo/'BEHAVIOR-1K/datasets/omnigibson-robot-assets').symlink_to(data_root/'omnigibson-robot-assets',target_is_directory=True)
        report['conditions'].update(restore_completed_archives(config.get('resume_archives',[]),output,opener))
        report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
        assert '5090' in report['gpu'] or 'N12LP' in report['gpu']
        driver=report['gpu'].split(',')[1].strip()
        libs=[p for d in ['/usr/local/nvidia/lib64','/usr/lib/x86_64-linux-gnu','/usr/lib64','/lib/x86_64-linux-gnu'] for p in Path(d).glob('libEGL_nvidia.so.'+driver)]
        assert libs
        vk=local/'nvidia_icd.json';egl=local/'egl.json'
        vk.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(libs[0]),'api_version':'1.3.242'}}))
        egl.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(libs[0])}}))
        repair=runtime/'library_snapshots/behavior-libxml-repair3-20261008-011710'
        ready=json.loads((repair/'_READY').read_text())
        assert hashlib.sha256((repair/'_READY').read_bytes()).hexdigest()=='66e5e790156270c67a266b49b115228f8159f032b2f336b61f4a32b511a0a8ec'
        env=dict(os.environ)
        env.update(SERF_EVAL_TASK_ID=str(config['task_id']),OMNIGIBSON_DATA_PATH=str(data_root),OMNIGIBSON_HEADLESS='1',OMNI_KIT_ACCEPT_EULA='YES',
            OMNIGIBSON_DISABLE_TORCH_COMPILE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',
            JAX_PLATFORMS='cpu',
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONHASHSEED='0',
            PYTHONPATH=':'.join(map(str,[repo/'src',repo/'BEHAVIOR-1K/OmniGibson',repo/'BEHAVIOR-1K/bddl',repo/'BEHAVIOR-1K/joylo'])),
            VK_ICD_FILENAMES=str(vk),__EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
            LD_LIBRARY_PATH=':'.join([ready['library_path'],str(runtime/'graphics-runtime/usr/lib/x86_64-linux-gnu'),str(runtime/'envs/b1k-eval/lib'),env.get('LD_LIBRARY_PATH','')]),
            SERF_COTRACKER_REPO=str(repo/'external/co-tracker'),SERF_COTRACKER_WEIGHTS=str(cotracker),
            TMPDIR=str(local),XDG_CACHE_HOME=str(local/'cache'))
        if config.get('site_path'):env['PYTHONPATH']=config['site_path']+':'+env['PYTHONPATH']
        if config.get('mode')=='gate':
            report['phase']='tracking_gate';publish()
            app=local/'app-gate';app.mkdir();env['OMNIGIBSON_APPDATA_PATH']=str(app)
            dest=output/'gate'
            with (output/'gate.log').open('w') as log:
                process=subprocess.Popen([str(runtime/'envs/b1k-eval/bin/python'),'-u',str(local/'run_case.py'),
                    '--repo',str(repo),'--condition','serf','--index','2','--output',str(dest),
                    '--shared-initial',config['gate_initial'],'--maps',config['maps_root'],
                    '--gate-replay',config['gate_actions'],'--gate-steps','4000'],
                    env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        heartbeat(process,dest)
                        time.sleep(2)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=20)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
            if (dest/'gate_result.json').exists():report['gate']=json.loads((dest/'gate_result.json').read_text())
            if process.returncode:raise RuntimeError('Tracking gate failed; see gate.log and gate_result.json')
            assert report['gate']['passed']
            report.update(phase='complete',complete=True)
            return
        if config.get('mode')=='full':
            assert config['asset_preflight_passed'] is True
            if shutil.disk_usage(local).free<20*1024**3:raise RuntimeError('Less than 20 GiB of node-local scratch space')
            probe_env=dict(env)
            if config.get('policy_dependency_path'):
                probe_env['PYTHONPATH']=config['policy_dependency_path']+':'+probe_env['PYTHONPATH']
            probe="import sys,types,importlib;from pathlib import Path;repo=Path("+repr(str(repo))+");sys.path[:0]=[str(repo/'src'),str(repo/'external/openpi/src'),str(repo/'external/openpi/packages/openpi-client/src')];og=types.ModuleType('omnigibson');og.__path__=[str(repo/'BEHAVIOR-1K/OmniGibson/omnigibson')];sys.modules['omnigibson']=og;"
            probe+="modules=['omnigibson.learning.utils.network_utils','b1k.policies.policy_config','serf_b1k.policies.policy_config','b1k.shared.eval_b1k_wrapper','serf_b1k.shared.eval_b1k_wrapper'];[(importlib.import_module(name),print('IMPORTED='+name,flush=True)) for name in modules]"
            if config.get('check_serf_construction'):
                probe+=";import jax;from flax import nnx;from serf_b1k.training.config import get_config;cfg=get_config("+repr(f"pi_serf_behavior_b1k_fast--4d_env_robot_feat_map--50t_lora--task-{config['task_id']:04d}")+");nnx.eval_shape(cfg.model.create,jax.random.key(0));print('SERF_MODEL_CONSTRUCTION_PASSED',flush=True)"
            report['phase']='verify_serving_imports';publish();event('verify_serving_imports')
            with (output/'serving_imports.log').open('w') as f:
                checked=subprocess.run([str(runtime/'envs/b1k-pi05/bin/python'),'-u','-c',probe],env=probe_env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=300)
            if checked.returncode:
                raise RuntimeError('Serving import check failed:\n'+(output/'serving_imports.log').read_text(errors='replace')[-12000:])
            # Conversion belongs to the simulator client.  Its dependencies are
            # installed in b1k-eval, not in the intentionally separate JAX server.
            client_probe="import sys,types;from pathlib import Path;repo=Path("+repr(str(repo))+");og=types.ModuleType('omnigibson');og.__path__=[str(repo/'BEHAVIOR-1K/OmniGibson/omnigibson')];sys.modules['omnigibson']=og;sys.path.insert(0,"+repr(str(local))+");from transport_preflight import validate;validate()"
            report['phase']='verify_client_transport';publish();event('verify_client_transport')
            with (output/'client_transport.log').open('w') as f:
                checked=subprocess.run([str(runtime/'envs/b1k-eval/bin/python'),'-u','-c',client_probe],env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=180)
            if checked.returncode:
                raise RuntimeError('Client transport check failed:\n'+(output/'client_transport.log').read_text(errors='replace')[-12000:])
            report['phase']='download_checkpoints';publish()
            checkpoint_root=local/'checkpoints'
            def fetch(item):
                target=checkpoint_root/item['relative']
                assert target.resolve().is_relative_to(checkpoint_root.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                h=hashlib.sha256();size=0
                with opener.open(item['url'],timeout=120) as r,target.open('xb') as f:
                    for block in iter(lambda:r.read(8*1024**2),b''):
                        f.write(block);h.update(block);size+=len(block)
                assert size==item['size'] and h.hexdigest()==item['sha256'],item['relative']
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(fetch,item) for item in config['checkpoints']]
                while True:
                    _,pending=concurrent.futures.wait(futures,timeout=5)
                    report['downloaded_files']=sum(f.done() and f.exception() is None for f in futures)
                    heartbeat()
                    if not pending:break
                for future in futures:future.result()
            event('checkpoints_verified',files=len(config['checkpoints']))
            env.update(OPENPI_DATA_HOME=str(runtime/'openpi-cache'),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                       XLA_PYTHON_CLIENT_MEM_FRACTION='0.5',XLA_PYTHON_CLIENT_ALLOCATOR='platform')
            for index in config['indices']:
                for condition in config.get('conditions_by_index',{}).get(str(index),['image','serf']):
                    assert condition in {'image','serf'}
                    key=f'{config["instance_ids"][index]}_{condition}'
                    report['phase']='evaluate_'+key;publish()
                    dest=output/key;dest.mkdir()
                    checkpoint=checkpoint_root/config['checkpoint_folders'][condition]
                    server_env=dict(env);server_env['JAX_PLATFORMS']='cuda'
                    if config.get('policy_dependency_path'):
                        dependency=Path(config['policy_dependency_path'])
                        assert (dependency/'addict/__init__.py').is_file()
                        server_env['PYTHONPATH']=str(dependency)+':'+server_env['PYTHONPATH']
                    app=local/('app-'+key);app.mkdir();env['OMNIGIBSON_APPDATA_PATH']=str(app)
                    server_log=(dest/'server.log').open('w')
                    bootstrap="import runpy,sys,time;print('POLICY_BOOTSTRAP_STARTED',time.time(),flush=True);script=sys.argv[1];sys.path.insert(0,str(__import__('pathlib').Path(script).parent));sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')"
                    if config.get('startup_diagnostics'):
                        bootstrap="import faulthandler;faulthandler.dump_traceback_later(300,repeat=True);"+bootstrap
                    server=subprocess.Popen([str(runtime/'envs/b1k-pi05/bin/python'),'-u','-X','faulthandler','-X','importtime','-c',bootstrap,str(local/'serve_policy.py'),
                        '--repo',str(repo),'--condition',condition,'--checkpoint',str(checkpoint),*(['--audit-output',str(dest/'inputs')] if config.get('audit_steps') else [])],
                        env=server_env,cwd=repo,stdout=server_log,stderr=subprocess.STDOUT)
                    event('policy_spawned',case=key,pid=server.pid)
                    try:
                        deadline=time.monotonic()+600
                        while True:
                            if server.poll() is not None:raise RuntimeError(f'Policy server exited before readiness: {key}; exit={server.returncode}')
                            heartbeat(server,dest)
                            try:
                                with opener.open('http://127.0.0.1:8000/healthz',timeout=2) as r:
                                    if r.status==200:break
                            except OSError:pass
                            if time.monotonic()>deadline:raise TimeoutError('Policy readiness timeout: '+key)
                            time.sleep(2)
                        report['policy_ready']={'case':key,'time':time.time(),'pid':server.pid}
                        event('policy_ready',case=key);heartbeat(server,dest,force=True)
                        with (dest/'evaluation.log').open('w') as log:
                            eval_bootstrap=[]
                            if config.get('startup_diagnostics'):
                                eval_bootstrap=['-c',"import faulthandler,runpy,sys;faulthandler.dump_traceback_later(300,repeat=True);script=sys.argv[1];sys.path.insert(0,str(__import__('pathlib').Path(script).parent));sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')"]
                            evaluation=subprocess.Popen([str(runtime/'envs/b1k-eval/bin/python'),'-u','-X','faulthandler',*eval_bootstrap,str(local/'run_case.py'),
                                '--repo',str(repo),'--condition',condition,'--index',str(index),'--output',str(dest),
                                '--shared-initial',str(config.get('shared_initial_dirs',{}).get(str(index),output/f'shared_initial_{index}')),'--maps',config['maps_root'],*(['--audit-steps',str(config['audit_steps'])] if config.get('audit_steps') else [])],
                                env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
                            try:
                                last_progress_time=time.monotonic()
                                last_progress_stamp=None
                                while evaluation.poll() is None:
                                    heartbeat(evaluation,dest)
                                    progress_file=dest/'progress.json'
                                    progress_stamp=progress_file.stat().st_mtime_ns if progress_file.exists() else None
                                    if progress_stamp != last_progress_stamp:
                                        last_progress_stamp=progress_stamp;last_progress_time=time.monotonic()
                                    # Scene load + first compilation can consume the startup
                                    # allowance. Once steps appear, retain the 20-minute guard.
                                    stall_limit=3600 if config.get('startup_diagnostics') and progress_stamp is None else 1200
                                    if time.monotonic()-last_progress_time > stall_limit:
                                        raise TimeoutError(f'No closed-loop progress for {stall_limit//60} minutes: {key}')
                                    if server.poll() is not None:raise RuntimeError(f'Policy server stopped during evaluation: {key}; exit={server.returncode}')
                                    with (dest/'server.log').open('rb') as f:
                                        f.seek(max(0,(dest/'server.log').stat().st_size-16000));tail=f.read()
                                    if b'Error in connection from ' in tail:
                                        raise RuntimeError(f'Policy inference failed: {key}; see server.log')
                                    time.sleep(2)
                            finally:
                                if evaluation.poll() is None:
                                    evaluation.terminate()
                                    try:evaluation.wait(timeout=15)
                                    except subprocess.TimeoutExpired:evaluation.kill();evaluation.wait()
                        event('evaluation_exited',case=key,returncode=evaluation.returncode)
                        if evaluation.returncode:raise RuntimeError(f'Evaluation process failed: {key}; exit={evaluation.returncode}')
                        if not (dest/'result.json').exists():
                            detail=(dest/'evaluation.log').read_text(errors='replace')[-14000:]
                            raise RuntimeError('Evaluator exited without a result file: '+key+'\n'+detail)
                        result=json.loads((dest/'result.json').read_text());assert result['complete'] is True
                        if config.get('episode_puts'):
                            case_archive=local/f'case_{key}.tar.gz'
                            with tarfile.open(case_archive,'w:gz') as tar:
                                for file in dest.rglob('*'):
                                    if file.is_file():tar.add(file,arcname=str(file.relative_to(output)),recursive=False)
                                shared=output/f'shared_initial_{index}'
                                for file in shared.rglob('*'):
                                    if file.is_file():tar.add(file,arcname=str(file.relative_to(output)),recursive=False)
                            with opener.open(urllib.request.Request(config['episode_puts'][key],data=case_archive.read_bytes(),method='PUT'),timeout=180):pass
                            result['episode_archive_uploaded']=True
                            case_archive.unlink()
                        report['conditions'][key]=result;publish()
                    finally:
                        server.terminate()
                        try:server.wait(timeout=15)
                        except subprocess.TimeoutExpired:server.kill();server.wait()
                        server_log.close()
            report.update(phase='complete',complete=True)
            return
        for condition in ['image','serf']:
            report['phase']='initialize_'+condition;publish()
            app=local/('app-'+condition);app.mkdir()
            env['OMNIGIBSON_APPDATA_PATH']=str(app)
            dest=output/condition
            with (output/(condition+'.log')).open('w') as log:
                result=subprocess.run([str(runtime/'envs/b1k-eval/bin/python'),str(local/'run_case.py'),
                    '--repo',str(repo),'--condition',condition,'--index','0','--output',str(dest),
                    '--shared-initial',str(output/'shared_initial'),'--maps',config['maps_root'],'--initialize-only'],
                    env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
            report['conditions'][condition]={'returncode':result.returncode}
            if (dest/'initialize_result.json').exists():
                report['conditions'][condition]['result']=json.loads((dest/'initialize_result.json').read_text())
            if result.returncode:
                report['conditions'][condition]['tail']=(output/(condition+'.log')).read_text()[-10000:]
                raise RuntimeError(condition+' initialization failed')
        report.update(phase='complete',complete=True)
    except BaseException:
        report['error']=traceback.format_exc()
    finally:
        publish()
        archive=local/'initialization_artifacts.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in output.rglob('*'):
                if path.is_file():tar.add(path,arcname=str(path.relative_to(output)),recursive=False)
        with archive.open('rb') as f:
            with opener.open(urllib.request.Request(config['artifacts_put'],data=f.read(),method='PUT'),timeout=90):pass
        print(json.dumps(report),flush=True)
    if not report['complete']:raise SystemExit(1)
