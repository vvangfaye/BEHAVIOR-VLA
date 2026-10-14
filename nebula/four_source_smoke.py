#!/usr/bin/env python3
"""Isolated Nebula acceptance run: real four-source batches, exactly two updates."""
from __future__ import annotations
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
_DIRECT_TOOL_READY = False

def read(path): return json.loads(Path(path).read_text())
def configuration():
    path=REPO/'nebula/worker_job.json'
    return read(path if path.exists() else REPO/'nebula/smoke_job.json')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def log(message): print(time.strftime('%Y-%m-%d %H:%M:%S'),message,flush=True)
def run(command,**kwargs):
    return subprocess.run([str(x) for x in command],check=True,**kwargs)
def clean_python_environment():
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD'):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    return env
def copy_verified(source,target,size=None,digest=None):
    global _DIRECT_TOOL_READY
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise ValueError('Fresh run target already exists: '+str(target))
    tool=REPO/'nebula/ossutil'
    direct=tool.is_file() and str(source).startswith('/data/oss_bucket_0/')
    if not direct and size is not None and source.stat().st_size!=size:raise ValueError('Source size changed: '+str(source))
    partial=target.with_suffix(target.suffix+'.partial')
    started=time.monotonic()
    if direct:
        cfg=configuration()
        if not _DIRECT_TOOL_READY:
            if sha(tool)!=cfg['ossutil_sha256']:raise ValueError('Frozen OSS transfer tool changed')
            tool.chmod(0o755);_DIRECT_TOOL_READY=True
        uri='oss://traffic-model/'+str(source)[len('/data/oss_bucket_0/'):]
        transfer=Path('/var/tmp/wangfaye/four-source-smoke')/cfg['run_id']/('.transfer-'+str(os.getpid()))
        transfer.mkdir(parents=True,exist_ok=True)
        command=[str(tool),'cp',uri,str(partial),'--endpoint',cfg['download_endpoint'],
            '--sign-version','v1','--parallel','8','--checkpoint-dir',str(transfer/'checkpoints'),
            '--output-dir',str(transfer/'output'),'--retry-times','10','--force','--no-progress']
        with tempfile.TemporaryFile(mode='w+t') as output:
            cp=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,text=True)
            try:
                while True:
                    try:cp.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        elapsed=time.monotonic()-started
                        if elapsed>1800:raise TimeoutError('Direct OSS download exceeded 30 minutes: '+target.name)
                        if size and size>=1024**3:
                            current=partial.stat().st_size if partial.exists() else 0
                            log('Downloading '+target.name+': '+str(current)+'/'+str(size)+' bytes, '+str(round(elapsed))+' seconds')
            finally:
                if cp.poll() is None:cp.kill();cp.wait()
            if cp.returncode:
                output.seek(0);message=output.read()
                for key in ('OSS_ACCESS_KEY_ID','OSS_ACCESS_KEY_SECRET'):
                    if os.environ.get(key):message=message.replace(os.environ[key],'[REDACTED]')
                raise RuntimeError('Direct OSS download failed: '+message[-1800:])
    else:
        shutil.copyfile(source,partial)
    if size is not None and partial.stat().st_size!=size:raise ValueError('Incomplete copy')
    if digest is not None and sha(partial)!=digest:raise ValueError('Source SHA256 changed: '+str(source))
    partial.replace(target)
    if target.stat().st_size>=1024**3:
        elapsed=time.monotonic()-started
        log('Downloaded '+target.name+': '+str(target.stat().st_size)+' bytes in '+str(round(elapsed,1))+' seconds')

def register_rank(base, count):
    if count == 1:return 0,None
    rendezvous=base/'rendezvous';rendezvous.mkdir(parents=True,exist_ok=True)
    ip=run(['hostname','-i'],capture_output=True,text=True).stdout.split()[0]
    if ip.startswith('127.') or ':' in ip:raise ValueError('Need a routable IPv4 worker address')
    path=rendezvous/(ip+'.json')
    with path.open('x') as f:json.dump(dict(ip=ip,host=socket.gethostname(),pid=os.getpid()),f)
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        paths=list(rendezvous.glob('*.json'))
        if len(paths)>count:raise ValueError('Too many workers in isolated smoke run')
        if len(paths)==count:
            try:ips=sorted(read(p)['ip'] for p in paths)
            except (json.JSONDecodeError,OSError):time.sleep(1);continue
            return ips.index(ip),ips[0]+':53791'
        time.sleep(2)
    raise TimeoutError('Smoke rank rendezvous timed out')

def bootstrap_worker(cfg,rank,count,base_output,scratch,output):
    hardware=run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
    gpus=[x.rsplit(',',1) for x in hardware.strip().splitlines()]
    write(output/'hardware.json',dict(gpus=gpus,free_disk_bytes=shutil.disk_usage(scratch).free))
    log('Allocated GPUs: '+json.dumps(gpus))
    if len(gpus)!=8 or any(int(x[1])<=0 for x in gpus):
        raise ValueError('Smoke requires eight allocated CUDA GPUs with reported memory')
    if shutil.disk_usage(scratch).free<150*1024**3:raise ValueError('Need 150 GiB scratch for staging and smoke checkpoint')
    mount=Path('/data/oss_bucket_0')
    archive=scratch/'environment.tar.gz';spec=cfg['environment']
    log('Copying frozen policy environment')
    copy_verified(mount/spec['key'],archive,spec['bytes'])
    log('Extracting frozen policy environment')
    envroot=scratch/'environment';envroot.mkdir()
    run(['tar','-xzf',archive,'-C',envroot]);archive.unlink()
    python=envroot/'bin/python'
    if (envroot/'bin/conda-unpack').exists():
        run([python,'-E','-s',envroot/'bin/conda-unpack'],env=clean_python_environment())
    cache=scratch/'openpi-cache';token=cache/'big_vision/paligemma_tokenizer.model';token.parent.mkdir(parents=True)
    if (REPO/'inputs/paligemma_tokenizer.model').exists():
        copy_verified(REPO/'inputs/paligemma_tokenizer.model',token,digest=cfg['tokenizer_sha256'])
    else:
        log('Extracting only tokenizer from existing OpenPI cache')
        with tarfile.open(mount/cfg['cache']['key'],'r:') as source:
            members=[m for m in source if m.isfile() and m.name.endswith('big_vision/paligemma_tokenizer.model')]
            if len(members)!=1:raise ValueError('Tokenizer missing/ambiguous in frozen cache')
            with source.extractfile(members[0]) as r,token.open('wb') as w:shutil.copyfileobj(r,w)
    checkpoint=scratch/'checkpoint1688977';base=mount/cfg['checkpoint_key']
    entries=[]
    for line in (REPO/'inputs/checkpoint.SHA256SUMS').read_text().splitlines():
        digest,name=line.split(None,1);name=name.lstrip('*');entries.append((digest,name))
    def cp_ckpt(item):
        digest,name=item
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe checkpoint manifest')
        copy_verified(base/name,checkpoint/name,digest=digest)
    log('Copying and hashing checkpoint 1688977 inference weights only')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(cp_ckpt,entries))
    write(output/'checkpoint_source_verified.json',dict(step=1688977,files=len(entries),manifest_sha256=sha(REPO/'inputs/checkpoint.SHA256SUMS'),old_optimizer_loaded=False))
    runtime_env=clean_python_environment()
    for key in list(runtime_env):
        if key.startswith(('OPENPI_','JAX_','TRAIN_','LR_')) or key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD'):
            runtime_env.pop(key,None)
    runtime_env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join(str(REPO/x) for x in ('openpi/src','openpi/packages/openpi-client/src','vendor/lerobot/src')),
        OPENPI_DATA_HOME=str(cache),WANDB_MODE='disabled',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',
        OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.90')
    # The accepted LeRobot source keeps timestamp comparisons at float64;
    # the data and time tolerance are unchanged.
    run([python,REPO/'sco/install_reader_timestamp_fix.py',REPO/'vendor/lerobot'],env=runtime_env)
    run([python,REPO/'sco/install_reader_episode_file_fix.py',REPO/'vendor/lerobot'],env=runtime_env)
    runtime=run([python,'-c','import jax,flax,optax,torch,pyarrow,lerobot; print(jax.__version__,flax.__version__,optax.__version__,torch.__version__,pyarrow.__version__); print(jax.devices())'],env=runtime_env,capture_output=True,text=True)
    log('Runtime: '+runtime.stdout.strip())
    write(output/'runtime.json',dict(stdout=runtime.stdout,tokenizer_sha256=sha(token)))
    log('Staging four-source data')
    run([python,REPO/'nebula/four_source_smoke.py','stage',scratch,output,str(rank),str(count)],env=runtime_env)
    return python,runtime_env,checkpoint

def execute():
    for name,digest in read(REPO/'bundle_manifest.json').items():
        if sha(REPO/name)!=digest:raise ValueError('Frozen smoke bundle changed: '+name)
    cfg=read(REPO/'nebula/smoke_job.json')
    if cfg['updates']!=2 or cfg['formal_training_authorized'] is not False:
        raise ValueError('This worker only accepts the two-update smoke contract')
    count=cfg.get('process_count',1)
    if count not in (1,2):raise ValueError('Only one or two eight-GPU workers are supported')
    base_output=Path(cfg['output']);base_output.mkdir(parents=True,exist_ok=True)
    rank,coordinator=register_rank(base_output,count)
    output=base_output/('rank-'+str(rank));output.mkdir(exist_ok=False)
    scratch=Path('/var/tmp/wangfaye/four-source-smoke')/cfg['run_id']/('rank-'+str(rank));scratch.mkdir(parents=True,exist_ok=False)
    write(output/'started.json',dict(run_id=cfg['run_id'],hostname=os.uname().nodename,pid=os.getpid(),started=time.time(),formal_training=False))
    try:
        python,runtime_env,checkpoint=bootstrap_worker(cfg,rank,count,base_output,scratch,output)
        inputs=REPO/'inputs/compiled';argv=read(inputs/'training_argv.json')['argv']
        replacements={'--exp-name':cfg['run_id'],'--num-train-steps':'2','--save-interval':'1','--keep-period':'1','--log-interval':'1',
            '--data.subtask-mixture-path':str(inputs/'mixture.json'),
            '--data.subtask-mixture-roots-path':str(scratch/'datasets/roots.json'),
            '--learning-rate-contract-path':str(inputs/'learning_rate_contract.json'),
            '--data.assets.assets-dir':str(checkpoint/'assets'),
            '--weight-loader.params-path':str(checkpoint/'params')}
        for key,value in replacements.items():argv[argv.index(key)+1]=value
        checkpoint_base=base_output/'checkpoints' if count>1 else scratch/'checkpoints'
        argv+=['--checkpoint-base-dir',str(checkpoint_base),'--assets-base-dir',str(scratch/'assets'),
            '--fsdp-devices','8','--num-workers','0','--no-wandb-enabled']
        write(output/'effective_training.json',dict(argv=argv,updates=2,formal_training=False,global_batch=192,
            gpus=8*count,processes=count,rank=rank,learning_rate_decay_steps=20000,initial_checkpoint=1688977))
        runtime_env.update(OPENPI_SMOKE_TWO_STEPS='1',OPENPI_SMOKE_REPORT=str(base_output/'smoke_metrics.jsonl'),
            OPENPI_VERIFY_FINAL_CHECKPOINT='1',OPENPI_REQUIRE_FINITE_METRICS='1',OPENPI_SUCCESS_PATH=str(base_output/'checkpoint_restore_ok.json'))
        if cfg.get('checkpoint_local_readback'):
            runtime_env.update(OPENPI_CHECKPOINT_LOCAL_READBACK='/var/tmp/wangfaye/four-source-readback/'+cfg['run_id'],
                OPENPI_OSSUTIL_PATH=str(REPO/'nebula/ossutil'),OPENPI_OSSUTIL_SHA256=cfg['ossutil_sha256'],
                OPENPI_DOWNLOAD_ENDPOINT=cfg['download_endpoint'])
        if count>1:
            runtime_env.update(OPENPI_PROCESS_COUNT=str(count),OPENPI_PROCESS_ID=str(rank),
                OPENPI_COORDINATOR_ADDRESS=coordinator,OPENPI_LOCAL_DEVICE_IDS='0,1,2,3,4,5,6,7',
                OPENPI_EXPECTED_GLOBAL_DEVICES=str(8*count),OPENPI_DISTRIBUTED_TIMEOUT='1200')
        log('Starting exactly two optimizer updates; no formal training')
        with (output/'trainer.log').open('w') as stream:
            run([python,REPO/'openpi/scripts/b1k/train_b1k.py',*argv],env=runtime_env,stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
        deadline=time.monotonic()+180
        while True:
            try:
                reports=[json.loads(x) for x in (base_output/'smoke_metrics.jsonl').read_text().splitlines()]
                if (base_output/'checkpoint_restore_ok.json').exists() and any(x['kind']=='checkpoint_restore' for x in reports):break
            except (OSError,json.JSONDecodeError):pass
            if time.monotonic()>deadline:raise TimeoutError('Global checkpoint proof did not become visible')
            time.sleep(2)
        if len([x for x in reports if x['kind']=='optimizer_update'])!=2 or not (base_output/'checkpoint_restore_ok.json').exists():
            raise ValueError('Two updates and checkpoint restore not proven')
        restore=[x for x in reports if x['kind']=='checkpoint_restore']
        if len(restore)!=1 or not restore[0]['all_state_leaves_exact'] or restore[0]['optimizer_clock']!=2:
            raise ValueError('Checkpoint restore proof missing')
        ckpt=checkpoint_base/'pi05_b1k'/cfg['run_id']
        write(output/'_RANK_SUCCESS.json',dict(rank=rank,completed_updates=2,checkpoint_restore=True))
        deadline=time.monotonic()+180
        while len(list(base_output.glob('rank-*/_RANK_SUCCESS.json')))<count:
            if list(base_output.glob('rank-*/failure.json')):raise RuntimeError('Peer smoke worker failed')
            if time.monotonic()>deadline:raise TimeoutError('Peer smoke completion timed out')
            time.sleep(2)
        if rank==0:write(base_output/'_SMOKE_SUCCESS.json',dict(run_id=cfg['run_id'],completed_updates=2,real_four_source_batches=2,
            checkpoint_restore=True,formal_training_started=False,discard_smoke_weights_for_formal_training=True,
            smoke_checkpoint_local=str(ckpt),processes=count,gpus=8*count,finished=time.time()))
        log('Two-update GPU smoke and exact checkpoint restore passed')
    except BaseException as exc:
        # Log files are private to this bounded run; never serialize env/argv secrets.
        write(output/'failure.json',dict(error_type=type(exc).__name__,message=str(exc)[:2000],formal_training=False,time=time.time()))
        raise

def stage(scratch,output,rank=0,process_count=1):
    from sco.prepare_retraining_data import safe_extract
    from openpi.training.subtask_mixture import MixturePlan
    from openpi.training.subtask_mixture_data import resolve_roots,verify_metadata
    scratch=Path(scratch);output=Path(output);mount=Path('/data/oss_bucket_0');dest=scratch/'datasets';dest.mkdir()
    manifest=read(REPO/'inputs/data_manifest.json');plan=MixturePlan.load(REPO/'inputs/compiled/mixture.json')
    expert=manifest['original_expert']['target'];packed=dest/'expert.tar.gz'
    copy_verified(mount/expert['key'],packed,expert['bytes'],expert['sha256'])
    unpack=dest/'expert';safe_extract(packed,unpack);packed.unlink()
    root=unpack/'expert_original';source=read(unpack/'source_objects.json')
    for name in ('data','videos'):(root/name).symlink_to(mount/source['prefix']/name,target_is_directory=True)
    if configuration().get('cache_smoke_original_files'):
        spec=importlib.util.spec_from_file_location('bounded_smoke_data_cache',REPO/'nebula/smoke_data_cache.py')
        cache_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache_module)
        from openpi.training.lerobot_compat import LeRobotDatasetMetadata
        meta=LeRobotDatasetMetadata(repo_id=plan.roots['expert_original']['repo_id'],root=str(root))
        cache_plan=cache_module.selected_original_files(plan,meta,source,rank=int(rank),process_count=int(process_count))
        cache_module.cache_original(root,cache_plan,source,mount,copy_verified=copy_verified,write=write,log=log,output=output)
    if configuration().get('cache_training_original_files'):
        spec=importlib.util.spec_from_file_location('bounded_training_data_cache',REPO/'nebula/smoke_data_cache.py')
        cache_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache_module)
        from openpi.training.lerobot_compat import LeRobotDatasetMetadata
        meta=LeRobotDatasetMetadata(repo_id=plan.roots['expert_original']['repo_id'],root=str(root))
        cache_plan=cache_module.selected_training_files(plan,meta,source,rank=int(rank),process_count=int(process_count))
        cache_module.cache_original(root,cache_plan,source,mount,copy_verified=copy_verified,write=write,log=log,output=output)
    roots={'expert_original':str(root)}
    def one(receipt):
        rid=receipt['root_id']
        if rid not in plan.roots:return None
        item=receipt['target'];packed=dest/(rid+'.tar')
        copy_verified(mount/item['key'],packed,item['bytes'],item['sha256'])
        target=dest/rid;safe_extract(packed,target);packed.unlink()
        reports=list(target.rglob('export_report.json'))
        if len(reports)!=1:raise ValueError('Ambiguous dataset root '+rid)
        return rid,str(reports[0].parent)
    count=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(one,manifest['archives']):
            if result:roots[result[0]]=result[1];count+=1
            if count and count%50==0:log('Staged '+str(count)+' collected roots')
    if set(roots)!=set(plan.roots):raise ValueError('Four-source roots incomplete')
    write(dest/'roots.json',roots)
    verify_metadata(plan,resolve_roots(plan,dest/'roots.json'))
    write(output/'data_staged.json',dict(roots=len(roots),mixture_fingerprint=plan.fingerprint,
        metadata_hashes_verified=True,archives_sha256_verified=True,
        original_data='full_original_local_cache' if configuration().get('cache_training_original_files') else ('bounded_exact_local_cache' if configuration().get('cache_smoke_original_files') else 'existing_OSS_mount'),
        real_batch_validation='pending_in_GPU_trainer'))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='stage':stage(*sys.argv[2:])
    else:execute()
