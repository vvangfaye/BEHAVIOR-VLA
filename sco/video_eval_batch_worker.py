"""Append to the frozen single-episode worker; reuse one server/evaluator for <=10 instances.

The original evaluator and model code are unchanged. All episode outputs are
published separately; the shared queue and reset observations preserve provenance.
"""

BATCH_PROGRESS_HOOK = r'''
import importlib.abc,importlib.machinery,json,os,sys,time
from pathlib import Path
if os.environ.get("VIDEO_EVAL_PROGRESS") and sys.argv[0].endswith("eval_b1k_queue.py"):
    _path=Path(os.environ["VIDEO_EVAL_PROGRESS"])
    _episodes=json.loads(os.environ["VIDEO_EVAL_EPISODES"])
    _total=0
    _events=[]
    def _atomic(path,value):
        tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value));os.replace(tmp,path)
    class _Loader(importlib.abc.Loader):
        def __init__(self,original):self.original=original
        def create_module(self,spec):return self.original.create_module(spec)
        def exec_module(self,module):
            self.original.exec_module(module)
            load,reset,step=module.Evaluator.load_task_instance,module.Evaluator.reset,module.Evaluator.step
            def observed_load(evaluator,instance_id,*args,**kwargs):
                result=load(evaluator,instance_id,*args,**kwargs)
                evaluator._video_batch_instance=int(instance_id)
                _events.append(dict(event='load',instance_id=int(instance_id),pid=os.getpid(),epoch=time.time()))
                _atomic(_path.parent/'reuse_evidence.json',_events)
                return result
            def observed_reset(evaluator,*args,**kwargs):
                result=reset(evaluator,*args,**kwargs)
                evaluator._video_batch_observed=None
                _events.append(dict(event='reset',instance_id=getattr(evaluator,'_video_batch_instance',None),prepare_for_task_instance=bool(kwargs.get('prepare_for_task_instance')),pid=os.getpid(),epoch=time.time()))
                _atomic(_path.parent/'reuse_evidence.json',_events)
                return result
            def observed_step(evaluator,*args,**kwargs):
                global _total
                result=step(evaluator,*args,**kwargs);_total+=1
                now=time.monotonic();prior=getattr(evaluator,'_video_batch_observed',None)
                terminal=bool(result[0] or result[1])
                if prior is None or now-prior>=5 or terminal:
                    instance=int(evaluator._video_batch_instance);ep=_episodes[str(instance)]
                    value=dict(state='progress',steps=int(evaluator._rollout_steps),terminal=terminal,
                               run_id=ep['run_id'],pid=os.getpid(),observed_at=time.time(),instance_id=instance)
                    _atomic(Path(ep['path']),value)
                    _atomic(_path,dict(value,steps=_total,run_id=os.environ['VIDEO_EVAL_RUN_ID']))
                    evaluator._video_batch_observed=now
                return result
            module.Evaluator.load_task_instance=observed_load
            module.Evaluator.reset=observed_reset
            module.Evaluator.step=observed_step
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname!='omnigibson.eval.evaluator':return None
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec and spec.loader:spec.loader=_Loader(spec.loader)
            return spec
    sys.meta_path.insert(0,_Finder())
'''


def validate_batch(payload):
    children = payload['episodes']
    if not 2 <= len(children) <= 10:
        raise ValueError('A batch must contain2..10 episodes')
    names, instances, roots = set(), set(), set()
    first = children[0]
    for child in children:
        validate_payload(child)
        for field in ('checkpoint_step', 'task', 'checkpoint', 'code', 'runtime',
                      'source_manifest_sha256', 'checkpoint_ready_sha256'):
            if child[field] != first[field]:
                raise ValueError('Batch mixes task, source or checkpoint')
        names.add(child['run_id']); instances.add(child['instance_id']); roots.add(child['assignment_root'])
        if child['pilot'] or child.get('recovery_chain'):
            raise ValueError('Only unsubmitted production episodes may be batched')
    if len(names) != len(children) or len(instances) != len(children) or len(roots) != len(children):
        raise ValueError('Repeated batch episode')
    group = Path(payload['assignment_root'])
    if group.parent.name != 'batches' or not re.fullmatch(r'[A-Za-z0-9_-]+', payload['batch_name']):
        raise ValueError('Invalid batch root/name')
    if group.name != payload['batch_name']:
        raise ValueError('Batch directory identity mismatch')
    for root in roots:
        if Path(root).parent != group.parent.parent / 'episodes':
            raise ValueError('Episode directory outside batch campaign')
    return children


def _batch_failure(child, error):
    root = Path(child['assignment_root'])
    if (root / '_SUCCESS').exists():
        return
    prior = json.loads((root / 'receipt.json').read_text()) if (root / 'receipt.json').exists() else {}
    if prior.get('state') in ('artifacts_verified', 'complete', 'delivery_failed'):
        return
    receipt = dict(checkpoint_step=2000000, task=child['task'], instance_id=child['instance_id'],
                   run_id=child['run_id'], execution_name=child['execution_name'], rollout_id=0,
                   source_manifest_sha256=child['source_manifest_sha256'],
                   checkpoint_ready_sha256=child['checkpoint_ready_sha256'], state='failed',
                   error_type=type(error).__name__, error=str(error)[:200], verified=False,
                   batch_context=child['batch_context'])
    write_json(root / 'receipt.json', receipt)
    try:
        put_file(root / 'receipt.json', child['artifact_puts']['receipt'])
    except Exception:
        pass


def run(payload):
    children = validate_batch(payload)
    group = Path(payload['assignment_root'])
    # Validate the complete reservation before creating any episode directory.
    if group.exists() or any(Path(c['assignment_root']).exists() for c in children):
        raise ValueError('Batch would overwrite an existing attempt')
    source = verify_inputs(children[0])
    group.mkdir(parents=True, exist_ok=False)
    executed_source = _EXECUTED_WORKER_SOURCE
    if hashlib.sha256(executed_source).hexdigest() != payload['batch_worker_sha256']:
        raise ValueError('Batch executed source mismatch')
    (group / 'batch_worker.py').write_bytes(executed_source)
    safe = dict(payload, episodes=[validate_payload(c) for c in children])
    write_json(group / 'payload.json', safe)
    for child in children:
        root = Path(child['assignment_root']); root.mkdir(parents=True, exist_ok=False)
        child['batch_context'] = dict(batch_name=payload['batch_name'], root=str(group),
            instance_ids=[c['instance_id'] for c in children], batch_worker_sha256=payload['batch_worker_sha256'])
        write_json(root / 'payload.json', validate_payload(child))
    cache = Path('/var/tmp/behavior-1k-runtime') / payload['batch_name']
    cache.mkdir(parents=True, exist_ok=False)
    if cache.stat().st_dev == Path(children[0]['runtime']).stat().st_dev or shutil.disk_usage(cache).free < 40 * 1024**3:
        raise ValueError('Batch cache must be local with at least40GiB free')
    hook_dir = cache / 'progress_hook'; hook_dir.mkdir()
    (hook_dir / 'sitecustomize.py').write_text(BATCH_PROGRESS_HOOK)
    env, gpu_rows = _environment(children[0], source, group, cache, hook_dir)
    env.update(INSTANCE_INDICES=' '.join(str(c['instance_id']-301) for c in children),
               PI05_EVAL_QUEUE_RUN_ID=payload['batch_name'], VIDEO_EVAL_RUN_ID=payload['batch_name'],
               VIDEO_EVAL_PROGRESS=str(group / 'native_progress.json'),
               VIDEO_EVAL_EPISODES=json.dumps({str(c['instance_id']):dict(run_id=c['run_id'],
                   path=str(Path(c['assignment_root'])/'native_progress.json')) for c in children}))
    sys.path.insert(0, str(Path(children[0]['code']) / 'scripts'))
    from pi05_eval_queue import EvaluationQueue
    queue = EvaluationQueue(group / 'queue.sqlite3')
    queue.initialize([(c['task'], c['instance_id']) for c in children], max_attempts=1,
                     metadata=dict(run_id=payload['batch_name'], mode='public_test', num_rollouts=1,
                         require_videos=True, checkpoint_step=2000000,
                         source_manifest_sha256=children[0]['source_manifest_sha256']))
    command = ['bash', str(Path(children[0]['code']) / 'scripts/eval_b1k.sh'), children[0]['task']]
    execution = dict(command=command, env={k:env[k] for k in (
        'MODE','PROMPT_MODE','NUM_ROLLOUTS','INSTANCE_INDICES','WRITE_VIDEO','PI05_CKPT',
        'SUBTASK_PREDICTION','EXECUTE_HORIZON','MAX_TOKEN_LEN','MAX_SUBTASK_LEN','PI05_REPO_ID',
        'PI05_MEM_FRACTION','XLA_PYTHON_CLIENT_PREALLOCATE','OMNIGIBSON_DISABLE_TORCH_COMPILE',
        'SERVER_GPU','EVAL_GPU','PORT','OMNIGIBSON_DATA_PATH','PI05_ENV','EVAL_ENV','PYTHONPATH',
        'PYTHONNOUSERSITE','LD_LIBRARY_PATH','VK_ICD_FILENAMES','__EGL_VENDOR_LIBRARY_FILENAMES')},
        wrapper='omnigibson.eval.wrappers.DefaultWrapper',robot_config='omnigibson/eval/r1pro.yaml',
        max_steps=None,video_fps=30,native_progress_hook_sha256=sha256(hook_dir/'sitecustomize.py'))
    write_json(group/'execution.json',execution)
    watcher = ProgressWatchdog(time.monotonic())
    process = None; failure = None
    try:
        with (group/'driver.log').open('w',buffering=1) as log:
            process=subprocess.Popen(command,env=env,cwd=children[0]['code'],stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                progress=json.loads((group/'native_progress.json').read_text()) if (group/'native_progress.json').exists() else {}
                reason=watcher.observe(time.monotonic(),progress)
                if reason:raise TimeoutError(reason)
                time.sleep(5)
            if process.returncode:raise RuntimeError('Batch evaluator exited unsuccessfully')
    except BaseException as error:
        failure=error
    finally:
        if process is not None:terminate_group(process)
    snapshot=queue.snapshot(readonly=True)
    write_json(group/'queue_snapshot.json',snapshot)
    for child in children:
        try:
            _batch_finalize(child,group,source,env,hook_dir,queue,snapshot,gpu_rows,execution)
        except BaseException as error:
            _batch_failure(child,error);failure=failure or error
    if failure is not None:
        raise RuntimeError('Batch ended with preserved incomplete or undelivered episodes') from None
    write_json(group/'_SUCCESS',dict(batch_name=payload['batch_name'],episodes=len(children),
                                   batch_worker_sha256=payload['batch_worker_sha256']))


def _batch_finalize(payload, group, source, env, hook_dir, original_queue, snapshot, gpu_rows, execution):
    from pi05_eval_queue import EvaluationQueue
    root=Path(payload['assignment_root']);code=Path(payload['code']);runtime=Path(payload['runtime'])
    selected=dict(snapshot)
    selected['jobs']=[r for r in snapshot['jobs'] if r['task']==payload['task'] and r['instance_id']==payload['instance_id']]
    selected['attempts']=[r for r in snapshot['attempts'] if r['task']==payload['task'] and r['instance_id']==payload['instance_id']]
    selected['total']=len(selected['jobs'])
    selected['counts']={state:sum(row['state']==state for row in selected['jobs']) for state in snapshot['counts']}
    if len(selected['jobs'])!=1 or selected['jobs'][0]['state']!='succeeded' or selected['jobs'][0]['attempts']!=1:
        raise ValueError('Batch episode does not have one completed original attempt')
    class SingleEpisodeView(EvaluationQueue):
        def snapshot(self,**kwargs):return selected
        def metadata(self,**kwargs):return original_queue.metadata(**kwargs)
    queue=SingleEpisodeView(original_queue.db_path)
    for kind,extension in (('json','.json'),('videos','.mp4')):
        stem=f"{payload['task']}_{payload['instance_id']}_0{extension}"
        original=group/'results/pi05'/payload['task']/kind/stem
        target=root/'results/pi05'/payload['task']/kind/stem;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,target)
        if sha256(original)!=sha256(target):raise ValueError('Batch original copy changed')
    write_json(root/'execution.json',dict(execution,batch_context=payload['batch_context']))
    write_json(root/'batch_queue_snapshot.json',snapshot)
    shutil.copyfile(group/'reuse_evidence.json',root/'batch_reuse.json')
    shutil.copyfile(group/'driver.log',root/'driver.log')
    shutil.copytree(group/'logs',root/'logs')
    progress=json.loads((root/'native_progress.json').read_text())
    receipt=dict(schema_version=1,checkpoint_step=2000000,task=payload['task'],instance_id=payload['instance_id'],
        rollout_id=0,run_id=payload['run_id'],execution_name=payload['execution_name'],pilot=False,
        source_manifest_sha256=payload['source_manifest_sha256'],checkpoint_ready_sha256=payload['checkpoint_ready_sha256'],
        started_at=selected['attempts'][0]['started_at'],state='running',gpu_rows=gpu_rows,
        batch_context=payload['batch_context'])
    reuse=json.loads((root/'batch_reuse.json').read_text())
    pids={event['pid'] for event in reuse}
    reset_ids={event['instance_id'] for event in reuse if event['event']=='reset' and event['instance_id'] is not None and not event.get('prepare_for_task_instance')}
    server_pids=re.findall(r'policy server pid=([0-9]+)',(root/'driver.log').read_text())
    if len(pids)!=1 or len(server_pids)!=1 or payload['instance_id'] not in reset_ids:
        raise ValueError('Cannot prove model/evaluator reuse and independent reset')
    receipt['batch_reuse']=dict(single_model_server=True,single_evaluator_process=True,
        server_pid=int(server_pids[0]),evaluator_pid=next(iter(pids)),reset_instance_ids=sorted(reset_ids),
        loaded_instance_ids=[e['instance_id'] for e in reuse if e['event']=='load'])
    snapshot=selected

    artifact = queue.create_result_manifest(root / "results", root / "RESULT_MANIFEST.json", model="pi05", num_rollouts=1, require_videos=True)
    if artifact["rollout_count"] != 1:
        raise ValueError("Single-episode output mismatch")
    row = artifact["rollouts"][0]
    result_path = contained_file(root / "results", row["json"]["path"])
    video_path = contained_file(root / "results", row["video"]["path"])
    value = json.loads(result_path.read_bytes())
    steps, q = validate_result(value, payload, progress)
    attempt = snapshot["attempts"]
    if len(attempt) != 1 or attempt[0]["state"] != "succeeded":
        raise ValueError("Unexpected attempt history")
    attempt_root = Path(attempt[0]["attempt_dir"])
    if sha256(attempt_root / "json" / result_path.name) != sha256(result_path) or sha256(attempt_root / "videos" / video_path.name) != sha256(video_path):
        raise ValueError("JSON and video must both match the single original attempt")
    frames_dir = root / "video_verification"
    frames_dir.mkdir()
    video_validation = json.loads(subprocess.check_output(
        [str(runtime / "envs/b1k-eval/bin/python"), "-c", VIDEO_CHECK, str(video_path), str(steps), str(frames_dir)],
        env=env, text=True, timeout=1800))
    write_json(frames_dir / "validation.json", video_validation)
    bundle = root / "bundle"
    bundle.mkdir()
    originals = {"json/" + result_path.name: result_path, "videos/" + video_path.name: video_path,
                 "metadata/source_manifest.json": code / "_PUBLIC_TEST_SOURCE_READY",
                 "metadata/checkpoint_ready.json": Path(payload["checkpoint"]) / "_READY",
                 "metadata/execution.json": root / "execution.json", "metadata/payload.json": root / "payload.json",
                 "metadata/native_progress.json": root / "native_progress.json", "metadata/RESULT_MANIFEST.json": root / "RESULT_MANIFEST.json",
                 "config/r1pro.yaml": code / "BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
                 "wrapper/default_wrapper.py": code / "BEHAVIOR-1K/OmniGibson/omnigibson/eval/wrappers/default_wrapper.py",
                 "code/progress_sitecustomize.py": hook_dir / "sitecustomize.py", "logs/driver.log": root / "driver.log",
                 "code/batch_worker.py": group / "batch_worker.py",
                     "metadata/batch_queue_snapshot.json": root / "batch_queue_snapshot.json",
                     "metadata/batch_reuse.json": root / "batch_reuse.json"}
    write_json(root / "queue_snapshot.json", snapshot)
    originals["metadata/queue_snapshot.json"] = root / "queue_snapshot.json"
    for directory in (root / "logs", frames_dir):
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    originals[path.relative_to(root).as_posix()] = path
    for relative in ("scripts/eval_b1k.sh", "scripts/eval_b1k_queue.py", "scripts/pi05_eval_queue.py",
                     "BEHAVIOR-1K/OmniGibson/omnigibson/eval/evaluator.py",
                     "BEHAVIOR-1K/OmniGibson/omnigibson/eval/eval.py"):
        originals["code/" + relative] = contained_file(code, relative)
    files = []
    for relative, path in originals.items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original_hash = sha256(path)
        shutil.copyfile(path, target)
        if sha256(target) != original_hash:
            raise ValueError("Artifact changed during packaging")
        files.append(dict(path=relative, bytes=target.stat().st_size, sha256=original_hash))
    receipt.update(state="artifacts_verified", completed_at=time.time(), model_success=value["success"], q_score=q,
                   verified=False, metrics=dict(success=value["success"], q_score=q),
                   steps=steps, video_validation=video_validation, rollout_count=1,
                   canonical_results_kept_regardless_of_score=True, same_attempt_json_and_video=True,
                   full_video_decode_verified=True, files=files)
    write_json(bundle / "ARTIFACT_MANIFEST.json", receipt)
    archive = root / "artifact.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(bundle, arcname="episode", recursive=True)
    receipt["archive"] = dict(filename=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive))
    receipt["archive_sha256"] = receipt["archive"]["sha256"]
    receipt["archive_size"] = receipt["archive"]["bytes"]
    write_json(root / "receipt.json", receipt)
    put_file(archive, payload["artifact_puts"]["archive"])
    verify_remote_archive(payload["artifact_gets"]["archive"], receipt["archive_size"], receipt["archive_sha256"])
    receipt.update(archive_readback_verified=True, verified=True, state="complete")
    write_json(root / "receipt.json", receipt)
    put_file(root / "receipt.json", payload["artifact_puts"]["receipt"])
    write_json(root / "_SUCCESS", receipt)
    print("STATUS_JSON=" + json.dumps({k: v for k, v in receipt.items() if k not in ("files", "gpu_rows")}), flush=True)
    return receipt
