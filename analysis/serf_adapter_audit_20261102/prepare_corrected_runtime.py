"""Create an isolated corrected candidate; frozen campaigns stay unchanged."""
from pathlib import Path
import ast
import gzip
import hashlib
import importlib.util
import io
import json
import shutil
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'analysis/serf_multitask_20261029'
AUTHOR=ROOT/'analysis/serf_reproduction_20261025/repos/SERF-VLA'


def replace(path, old, new):
    text=path.read_text();assert old in text, (path,old[:100])
    path.write_text(text.replace(old,new))


def main():
    runtime=HERE/'runtime'
    assert not runtime.exists(), 'Do not overwrite a submitted candidate'
    shutil.copytree(OLD/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__'))
    spec=importlib.util.spec_from_file_location('author_goal_patch',AUTHOR/'scripts/setup/patch_collecting_childrens_toys_bddl.py')
    patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
    goal=runtime/'repo/BEHAVIOR-1K/bddl/bddl/activity_definitions/collecting_childrens_toys/problem0.bddl'
    original=goal.read_text();start,end=patch.find_goal_span(original)
    goal.write_text(original[:start]+patch.GOAL+original[end:])
    patch.patch_file(goal,check=True)
    case=runtime/'run_case.py'
    replace(case,"p.add_argument('--initialize-only',action='store_true')",
            "p.add_argument('--initialize-only',action='store_true')\np.add_argument('--audit-steps',type=int,default=0)")
    replace(case,"'write_video=true','write_third_person_video=false','record_step_q_score=true']",
            "'write_video=true','write_third_person_video=false',f'record_step_q_score={str(TASK_ID != 26).lower()}']")
    old="    elif args.gate_replay:\n        evaluator.prepare_episode(instance_id=instance,episode_id=0)"
    new="""    elif args.audit_steps:
        assert 0 < args.audit_steps <= 200
        metrics_path,_,_,_=base.resolve_eval_output_paths(config)
        evaluator.prepare_episode(instance_id=instance,episode_id=0)
        evaluator.reset()
        evaluator.prepare_episode_artifacts(instance_id=instance,episode_id=0)
        evaluator.start_episode_logging(metrics_path=metrics_path,instance_id=instance,episode_id=0)
        terminated=truncated=False
        for _ in range(args.audit_steps):
            terminated,truncated=evaluator.step()
            if terminated or truncated:break
        result={'task_id':TASK_ID,'instance':instance,'condition':args.condition,
                'kind':'adapter_audit_unscored','diagnostic_only':True,
                'simulator_steps':int(evaluator.env._current_step),
                'score':None,'success':None,'complete':True,
                'artifacts':{},'wall_seconds':time.time()-t0,
                'native_terminated':bool(terminated),'native_truncated':bool(truncated)}
        assert result['simulator_steps']>0
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print('SERF_ADAPTER_AUDIT_RESULT='+json.dumps(result),flush=True)
    elif args.gate_replay:
        evaluator.prepare_episode(instance_id=instance,episode_id=0)"""
    replace(case,old,new)
    server=runtime/'serve_policy.py'
    replace(server,"p.add_argument('--port',type=int,default=8000)",
            "p.add_argument('--port',type=int,default=8000)\np.add_argument('--audit-output',type=Path)")
    replace(server,"        action=wrapped.act(obs)","""        if args.audit_output is not None and expected in [1,21,101,181]:
            import hashlib,json,numpy as np
            args.audit_output.mkdir(parents=True,exist_ok=True)
            arrays={k:np.asarray(v) for k,v in obs.items()}
            summary={k:{'shape':list(v.shape),'dtype':str(v.dtype),
                        'finite':bool(np.isfinite(v).all()),
                        'sha256':hashlib.sha256(v.tobytes()).hexdigest()}
                     for k,v in arrays.items()}
            (args.audit_output/f'input_{expected-1}.json').write_text(json.dumps(summary,indent=2))
            np.savez_compressed(args.audit_output/f'input_{expected-1}.npz',**arrays)
        action=wrapped.act(obs)""")
    worker=HERE/'gpu_worker.py';shutil.copyfile(OLD/'gpu_worker.py',worker)
    # runpy from python -c does not add the script directory to sys.path.
    replace(worker,"script=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')",
            "script=sys.argv[1];sys.path.insert(0,str(__import__('pathlib').Path(script).parent));sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')")
    replace(worker,"'--repo',str(repo),'--condition',condition,'--checkpoint',str(checkpoint)],",
            "'--repo',str(repo),'--condition',condition,'--checkpoint',str(checkpoint),*(['--audit-output',str(dest/'inputs')] if config.get('audit_steps') else [])],")
    replace(worker,"'--shared-initial',str(config.get('shared_initial_dirs',{}).get(str(index),output/f'shared_initial_{index}')),'--maps',config['maps_root']],",
            "'--shared-initial',str(config.get('shared_initial_dirs',{}).get(str(index),output/f'shared_initial_{index}')),'--maps',config['maps_root'],*(['--audit-steps',str(config['audit_steps'])] if config.get('audit_steps') else [])],")
    # Limited retries cover transient DNS/connection failures; no inference request is retried.
    replace(worker,"    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))", """    raw_opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    class StorageOpener:
        def open(self,request,timeout=None):
            for attempt in range(3):
                try:return raw_opener.open(request,timeout=timeout)
                except urllib.error.URLError:
                    if attempt==2:raise
                    time.sleep(2**attempt)
    opener=StorageOpener()""")
    manifest={}
    with (HERE/'source.tar.gz').open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for f in sorted(runtime.rglob('*')):
            if not f.is_file():continue
            b=f.read_bytes();rel=str(f.relative_to(runtime))
            if f.suffix=='.py':ast.parse(b,filename=rel)
            manifest[rel]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
            entry=tarfile.TarInfo(rel);entry.size=len(b);entry.mode=0o644;entry.mtime=0
            tar.addfile(entry,io.BytesIO(b))
    ast.parse(worker.read_text())
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    receipt={'source_sha256':hashlib.sha256((HERE/'source.tar.gz').read_bytes()).hexdigest(),
             'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),
             'bytes':(HERE/'source.tar.gz').stat().st_size,'files':len(manifest),
             'corrected_goal_sha256':hashlib.sha256(goal.read_bytes()).hexdigest()}
    (HERE/'source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
