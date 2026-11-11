"""At most one CPU check, one two-step GPU smoke, and one authorized formal job."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    s=module('quality_dispatch',ROOT/'dispatch_support.py')
    with (ROOT/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        auth=s.read(ROOT/'execution_authorization.json')
        audit=s.read(ROOT/'metadata_sampling_validation.json')
        assert auth['formal_training_allowed'] and auth['max_completed_updates']==5000
        assert (auth['workers'],auth['gpus_per_worker'])==(2,8)
        assert auth['peak_learning_rate']==1e-6 and auth['fixed_decay_end']==20000
        assert audit['passed'] and audit['samples']==960000 and audit['max_full_episode_draws']<=256
        assert audit['max_full_window_repetitions']<=2 and audit['samples_per_task']==9600
        preparation=s.read(ROOT/'preparation.json')
        fp=audit['mixture_fingerprint']
        assert all(x['mixture_fingerprint']==fp for x in preparation['stages'].values())
        sys.path.insert(0,str(ROOT/'formal_stage/bundle'))
        client=module('quality_oss',ROOT/'formal_stage/bundle/nebula/submit_oss.py').OSSClient()
        def stage_receipt(stage,folder,entry,cluster,output,workers,gpus,seconds,run_id):
            path=folder/'receipt.json'
            if path.exists():
                receipt=s.read(path)
                if (receipt.get('state')!='submitted' or len(receipt.get('identifiers',[]))!=1
                        or receipt.get('run_id')!=run_id or receipt.get('output')!=output
                        or receipt.get('gpus')!=gpus or receipt.get('workers')!=workers):
                    raise RuntimeError('Existing stage identity ambiguous; no resubmission')
                return receipt
            if (folder/'intent.json').exists():
                raise RuntimeError('Unknown creation intent; reconcile exact task before continuing')
            return s.create(client,stage,folder,entry,cluster,output,workers,gpus,seconds,run_id)
        cpu=ROOT/'cpu';job=s.read(cpu/'bundle/nebula/smoke_job.json')
        assert job['updates']==0 and job['process_count']==1 and job['mixture_fingerprint']==fp
        assert hashlib.sha256(client.read(job['data_manifest_key'])).hexdigest()==job['data_manifest_sha256']
        for field in ['environment','cache']:
            spec=job[field];head=client.bucket.head_object(spec['key'])
            assert head.content_length==spec['bytes'] and head.etag==spec['etag']
        amendment=ROOT/'combined_gpu_acceptance.json'
        if amendment.exists():
            approved=s.read(amendment)
            assert approved['mode']=='same_two_global_batches_on_16GPU_smoke'
            assert approved['mixture_fingerprint']==fp and approved['cpu_iid']=='239591'
            failed=s.status(approved['cpu_iid']);s.save(cpu/'final_status.json',failed)
            if failed['application_status'] not in ('FAIL','FAILED','FINISHED','KILLED','STOPPED','SUCCESS','SUCCEEDED'):
                raise RuntimeError('Standalone CPU task is not terminal; no new stage')
            errors=[s.read(p) for p in cpu.glob('failure_*.json')]
            if not any('Not enough local disk' in str(e.get('details','')) for e in errors):
                raise RuntimeError('Expected isolated CPU disk failure not proven')
            s.save(ROOT/'smoke/cpu_preflight_disposition.json',dict(
                standalone_cpu_passed=False,reason='local cache disk shortage',
                full_real_batches_required_on_GPU=2,global_batch=192,cpu_task=failed))
        else:
            receipt=stage_receipt('cpu',cpu,'four_source_cpu_probe.py',cpu/'bundle/probe_cluster.json',job['output'],1,1,10800,job['run_id'])
            proof=s.wait_receipt(client,'cpu',cpu,receipt,'bootstrap_result.json',14400)
            s.cpu_validate(proof,fp);s.save(ROOT/'smoke/cpu_batch_preflight.json',proof)
        smoke=ROOT/'smoke';job=s.read(smoke/'bundle/nebula/smoke_job.json')
        assert job['updates']==2 and job['process_count']==2 and job['mixture_fingerprint']==fp
        helper=module('quality_source_guard',ROOT/'submit_two_source_smoke.py')
        s.save(smoke/'checkpoint_source_preflight.json',helper.verify_checkpoint_source(client,smoke/'bundle',job))
        receipt=stage_receipt('smoke',smoke,'nebula/four_source_smoke.py',smoke/'bundle/nebula/smoke_cluster.json',job['output'],2,16,14400,job['run_id'])
        s.wait_receipt(client,'smoke',smoke,receipt,'_SMOKE_SUCCESS.json',18000)
        verifier=module('quality_smoke_verifier',ROOT/'submit_two_source_training.py')
        verified=verifier.verify_smoke(client,client.key(job['output']))
        assert verified['verified_mixture_fingerprint']==fp
        s.save(smoke/'verified_gpu_smoke.json',verified)
        formal=ROOT/'formal_stage';b=formal/'bundle';job=s.read(b/'nebula/worker_job.json')
        assert job['stage_end_step']==job['max_steps']==5000 and not job['resume']
        assert job['mixture_fingerprint']==fp
        s.save(b/'inputs/gpu_smoke_success.json',verified['success'])
        module('quality_formal_guard',b/'nebula/train_four_source.py').validate(job,s.read(b/'inputs/formal_authorization.json'),verified['success'])
        s.save(formal/'original_source_preflight.json',verifier.verify_original_objects(client,s.read(b/'inputs/original_source_objects.json')))
        s.save(formal/'checkpoint_source_preflight.json',helper.verify_checkpoint_source(client,b,job))
        receipt=stage_receipt('formal',formal,'nebula/train_four_source.py',b/'nebula/training_cluster.json',job['output'],2,16,153600,job['run_id']+'-'+job['attempt_id'])
        s.save(ROOT/'gated_training_status.json',dict(stage='formal',state='submitted',receipt=receipt,
            observed_at=time.time(),all_new_CPU_GPU_gates_passed=True,formal_updates=5000,automatic_extension=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        module('quality_error_dispatch',ROOT/'dispatch_support.py').save(ROOT/'gated_training_status.json',
            dict(state='halted',error_type=type(error).__name__,reason=str(error)[:500],observed_at=time.time(),automatic_retry=False))
        raise
