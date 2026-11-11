"""Direct 1000-update expert control under explicit user waiver of smoke."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'formal_stage/bundle'))
import dispatch_support as dispatch
from nebula.submit_oss import OSSClient


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def main():
    with (ROOT/'submission.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        folder=ROOT/'formal_stage';bundle=folder/'bundle'
        if (folder/'intent.json').exists() or (folder/'receipt.json').exists():
            raise RuntimeError('Existing formal submission evidence; never submit twice')
        job=dispatch.read(bundle/'nebula/worker_job.json')
        assert job['run_id']=='b1k-expert-control-1688977-20261016'
        assert job['stage_end_step']==job['max_steps']==1000 and not job['resume']
        for name,digest in dispatch.read(bundle/'bundle_manifest.json').items():
            assert dispatch.digest(bundle/name)==digest,name
        client=OSSClient()
        authorization=dispatch.read(bundle/'inputs/formal_authorization.json')
        assert job.get('skip_gpu_smoke') is True
        assert authorization.get('smoke_waiver_instruction')=='不需要烟测，直接纯专家排队'
        worker=module('formal_worker',bundle/'nebula/train_four_source.py')
        worker.validate(job,authorization,{})
        assert hashlib.sha256(client.read(job['data_manifest_key'])).hexdigest()==job['data_manifest_sha256']
        guard=module('checkpoint_guard',ROOT/'submit_two_source_smoke.py')
        dispatch.save(folder/'checkpoint_source_preflight.json',guard.verify_checkpoint_source(client,bundle,job))
        inventory=module('inventory_guard',ROOT/'verify_source_inventory.py')
        dispatch.save(folder/'original_source_preflight.json',inventory.verify_original_objects(client,dispatch.read(bundle/'inputs/original_source_objects.json')))
        for field in ('environment','cache'):
            head=client.bucket.head_object(job[field]['key'])
            assert head.content_length==job[field]['bytes'] and head.etag==job[field]['etag']
        (ROOT/'direct_flow').mkdir(exist_ok=True)
        dispatch.HERE=ROOT/'direct_flow'
        dispatch.create(client,'expert_control_direct_formal',folder,'nebula/train_four_source.py',
            bundle/'nebula/training_cluster.json',job['output'],2,16,21600,job['run_id']+'-'+job['attempt_id'])


if __name__=='__main__':main()
