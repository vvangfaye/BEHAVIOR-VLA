"""Submit this unique two-step expert control once; never start formal training."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'smoke/bundle'))
import dispatch_support as dispatch
from nebula.submit_oss import OSSClient


def main():
    with (ROOT/'submission.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder = ROOT/'smoke'
        bundle = folder/'bundle'
        job = dispatch.read(bundle/'nebula/smoke_job.json')
        assert job['run_id'] == 'b1k-expert-control-1688977-20261016-smoke'
        assert job['updates'] == 2 and job['process_count'] == 2
        assert job['formal_training_authorized'] is False
        assert job['expected_source_counts'] == {'replay':192,'full_episode':0}
        if (folder/'intent.json').exists() or (folder/'receipt.json').exists():
            raise RuntimeError('Existing submission evidence: reconcile, never submit again')
        for name,digest in dispatch.read(bundle/'bundle_manifest.json').items():
            assert dispatch.digest(bundle/name)==digest, name
        client = OSSClient()
        assert hashlib.sha256(client.read(job['data_manifest_key'])).hexdigest()==job['data_manifest_sha256']
        for field in ('environment','cache'):
            head = client.bucket.head_object(job[field]['key'])
            assert head.content_length == job[field]['bytes'] and head.etag == job[field]['etag']
        spec=importlib.util.spec_from_file_location('source_guard',ROOT/'submit_two_source_smoke.py')
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        dispatch.save(folder/'checkpoint_source_preflight.json',guard.verify_checkpoint_source(client,bundle,job))
        dispatch.create(client,'expert_control_smoke',folder,'nebula/four_source_smoke.py',
            bundle/'nebula/smoke_cluster.json',job['output'],2,16,14400,job['run_id'])


if __name__ == '__main__':
    main()
