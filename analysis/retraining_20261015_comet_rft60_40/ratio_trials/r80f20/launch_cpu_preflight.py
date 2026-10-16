"""Submit at most one CPU-only data check. Never dispatch GPU smoke or training."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CPU = ROOT/'gates/cpu'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    auth = json.loads((ROOT/'cpu_preflight_authorization.json').read_text())
    if not (auth['cpu_preflight_allowed'] and auth['maximum_new_jobs'] == 1
            and auth['allocated_gpus'] == 1 and auth['optimizer_updates'] == 0
            and auth['formal_training_allowed'] is False):
        raise ValueError('CPU-only scope required')
    if (ROOT/'STOP').exists() or (CPU/'intent.json').exists() or (CPU/'receipt.json').exists():
        raise ValueError('STOP or existing attempt; reconcile instead of resubmitting')
    bundle = CPU/'bundle'
    sys.path.insert(0, str(bundle))
    sys.path.insert(0, str(bundle/'openpi/src'))
    from openpi.training.subtask_mixture import MixturePlan
    plan = MixturePlan.load(bundle/'inputs/compiled/mixture.json')
    job = json.loads((bundle/'nebula/smoke_job.json').read_text())
    probe = json.loads((bundle/'probe_job.json').read_text())
    if (plan.fingerprint != auth['mixture_fingerprint'] or job['mixture_fingerprint'] != plan.fingerprint
            or job['updates'] != 0 or job['process_count'] != 1
            or job['formal_training_authorized'] is not False
            or probe['run_id'] != job['run_id'] or probe['output'] != job['output']):
        raise ValueError('CPU data-check identity changed')
    expected = json.loads((bundle/'bundle_manifest.json').read_text())
    support = module('ratio_dispatch_support', ROOT/'dispatch_support.py')
    for relative, digest in expected.items():
        if support.digest(bundle/relative) != digest:
            raise ValueError('Prepared CPU source changed: '+relative)
    client = module('ratio_oss', bundle/'nebula/submit_oss.py').OSSClient()
    # Validate frozen remote runtime and data before reserving the unique run.
    for field in ('environment', 'cache'):
        item = job[field]
        head = client.bucket.head_object(item['key'])
        if head.content_length != item['bytes'] or head.etag != item['etag']:
            raise ValueError('Frozen runtime changed: '+field)
    import hashlib
    if hashlib.sha256(client.read(job['data_manifest_key'])).hexdigest() != job['data_manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    support.create(client, 'cpu-r80f20', CPU, 'four_source_cpu_probe.py',
                   bundle/'probe_cluster.json', job['output'], 1, 1,
                   auth['maximum_runtime_seconds'], job['run_id'])


if __name__ == '__main__':
    main()
