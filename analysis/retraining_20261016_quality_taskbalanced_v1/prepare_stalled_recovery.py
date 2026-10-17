"""Prepare two single-episode restarts; do not stop or submit GPU jobs."""
import base64
import hashlib
import json
import shutil
from pathlib import Path

from run_dual_evaluation import module, read, save

ROOT = Path(__file__).resolve().parent
CASES = (
    ('paired_fixed20_step5000_5x4', 'paired_fixed20_step5000_stall_recovery_g1',
     'b1k-qbal5k-1017-c01-g1', 'df17ef54250945c68fa5e3e817ca8054'),
    ('expert_fixed20_step1000_5x4', 'expert_fixed20_step1000_stall_recovery_g1',
     'b1k-expert1k-1017-c01-g1', '5359bb8538924e4aaa864fd9fa706c54'),
)


def main():
    requests = []
    for original, recovery, name, uid in CASES:
        src, dst = ROOT / original, ROOT / recovery
        if (dst / 'recovery_manifest.json').exists():
            raise RuntimeError('Preparation already exists; inspect rather than regenerate signed source')
        old = read(src / 'intents/batch-01.json')
        assert old['uid'] == uid and old['indices'][-1] == 1
        assert not (src / 'reports/candidate-01.json').exists()
        for folder in ('private', 'intents', 'reports', 'observations'):
            (dst / folder).mkdir(parents=True, exist_ok=True)
        shutil.copytree(src / 'frozen', dst / 'frozen')
        shutil.copyfile(src / 'frozen_sha256.json', dst / 'frozen_sha256.json')
        plan = read(src / 'plan.json')
        plan['afs_root'] += '/recovery-stall-g1'
        plan['object_prefix'] += '/recovery-stall-g1'
        save(dst / 'plan.json', plan)
        ctl = module('prepare_' + recovery, src / 'controller.py')
        ctl.ROOT = dst
        ctl.base.ROOT = dst
        client = ctl.aoss_client()
        def signed(suffix):
            return client.generate_presigned_url('put_object', Params={
                'Bucket': '260010165', 'Key': plan['object_prefix'] + suffix}, ExpiresIn=172800)
        bundle = hashlib.sha256((dst / 'frozen_sha256.json').read_bytes()).hexdigest()
        payload = dict(id='candidate-01', mode='evaluate', job_name=name, role='candidate',
                       pair=plan['pairs'][1], plan=plan, worker_bundle_sha256=bundle,
                       heartbeat_put_url=signed('/telemetry/candidate-01/heartbeat.json'),
                       diagnostics_put_url=signed('/telemetry/candidate-01/diagnostics.json'))
        source = ctl.base.bootstrap(plan, payload, signed('/reports/candidate-01.json'))
        compile(source, name, 'exec')
        raw = source.encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = dst / 'private/candidate-01.startup.py'
        path.write_bytes(raw)
        path.chmod(0o600)
        key = plan['object_prefix'] + '/private-startup/candidate-01.py'
        client.put_object(Bucket='260010165', Key=key, Body=raw,
                          ContentType='text/x-python', Metadata={'sha256': digest})
        target = plan['afs_root'] + '/bootstraps/candidate-01.py'
        save(dst / 'recovery_manifest.json', dict(original_folder=original, old_intent='batch-01',
             old_uid=uid, id='candidate-01', index=1, name=name, generation=1, attempt=2,
             source_sha256=digest, source_path=target, bundle_sha256=bundle,
             remedy='Fresh independent worker; identical frozen evaluation and checkpoint'))
        requests.append(dict(destination=target, sha256=digest,
            url=client.generate_presigned_url('get_object', Params={
                'Bucket':'260010165', 'Key':key}, ExpiresIn=7200)))
    save(ROOT / 'private_stall_staging_requests.json', requests)
    print(json.dumps({'prepared': len(requests), 'gpu_jobs_created': 0,
                      'frozen_sources_unchanged': True}))


if __name__ == '__main__':
    main()
