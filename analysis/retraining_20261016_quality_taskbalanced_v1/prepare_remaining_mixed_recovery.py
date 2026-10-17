"""Isolate only the two remaining members of the confirmed stalled g1 batch."""
import hashlib
import json
import shutil
from pathlib import Path

from run_dual_evaluation import module, read, save

ROOT = Path(__file__).resolve().parent


def main():
    original = ROOT / 'paired_fixed20_step5000_5x4'
    old_folder = 'paired_fixed20_step5000_recovery_g1'
    old = read(ROOT / old_folder / 'intents/batch-04.json')
    assert old['uid'] == '5a3bd51080b540709e4507be86699fb5'
    requests = []
    for index in (11, 5):
        folder = 'paired_fixed20_step5000_c%02d_recovery_g2' % index
        dst = ROOT / folder
        if (dst / 'recovery_manifest.json').exists():
            raise RuntimeError('Existing preparation; do not regenerate or resubmit')
        id = 'candidate-%02d' % index
        assert not (original / 'reports' / (id + '.json')).exists()
        for name in ('private', 'reports', 'observations', 'intents'):
            (dst / name).mkdir(parents=True, exist_ok=True)
        shutil.copytree(original / 'frozen', dst / 'frozen')
        shutil.copyfile(original / 'frozen_sha256.json', dst / 'frozen_sha256.json')
        plan = read(original / 'plan.json')
        suffix = '/recovery-stall-g2-c%02d' % index
        plan['afs_root'] += suffix
        plan['object_prefix'] += suffix
        save(dst / 'plan.json', plan)
        ctl = module('prepare_c%02d' % index, original / 'controller.py')
        ctl.ROOT = dst
        ctl.base.ROOT = dst
        client = ctl.aoss_client()
        def signed(suffix):
            return client.generate_presigned_url('put_object', Params={
                'Bucket':'260010165', 'Key':plan['object_prefix']+suffix}, ExpiresIn=172800)
        bundle = hashlib.sha256((dst/'frozen_sha256.json').read_bytes()).hexdigest()
        name = 'b1k-qbal5k-1017-c%02d-g2' % index
        payload = dict(id=id, mode='evaluate', role='candidate', job_name=name,
                       pair=plan['pairs'][index], plan=plan, worker_bundle_sha256=bundle,
                       heartbeat_put_url=signed('/telemetry/'+id+'/heartbeat.json'),
                       diagnostics_put_url=signed('/telemetry/'+id+'/diagnostics.json'))
        source = ctl.base.bootstrap(plan, payload, signed('/reports/'+id+'.json'))
        compile(source, name, 'exec')
        raw = source.encode()
        digest = hashlib.sha256(raw).hexdigest()
        p = dst/'private'/ (id+'.startup.py')
        p.write_bytes(raw)
        p.chmod(0o600)
        key = plan['object_prefix']+'/private-startup/'+id+'.py'
        client.put_object(Bucket='260010165', Key=key, Body=raw,
                          ContentType='text/x-python', Metadata={'sha256':digest})
        target = plan['afs_root']+'/bootstraps/'+id+'.py'
        save(dst/'recovery_manifest.json', dict(original_folder=original.name,
             old_folder=old_folder, old_intent='batch-04', old_uid=old['uid'],
             id=id, index=index, name=name, generation=2, attempt=3,
             prior_completed_rollouts=0, prior_started_rollouts=1 if index==11 else 0,
             source_sha256=digest, source_path=target, bundle_sha256=bundle,
             remedy='Fresh independent worker; identical frozen evaluation and checkpoint'))
        requests.append(dict(destination=target,sha256=digest,
            url=client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=7200)))
    save(ROOT/'private_remaining_mixed_staging_requests.json',requests)
    print(json.dumps({'prepared_ids':['candidate-11','candidate-05'],'gpu_jobs_created':0}))


if __name__ == '__main__':
    main()
