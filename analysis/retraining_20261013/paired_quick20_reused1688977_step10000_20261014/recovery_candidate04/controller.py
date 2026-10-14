"""Single bounded generation-1 recovery; original training is never accessed."""
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
spec = importlib.util.spec_from_file_location('original_controller', CAMPAIGN/'controller.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
ID = 'candidate-04-g1'
NAME = 'b1k-q20k10-1014-' + ID
ORIGINAL_UID = '96a9f2f9f1c5470d850f886c37c1c7c4'


def immutable_identity():
    files = [CAMPAIGN/'plan.json', CAMPAIGN/'frozen_sha256.json', CAMPAIGN/'baseline_accepted.json',
             CAMPAIGN/'step5000_candidate_accepted.json', CAMPAIGN/'step5000_comparison.json']
    files += sorted((CAMPAIGN/'reports').glob('candidate-*.json'))
    files += sorted((CAMPAIGN/'frozen').glob('*.py'))
    return {str(p.relative_to(CAMPAIGN)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def verify_recovery_report(report, plan, intent):
    if (report['id'] != ID or report['protocol_sha256'] != plan['protocol_sha256']
            or report['worker_bundle_sha256'] != intent['worker_bundle_sha256']):
        raise ValueError('Recovery report/source identity mismatch')
    if report['state'] != 'accepted':
        return False
    record = report['record']
    export = c.load(CAMPAIGN/'reports/export.json')
    if (report['checkpoint_manifest_sha256'] != export['manifest_sha256']
            or [record['task'], record['instance_id']] != plan['pairs'][4]
            or record['policy_seed'] != 0 or record['environment_seed'] != 0
            or not record['artifact_accepted'] or not record['policy_seed_acknowledged']):
        raise ValueError('Recovery checkpoint/episode/seed mismatch')
    return True


def read_object(client, key):
    from botocore.exceptions import ClientError
    try:
        response = client.get_object(Bucket='260010165', Key=key)
        try:
            return json.loads(response['Body'].read(8*1024**2))
        finally:
            response['Body'].close()
    except ClientError as error:
        if error.response['Error']['Code'] in ('404', 'NoSuchKey'):
            return None
        raise


def submit_once(client, plan):
    intent_path = ROOT/'intent.json'
    if intent_path.exists():
        intent = c.load(intent_path)
        if not intent['creation_attempted']:
            raise RuntimeError('Preserved intent lacks proven creation state; inspect before proceeding')
        obs = c.observe(intent)
        if not intent.get('uid'):
            intent.update(uid=obs['uid'], state=obs['state'], verified_at=obs['observed_at'])
            c.save(intent_path, intent)
        return intent
    if (CAMPAIGN/'STOP').exists() or (ROOT/'STOP').exists() or time.time() >= plan['deadline_epoch']:
        raise RuntimeError('Admission stopped or deadline reached')
    original = c.load(CAMPAIGN/'intents/candidate-04.json')
    obs = c.observe(original)
    workers = subprocess.run([c.CLI, 'acp', 'jobs', 'get-workers', original['name'],
                              '--workspace-name=share-space'], env=c.environment(),
                             capture_output=True, text=True, check=True, timeout=30).stdout
    if original['uid'] != ORIGINAL_UID or obs['state'] != 'FAILED' or 'Deleted' not in workers:
        raise RuntimeError('Old writer not proven stopped')
    c.save(ROOT/'old_writer_terminal_before_recovery.json', dict(observation=obs, workers=workers))
    diagnostic = c.load(ROOT/'diagnostic_report.json')
    raw_result = diagnostic['files']['episodes/candidate-04/results/json/putting_up_Christmas_decorations_inside_301_0.json']
    if raw_result['exists']:
        raise RuntimeError('Existing result needs artifact recovery, not a new episode')
    if c.read_report(client, plan, 'candidate-04') is not None:
        raise RuntimeError('Original cloud report arrived; review before recovery')
    template = c.cli_json(['acp', 'jobs', 'describe', plan['gpu_copy_template'],
                          '--workspace-name=share-space', '-o', 'json'])
    c.verify_job(template, dict(name=plan['gpu_copy_template'], uid=plan['gpu_copy_template_uid'],
                                gpu=1, image=plan['image']))
    if template['state'] != 'SUCCEEDED':
        raise RuntimeError('Proven own template changed')
    prefix = plan['object_prefix']+'/recovery-candidate04/'
    keys = dict(report=prefix+'candidate-04-g1.json', heartbeat=prefix+'heartbeat-g1.json',
                diagnostics=prefix+'exit-diagnostics-g1.json')
    urls = {k: client.generate_presigned_url('put_object', Params={'Bucket':'260010165','Key':v},
                                            ExpiresIn=172800) for k,v in keys.items()}
    bundle_sha = c.sha_bytes((CAMPAIGN/'frozen_sha256.json').read_bytes())
    payload = dict(id=ID, mode='evaluate', job_name=NAME, plan=plan, role='candidate', pair=plan['pairs'][4],
                   worker_bundle_sha256=bundle_sha, recovery_generation=1, attempt=2, original_uid=ORIGINAL_UID)
    child = c.bootstrap(plan, payload, urls['report'])
    config = dict(id=ID, original_uid=ORIGINAL_UID, python=plan['runtime']+'/envs/b1k-eval/bin/python',
                  worker_root=plan['afs_root']+'/workers/'+ID,
                  episode_root=plan['afs_root']+'/episodes/'+ID,
                  monitor_root=plan['afs_root']+'/recovery/candidate-04-g1',
                  child_source=base64.b64encode(zlib.compress(child.encode(),9)).decode(),
                  child_sha256=c.sha_bytes(child.encode()), heartbeat_put_url=urls['heartbeat'],
                  diagnostics_put_url=urls['diagnostics'])
    startup = (ROOT/'supervisor.py').read_text() + '\nraise SystemExit(supervisor(' + repr(config) + '))\n'
    (ROOT/'private').mkdir(exist_ok=True)
    private = ROOT/'private/recovery.startup.py'
    private.write_text(startup); os.chmod(private, 0o600)
    intent = dict(id=ID, name=NAME, mode='evaluate', gpu=1, image=plan['image'], created_intent_at=time.time(),
                  startup_sha256=c.sha_bytes(startup.encode()), creation_attempted=False, uid=None,
                  attempts=2, generation=1, original_uid=ORIGINAL_UID, additional_attempts_authorized=1,
                  worker_bundle_sha256=bundle_sha, protocol_sha256=plan['protocol_sha256'],
                  report_key=keys['report'], heartbeat_key=keys['heartbeat'], diagnostics_key=keys['diagnostics'],
                  maximum_runtime_seconds=28800, training_modified=False)
    c.new(intent_path, intent)
    command = 'timeout --signal=TERM --kill-after=30 28800 '+shlex.quote(config['python'])+' -c '+shlex.quote(startup)
    args = [c.CLI,'acp','jobs','copy','--copy-job-name='+plan['gpu_copy_template'],
            '--workspace-name=share-space','--name='+NAME,'--job-name='+NAME,
            '--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type=spot','--retry-times=0',
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
            '--container-image-url='+plan['image'],'--env=NVIDIA_DRIVER_CAPABILITIES:all','--command='+command]
    intent.update(creation_attempted=True, creation_started_at=time.time(), state='creation_unknown')
    c.save(intent_path, intent)
    try:
        cp = subprocess.run(args, env=c.environment(), capture_output=True, text=True, timeout=90)
        intent['cli_exit_code'] = cp.returncode
    except subprocess.TimeoutExpired:
        intent['cli_outcome'] = 'timeout_unknown'
    c.save(intent_path, intent)
    obs = c.observe(intent)
    intent.update(uid=obs['uid'],state=obs['state'],verified_at=obs['observed_at'])
    c.save(intent_path, intent)
    return intent


def finalize(plan, report, intent):
    rows = {}
    for i in range(20):
        id = f'candidate-{i:02d}'
        if i == 4:
            selected = report
        else:
            selected = c.load(CAMPAIGN/'reports'/f'{id}.json')
        if selected['state'] != 'accepted':
            raise ValueError('Cannot complete comparison with unaccepted artifacts')
        rows[id] = dict(report=selected)
    c.finalize(plan, rows)
    comparison = c.load(CAMPAIGN/'comparison.json')
    comparison.update(new_candidate_rollouts=21, accepted_candidate_episodes=20,
                      incomplete_original_attempts=1, additional_recovery_attempts=1,
                      recovery=dict(task=plan['pairs'][4][0],instance_id=301,original_uid=ORIGINAL_UID,
                                    recovery_uid=intent['uid'],generation=1,attempts=2,
                                    report_path=str(ROOT/'report.json'),report_sha256=c.sha_bytes((ROOT/'report.json').read_bytes())),
                      training_modified=False,training_restarted=False)
    c.save(CAMPAIGN/'comparison.json', comparison)
    c.save(CAMPAIGN/'status.json', dict(observed_at=time.time(),state='comparison_complete_after_single_recovery',
               accepted_episodes=20,created_gpu_jobs=21,pending_gpu_jobs=0,active_gpu_jobs=0,running_gpu_jobs=0,
               incomplete_original_attempts=1,reused_baseline_episodes=20,training_iid_unchanged=234354,
               training_target_unchanged=20000,recovery_uid=intent['uid']))
    handoff=c.load(CAMPAIGN/'HANDOFF.json')
    handoff.update(state='comparison_complete_after_single_recovery',accepted_candidate_episodes=20,
                   remaining_candidate_episodes=0,recovery_uid=intent['uid'],latest_actual_check=time.time(),
                   controller_active=False,training_modified_or_restarted=False)
    c.save(CAMPAIGN/'HANDOFF.json',handoff)


def main():
    with (CAMPAIGN/'.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        auth = c.load(ROOT/'authorization.json')
        if auth['additional_evaluation_attempts_max'] != 1 or auth['training_modify_stop_or_restart']:
            raise ValueError('Recovery authorization differs')
        plan=c.load(CAMPAIGN/'plan.json')
        identity=immutable_identity()
        previous=ROOT/'immutable_original_sha256.json'
        if previous.exists() and c.load(previous) != identity:
            raise ValueError('Original results/source changed')
        c.save(previous,identity)
        c.save(ROOT/'launch.json',dict(pid=os.getpid(),started_at=time.time(),command=str(Path(__file__).resolve()),
                                      maximum_additional_jobs=1,recurring_automation=False,training_modified=False))
        client=c.aoss_client()
        intent=submit_once(client,plan)
        missing_since=None
        while True:
            if immutable_identity()!=identity:
                raise ValueError('Original frozen evaluator or19 accepted reports changed')
            if time.time()>intent['creation_started_at']+28800+600:
                c.save(ROOT/'status.json',dict(observed_at=time.time(),uid=intent['uid'],
                       state='observer_runtime_deadline_reached_no_further_creation',training_modified=False))
                return
            try:
                report=read_object(client,intent['report_key'])
                heartbeat=read_object(client,intent['heartbeat_key'])
                obs=c.observe(intent)
            except (ValueError, KeyError, TypeError):
                raise
            except Exception as error:
                # An observation error never authorizes a second creation.
                c.save(ROOT/'last_observation_error.json',dict(observed_at=time.time(),
                       error_type=type(error).__name__,uid=intent['uid'],new_creation=False,training_modified=False))
                time.sleep(45)
                continue
            c.save(ROOT/'observation.json',obs)
            if heartbeat:
                c.save(ROOT/'heartbeat.json',heartbeat)
            status=dict(observed_at=time.time(),state=obs['state'],uid=intent['uid'],
                        original_uid=ORIGINAL_UID,generation=1,attempts=2,accepted_episodes=19,
                        report_present=report is not None,training_iid=234354,training_target=20000,
                        training_modified=False,training_restarted=False)
            c.save(ROOT/'status.json',status)
            if report:
                c.save(ROOT/'report.json',report)
                if verify_recovery_report(report,plan,intent):
                    finalize(plan,report,intent)
                    status.update(state='accepted_and_comparison_complete',accepted_episodes=20)
                else:
                    status.update(state='recovery_failed_no_further_retry')
                c.save(ROOT/'status.json',status)
                return
            if obs['state'] in c.TERMINAL:
                if missing_since is None:
                    missing_since=time.time()
                if time.time()-missing_since>=180:
                    diagnostics=read_object(client,intent['diagnostics_key'])
                    if diagnostics:
                        c.save(ROOT/'exit_diagnostics.json',diagnostics)
                    status.update(state='terminal_without_report_no_further_retry')
                    c.save(ROOT/'status.json',status)
                    return
            if (CAMPAIGN/'STOP').exists() or (ROOT/'STOP').exists():
                status.update(state='observer_stopped_by_human')
                c.save(ROOT/'status.json',status)
                return
            if time.time()>intent['creation_started_at']+28800+600:
                status.update(state='observer_runtime_deadline_reached_no_further_creation')
                c.save(ROOT/'status.json',status)
                return
            time.sleep(45)


if __name__ == '__main__':
    main()
