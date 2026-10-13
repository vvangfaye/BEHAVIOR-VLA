"""One-shot, resumable transfer/build and private publication on the approved host.

Registry authentication is sent only through encrypted SSH stdin and kept in
memory for the approved push; no remote Docker login or credential file is written.
"""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1] / 'eval_results/sco_submission1000_2000000_20261013'
B = ROOT / 'serving_bundle'
REMOTE = '/home/faye/behavior-video2m-20261013'
RB = REMOTE + '/serving_bundle'
HOST = 'slai-wy-local'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=20', HOST]


def save(name, value):
    tmp = B / ('.' + name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(B / name)


def state(phase, **kw):
    save('linux_handoff_status.json', dict(state=phase, host=HOST, observed_epoch=time.time(), **kw))


def remote_json(name):
    result = subprocess.run(SSH + ['cat ' + RB + '/' + name], capture_output=True, text=True)
    return json.loads(result.stdout) if result.returncode == 0 else None


def transfer(args, phase):
    for attempt in range(1, 5):
        state(phase, attempt=attempt)
        with (B / 'linux_transfer.log').open('a') as log:
            p = subprocess.Popen(['rsync', '-a', '--partial', '--progress', '-e',
                                  'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=20'] + args,
                                 stdout=log, stderr=subprocess.STDOUT)
            save('linux_transfer_process.json', dict(pid=p.pid, attempt=attempt, started_epoch=time.time()))
            code = p.wait()
        if code == 0:
            return
        time.sleep(15)
    raise RuntimeError('SSH transfer failed after bounded retries; partial file retained')


def main():
    with (B / 'linux_handoff.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save('linux_handoff_process.json', dict(pid=os.getpid(), host=HOST, started_epoch=time.time()))
        subprocess.run(SSH + ['mkdir -p ' + RB + '; chmod 700 ' + REMOTE], check=True)
        transfer([str(B / 'linux_transfer') + '/', HOST + ':' + RB + '/'], 'transferring_helpers')
        # preparation_ready belongs to the campaign root, not the bundle directory.
        subprocess.run(SSH + ['cp ' + RB + '/preparation_ready.json ' + REMOTE + '/preparation_ready.json'], check=True)
        transfer([str(B / 'build-context.tar'), HOST + ':' + RB + '/build-context.tar'], 'transferring_verified_context')
        launch = """import pathlib,fcntl,json,subprocess,time,os
b=pathlib.Path('/home/faye/behavior-video2m-20261013/serving_bundle')
with (b/'launch.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 p=b/'linux_driver_process.json'
 if p.exists():
  d=json.loads(p.read_text())
  try: os.kill(d['pid'],0); print(json.dumps({'existing':True,**d})); raise SystemExit(0)
  except ProcessLookupError: pass
 if (b/'LINUX_VERIFIED.json').exists(): print(json.dumps({'verified':True})); raise SystemExit(0)
 if (b/'LINUX_ERROR.json').exists(): raise RuntimeError('Existing build error requires inspection')
 with (b/'linux_driver.log').open('a') as log:
  p=subprocess.Popen(['python3',str(b/'linux_driver.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  d={'pid':p.pid,'started_epoch':time.time()}; (b/'linux_driver_launch.json').write_text(json.dumps(d)); print(json.dumps(d))
"""
        out = subprocess.check_output(SSH + ['python3 -c ' + __import__('shlex').quote(launch)], text=True)
        save('linux_driver_launch.json', json.loads(out))
        state('building_and_verifying_on_linux')
        while not remote_json('LINUX_VERIFIED.json'):
            error = remote_json('LINUX_ERROR.json')
            if error:
                save('linux_remote_error.json', error)
                raise RuntimeError('Linux build or full content verification failed')
            for n in ['image_build_status.json', 'image_content_verify_status.json']:
                d = remote_json(n)
                if d:
                    save('linux_' + n, d)
            time.sleep(45)
        for n in ['TRANSFER_VERIFIED.json', 'IMAGE_BUILT.json', 'CPU_IMPORT_CHECK.json', 'IMAGE_CONTENT_VERIFIED.json', 'LINUX_VERIFIED.json']:
            d = remote_json(n)
            assert d
            d['build_host'] = HOST
            save(n, d)
        built = json.loads((B / 'IMAGE_BUILT.json').read_text())
        assert json.loads((B / 'CPU_IMPORT_CHECK.json').read_text())['exit_code'] == 0
        verified = json.loads((B / 'IMAGE_CONTENT_VERIFIED.json').read_text())
        assert verified['image_id'] == built['image_id'] and verified['verified']
        auth = json.loads((B / 'upload_authorization.json').read_text())
        assert auth['must_remain_private'] and auth['repository'] == 'docker.io/faye0138/behavior-pi05'
        registry = runpy.run_path(str(B / 'linux_transfer/local_private_registry.py'))
        c = registry['get_credentials']()
        registry['private_check'](c)
        existing = registry['remote_config'](c)
        if existing:
            assert existing['config_digest'] == built['image_id'], 'Refuse to overwrite another image'
        else:
            # An interrupted attempt must be inspected, never blindly repeated.
            assert not (B / 'linux_private_push_intent.json').exists()
            save('linux_private_push_intent.json', dict(image_id=built['image_id'], target=registry['TARGET'], observed_epoch=time.time()))
            state('pushing_verified_linux_image')
            payload = json.dumps(dict(username=c['Username'], password=c['Secret'], image_id=built['image_id'])).encode()
            with (B / 'linux_private_push.log').open('a') as log:
                p = subprocess.Popen(SSH + ['python3 ' + RB + '/remote_push.py'], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
                save('linux_private_push_process.json', dict(pid=p.pid, host=HOST, started_epoch=time.time()))
                p.communicate(payload)
                assert p.returncode == 0, 'Push interrupted or failed; reconcile exact remote tag before retry'
            existing = registry['remote_config'](c)
        registry['private_check'](c)
        assert existing and existing['config_digest'] == built['image_id']
        save('IMAGE_PUBLISHED.json', dict(repository='docker.io/faye0138/behavior-pi05', tag='2000000-video-20261013',
             digest=existing['digest'], immutable_reference='docker.io/faye0138/behavior-pi05@' + existing['digest'],
             image_id=built['image_id'], is_private=True, manifest_verified=True, build_host=HOST,
             source_manifest_sha256=built['source_manifest_sha256'], checkpoint_ready_sha256=built['checkpoint_ready_sha256'],
             gpu_validation_pending=True, organizer_access_pending=True, observed_epoch=time.time()))
        state('private_image_published_gpu_and_delivery_pending')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        state('attention_required', error_type=type(e).__name__, message=str(e)[:250])
        raise
