"""Publish the verified Linux image once, using the host's existing CCR login."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

B = Path('/home/faye/behavior-video2m-20261013/serving_bundle')
TARGET = 'registry.cn-sh-01.sensecore.cn/ccr-zhicheng-01/behavior-pi05:2000000-video-20261013'
SOURCE = '8fabbeb8d3b1687f80dec7c0f1a164d7be102cf97957005d09fec1710883473b'
CHECKPOINT = 'c4c23d2d479320755ed6da26972907e6050594020fa053a2b2c64df1e321373b'


def save(name, value):
    temporary = B / ('.' + name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(B / name)


def status(state, **values):
    save('ccr_publish_status.json', dict(state=state, observed_epoch=time.time(), **values))


def read(name):
    return json.loads((B / name).read_text())


def verified_image_id(built, content, cpu, linux):
    assert linux.get('verified') is True
    assert cpu.get('exit_code') == 0
    assert built['platform'] == 'linux/amd64'
    assert built['source_manifest_sha256'] == SOURCE
    assert built['checkpoint_ready_sha256'] == CHECKPOINT
    assert content['verified'] is True and content['image_id'] == built['image_id']
    return built['image_id']


def checked_manifest(raw, digest):
    # imagetools may append one display newline to the registry's raw JSON.
    candidates = (raw, raw[:-1]) if raw.endswith(b'\n') else (raw,)
    exact = next((x for x in candidates if 'sha256:' + hashlib.sha256(x).hexdigest() == digest), None)
    if exact is None:
        raise RuntimeError('Registry raw manifest does not match its descriptor digest')
    return json.loads(exact)


def registry_manifest(reference):
    p = subprocess.run(['docker', 'buildx', 'imagetools', 'inspect', reference,
                        '--format', '{{json .Manifest}}'], capture_output=True, timeout=180)
    if p.returncode:
        error = p.stderr.decode(errors='replace').lower()
        if 'not found' in error or 'manifest unknown' in error:
            return None
        raise RuntimeError('Registry inspection failed; target existence unknown')
    descriptor = json.loads(p.stdout)
    raw = subprocess.check_output(['docker', 'buildx', 'imagetools', 'inspect', reference, '--raw'], timeout=180)
    manifest = checked_manifest(raw, descriptor['digest'])
    if 'manifests' in manifest:
        children = [m for m in manifest['manifests']
                    if m.get('platform', {}).get('os') == 'linux'
                    and m.get('platform', {}).get('architecture') == 'amd64']
        assert len(children) == 1, 'Expected exactly one Linux/amd64 image'
        child = children[0]
        ref = TARGET.split(':')[0] + '@' + child['digest']
        raw = subprocess.check_output(['docker', 'buildx', 'imagetools', 'inspect', ref, '--raw'], timeout=180)
        manifest = checked_manifest(raw, child['digest'])
    return {'digest': descriptor['digest'], 'config_digest': manifest['config']['digest']}


def main():
    with (B / 'ccr_publish.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save('ccr_publish_process.json', dict(pid=os.getpid(), target=TARGET, started_epoch=time.time()))
        while not (B / 'LINUX_VERIFIED.json').exists():
            if (B / 'LINUX_ERROR.json').exists():
                raise RuntimeError('Linux build/verification failed; no CCR publication attempted')
            status('waiting_verified_linux_image')
            time.sleep(45)
        built = read('IMAGE_BUILT.json')
        image_id = verified_image_id(built, read('IMAGE_CONTENT_VERIFIED.json'),
                                     read('CPU_IMPORT_CHECK.json'), read('LINUX_VERIFIED.json'))
        actual = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image_id]))[0]
        assert actual['Id'] == image_id and actual['Os'] == 'linux' and actual['Architecture'] == 'amd64'
        existing = registry_manifest(TARGET)
        if existing is None:
            if (B / 'ccr_publish_intent.json').exists():
                raise RuntimeError('Prior push has uncertain outcome; inspect exact tag before any retry')
            save('ccr_publish_intent.json', dict(target=TARGET, image_id=image_id, observed_epoch=time.time()))
            subprocess.run(['docker', 'tag', image_id, TARGET], check=True)
            status('pushing_verified_image', image_id=image_id)
            with (B / 'ccr_docker_push.log').open('ab') as log:
                p = subprocess.Popen(['docker', 'push', TARGET], stdout=log, stderr=subprocess.STDOUT)
                save('ccr_docker_push_process.json', dict(pid=p.pid, target=TARGET, started_epoch=time.time()))
                code = p.wait()
            save('ccr_docker_push_exit.json', dict(exit_code=code, observed_epoch=time.time()))
            # A failed client exit may still have published the image. Reconcile,
            # but never initiate a second push in this process.
            existing = registry_manifest(TARGET)
        if existing is None:
            raise RuntimeError('CCR target still absent after the one allowed push')
        assert existing['config_digest'] == image_id, 'Refuse to overwrite or accept a different image'
        receipt = dict(target=TARGET, image_id=image_id, digest=existing['digest'],
                       immutable_reference=TARGET.split(':')[0] + '@' + existing['digest'],
                       manifest_verified=True, source_manifest_sha256=SOURCE,
                       checkpoint_ready_sha256=CHECKPOINT, build_host='slai-wy-local',
                       namespace_private_preflight=True, observed_epoch=time.time(),
                       gpu_validation_pending=True, organizer_access_pending=True)
        save('CCR_IMAGE_PUBLISHED.json', receipt)
        status('published_gpu_and_organizer_access_pending', **receipt)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        raise SystemExit('An existing CCR publisher holds the lock')
    except Exception as error:
        status('attention_required', error_type=type(error).__name__, message=str(error)[:250])
        raise
