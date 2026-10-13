"""Upload the existing verified image to its authorized CCR tag without daemon proxy.

No rebuild, no global proxy changes, and no credentials in arguments or logs.
Opaque registry upload locations are retained only in a mode-0600 private state.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

B = Path('/home/faye/behavior-video2m-20261013/serving_bundle')
REGISTRY = 'registry.cn-sh-01.sensecore.cn'
REPO = 'ccr-zhicheng-01/behavior-pi05'
TAG = '2000000-video-20261013'
BASE = 'https://' + REGISTRY
TARGET = REGISTRY + '/' + REPO + ':' + TAG
CHUNK = 32 * 1024 * 1024
CTR = ['sudo', '-n', 'ctr', '--address', '/run/containerd/containerd.sock', '--namespace', 'moby', 'content', 'get']


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def save(name, value, private=False):
    p = B / name
    temporary = p.with_name('.' + p.name + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    temporary.replace(p)
    if private:
        p.chmod(0o600)


def read(name):
    return json.loads((B / name).read_text())


class Client:
    def __init__(self):
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.token = None
        self.token_time = 0

    def authenticate(self):
        auth = json.loads((Path.home() / '.docker/config.json').read_text())['auths'][REGISTRY]['auth']
        url = BASE + '/service/token?' + urllib.parse.urlencode({'service': 'harbor-registry', 'scope': 'repository:' + REPO + ':pull,push'})
        with self.opener.open(urllib.request.Request(url, headers={'Authorization': 'Basic ' + auth}), timeout=60) as r:
            self.token = json.load(r)['token']
        self.token_time = time.monotonic()

    def request(self, method, url, data=None, headers=None, authenticate=True):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname != REGISTRY:
            raise ValueError('Refuse registry credentials to an unexpected origin')
        if authenticate and (not self.token or time.monotonic() - self.token_time > 300):
            self.authenticate()
        h = dict(headers or {})
        if authenticate:
            h['Authorization'] = 'Bearer ' + self.token
        try:
            r = self.opener.open(urllib.request.Request(url, data=data, headers=h, method=method), timeout=240)
        except urllib.error.HTTPError as e:
            r = e
        with r:
            return r.code, dict(r.headers), r.read()

    def exists(self, digest):
        code, _, _ = self.request('HEAD', BASE + '/v2/' + REPO + '/blobs/' + digest)
        if code in (200, 307):
            return True
        if code == 404:
            return False
        raise RuntimeError('Blob presence query returned HTTP ' + str(code))

    def location(self, h, fallback=None):
        url = urllib.parse.urljoin(BASE, h.get('Location', fallback or ''))
        if urllib.parse.urlparse(url).hostname != REGISTRY:
            raise RuntimeError('Unexpected upload origin')
        return url


def progress(**values):
    save('ccr_direct_status.json', dict(observed_epoch=time.time(), target=TARGET, **values))


def upload(client, descriptor, done_bytes, total_bytes):
    digest, size = descriptor['digest'], descriptor['size']
    if client.exists(digest):
        return
    name = 'ccr_direct_session_' + digest.split(':')[1] + '.json'
    p = B / name
    if p.exists():
        state = read(name)
        assert state['digest'] == digest and state['size'] == size
        code, h, _ = client.request('GET', state['url'])
        if code != 204:
            raise RuntimeError('Saved upload session unavailable; reconcile before restarting (HTTP %s)' % code)
        state['url'] = client.location(h, state['url'])
        offset = int(h['Range'].split('-')[-1]) + 1
        if state['offset'] == 0 and h['Range'] == '0-0':
            offset = 0
        assert 0 <= offset <= size
        state['offset'] = offset
    else:
        code, h, _ = client.request('POST', BASE + '/v2/' + REPO + '/blobs/uploads/', b'')
        if code != 202:
            raise RuntimeError('Start upload returned HTTP ' + str(code))
        state = dict(digest=digest, size=size, offset=0, url=client.location(h), created_epoch=time.time())
    save(name, state, private=True)
    stream = subprocess.Popen(CTR + [digest], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    hasher = hashlib.sha256()
    position = 0
    try:
        while position < size:
            data = stream.stdout.read(min(CHUNK, size - position))
            if not data:
                raise RuntimeError('Local image blob stream ended early')
            hasher.update(data)
            next_position = position + len(data)
            if next_position <= state['offset']:
                position = next_position
                continue
            if position != state['offset']:
                data = data[state['offset'] - position:]
                position = state['offset']
            start = position
            for attempt in range(3):
                try:
                    code, h, _ = client.request('PATCH', state['url'], data, {'Content-Type': 'application/octet-stream', 'Content-Range': '%d-%d' % (start, next_position - 1)})
                    if code == 202:
                        if int(h['Range'].split('-')[-1]) + 1 != next_position:
                            raise RuntimeError('Server acknowledged unexpected offset')
                        state.update(url=client.location(h, state['url']), offset=next_position)
                        break
                    if code not in (408, 416, 429, 500, 502, 503, 504):
                        raise RuntimeError('Chunk upload returned HTTP ' + str(code))
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                # Ambiguous writes are reconciled against the same session,
                # never repeated blindly or uploaded into a second session.
                code, h, _ = client.request('GET', state['url'])
                if code != 204:
                    raise RuntimeError('Unable to reconcile upload offset: HTTP ' + str(code))
                observed = int(h['Range'].split('-')[-1]) + 1
                state['url'] = client.location(h, state['url'])
                if observed == next_position:
                    state['offset'] = observed
                    break
                if observed != start:
                    raise RuntimeError('Upload offset ambiguous; preserve session for inspection')
                if attempt == 2:
                    raise RuntimeError('Chunk failed after three reconciled attempts')
                time.sleep(2)
            save(name, state, private=True)
            position = next_position
            progress(state='uploading', blob=digest, blob_uploaded_bytes=position, blob_total_bytes=size, completed_or_cached_bytes=done_bytes, uploaded_or_cached_bytes=done_bytes + position, total_blob_bytes=total_bytes, transport='direct_https_chunked', chunk_bytes=CHUNK)
        if stream.wait(timeout=20) != 0 or hasher.hexdigest() != digest.split(':')[1]:
            raise RuntimeError('Local compressed image blob checksum failed')
    finally:
        if stream.poll() is None:
            stream.terminate()
            stream.wait(timeout=10)
    final_url = state['url'] + ('&' if '?' in state['url'] else '?') + urllib.parse.urlencode({'digest': digest})
    code, h, _ = client.request('PUT', final_url, b'', {'Content-Type': 'application/octet-stream'})
    if code != 201:
        if not client.exists(digest):
            raise RuntimeError('Blob commit incomplete: HTTP ' + str(code))
    elif h.get('Docker-Content-Digest') != digest:
        raise RuntimeError('Registry returned different blob digest')
    assert client.exists(digest)
    state.update(complete=True, offset=size, verified_epoch=time.time())
    save(name, state, private=True)


def main():
    with (B / 'ccr_publish.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save('ccr_direct_process.json', dict(pid=os.getpid(), started_epoch=time.time()))
        built, content = read('IMAGE_BUILT.json'), read('IMAGE_CONTENT_VERIFIED.json')
        assert read('LINUX_VERIFIED.json')['verified'] and read('CPU_IMPORT_CHECK.json')['exit_code'] == 0
        assert content['verified'] and content['image_id'] == built['image_id']
        raw = subprocess.check_output(CTR + [built['image_id']])
        assert 'sha256:' + hashlib.sha256(raw).hexdigest() == built['image_id']
        manifest = json.loads(raw)
        client = Client()
        tag_url = BASE + '/v2/' + REPO + '/manifests/' + TAG
        code, _, existing = client.request('GET', tag_url, headers={'Accept': manifest['mediaType']})
        if code == 200:
            if existing != raw:
                raise RuntimeError('Refuse overwriting different remote manifest')
        elif code == 404:
            descriptors = [manifest['config']] + manifest['layers']
            total = sum(d['size'] for d in descriptors)
            done = 0
            for descriptor in descriptors:
                upload(client, descriptor, done, total)
                done += descriptor['size']
            # Recheck immediately before publishing the exact original manifest.
            code, _, existing = client.request('GET', tag_url, headers={'Accept': manifest['mediaType']})
            if code == 200 and existing != raw:
                raise RuntimeError('Remote tag changed during upload')
            if code not in (200, 404):
                raise RuntimeError('Unable to reconcile tag before publication')
            if code == 404:
                code, h, _ = client.request('PUT', tag_url, raw, {'Content-Type': manifest['mediaType']})
                if code != 201 or h.get('Docker-Content-Digest') != built['image_id']:
                    raise RuntimeError('Manifest publication not acknowledged')
        else:
            raise RuntimeError('Tag inspection returned HTTP ' + str(code))
        code, h, remote = client.request('GET', tag_url, headers={'Accept': manifest['mediaType']})
        assert code == 200 and remote == raw and h['Docker-Content-Digest'] == built['image_id']
        receipt = dict(target=TARGET, image_id=built['image_id'], digest=built['image_id'], config_digest=manifest['config']['digest'], immutable_reference=REGISTRY + '/' + REPO + '@' + built['image_id'], manifest_verified=True, local_manifest_verified=True, source_manifest_sha256=built['source_manifest_sha256'], checkpoint_ready_sha256=built['checkpoint_ready_sha256'], build_host='slai-wy-local', namespace_private_preflight=True, transport='direct_https_chunked', gpu_validation_pending=True, organizer_access_pending=True, observed_epoch=time.time())
        save('CCR_IMAGE_PUBLISHED.json', receipt)
        save('ccr_publish_status.json', dict(state='published_gpu_and_organizer_access_pending', **receipt))
        progress(state='published', digest=built['image_id'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Exception strings may contain opaque signed upload URLs. Do not log.
        progress(state='attention_required', error_type=type(error).__name__)
        raise SystemExit(type(error).__name__)
