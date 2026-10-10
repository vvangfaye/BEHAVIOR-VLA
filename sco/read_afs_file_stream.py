"""Read one immutable AFS artifact through an owned live SCO worker in bounded frames."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import subprocess
import time
import zlib


def run(args):
    env = os.environ.copy()
    env.update(SCO_CONFIG='/private/tmp/behavior-sco-config', SCO_HOME='/private/tmp/behavior-sco-runtime', SCO_DATA_HOME='/private/tmp/behavior-sco-data')
    cli = '/private/tmp/behavior-sco-runtime/bin/sco'
    job = json.loads(subprocess.run([cli, 'acp', 'jobs', 'describe', args.name, '--workspace-name=share-space', '-o', 'json'], env=env, capture_output=True, text=True, check=True, timeout=25).stdout)
    assert job['uid'] == args.uid and job['state'] == 'RUNNING' and job['ownership']['user_name'] == '260010165'
    scope = '/mnt/afs/260010165/behavior_1k/eval_results/public-test2000-1688977-20261010/'
    assert args.remote_path.startswith(scope) and '..' not in Path(args.remote_path).parts
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(target) + '.part')
    state_path = args.state_path.resolve() if args.state_path else Path(str(target) + '.transfer.json')
    state_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    offset = 0
    if previous and part.exists():
        assert previous['remote_path'] == args.remote_path
        with part.open('rb') as f:
            for chunk in previous['chunks']:
                assert chunk['offset'] == offset
                raw = f.read(chunk['bytes'])
                assert len(raw) == chunk['bytes'] and hashlib.sha256(raw).hexdigest() == chunk['sha256']
                offset += len(raw)
        with part.open('r+b') as f:
            f.truncate(offset)
    elif part.exists():
        raise RuntimeError('Untracked partial copy; inspect before retry')
    remote = '''import base64,hashlib,json,time,zlib
from pathlib import Path
p=Path(REMOTE).resolve();assert str(p).startswith(SCOPE)
s=p.stat();h=hashlib.sha256()
with p.open('rb') as f:
 while True:
  b=f.read(4*1024*1024)
  if not b:break
  h.update(b)
sha=h.hexdigest();meta=dict(remote_path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha,offset=OFFSET)
print('ARTIFACT_META='+json.dumps(meta),flush=True)
h=hashlib.sha256()
with p.open('rb') as f:
 pos=0
 while True:
  b=f.read(256*1024)
  if not b:break
  h.update(b)
  if pos>=OFFSET:print('ARTIFACT_CHUNK='+json.dumps(dict(offset=pos,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),encoding='zlib+base64',data=base64.b64encode(zlib.compress(b,1)).decode())),flush=True)
  pos+=len(b)
assert p.stat().st_size==s.st_size and p.stat().st_mtime_ns==s.st_mtime_ns and h.hexdigest()==sha
print('ARTIFACT_END='+json.dumps(dict(bytes=pos,sha256=h.hexdigest())),flush=True)
'''.replace('REMOTE', repr(args.remote_path)).replace('SCOPE', repr(scope)).replace('OFFSET', str(offset))
    encoded = base64.b64encode(zlib.compress(remote.encode())).decode()
    wire = ("python3 - <<'BEHAVIOR_STREAM_READ'\nimport base64,zlib\nb=(\n" + ''.join(repr(encoded[i:i+500]) + '\n' for i in range(0, len(encoded), 500)) + ')\nexec(compile(zlib.decompress(base64.b64decode(b)),"immutable_afs_read","exec"))\nBEHAVIOR_STREAM_READ\n').encode()
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    proc = subprocess.Popen([cli, 'acp', 'jobs', 'exec', args.name, '--workspace-name=share-space', '--worker-name=pt-'+args.uid+'-worker-0'], stdin=slave, stdout=slave, stderr=slave, env=env)
    os.close(slave)
    pending = b''
    prompt = b''
    sent = False
    wire_offset = 0
    record = None
    stream = None
    complete = False
    started = time.monotonic()
    last_data = started
    progress_at = offset
    try:
        while time.monotonic()-started < args.timeout:
            readable, writable, _ = select.select([master], [master] if sent and wire_offset<len(wire) else [], [], 0.25)
            if readable:
                try:
                    data = os.read(master, 65536)
                except BlockingIOError:
                    continue
                except OSError:
                    break
                if not data:
                    break
                last_data = time.monotonic()
                pending += data
                if not sent:
                    prompt = (prompt + data)[-8192:]
                    clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', prompt.decode(errors='replace').replace('\r',''))
                    if re.search(r'(?:^|\n)[^\n]*[#$]\s*$', clean):
                        sent = True
            if writable:
                try:
                    wire_offset += os.write(master, wire[wire_offset:wire_offset+512])
                except BlockingIOError:
                    pass
            while b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                line = line.rstrip(b'\r').replace(b'\x01', b'')
                line = re.sub(rb'\x1b\[[0-?]*[ -/]*[@-~]', b'', line)
                for marker in (b'ARTIFACT_META=', b'ARTIFACT_CHUNK=', b'ARTIFACT_END='):
                    position = line.find(marker)
                    if position >= 0:
                        line = line[position:]
                        break
                if line.startswith(b'ARTIFACT_META='):
                    meta = json.loads(line[len(b'ARTIFACT_META='):])
                    assert meta['remote_path'] == args.remote_path and meta['offset'] == offset
                    if previous:
                        assert all(meta[k] == previous[k] for k in ('remote_path','bytes','mtime_ns','sha256'))
                    if target.exists():
                        assert target.stat().st_size == meta['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest() == meta['sha256']
                        complete = True
                        record = dict(meta, state='verified_existing_original')
                        state_path.write_text(json.dumps(record, indent=2)+'\n')
                        break
                    record = dict(meta, state='transferring', chunks=previous['chunks'] if previous else [], observer_uid=args.uid, observed_at=time.time())
                    stream = part.open('ab' if part.exists() else 'xb')
                    state_path.write_text(json.dumps(record, indent=2)+'\n')
                    print(json.dumps(dict(state='streaming',bytes=meta['bytes'],verified_resume_offset=offset)), flush=True)
                elif line.startswith(b'ARTIFACT_CHUNK='):
                    assert record is not None and stream is not None
                    chunk = json.loads(line[len(b'ARTIFACT_CHUNK='):])
                    assert 0 < chunk['bytes'] <= 256*1024
                    payload = base64.b64decode(chunk['data'], validate=True)
                    if chunk.get('encoding', 'base64') == 'zlib+base64':
                        decoder = zlib.decompressobj()
                        raw = decoder.decompress(payload, chunk['bytes']+1)
                        assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
                    else:
                        assert chunk.get('encoding', 'base64') == 'base64'
                        raw = payload
                    assert chunk['offset'] == offset and len(raw) == chunk['bytes'] and hashlib.sha256(raw).hexdigest() == chunk['sha256']
                    stream.write(raw)
                    stream.flush()
                    record['chunks'].append({k:chunk[k] for k in ('offset','bytes','sha256')})
                    offset += len(raw)
                    state_path.write_text(json.dumps(record, indent=2)+'\n')
                    if offset-progress_at >= 8*1024*1024:
                        print(json.dumps(dict(state='streaming',verified_bytes=offset,total_bytes=record['bytes'])),flush=True)
                        progress_at = offset
                elif line.startswith(b'ARTIFACT_END='):
                    end = json.loads(line[len(b'ARTIFACT_END='):])
                    assert record is not None and end['bytes'] == offset == record['bytes'] and end['sha256'] == record['sha256']
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                    stream = None
                    with part.open('rb') as f:
                        digest = hashlib.sha256()
                        for block in iter(lambda:f.read(4*1024*1024),b''):
                            digest.update(block)
                    assert part.stat().st_size == offset and digest.hexdigest() == end['sha256']
                    assert not target.exists()
                    os.link(part, target)
                    part.unlink()
                    record.update(state='original_bytes_verified',completed_at=time.time(),local_path=str(target),original_content_unchanged=True)
                    state_path.write_text(json.dumps(record, indent=2)+'\n')
                    complete = True
                    break
            if complete:
                break
            assert len(pending) < 2*1024*1024, 'Unexpected unbounded transport frame'
            if time.monotonic()-last_data > 120:
                break
        assert complete, 'Artifact observer interrupted; keep verified partial, do not restart model'
        print(json.dumps({k:record[k] for k in ('state','bytes','sha256')}),flush=True)
    finally:
        if stream:
            stream.close()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        os.close(master)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--remote-path', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state-path', type=Path)
    parser.add_argument('--timeout', type=float, default=900)
    run(parser.parse_args())
