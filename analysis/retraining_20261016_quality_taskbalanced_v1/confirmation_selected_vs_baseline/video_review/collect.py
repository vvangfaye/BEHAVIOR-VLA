"""Copy exactly the user-requested matched videos/results through own private storage."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
REFERENCE = CAMPAIGN.parent / 'heldout20_baseline_vs_expert1k'


def load(path):
    return json.loads(path.read_text())


def main():
    spec = importlib.util.spec_from_file_location('video_copy_transport', REFERENCE / 'controller.py')
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    control.ROOT = ROOT
    control.transport.ROOT = ROOT
    selection = load(ROOT / 'selection.json')
    plan = load(CAMPAIGN / 'plan.json')
    client = control.transport.aoss_client()
    intent_path = ROOT / 'intents/copy.json'
    prefix = selection['destination_prefix']
    if not intent_path.exists():
        files = []
        for row in selection['files']:
            files.append({**row, 'put_url': control.signed(client, prefix + '/media/' + row['relative_path'])})
        config = {'files': files, 'report_url': control.signed(client, prefix + '/receipt.json')}
        source = '''import hashlib,json,time,urllib.request
from pathlib import Path
config=CONFIG
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
completed=[]
for row in config['files']:
 p=Path(row['path'])
 assert p.is_file() and p.stat().st_size==row['bytes']
 raw=p.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==row['sha256']
 for attempt in range(3):
  try:
   with opener.open(urllib.request.Request(row['put_url'],data=raw,method='PUT'),timeout=120):pass
   break
  except OSError:
   if attempt==2:raise
   time.sleep(2**attempt)
 completed.append({k:row[k] for k in ('relative_path','bytes','sha256')})
 del raw
report={'complete':True,'files':completed,'source_sha256_verified':True,'evaluation_modified':False,'log_contents_exported':False}
with opener.open(urllib.request.Request(config['report_url'],data=json.dumps(report).encode(),method='PUT'),timeout=60):pass
print('requested video copy completed',flush=True)
'''.replace('CONFIG', repr(config))
        private = ROOT / 'private/copy_startup.py'
        private.write_text(source)
        private.chmod(0o600)
        command = 'timeout --signal=TERM --kill-after=15 900 python3 -u -c ' + shlex.quote(source)
        assert len(command.encode()) < 100000
        intent = {'id': 'copy', 'name': 'b1k-confirm-video-review-1017', 'uid': None,
                  'gpu': 0, 'image': plan['image'], 'creation_attempted': False,
                  'startup_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'selection_sha256': hashlib.sha256((ROOT / 'selection.json').read_bytes()).hexdigest()}
        control.transport.new(intent_path, intent)
        control.native_create(intent, command, cpu=True)
    intent = load(intent_path)
    assert intent['selection_sha256'] == hashlib.sha256((ROOT / 'selection.json').read_bytes()).hexdigest()
    state = control.observe(intent)
    control.save(ROOT / 'native_observation.json', state)
    try:
        response = client.get_object(Bucket='260010165', Key=prefix + '/receipt.json')
    except Exception as exc:
        if getattr(exc, 'response', {}).get('Error', {}).get('Code') not in ('NoSuchKey', '404'):
            raise
        print(json.dumps({'state': state['state'], 'uid': state['uid'], 'receipt_received': False}))
        return
    try:
        receipt = json.loads(response['Body'].read(1048576))
    finally:
        response['Body'].close()
    expected = [{k:row[k] for k in ('relative_path', 'bytes', 'sha256')} for row in selection['files']]
    assert receipt['complete'] and receipt['files'] == expected and receipt['source_sha256_verified']
    for row in expected:
        path = ROOT / 'media' / row['relative_path']
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            assert path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
            continue
        response = client.get_object(Bucket='260010165', Key=prefix + '/media/' + row['relative_path'])
        tmp = path.with_suffix(path.suffix + '.partial')
        h = hashlib.sha256()
        try:
            with tmp.open('xb') as stream:
                for block in iter(lambda: response['Body'].read(1048576), b''):
                    h.update(block)
                    stream.write(block)
        finally:
            response['Body'].close()
        assert tmp.stat().st_size == row['bytes'] and h.hexdigest() == row['sha256']
        tmp.replace(path)
    control.save(ROOT / 'collection_receipt.json', {**receipt, 'native': state, 'local_sha256_verified': True})
    print(json.dumps({'files_verified_locally': len(expected), 'bytes': sum(r['bytes'] for r in expected),
                      'native_state': state['state']}))


if __name__ == '__main__':
    main()
