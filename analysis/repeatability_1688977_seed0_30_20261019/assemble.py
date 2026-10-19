"""Pure construction used locally and by the CPU staging job."""
import base64
import hashlib
import json
import zlib


def pack(value):
    return base64.b64encode(zlib.compress(json.dumps(value).encode(), 9)).decode()


def assemble(data):
    plan, files, urls = data['plan'], data['files'], data['urls']
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    outputs = {}
    for group in plan['groups']:
        members = []
        for item in group['members']:
            identity = item['role'] + '-' + f"{item['index']:02d}"
            payload = {'id': identity, 'mode': 'evaluate', 'job_name': group['name'] + '-' + identity,
                       'role': item['role'], 'repeat': item['repeat'], 'pair': plan['pairs'][item['index']], 'plan': plan,
                       'worker_bundle_sha256': data['bundle_hash'],
                       'heartbeat_put_url': urls[identity]['heartbeat'],
                       'diagnostics_put_url': urls[identity]['diagnostics']}
            destination = plan['afs_root'] + '/workers/' + identity
            child = '''import base64,hashlib,json,os,sys,time,traceback,urllib.request,zlib
from pathlib import Path
root=Path(DEST);root.mkdir(parents=True,exist_ok=False)
fd=os.open(str(root/'worker.log'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
files=json.loads(zlib.decompress(base64.b64decode(FILES)))
for name,text in files.items():
 assert hashlib.sha256(text.encode()).hexdigest()==HASHES[name]
 (root/name).write_text(text)
payload=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
sys.path.insert(0,str(root))
try:
 import worker
 result=worker.main(payload)
except BaseException as exc:
 result={'state':'failed','error_type':type(exc).__name__}
 traceback.print_exc()
result.update(id=payload['id'],protocol_sha256=payload['plan']['protocol_sha256'],worker_bundle_sha256=payload['worker_bundle_sha256'],finished_at=time.time())
(root/'report.json').write_text(json.dumps(result,indent=2)+'\\n')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for attempt in range(4):
 try:
  with opener.open(urllib.request.Request(REPORT,data=json.dumps(result).encode(),method='PUT'),timeout=40):pass
  break
 except OSError:
  if attempt==3:raise
  time.sleep(2**attempt)
if result['state']!='accepted':raise SystemExit(1)
'''
            import re
            replacements = {'DEST': destination, 'FILES': pack(files), 'HASHES': hashes,
                            'PAYLOAD': pack(payload), 'REPORT': urls[identity]['report']}
            child = re.sub(r'\b(DEST|FILES|HASHES|PAYLOAD|REPORT)\b', lambda m: repr(replacements[m[0]]), child)
            task, instance = payload['pair']
            config = {'id': identity, 'original_uid': None,
                      'checkpoint_step': 1688977,
                      'python': plan['runtime'] + '/envs/b1k-eval/bin/python',
                      'worker_root': destination, 'episode_root': plan['afs_root'] + '/episodes/' + identity,
                      'monitor_root': plan['afs_root'] + '/supervision/' + identity,
                      'result_name': f'{task}_{instance}_0',
                      'child_source': base64.b64encode(zlib.compress(child.encode(), 9)).decode(),
                      'child_sha256': hashlib.sha256(child.encode()).hexdigest(),
                      'heartbeat_put_url': urls[identity]['heartbeat'],
                      'diagnostics_put_url': urls[identity]['diagnostics']}
            source = files['supervisor.py'] + '\nraise SystemExit(supervisor(' + repr(config) + '))\n'
            members.append({'id': identity, 'source': base64.b64encode(zlib.compress(source.encode(), 9)).decode(),
                            'sha256': hashlib.sha256(source.encode()).hexdigest(),
                            'report_url': urls[identity]['report'],
                            'report_path': destination + '/report.json'})
        config = {'id': group['id'], 'members': members, 'root': plan['afs_root'] + '/batches/' + group['id'],
                  'python': plan['runtime'] + '/envs/b1k-eval/bin/python',
                  'member_walltime_seconds': plan['member_walltime_seconds'],
                  'protocol_sha256': plan['protocol_sha256'], 'worker_bundle_sha256': data['bundle_hash'],
                  'report_url': urls[group['id']]['report']}
        outputs[group['id']] = files['batch_runner.py'] + '\nraise SystemExit(run(' + repr(config) + '))\n'
    return outputs
