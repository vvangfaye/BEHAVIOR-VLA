"""Read-only CPU probe: fetch the persisted logs of a failed validation learner from AFS (no GPU).

Same pattern as analysis/serf_reproduction_20261025/read_failed_logs.py: template b1k-storage-inv-1024,
spec n6ls.iu.i40.2c8g (0 GPU), 180 s limit; the job only reads files and PUTs one JSON to AOSS.
Usage: python probe_failure.py <validation-tag>   e.g. val-learn-5090-full-r1
"""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import deployment as d

tag = sys.argv[1]
row = json.loads((HERE / 'jobs' / tag / 'submission.json').read_text())
job = d.describe(row['job'])
assert job['uid'] == row['uid'] and job['state'] in d.TERMINAL
remote = d.campaign()['remote_root']
series = tag.rsplit('-r', 1)[0] + '-series' if '-r' in tag.split('val-learn-')[-1] else tag
roots = {'job_root': f'{remote}/jobs/{tag}', 'output': f'{remote}/validation/{series}/results'}
key = d.campaign()['aoss_prefix'] + f'validation/probe-{tag}/result.json'
receipt = HERE / 'jobs' / tag / 'probe_submission.json'
assert not receipt.exists(), 'Probe already submitted; read its result instead'
client = d.status_client()
try:
    url = client.generate_presigned_url('put_object', Params={'Bucket': '260010165', 'Key': key}, ExpiresIn=7200)
finally:
    client.close()
source = """import json,urllib.request
from pathlib import Path
report={}
for label,root in ROOTS.items():
 root=Path(root);files={}
 if root.exists():
  for pattern in ['driver.log','resources.jsonl','bootstrap_error.json','*.json','failures/*.json','worker_status/*.json','checks/*.json']:
   for path in sorted(root.glob(pattern)):
    if path.is_file() and str(path) not in files:
     with path.open('rb') as f:
      f.seek(max(0,path.stat().st_size-30000));data=f.read(30000)
     files[str(path.relative_to(root))]={'bytes':path.stat().st_size,'tail':data.decode(errors='replace')}
 report[label]={'root':str(root),'exists':root.exists(),'files':files}
data=json.dumps(report).encode()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(urllib.request.Request(PUT,data=data,method='PUT'),timeout=45):pass
print('probe uploaded',len(data),flush=True)
"""
source = 'ROOTS=' + repr(roots) + '\nPUT=' + repr(url) + '\n' + source
packed = base64.b64encode(zlib.compress(source.encode())).decode()
command = 'timeout --signal=TERM --kill-after=10s 180s python3 -u -c ' + shlex.quote(
    'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
name = f'b1k-mr-1102-probe-{tag[-12:]}'.replace('_', '-')
intent = {'job': name, 'source_job': row['job'], 'source_uid': row['uid'], 'gpus': 0, 'max_seconds': 180,
          'result_key': key, 'roots': roots, 'submitted_at': time.time()}
receipt.write_text(json.dumps(intent, indent=2) + '\n')
result = subprocess.run([d.CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space', '--copy-job-name=b1k-storage-inv-1024',
                         '--name=' + name, '--job-name=' + name, '--worker-spec=n6ls.iu.i40.2c8g', '--worker-nodes=1',
                         '--quota-type=reserved', '--retry-times=0', '--enable-fault-tolerance=false',
                         '--enable-anomaly-detection=false', '--command=' + command],
                        env=d.ENV, capture_output=True, text=True, timeout=90)
intent['returncode'] = result.returncode
intent['uid'] = d.describe(name)['uid']
receipt.write_text(json.dumps(intent, indent=2) + '\n')
print(json.dumps({k: intent[k] for k in ['job', 'uid', 'returncode', 'result_key']}))
