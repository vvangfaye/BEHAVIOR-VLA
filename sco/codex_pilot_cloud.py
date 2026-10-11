"""Finite, uniquely receipted ACP jobs for the isolated Codex collection pilot."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zlib

BUCKET = '260010165'
PREFIX = 'behavior1k/codex-pilot-20261011/'
CLI = '/private/tmp/behavior-sco-runtime/bin/sco'
AFS = '/mnt/afs/260010165/behavior_1k/codex_pilot_20261011'


def environment():
    return dict(os.environ, SCO_CONFIG='/private/tmp/behavior-sco-config',
                SCO_HOME='/private/tmp/behavior-sco-runtime', SCO_DATA_HOME='/private/tmp/behavior-sco-data')


def describe(name):
    p = subprocess.run([CLI, 'acp', 'jobs', 'describe', name, '--workspace-name=share-space', '-o', 'json'],
                       env=environment(), capture_output=True, text=True, timeout=30, check=True)
    d = json.loads(p.stdout)
    assert d['ownership']['user_name'] == BUCKET
    return d


def storage():
    sys.path.insert(0, '/private/tmp')
    from behavior_aoss_client import client
    return client()


def submit(source_path, receipt_path, *, label, gpu, seconds, image=None, quota='spot', python_binary='python3'):
    assert gpu in (0, 1) and 1 <= seconds <= 14400
    assert quota in ('spot', 'reserved')
    assert python_binary == 'python3' or Path(python_binary).is_absolute()
    assert label.replace('-', '').isalnum() and len(label) < 45
    source = Path(source_path).read_bytes()
    compile(source, str(source_path), 'exec')
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    name = 'behavior-codex-' + label + '-' + datetime.now(timezone.utc).strftime('%m%d-%H%M%S')
    key = PREFIX + 'jobs/' + name + '/driver.py'
    report_key = PREFIX + 'jobs/' + name + '/report.json'
    record = dict(name=name, state='intent', gpu_count=gpu, max_seconds=seconds, quota_type=quota,
                  source_sha256=hashlib.sha256(source).hexdigest(), source_key=key, report_key=report_key,
                  source_transport='verified_inline_zlib_base64', python_binary=python_binary)
    if image:
        assert image.startswith('registry.cn-sh-01') and 'sensecore.cn/' in image and ':' in image
        record['container_image'] = image
    with receipt_path.open('x') as f:
        json.dump(record, f, indent=2)
    s = storage()
    s.put_object(Bucket=BUCKET, Key=key, Body=source)
    assert hashlib.sha256(s.get_object(Bucket=BUCKET, Key=key)['Body'].read()).hexdigest() == record['source_sha256']
    report_put = s.generate_presigned_url('put_object', Params=dict(Bucket=BUCKET, Key=report_key), ExpiresIn=86400)
    startup = '''import urllib.request,urllib.error,hashlib,os,json,base64,zlib,re,traceback,socket,time
from pathlib import Path
for k in list(os.environ):
 if k.upper() in {'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY'}:os.environ.pop(k)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def upload_report(value):
 raw=json.dumps(value).encode()
 for attempt in range(4):
  try:
   with opener.open(urllib.request.Request(REPORT_URL,data=raw,method='PUT'),timeout=30):pass
   return
  except urllib.error.URLError as error:
   if not isinstance(error.reason,socket.gaierror) or error.reason.errno!=socket.EAI_AGAIN or attempt==3:raise
   time.sleep(0.5)
stage='bootstrap'
marker=Path(MARKER)
try:
 marker.parent.mkdir(parents=True,exist_ok=True)
 marker.write_text(json.dumps({'state':stage}))
 stage='source_decode'
 b=zlib.decompress(base64.b64decode(SOURCE_INLINE,validate=True))
 assert hashlib.sha256(b).hexdigest()==SOURCE_HASH
 stage='driver_execution'
 marker.write_text(json.dumps({'state':stage}))
 os.environ['CODEX_PILOT_REPORT_PUT']=REPORT_URL
 namespace={'__name__':'__main__','__file__':'codex_pilot_job'}
 exec(compile(b,'codex_pilot_job','exec'),namespace)
 if 'BOOTSTRAP_REPORT' in namespace:
  stage='driver_report_upload'
  try:
   upload_report(namespace['BOOTSTRAP_REPORT'])
  except Exception as report_error:
   transport={'state':'driver_completed_report_upload_failed','error_type':type(report_error).__name__}
   value=namespace['BOOTSTRAP_REPORT']
   if isinstance(value,dict):transport['driver_state']=value.get('state')
   marker.write_text(json.dumps(transport))
   print(json.dumps(transport),flush=True)
except BaseException as e:
 failure={'state':'bootstrap_failed','stage':stage,'error_type':type(e).__name__}
 detail=re.sub(r"(?:https?|oss)://[^\\s<>\\\"']+",'<URL redacted>',str(e),flags=re.I)
 failure['error_detail']=detail[-2000:]
 reason=getattr(e,'reason',None)
 failure['reason_type']=type(reason).__name__
 if isinstance(getattr(reason,'errno',None),int):failure['reason_errno']=reason.errno
 frames=traceback.extract_tb(e.__traceback__)
 if frames:failure['last_frame']={'file':Path(frames[-1].filename).name,'line':frames[-1].lineno}
 marker.write_text(json.dumps(failure))
 try:
  upload_report(failure)
 except Exception:pass
 print(json.dumps(failure),flush=True)
 raise SystemExit(1)
'''
    import re
    encoded_source = base64.b64encode(zlib.compress(source, 9)).decode('ascii')
    assert len(encoded_source) < 1024 * 1024, 'Inline source exceeds the bounded startup size'
    values = dict(MARKER=AFS+'/bootstrap/'+name+'.json', SOURCE_INLINE=encoded_source,
                  SOURCE_HASH=record['source_sha256'], REPORT_URL=report_put)
    startup=re.sub(r'\b(MARKER|SOURCE_INLINE|SOURCE_HASH|REPORT_URL)\b', lambda m:repr(values[m.group()]), startup)
    command = ('timeout --signal=TERM --kill-after=30 ' + str(seconds) + ' '
               + shlex.quote(python_binary) + ' -c ' + shlex.quote(startup))
    common = ['--workspace-name=share-space', '--name='+name, '--job-name='+name,
              '--worker-nodes=1', '--quota-type='+quota,
              '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs', '--command='+command]
    if gpu:
        template = describe('pt-y91th8k9')
        assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
        args = [CLI, 'acp', 'jobs', 'copy', '--copy-job-name=pt-y91th8k9',
                '--worker-spec=n12lp.nn.i10a.1', '--env=NVIDIA_DRIVER_CAPABILITIES:all'] + common
        if image:
            args.append('--container-image-url='+image)
    else:
        args = [CLI, 'acp', 'jobs', 'create', '--aec2-name=share-cluster', '--training-framework=pytorch',
                '--container-image-url='+(image or 'registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630'),
                '--worker-spec=n6ls.iu.i40.8c32g'] + common
    record['state'] = 'submission_unknown'
    receipt_path.write_text(json.dumps(record, indent=2)+'\n')
    try:
        p = subprocess.run(args, env=environment(), capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        raise RuntimeError('Submission outcome unknown; inspect the saved exact name before any retry') from None
    record['cli_exit_code'] = p.returncode
    receipt_path.write_text(json.dumps(record, indent=2)+'\n')
    # Do not print startup commands or signed URLs, even when the CLI fails.
    assert p.returncode == 0, 'Inspect the exact saved job name before any new submission'
    d = describe(name)
    assert d['scheduling']['quota_type'] == quota.upper(), 'Platform quota differs from the authorized request'
    if image:
        assert all(role['image_path'] == image for role in d['roles']), 'Platform image differs'
    record.update(uid=d['uid'], state='submitted', platform_state=d['state'])
    receipt_path.write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['submit', 'status', 'report'])
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--label')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--image', help='Previously verified user image URI')
    p.add_argument('--quota', choices=['spot', 'reserved'], default='spot')
    p.add_argument('--python', default='python3', dest='python_binary')
    a = p.parse_args()
    if a.action == 'submit':
        result = submit(a.source, a.receipt, label=a.label, gpu=a.gpu, seconds=a.seconds, image=a.image, quota=a.quota, python_binary=a.python_binary)
    else:
        r = json.loads(a.receipt.read_text())
        d = describe(r['name'])
        if r.get('uid'):
            assert d['uid'] == r['uid']
        result = {k:d.get(k) for k in ['name','uid','state','start_time','complete_time']}
        result['quota_type'] = d['scheduling']['quota_type']
        result['resource_pool'] = d['resource_pool']['name']
        if a.action == 'report':
            try:
                raw = storage().get_object(Bucket=BUCKET, Key=r['report_key'])['Body'].read()
            except Exception as error:
                if getattr(error, 'response', {}).get('Error', {}).get('Code') == 'NoSuchKey':
                    result['report_available'] = False
                    print(json.dumps(result, ensure_ascii=False))
                    raise SystemExit(0)
                raise
            out = a.receipt.with_name(a.receipt.stem+'_report.json')
            out.write_bytes(raw)
            report = json.loads(raw)
            result['report_path'] = str(out)
            result['report'] = {k:v for k,v in report.items() if k not in
                {'metadata_files','transfers','source_files_sha256','source_files','directories','required_files','selected_metadata'}}
            if isinstance(result['report'].get('auth_recipient'), dict):
                result['report']['auth_recipient'] = {k:v for k,v in result['report']['auth_recipient'].items()
                    if k != 'public_certificate_pem'}
            if 'transfers' in report:
                result['report']['transferred_files'] = len(report['transfers'])
                result['report']['transferred_bytes'] = sum(r['bytes'] for r in report['transfers'])
    print(json.dumps(result, ensure_ascii=False))
