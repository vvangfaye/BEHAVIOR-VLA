"""Read bounded native logs on the Mac; export only fixed progress fields."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import importlib.util,json,os,re,subprocess,time
r=Path('/Users/faye/code/Behavior-1k Challenge/analysis/huber50_submission_20261017_g0')
spec=importlib.util.spec_from_file_location('h50_cache_observer',r/'expert_huber50_candidate/submit_selected.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.shared.load_credentials()
cli='/Users/faye/.local/bin/ai-hub-cli';token='--token='+os.environ['AI_HUB_TOKEN']
native=subprocess.run([cli,'train','status','--iid','241869','--detail',token],capture_output=True,text=True,timeout=15)
if native.returncode or '{' not in native.stdout:raise ValueError('Native structured status unavailable')
data=json.loads(native.stdout[native.stdout.index('{'):]).get('data',{})
progress=[]
markers=[('environment_copy','Copying frozen policy environment'),
 ('environment_extract','Extracting frozen policy environment'),
 ('checkpoint_copy','Copying and hashing checkpoint'),
 ('data_staging','Staging four-source data'),('expert_cache','Caching exact original files'),
 ('training_start','Starting authorized formal fine-tuning')]
for worker in data.get('containers',[]):
 if worker.get('role')!='worker':continue
 row={'platform_rank':worker.get('rank'),'status':worker.get('status')}
 path=worker.get('log_storage_path','')
 if not path.startswith('oss://ados-nebula-v2/nebula-one-proxy/remote_log/'):
  row['progress_available']=False;progress.append(row);continue
 try:
  result=subprocess.run([cli,'train','logview','oss','--path='+path+'/stdout',
       '--bufferSize=8192','--fromHead=false',token],capture_output=True,text=True,timeout=10)
  if result.returncode or '{' not in result.stdout:raise ValueError('Log view unavailable')
  body=json.loads(result.stdout[result.stdout.index('{'):]).get('content','')
  if not isinstance(body,str):raise ValueError('Unexpected log format')
  row['progress_available']=True
  found=[(body.rfind(text),stage) for stage,text in markers if text in body]
  if found:row['stage']=max(found)[1]
  cached=re.findall(r'Cached original files\s+(\d+)/(\d+)',body)
  if cached:
   row.update(stage='expert_cache',cached_files=int(cached[-1][0]),total_files=int(cached[-1][1]))
 except (ValueError,subprocess.TimeoutExpired):row['progress_available']=False
 progress.append(row)
print(json.dumps({'iid':'241869','observed_at':time.time(),'workers':progress,'raw_logs_exported':False}))
'''


def main():
    command = shlex.quote('/Users/faye/code/Behavior-1k Challenge/.venv/bin/python') + ' -c ' + shlex.quote(REMOTE)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-o', 'ConnectionAttempts=1', 'macpro', command],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Read-only cache observation failed; no task mutation')
    data = json.loads(result.stdout)
    if data['iid'] != '241869':
        raise ValueError('Unexpected task identity')
    (ROOT / 'cache_progress.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data))


if __name__ == '__main__':
    main()
