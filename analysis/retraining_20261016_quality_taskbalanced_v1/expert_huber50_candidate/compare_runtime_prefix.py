"""Read-only timing comparison on matching, individually logged early updates."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import hashlib,importlib.util,json,statistics
r=Path('/Users/faye/code/Behavior-1k Challenge/analysis/huber50_submission_20261017_g0')
s=importlib.util.spec_from_file_location('h50_timing_auth',r/'expert_huber50_candidate/submit_selected.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.shared.load_credentials()
import sys
sys.path.insert(0,str(r/'expert_huber50_candidate/selected_stage/bundle'))
from nebula.submit_oss import OSSClient
c=OSSClient();rows={};hashes={}
for name,root in [('reference','expert_control/b1k-expert-control-1688977-20261016'),
                  ('huber50','expert_huber/b1k-expert-huber50-1688977-20261017')]:
 key='wangfaye/result/behavior-challenge/'+root+'/attempts/stage1000-g0/training_metrics.jsonl'
 size=c.head(key)
 if not size or size>2097152:raise ValueError('Timing metadata missing or oversized')
 raw=c.read(key);hashes[name]=hashlib.sha256(raw).hexdigest();rows[name]={}
 for line in raw.splitlines():
  try:row=json.loads(line)
  except ValueError:continue
  if row.get('kind')=='training_update' and row['completed_updates']<=200:rows[name][row['completed_updates']]=row
n=min(max(rows['reference']),max(rows['huber50']))
assert set(range(1,n+1)).issubset(rows['reference']) and set(range(1,n+1)).issubset(rows['huber50'])
results={}
for name,values in rows.items():
 windows={}
 for label,start in [('all_available_prefix',1),('excluding_first_ten',11)]:
  if n<start:continue
  subset=[values[i]['metrics'] for i in range(start,n+1)]
  waits=[x['data_wait_seconds'] for x in subset];steps=[x['step_seconds'] for x in subset]
  windows[label]={'first_update':start,'last_update':n,'updates':len(subset),
      'mean_data_wait_seconds':statistics.mean(waits),'mean_reported_step_seconds':statistics.mean(steps),
      'mean_cycle_seconds_approx':statistics.mean(a+b for a,b in zip(waits,steps))}
 results[name]=windows
print(json.dumps({'matched_updates':n,'timings':results,'metrics_snapshot_sha256':hashes,
 'interpretation':'Reported step_seconds excludes the preceding batch fetch. Their sum approximates the primary-process update cycle; distributed stragglers may also appear in step_seconds. This early prefix is not a whole-run ETA.'}))
'''


def main():
    command = shlex.quote('/Users/faye/code/Behavior-1k Challenge/.venv/bin/python') + ' -c ' + shlex.quote(REMOTE)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-o', 'ConnectionAttempts=1', 'macpro', command],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Read-only timing comparison failed')
    data = json.loads(result.stdout)
    (ROOT / 'runtime_prefix_comparison.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data))


if __name__ == '__main__':
    main()
