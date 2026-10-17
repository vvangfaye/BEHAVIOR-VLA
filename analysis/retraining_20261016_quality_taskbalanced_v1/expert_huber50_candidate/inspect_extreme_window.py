"""Read exact logged updates around the independently located extreme target."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import hashlib,importlib.util,json,sys
r=Path('/Users/faye/code/Behavior-1k Challenge/analysis/huber50_submission_20261017_g0')
s=importlib.util.spec_from_file_location('h50_extreme_auth',r/'expert_huber50_candidate/submit_selected.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.shared.load_credentials()
sys.path.insert(0,str(r/'expert_huber50_candidate/selected_stage/bundle'))
from nebula.submit_oss import OSSClient
c=OSSClient();out={}
for name,root in [('expert_mse','expert_control/b1k-expert-control-1688977-20261016'),
                  ('huber50','expert_huber/b1k-expert-huber50-1688977-20261017')]:
 key='wangfaye/result/behavior-challenge/'+root+'/attempts/stage1000-g0/training_metrics.jsonl'
 size=c.head(key)
 if not size or size>2097152:raise ValueError('Metrics missing or oversized')
 raw=c.read(key);rows=[]
 for line in raw.splitlines():
  try:row=json.loads(line)
  except ValueError:continue
  if row.get('kind')=='training_update' and 790<=row['completed_updates']<=840:rows.append(row)
 out[name]={'metrics_snapshot_sha256':hashlib.sha256(raw).hexdigest(),'rows':rows}
print(json.dumps(out))
'''


def main():
    command = shlex.quote('/Users/faye/code/Behavior-1k Challenge/.venv/bin/python') + ' -c ' + shlex.quote(REMOTE)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-o', 'ConnectionAttempts=1', 'macpro', command],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Read-only loss-window observation failed')
    data = json.loads(result.stdout)
    (ROOT / 'extreme_window_observation.json').write_text(json.dumps(data, indent=2) + '\n')
    keys = ('loss', 'action_loss', 'action_mse', 'action_robust_tail_fraction', 'grad_norm', 'nonfinite_count')
    print(json.dumps({name: [{'completed_updates': row['completed_updates'],
                             **{key: row['metrics'][key] for key in keys if key in row['metrics']}}
                            for row in value['rows']] for name, value in data.items()}))


if __name__ == '__main__':
    main()
