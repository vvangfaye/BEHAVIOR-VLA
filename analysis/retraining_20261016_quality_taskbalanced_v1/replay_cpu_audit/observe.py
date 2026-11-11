"""Read-only status observer for the one submitted replay diagnostic."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import sys,json,time
r=Path('/Users/faye/code/Behavior-1k Challenge/analysis/replay_cpu_loss_audit_20261017')
sys.path.insert(0,str(r))
import submit,dispatch_support
submit.load_credentials()
from submit_oss import OSSClient
c=OSSClient()
prefix=dispatch_support.read(r/'runtime_base.json')['output_key']
output={}
for name in ['started.json','runtime.json','progress.json','_SUCCESS.json','failure.json']:
 if c.head(prefix+'/'+name):output[name]=json.loads(c.read(prefix+'/'+name))
print(json.dumps({'observed_at':time.time(),'native':dispatch_support.status('241735'),'outputs':output}))
'''


def main():
    command = shlex.quote('/Users/faye/code/Behavior-1k Challenge/.venv/bin/python') + ' -c ' + shlex.quote(REMOTE)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-o', 'ConnectionAttempts=1', 'macpro', command],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Read-only CPU observation failed; no submission attempted')
    observed = json.loads(result.stdout)
    if observed['native']['iid'] != '241735':
        raise ValueError('Unexpected task identity')
    output = observed['outputs']
    if output.get('started.json', {}).get('run_id', 'b1k-replay-loss-audit-20261017-g0') != 'b1k-replay-loss-audit-20261017-g0':
        raise ValueError('Unexpected output identity')
    temp = ROOT / 'current_observation.json.tmp'
    temp.write_text(json.dumps(observed, indent=2) + '\n')
    temp.replace(ROOT / 'current_observation.json')
    print(json.dumps({'native': observed['native']['application_status'],
        'progress': output.get('progress.json'), 'failure': output.get('failure.json'),
        'complete': output.get('_SUCCESS.json'), 'runtime_prepared': 'runtime.json' in output}))


if __name__ == '__main__':
    main()
