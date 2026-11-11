"""Read-only committed-checkpoint probe; signed URLs stay in a private file."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
PREAMBLE = r'''
import importlib.util,os,sys
from pathlib import Path
r=Path('/Users/faye/code/Behavior-1k Challenge/analysis/huber50_submission_20261017_g0')
spec=importlib.util.spec_from_file_location('h50_checkpoint_auth',r/'expert_huber50_candidate/submit_selected.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.shared.load_credentials()
os.environ.pop('PAIRED_TRANSFER_PROBE',None)
sys.path.insert(0,str(r/'expert_huber50_candidate/selected_stage/bundle'))
'''


def main():
    source = ROOT / 'checkpoint_probe.py'
    proof = json.loads((ROOT / 'checkpoint_probe_preparation.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != proof['probe_sha256']:
        raise ValueError('Checkpoint probe source changed')
    command = shlex.quote('/Users/faye/code/Behavior-1k Challenge/.venv/bin/python') + ' -'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-o', 'ConnectionAttempts=1', 'macpro', command],
                            input=PREAMBLE + source.read_text(), capture_output=True, text=True, timeout=150)
    if result.returncode:
        raise RuntimeError('Read-only checkpoint probe failed; no export submitted')
    snapshot = json.loads(result.stdout)
    if snapshot['run_id'] != proof['expected_run_id']:
        raise ValueError('Unexpected source checkpoint identity')
    if snapshot['state'] == 'committed_checkpoint1000':
        plan = json.loads((ROOT / 'plan.json').read_text())
        if snapshot['formal_contract'] != plan['formal_contract'] or snapshot['consumed_step'] != 1000:
            raise ValueError('Committed checkpoint differs from the frozen training contract')
        private = ROOT / 'private'
        private.mkdir(exist_ok=True)
        private.chmod(0o700)
        path = private / 'checkpoint_snapshot.json'
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(snapshot, stream)
    status = {'observed_at': snapshot['observed_at'], 'state': snapshot['state'],
              'run_id': snapshot['run_id'], 'completed_updates': snapshot.get('completed_updates'),
              'export_submitted': False}
    (ROOT / 'checkpoint_probe_status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
