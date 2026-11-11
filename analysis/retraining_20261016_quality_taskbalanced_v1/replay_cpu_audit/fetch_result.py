"""Fetch only the complete diagnostic JSON after the CPU job is terminal."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import sys,json,hashlib
r=Path('/Users/faye/code/Behavior-1k Challenge/analysis/replay_cpu_loss_audit_20261017')
sys.path.insert(0,str(r))
import submit,dispatch_support
submit.load_credentials()
from submit_oss import OSSClient
native=dispatch_support.status('241735')
if native['application_status'] not in ('FINISHED','SUCCESS','SUCCEEDED'):
 raise RuntimeError('Diagnostic is not successfully terminal')
c=OSSClient()
prefix=dispatch_support.read(r/'runtime_base.json')['output_key']
success=json.loads(c.read(prefix+'/_SUCCESS.json'))
raw=c.read(prefix+'/result.json')
if not success.get('complete') or hashlib.sha256(raw).hexdigest()!=success['result_sha256']:
 raise ValueError('Diagnostic result digest differs')
print(json.dumps({'native':native,'success':success,'result_text':raw.decode('utf-8')}))
'''


def main():
    state = json.loads((ROOT / "current_observation.json").read_text())
    if "_SUCCESS.json" not in state["outputs"]:
        print(json.dumps({"fetched": False, "reason": "success_receipt_not_observed"}))
        return
    command = shlex.quote('/Users/faye/code/Behavior-1k Challenge/.venv/bin/python') + ' -c ' + shlex.quote(REMOTE)
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                               '-o', 'ConnectionAttempts=1', 'macpro', command],
                              capture_output=True, text=True, timeout=60)
    if response.returncode:
        raise RuntimeError('Diagnostic fetch failed; no result accepted')
    envelope = json.loads(response.stdout)
    raw = envelope.pop("result_text").encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != envelope["success"]["result_sha256"]:
        raise ValueError("Fetched bytes changed")
    for name, data in (("result.json", raw), ("fetch_proof.json", (json.dumps(envelope, indent=2) + "\n").encode())):
        temp = ROOT / (name + ".tmp")
        temp.write_bytes(data)
        temp.replace(ROOT / name)
    print(json.dumps({"fetched": True, "bytes": len(raw)}))


if __name__ == "__main__":
    main()
