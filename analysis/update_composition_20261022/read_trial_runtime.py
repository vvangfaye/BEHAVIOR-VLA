"""Bounded, read-only inspection through this trial's own live worker.

The local SCO terminal client is terminated after the reply; the remote model
and simulator processes are not signalled. No commands outside the allowlisted
trial directory are executed.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import select
import shlex
import subprocess
import time

from submit import cli

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('meta','maml','fixed_crossplay','rl_bc','before_after'), required=True)
    parser.add_argument('--role', choices=('trainer','renderer'), default='trainer')
    args = parser.parse_args()
    intent = json.loads((HERE/'split_trial_submissions'/args.method/args.role/'submission.json').read_text())
    response = cli('acp','jobs','describe',intent['job_name'],'--workspace-name=share-space','-o','json')
    response.check_returncode()
    job = json.loads(response.stdout)
    if job['uid'] != intent['uid'] or job['state'] != 'RUNNING':
        raise RuntimeError('Exact trial worker is not confirmed running')
    root = '/mnt/afs/260010165/behavior_1k/update_composition_20261022/trial_h100/'+args.method
    script = f'''
import base64,json,pathlib
root=pathlib.Path({root!r})
result={{}}
for name in ('messages/status.json','trainer/output/progress.json','trainer/output/result.json','renderer/output/progress.json','renderer/output/result.json'):
 p=root/name
 if p.exists():
  raw=p.read_bytes()
  result[name]=json.loads(raw) if len(raw)<524288 else {{'oversized':len(raw)}}
for name in ('trainer/worker.log','renderer/worker.log'):
 p=root/name
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-3000));result[name]=f.read(3000).decode(errors='replace')
episodes=sorted((root/'trainer/output').glob('iteration_*_episode_*'))
for directory in episodes[-2:]:
 p=directory/'simulator.log'
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-6000));result[str(p.relative_to(root))]=f.read(6000).decode(errors='replace')
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
'''
    encoded = base64.b64encode(script.encode()).decode()
    command = 'python3 -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')+'\nexit\n'
    process = subprocess.Popen(['sco','acp','jobs','exec',intent['job_name'],
        '--workspace-name=share-space','--worker-name=pt-'+intent['uid']+'-worker-0'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buffer = b''
    command_sent = False
    try:
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            ready,_,_ = select.select([process.stdout],[],[],1)
            if ready:
                chunk = os.read(process.stdout.fileno(),65536)
                if not chunk:
                    break
                buffer += chunk
                if len(buffer)>2*1024**2:
                    raise ValueError('Remote inspection output exceeded limit')
                # SCO first opens a shell and runs its own PID handshake. Input
                # sent before the prompt can race that initialization.
                if not command_sent and re.search(rb'(?:# |\$ )',buffer):
                    process.stdin.write(command.encode())
                    process.stdin.flush()
                    command_sent = True
                # SCO emits the websocket channel byte at frame boundaries,
                # including boundaries inside a single printed JSON payload.
                normalized = buffer.translate(None,b'\x00\x01\x02\x03\x04')
                match = re.search(rb'B1KRUNTIMEBEGIN([A-Za-z0-9+/=]+)B1KRUNTIMEEND',normalized)
                if match:
                    value=json.loads(base64.b64decode(match.group(1)))
                    path=HERE/'split_trial_submissions'/args.method/'runtime_inspection.json'
                    path.write_text(json.dumps(value,indent=2)+'\n')
                    print(json.dumps(value,indent=2))
                    return
        phase = 'awaiting_result' if command_sent else 'awaiting_shell_prompt'
        error_info=dict(phase=phase,bytes_received=len(buffer),
                        begin_marker_seen=b'B1KRUNTIMEBEGIN' in buffer,
                        end_marker_seen=b'B1KRUNTIMEEND' in buffer)
        (HERE/'split_trial_submissions'/args.method/'runtime_read_status.json').write_text(
            json.dumps(error_info,indent=2)+'\n')
        raise TimeoutError('Read-only runtime inspection timed out in '+phase+'; no job state inferred')
    finally:
        process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=3)


if __name__=='__main__':main()
