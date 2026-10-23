"""Bounded transport for reviewed storage scripts; does not signal remote research processes."""
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



HERE = Path(__file__).resolve().parent


def main():
    intent = dict(job_name='b1k-comp1022-meta-renderer-s17-r03',uid='f9bde6f89593443d8ba306c7109ea7f8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',default=intent['job_name'])
    parser.add_argument('--uid',default=intent['uid'])
    parser.add_argument('--script',default=str(HERE/'remote_scan.py'))
    parser.add_argument('--out',default=str(HERE/'inventory.json'))
    args=parser.parse_args()
    intent={'job_name':args.job,'uid':args.uid}
    script = Path(args.script).read_text()
    encoded = base64.b64encode(script.encode()).decode()
    command = 'python3 -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')+'\nexit\n'
    process = subprocess.Popen(['sco','acp','jobs','exec',intent['job_name'],
        '--workspace-name=share-space','--worker-name=pt-'+intent['uid']+'-worker-0'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buffer = b''
    command_sent = False
    try:
        deadline = time.monotonic()+55
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
                    path=Path(args.out)
                    path.write_text(json.dumps(value,indent=2)+'\n')
                    print(json.dumps({'saved':str(path),'keys':list(value)}))
                    return
        phase = 'awaiting_result' if command_sent else 'awaiting_shell_prompt'
        error_info=dict(phase=phase,bytes_received=len(buffer),
                        begin_marker_seen=b'B1KRUNTIMEBEGIN' in buffer,
                        end_marker_seen=b'B1KRUNTIMEEND' in buffer)
        (HERE/'runtime_read_status.json').write_text(
            json.dumps(error_info,indent=2)+'\n')
        raise TimeoutError('Read-only runtime inspection timed out in '+phase+'; no job state inferred')
    finally:
        process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=3)


if __name__=='__main__':main()
