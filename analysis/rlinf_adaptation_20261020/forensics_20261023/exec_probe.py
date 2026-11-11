"""Run a bounded diagnostic script on one verified existing worker."""
import argparse
import base64
import json
import os
import re
import select
import shlex
import subprocess
import time
import zlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True)
    p.add_argument('--uid',required=True)
    p.add_argument('--script',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=60)
    a=p.parse_args()
    r=subprocess.run(['sco','acp','jobs','describe',a.job,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,timeout=40,check=True)
    job=json.loads(r.stdout)
    assert job['uid']==a.uid and job['state']=='RUNNING' and job['ownership']['user_name']=='260010165'
    source=a.script.read_text()
    encoded=base64.b64encode(zlib.compress(source.encode())).decode()
    command='python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))')+'\nexit\n'
    process=subprocess.Popen(['sco','acp','jobs','exec',a.job,'--workspace-name=share-space','--worker-name=pt-'+a.uid+'-worker-0'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    pending=b'';sent=False;deadline=time.monotonic()+a.timeout
    try:
        while time.monotonic()<deadline:
            ready,_,_=select.select([process.stdout],[],[],1)
            if not ready:continue
            data=os.read(process.stdout.fileno(),65536)
            if not data:break
            pending+=data
            pending=pending.translate(None,b'\x00\x01\x02\x03\x04')
            if not sent and re.search(rb'(?:# |\$ )',pending):
                process.stdin.write(command.encode());process.stdin.flush();sent=True;pending=b''
            match=re.search(rb'B1KPROBE_BEGIN([A-Za-z0-9+/=]+)B1KPROBE_END',pending)
            if match:
                result=json.loads(base64.b64decode(match.group(1)))
                a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
                print(json.dumps(result,ensure_ascii=False));return
            if len(pending)>2*1024**2:raise RuntimeError('bounded output exceeded')
        raise RuntimeError('No diagnostic receipt; '+re.sub(r'https?://\S+','[URL omitted]',pending[-1000:].decode(errors='replace')))
    finally:
        process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':main()
