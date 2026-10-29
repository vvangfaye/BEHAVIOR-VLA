"""Small standalone supervisor: persist cgroup evidence if a child is SIGKILLed."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def snapshot():
    out={'time':time.time(),'cgroup':{},'process_rss_kib':[]}
    root=Path('/sys/fs/cgroup')
    for name in ('memory.current','memory.peak','memory.max','memory.events',
                 'memory/memory.usage_in_bytes','memory/memory.max_usage_in_bytes',
                 'memory/memory.limit_in_bytes','memory/memory.failcnt'):
        path=root/name
        if path.is_file():
            out['cgroup'][name]=path.read_text()[:4000].strip()
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            fields={}
            for line in path.read_text().splitlines():
                key,_,value=line.partition(':')
                if key in ('Name','Pid','PPid','VmRSS','VmHWM'):
                    fields[key]=value.strip()
            if int(fields.get('VmRSS','0').split()[0])>100000:
                out['process_rss_kib'].append(fields)
        except (OSError,ValueError):
            pass
    return out


def write(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,allow_nan=False)+'\n');tmp.replace(path)


def run(command, root, env):
    root=Path(root)
    with (root/'driver.log').open('x') as log:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        previous={}
        def forward(sig,*_):
            if process.poll() is None:
                process.send_signal(sig)
        for sig in (signal.SIGTERM,signal.SIGINT):
            previous[sig]=signal.signal(sig,forward)
        try:
            before=snapshot();write(root/'resource_before.json',before)
            with (root/'resources.jsonl').open('x',buffering=1) as output:
                while True:
                    current=snapshot()
                    output.write(json.dumps(current,allow_nan=False)+'\n')
                    write(root/'resource_latest.json',current)
                    try:
                        code=process.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            after=snapshot()
            receipt={'returncode':code,'signal':signal.Signals(-code).name if code<0 else None,
                     'before':before,'after':after}
            write(root/'driver_exit.json',receipt)
            return receipt
        finally:
            for sig,handler in previous.items():
                signal.signal(sig,handler)
            if process.poll() is None:
                process.terminate()


if __name__=='__main__':
    import sys
    result=run(sys.argv[1:],Path.cwd(),dict(os.environ))
    raise SystemExit(result['returncode'] if result['returncode']>=0 else 128-result['returncode'])
