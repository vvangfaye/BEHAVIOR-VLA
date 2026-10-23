"""Read-only inspection of this experiment's AFS artifacts; no model or simulator."""
import hashlib
import json
import os
import socket
import time
import urllib.request
from pathlib import Path


def inspect(config):
    started=time.time()
    result={'hostname':socket.gethostname(),'read_only':True,'checks':[],'roots':{}}
    for name in ['base','retry']:
        root=Path(config[name])
        data={'exists':root.exists()}
        try:
            data['entries']=[{'name':p.name,'directory':p.is_dir(),'bytes':p.stat().st_size} for p in sorted(root.iterdir())]
            st=os.statvfs(root)
            data['available_bytes']=st.f_bavail*st.f_frsize
        except Exception as error:data['error']=repr(error)
        result['roots'][name]=data
    for relative,expected in config['hashes'].items():
        path=Path(config['base'])/relative
        record={'relative':relative,'exists':path.exists(),'expected':expected}
        try:
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
            record.update(actual=h.hexdigest(),bytes=path.stat().st_size,match=h.hexdigest()==expected)
        except Exception as error:record['error']=repr(error)
        result['checks'].append(record)
    root=Path(config['retry'])
    result['reports']={}
    for relative in ['runtime_receipt.json','results/status.json','results/baseline.log','results/jax/301/initialization.json','results/jax/301/simulator.log']:
        path=root/relative
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-16000))
                text=stream.read().decode(errors='replace')
            result['reports'][relative]=text
    result['elapsed_seconds']=time.time()-started
    return result


def main(config):
    try:result=inspect(config)
    except BaseException as error:result={'probe_failed':repr(error)}
    raw=json.dumps(result,ensure_ascii=False).encode()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(config['put'],data=raw,method='PUT'),timeout=120) as response:
        assert response.status in (200,201,204)
    print(json.dumps({'uploaded':True,'checks':len(result.get('checks',[])),'mismatches':[x['relative'] for x in result.get('checks',[]) if not x.get('match')]}),flush=True)
