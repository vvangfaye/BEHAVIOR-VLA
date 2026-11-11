"""Bounded CPU-only account inventory, outside the GPU bootstrap path."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def run(config):
    assert config['account_root']=='/mnt/afs/260010165'
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=False)
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    def publish(value):
        write(root/'status.json',value)
        request=urllib.request.Request(config['result_put'],data=json.dumps(value).encode(),method='PUT')
        with urllib.request.urlopen(request,timeout=45):pass
    started=time.time()
    publish({'state':'scanning','started_at':started,'gpu_count':0})
    try:
        with (root/'du.stdout').open('wb') as out,(root/'du.stderr').open('wb') as err:
            process=subprocess.run(['nice','-n','15','du','-sx','--block-size=1',config['account_root']],
                                   stdout=out,stderr=err,timeout=6600)
        if process.returncode:raise RuntimeError('Account inventory was incomplete; see bounded CPU audit stderr')
        used=int((root/'du.stdout').read_text().split()[0])
        evidence={'state':'complete','complete':True,'experiment':config['experiment'],
                  'account_root':config['account_root'],'account_allocated_bytes':used,
                  'quota_lower_bound_bytes':config['quota_lower_bound_bytes'],
                  'quota_source':config['quota_source'],'new_task_reserve_bytes':config['reserve_bytes'],
                  'headroom_margin_bytes':10*1024**3,'observed_at':time.time(),
                  'inventory_seconds':time.time()-started,'gpu_count':0}
        evidence['passed']=used+config['reserve_bytes']+evidence['headroom_margin_bytes']<=config['quota_lower_bound_bytes']
        write(config['proof_path'],evidence);publish(evidence)
    except BaseException as error:
        publish({'state':'failed','complete':False,'error_type':type(error).__name__,
                 'error':str(error)[:500],'elapsed_seconds':time.time()-started,'gpu_count':0})
        raise
