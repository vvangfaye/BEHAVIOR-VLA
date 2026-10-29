"""Keep finite preparation jobs alive, without duplicate writers or silent retries."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
JOBS={'small':('small_upload.json','prepare_campaign.py',['--part','small'],'prepare_small.log'),
      'checkpoints':('checkpoints_upload.json','prepare_campaign.py',['--part','checkpoints'],'prepare_checkpoints.log'),
      'scenes':('data_bundle_upload.json','prepare_scene_bundle.py',[],'prepare_scenes.log')}


def live_job(script,args):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:parts=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(x.endswith(('/serf_multitask_20261029/'+script).encode()) for x in parts) and all(x.encode() in parts for x in args):result.append(int(p.name))
    return result


def main():
    with (HERE/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status={'started_at':time.time(),'pid':os.getpid(),'jobs':{}}
        while time.time()-status['started_at']<8*3600:
            all_done=True
            for label,(receipt,script,args,log_name) in JOBS.items():
                path=HERE/receipt
                if path.exists() and json.loads(path.read_text()).get('complete'):
                    status['jobs'][label]={'complete':True};continue
                all_done=False;pids=live_job(script,args)
                assert len(pids)<=1,(label,pids)
                if not pids:
                    log=HERE/log_name
                    text=log.read_text(errors='replace')[-5000:] if log.exists() else ''
                    if 'Traceback (most recent call last)' in text:
                        status['error']=label+' preparation failed; see '+log_name
                        (HERE/'preparation_status.json').write_text(json.dumps(status,indent=2)+'\n');return
                    env=dict(os.environ);env['SSL_CERT_FILE']='/etc/ssl/certs/ca-certificates.crt'
                    with log.open('ab',buffering=0) as output:
                        child=subprocess.Popen([sys.executable,'-u',str(HERE/script),*args],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True)
                    pids=[child.pid]
                status['jobs'][label]={'complete':False,'pids':pids}
            status['updated_at']=time.time();temp=HERE/'preparation_status.json.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(HERE/'preparation_status.json')
            if all_done:return
            time.sleep(15)
        status['error']='Preparation exceeded eight-hour bound';(HERE/'preparation_status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':main()
