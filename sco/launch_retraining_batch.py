"""Launch one detached, finite CPU data-preparation batch on macpro."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_retraining_data import Preparation, save


def main(cache,output):
    lock=(output/'batch_launch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=output/'batch_launch.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        command=subprocess.run(['ps','-p',str(old['pid']),'-o','args='],capture_output=True,text=True).stdout
        if str(cache/'code/sco/run_retraining_preparation.py') in command:
            print(json.dumps(old));return
    if (output/'STOP').exists():raise RuntimeError('Data preparation STOP is present')
    prep=Preparation(cache,output);fingerprint=prep.inputs['fingerprint'];prep.lock.close()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',
        PYTHONPATH=':'.join(str(cache/p) for p in ['code/sco','code/scripts','code/openpi/src','lerobot/src']))
    command=[sys.executable,str(cache/'code/sco/run_retraining_preparation.py'),'--cache',str(cache),'--output',str(output)]
    child=subprocess.Popen(command,env=env,stdout=(output/'batch.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    result={'pid':child.pid,'command':command,'launched_at':time.time(),'source_fingerprint':fingerprint,
            'training_allowed':False,'cpu_only':True,'finite_sources':463}
    save(receipt,result);print(json.dumps(result))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
