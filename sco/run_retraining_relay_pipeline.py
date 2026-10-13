"""Wait for the full-episode pilot, then process the finite frozen source list."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shlex

from prepare_retraining_data import save
from relay_retraining_sources import remote, REMOTE_CACHE, REMOTE_OUTPUT, SOCKET


def main(args):
    inputs=json.loads(args.sources.read_text())
    assert inputs['authorization']['training_allowed'] is False
    sequence=(args.output/'pipeline.lock').open('a+')
    fcntl.flock(sequence,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(args.output/'pipeline_status.json',{'pid':os.getpid(),'phase':'waiting_for_full_episode_pilot',
        'source_fingerprint':inputs['fingerprint'],'updated_at':time.time(),'training_executed':False})
    # The existing pilot owns this lock until its actual process exits.
    relay=(args.output/'relay.lock').open('a+')
    fcntl.flock(relay,fcntl.LOCK_EX)
    pilot=next(r for r in inputs['source_archives'] if r['kind']=='full_episode')
    result=remote(pilot,prepare_only=True)
    if not result.get('already_accepted'):
        raise RuntimeError('Full-episode pilot did not pass; inspect its diagnostic before expanding')
    if (args.output/'STOP').exists():raise RuntimeError('Data preparation STOP is present')
    relay.close()
    if args.remote_direct:
        command=shlex.join([REMOTE_CACHE+'/env/bin/python',REMOTE_CACHE+'/code/sco/launch_retraining_batch.py',
                            REMOTE_CACHE,REMOTE_OUTPUT])
        paths=':'.join(REMOTE_CACHE+'/'+p for p in ['code/sco','code/scripts','code/openpi/src','lerobot/src'])
        command=shlex.join(['env','PYTHONPATH='+paths])+' '+command
        result=subprocess.run(['ssh','-S',SOCKET,'macpro',command],capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Remote batch launch failed: '+result.stderr[-1500:])
        launch=json.loads(result.stdout.strip().splitlines()[-1])
        save(args.output/'pipeline_status.json',{'pid':os.getpid(),'phase':'remote_batch_running',
            'remote_launch':launch,'source_fingerprint':inputs['fingerprint'],
            'updated_at':time.time(),'training_executed':False})
        return
    command=[sys.executable,str(Path(__file__).with_name('relay_retraining_sources.py')),
        '--sources',str(args.sources),'--cache',str(args.cache),'--output',str(args.output)]
    child=subprocess.Popen(command)
    save(args.output/'pipeline_status.json',{'pid':os.getpid(),'worker_pid':child.pid,'phase':'all_frozen_archives',
        'source_fingerprint':inputs['fingerprint'],'source_count':len(inputs['source_archives']),
        'updated_at':time.time(),'training_executed':False})
    code=child.wait()
    save(args.output/'pipeline_status.json',{'pid':os.getpid(),'phase':'finished' if code==0 else 'needs_diagnostic',
        'returncode':code,'updated_at':time.time(),'training_executed':False})
    raise SystemExit(code)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--remote-direct',action='store_true')
    a=p.parse_args()
    try:main(a)
    except Exception as error:
        save(a.output/'pipeline_status.json',{'pid':os.getpid(),'phase':'needs_diagnostic','type':type(error).__name__,
            'message':str(error)[:1000],'updated_at':time.time(),'training_executed':False})
        raise SystemExit(type(error).__name__+': '+str(error)) from None
