"""Finite data relay run by the user's allowlisted system Python executable.

The approved local process reads AOSS directly. macpro receives verified files
over SSH and performs CPU-only export and target-OSS validation. It never uses
the blocked macpro AOSS route or any proxy.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request

from prepare_retraining_data import download, save

REMOTE_CACHE='/Users/faye/Library/Caches/behavior-data-prep-20261013-v1'
REMOTE_OUTPUT='/Users/faye/code/Behavior-1k Challenge/analysis/retraining_20261013/preparation'
SOCKET='/private/tmp/behavior-data-prep-tunnel.sock'


def remote(row, *, prepare_only=False, log=None):
    argv=[REMOTE_CACHE+'/env/bin/python',REMOTE_CACHE+'/code/sco/accept_relayed_retraining_source.py',
          '--cache',REMOTE_CACHE,'--output',REMOTE_OUTPUT,'--key',row['key']]
    if prepare_only:argv.append('--prepare-only')
    paths=':'.join(REMOTE_CACHE+'/'+p for p in ['code/sco','code/scripts','code/openpi/src','lerobot/src'])
    command=shlex.join(['env','PYTHONPATH='+paths,'CUDA_VISIBLE_DEVICES=',*argv])
    result=subprocess.run(['ssh','-S',SOCKET,'macpro',command],capture_output=True,text=True)
    if log:
        log.write(result.stdout+result.stderr);log.flush()
    if result.returncode:
        raise RuntimeError('Remote data preparation failed; see relay worker log')
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(args):
    inputs=json.loads(args.sources.read_text())
    assert inputs['authorization']['training_allowed'] is False
    args.cache.mkdir(parents=True,exist_ok=True);os.chmod(args.cache,0o700)
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'relay.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    context=ssl.create_default_context()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=context))
    rows=[r for r in inputs['source_archives'] if args.kind is None or r['kind']==args.kind]
    if args.limit:rows=rows[:args.limit]
    for number,row in enumerate(rows):
        if (args.output/'STOP').exists():raise RuntimeError('Data preparation STOP is present')
        state={'pid':os.getpid(),'phase':'checking_remote_receipt','source_id':row['id'],
               'index':number,'selected_archives':len(rows),'updated_at':time.time(),
               'transport':'allowlisted_local_python_direct_AOSS_then_SSH','training_executed':False}
        save(args.output/'relay_status.json',state)
        record=remote(row,prepare_only=True)
        if record['already_accepted']:continue
        if shutil.disk_usage(args.cache).free < row['bytes']+16*1024**3:
            raise RuntimeError('Insufficient relay scratch space')
        path=args.cache/(record['id']+'.tar')
        private_url=args.cache/(record['id']+'.url.json')
        # Signing is offline; only the allowlisted system Python opens AOSS.
        signer=os.environ.get('RETRAINING_SIGNER_PYTHON','/private/tmp/behavior-sco-tests/bin/python')
        signed=subprocess.run([signer,str(Path(__file__).with_name('sign_retraining_source.py')),
            '--sources',str(args.sources),'--key',row['key'],'--output',str(private_url)],
            capture_output=True,text=True)
        if signed.returncode:raise RuntimeError('Private source URL refresh failed')
        fresh=json.loads(private_url.read_text())
        if fresh['key']!=row['key']:raise ValueError('Refreshed source scope differs')
        row={**row,'signed_get_url':fresh['signed_get_url']}
        state.update(phase='downloading_source',updated_at=time.time());save(args.output/'relay_status.json',state)
        def report_progress(done,total):
            state.update(downloaded_bytes=done,total_bytes=total,updated_at=time.time())
            save(args.output/'relay_status.json',state)
        download(row,path,opener,progress=report_progress)
        state.update(phase='sending_to_macpro',updated_at=time.time());save(args.output/'relay_status.json',state)
        subprocess.run(['scp','-o','ControlPath='+SOCKET,str(path),
                        'macpro:'+REMOTE_CACHE+'/work/'+record['id']+'/source.tar.relay'],check=True)
        state.update(phase='remote_export_and_readback',updated_at=time.time());save(args.output/'relay_status.json',state)
        with (args.output/'relay_worker.log').open('a') as log:accepted=remote(row,log=log)
        if accepted.get('accepted') is not True:raise ValueError('Archive was not accepted')
        path.unlink()
        private_url.unlink()
        save(args.output/'relay_receipts'/(record['id']+'.json'),{**accepted,'source_sha256':row['sha256'],
            'verified_at':time.time(),'network_executable':sys.executable,'proxy_used':False,'training_executed':False})
    save(args.output/'relay_status.json',{'pid':os.getpid(),'phase':'selected_archives_complete',
         'selected_archives':len(rows),'updated_at':time.time(),'training_executed':False})
    paths=':'.join(REMOTE_CACHE+'/'+p for p in ['code/sco','code/scripts','code/openpi/src','lerobot/src'])
    command=shlex.join(['env','PYTHONPATH='+paths,REMOTE_CACHE+'/env/bin/python',
                       REMOTE_CACHE+'/code/sco/finalize_retraining_data.py',REMOTE_CACHE,REMOTE_OUTPUT])
    result=subprocess.run(['ssh','-S',SOCKET,'macpro',command],capture_output=True,text=True)
    with (args.output/'relay_worker.log').open('a') as log:log.write(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Final data manifest audit failed; inspect worker log')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--kind',choices=['subtask','full_episode']);p.add_argument('--limit',type=int)
    a=p.parse_args()
    try:main(a)
    except Exception as error:
        message=re.sub(r'https?://[^\s\"\']+','[redacted URL]',str(error))
        save(a.output/'relay_failure.json',{'type':type(error).__name__,'message':message[:2000],'time':time.time(),'training_executed':False})
        raise SystemExit(type(error).__name__+': '+message) from None
