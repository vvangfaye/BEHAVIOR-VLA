"""Finite CPU-only source-cloud to destination-OSS transfer with real readback."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import sys
import time
import urllib.request
import urllib.error

from prepare_retraining_data import download,error_detail,safe_extract,save,sha,verify_dataset


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=ssl.create_default_context()))


def put(o,url,path,headers=None):
    with Path(path).open('rb') as stream:
        request=urllib.request.Request(url,data=stream,method='PUT',headers={'Content-Length':str(Path(path).stat().st_size),**(headers or {})})
        with o.open(request,timeout=600):pass


def readback(o,spec,path):
    target=spec['target'];h=hashlib.sha256();n=0
    with o.open(spec['get'],timeout=600) as r, Path(path).open('wb') as out:
        etag=r.headers.get('ETag','').strip('"')
        while block:=r.read(8*1024**2):
            out.write(block);h.update(block);n+=len(block)
            if n>target['bytes']:raise ValueError('Destination longer than signed source')
    assert n==target['bytes'] and h.hexdigest()==target['sha256'],'Destination full SHA differs'
    return {**target,'etag':etag,'full_destination_readback':True}


def transfer(root,item):
    root=Path(root);work=root/'work'/item['identity'];work.mkdir(parents=True,exist_ok=True)
    o=opener();proof=item['proof'];report=root/'progress'/(item['identity']+'.json')
    def state(phase):save(report,{'phase':phase,'source_key':proof['source_key'],'time':time.time(),'training_executed':False})
    try:
        state('source_download')
        download(item['source'],work/'source.tar',o,workers=4)
        state('destination_upload')
        data=item['destinations']['data']
        try: target=readback(o,data,work/'destination.tar')
        except urllib.error.HTTPError as e:
            if e.code!=404:raise
            try:put(o,data['put'],work/'source.tar',data['headers'])
            except urllib.error.HTTPError as e:
                if e.code not in (409,412):raise
            target=readback(o,data,work/'destination.tar')
        state('destination_real_reader')
        unpacked=work/'unpacked'
        if unpacked.exists():shutil.rmtree(unpacked)
        safe_extract(work/'destination.tar',unpacked,max_bytes=16*1024**3)
        reports=list(unpacked.rglob('export_report.json'));assert len(reports)==1
        export=json.loads(reports[0].read_text())
        verification=verify_dataset(reports[0].parent,export['repo_id'],permit_empty_labels=proof['kind']=='full_episode')
        assert verification['episodes']==proof['acceptance']['episodes']
        assert verification['frames']==proof['acceptance']['frames']
        state('index_upload')
        indexfile=work/'index.json';indexfile.write_text(item['index_body'])
        index=item['destinations']['index'];assert sha(indexfile)==index['target']['sha256']
        try: itarget=readback(o,index,work/'index.readback.json')
        except urllib.error.HTTPError as e:
            if e.code!=404:raise
            try:put(o,index['put'],indexfile,index['headers'])
            except urllib.error.HTTPError as e:
                if e.code not in (409,412):raise
            itarget=readback(o,index,work/'index.readback.json')
        receipt={**proof,'target':target,'index_target':itarget,'source_cloud_target':proof['target'],
                 'source_cloud_acceptance':proof['acceptance'],'destination_reader_acceptance':verification,
                 'verified_at':time.time(),'training_executed':False,'transfer_route':'source_cloud_direct_to_OSS'}
        receiptfile=root/'receipts'/(item['identity']+'.json');save(receiptfile,receipt)
        put(o,item['receipt_put'],receiptfile)
        state('accepted');shutil.rmtree(work)
        return item['identity']
    except BaseException as error:
        save(report,{'phase':'needs_diagnostic',**error_detail(error),'source_key':proof['source_key']});raise


def run(root):
    root=Path(root);control=json.loads((root/'private_control.json').read_text());o=opener()
    assert control['training_allowed'] is False
    assert not os.environ.get('CUDA_VISIBLE_DEVICES')
    futures={};done={p.stem for p in (root/'receipts').glob('*.json')};start=time.time();queue={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        while time.time()<control['deadline_epoch']:
            with o.open(control['queue_get'],timeout=60) as r:queue=json.loads(r.read())
            assert queue['fingerprint']==control['fingerprint'] and not queue['training_allowed']
            active=set(futures.values())
            for item in queue['items']:
                if len(futures)>=4:break
                if item['identity'] not in done|active:
                    futures[pool.submit(transfer,str(root),item)]=item['identity'];active.add(item['identity'])
            complete,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
            for f in complete:
                identity=futures.pop(f);assert f.result()==identity;done.add(identity)
            status={'phase':'direct_oss_transfer','accepted':len(done),'active':list(futures.values()),'known_items':len(queue['items']),
                    'time':time.time(),'started_at':start,'training_executed':False}
            p=root/'status.json';save(p,status);put(o,control['status_put'],p)
            if queue.get('all_sources_prepared') and len(done)==len(queue['items']) and not futures:return
            if not futures:time.sleep(15)
    raise RuntimeError('Finite direct transfer deadline reached')


if __name__=='__main__':
    try:run(sys.argv[1])
    except BaseException as e:
        root=Path(sys.argv[1]);p=root/'failure.json';save(p,error_detail(e))
        control=json.loads((root/'private_control.json').read_text())
        try:put(opener(),control['status_put'],p)
        except Exception:pass
        raise SystemExit(1)
