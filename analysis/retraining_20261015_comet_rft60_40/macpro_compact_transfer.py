"""Finite CPU transfer of already converted data; no training or cloud jobs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import ssl
import sys
import time
import urllib.request
import urllib.parse
import configparser
import truststore

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),'/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/lerobot/src']
import direct_oss_worker as worker
from prepare_retraining_data import save,error_detail


def native_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)))


def one(item,deadline):
    if time.time()>=deadline or (ROOT/'STOP').exists():
        raise RuntimeError('Finite transfer deadline or human STOP')
    return worker.transfer(str(ROOT),item)


def main():
    worker.opener=native_opener
    # Use the previously configured OSS SDK for file-body PUTs on macOS.
    # Signed AOSS receipt publication still uses the bounded original uploader.
    import oss2
    config=configparser.ConfigParser(interpolation=None)
    config.read(Path.home()/'.ossutilconfig');credentials=config['Credentials']
    bucket=oss2.Bucket(oss2.Auth(credentials['accesskeyid'],credentials['accesskeysecret']),
        'https://oss-cn-zhangjiakou.aliyuncs.com','traffic-model',connect_timeout=30)
    original_put=worker.put
    def put(o,url,path,headers=None):
        parsed=urllib.parse.urlparse(url)
        if parsed.hostname=='traffic-model.oss-cn-zhangjiakou.aliyuncs.com':
            key=urllib.parse.unquote(parsed.path.lstrip('/'))
            assert key.startswith('wangfaye/embodied/datasets/behavior-1k/retrain_20261015_v2/')
            bucket.put_object_from_file(key,str(path),headers=headers or {})
        else:original_put(o,url,path,headers)
    worker.put=put
    queue=json.loads((ROOT/'private_queue.json').read_text())
    assert not queue['training_allowed'] and len(queue['items'])==66
    deadline=queue['deadline_epoch']
    with (ROOT/'lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (ROOT/'receipts').mkdir(exist_ok=True)
        done={p.stem for p in (ROOT/'receipts').glob('*.json')}
        items=[item for item in queue['items'] if item['identity'] not in done]
        save(ROOT/'status.json',dict(phase='compact_export_transfer',accepted=len(done),expected=66,
             original_prior_accepted=16,pid=os.getpid(),started_at=time.time(),training_started=False))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(one,item,deadline):item['identity'] for item in items}
            for future in as_completed(futures):
                done.add(future.result())
                save(ROOT/'status.json',dict(phase='compact_export_transfer',accepted=len(done),expected=66,
                     original_prior_accepted=16,pid=os.getpid(),updated_at=time.time(),training_started=False))
        assert len(done)==66
        save(ROOT/'status.json',dict(phase='all_66_compact_exports_verified',accepted=66,expected=66,
             original_prior_accepted=16,pid=os.getpid(),completed_at=time.time(),training_started=False))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(ROOT/'failure.json',dict(error_detail(error),training_started=False));raise
