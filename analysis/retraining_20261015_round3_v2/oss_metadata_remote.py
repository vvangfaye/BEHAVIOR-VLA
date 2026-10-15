"""Runs on macpro: scoped OSS metadata/signing; permanent keys never leave host."""
from concurrent.futures import ThreadPoolExecutor
import configparser
import hashlib
import json
from pathlib import Path
import sys
import time
import oss2

PREFIX='wangfaye/embodied/datasets/behavior-1k/retrain_20261015_v2/'
PARENT='wangfaye/embodied/datasets/behavior-1k/retrain_20261013_v1/'
PARENT_SHA='bcfef52fd6e8be6892c409161e01e2e773aa2bcee48702544124042dc86b6357'

def encoded(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()

def main(request):
    config=configparser.ConfigParser(interpolation=None)
    config.read(Path.home()/'.ossutilconfig');c=config['Credentials']
    bucket=oss2.Bucket(oss2.Auth(c['accesskeyid'],c['accesskeysecret']),
                       'https://oss-cn-zhangjiakou.aliyuncs.com','traffic-model')
    def read(key,maximum=16*1024**2):
        response=bucket.get_object(key)
        try:
            body=response.read(maximum+1)
            assert len(body)<=maximum,'Metadata size bound exceeded'
            return body
        finally:response.resp.response.close()
    def put_new(key,body):
        assert key.startswith(PREFIX)
        digest=hashlib.sha256(body).hexdigest()
        try:
            bucket.put_object(key,body,headers={'x-oss-forbid-overwrite':'true','x-oss-meta-sha256':digest})
        except oss2.exceptions.ServerError as error:
            if error.status not in (409,412):raise
        assert read(key)==body,'Destination metadata differs'
        head=bucket.head_object(key)
        return dict(bucket='traffic-model',key=key,bytes=len(body),sha256=digest,
                    etag=head.etag,full_destination_readback=True)
    if request['operation']=='sign':
        result=[]
        for item in request['items']:
            proof=item['proof'];assert proof['kind']=='full_episode' and proof['accepted']
            target=dict(bucket='traffic-model',key=PREFIX+'full_episode/exports/'+proof['source_sha256']+'.tar',
                        bytes=proof['target']['bytes'],sha256=proof['target']['sha256'])
            index=item['index'];index['data_target']=target
            body=encoded(index)
            index_target=dict(bucket='traffic-model',key=PREFIX+'indexes/'+proof['root_id']+'.json',
                              bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
            destinations={}
            for label,t in [('data',target),('index',index_target)]:
                headers={'Content-Type':'application/octet-stream','x-oss-forbid-overwrite':'true','x-oss-meta-sha256':t['sha256']}
                destinations[label]=dict(target=t,put=bucket.sign_url('PUT',t['key'],129600,headers=headers),
                                         get=bucket.sign_url('GET',t['key'],129600),headers=headers)
            result.append(dict(item,index_body=body.decode(),destinations=destinations))
        return result
    parent=read(PARENT+'data_preparation_manifest.json')
    assert hashlib.sha256(parent).hexdigest()==PARENT_SHA,'Accepted parent manifest changed'
    old=json.loads(parent)
    if request['operation']=='old_content':
        def one(receipt):
            target=receipt['index_target'];assert target['key'].startswith(PARENT+'indexes/')
            body=read(target['key']);assert hashlib.sha256(body).hexdigest()==target['sha256']
            index=json.loads(body)
            selected=index.get('action_only_episodes') if 'action_only_episodes' in index else (index.get('index') or {}).get('episodes',[])
            return dict(key=target['key'],episodes=[{k:e[k] for k in ['task_name','task_instance_id','content_sha256']} for e in selected if e.get('valid_anchors',1)>0])
        with ThreadPoolExecutor(max_workers=12) as pool:items=list(pool.map(one,old['archives']))
        return dict(parent_sha256=PARENT_SHA,observed_at=time.time(),indexes=items)
    assert request['operation']=='publish'
    manifest=request['manifest']
    assert manifest['data_ready'] and manifest['parent_manifest']['sha256']==PARENT_SHA
    assert len(manifest['archives'])==len(old['archives'])+82
    assert manifest['archives'][:len(old['archives'])]==old['archives'],'Parent archive references changed'
    for receipt in manifest['archives'][len(old['archives']):]:
        for name in ['target','index_target']:
            target=receipt[name];assert target['key'].startswith(PREFIX)
            head=bucket.head_object(target['key'])
            assert head.content_length==target['bytes']
            assert head.headers.get('x-oss-meta-sha256')==target['sha256']
    target=put_new(PREFIX+'data_preparation_manifest.json',encoded(manifest))
    ready=dict(manifest=target,data_ready=True,training_allowed=False,training_executed=False)
    marker=put_new(PREFIX+'_DATA_READY.json',encoded(ready))
    return dict(manifest=target,marker=marker,data_ready=True)

if __name__=='__main__':
    print(json.dumps(main(json.load(sys.stdin)),allow_nan=False))
