"""Recover the small original reports without waiting for the multi-GB tensors."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

required={'gate.json','status.json','gpu_fp32_probe.json','handoff_receipt.json',
          'jax/device.json','jax/301/result.json','jax/302/result.json',
          'ppo/result.json'}
client=aoss_client()
key='behavior1k/rlinf-adapter-20261023/b1k-rlinf-1023-r07/result.tar.gz'
saved={}
try:
    head=client.head_object(Bucket='260010165',Key=key)
    response=client.get_object(Bucket='260010165',Key=key,Range='bytes=0-67108863',IfMatch=head['ETag'])
    body=response['Body']
    try:
        with tarfile.open(fileobj=body,mode='r|gz') as archive:
            for member in archive:
                rel=Path(member.name)
                assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
                if str(rel) in required:
                    data=archive.extractfile(member).read()
                    json.loads(data)
                    target=HERE/'retrieved'/rel
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():assert target.read_bytes()==data
                    else:target.write_bytes(data)
                    saved[str(rel)]=hashlib.sha256(data).hexdigest()
                if required.issubset(saved):break
                if member.size>64*1024**2:break
    finally:
        body.close()
    assert required.issubset(saved), f'Reports not all reached in prefix: {required-set(saved)}'
    result={'passed':True,'source_bucket':'260010165','source_key':key,
            'archive_bytes':head['ContentLength'],'etag':head['ETag'],
            'reports_sha256':saved,'tensor_download':'Continuing separately in monitor.py'}
    (HERE/'reports_retrieval.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'reports':len(saved),'archive_bytes':head['ContentLength']}))
finally:
    client.close()
