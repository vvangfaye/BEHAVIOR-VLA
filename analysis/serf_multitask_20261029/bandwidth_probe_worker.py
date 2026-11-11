import json,time,urllib.request
from pathlib import Path


def run(config):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result={'complete':False,'kind':'public checkpoint range throughput probe'}
    started=time.monotonic()
    try:
        request=urllib.request.Request(config['url'],headers={'Range':'bytes=0-8388607'})
        count=0
        with opener.open(request,timeout=15) as response:
            while count<8388608:
                block=response.read(min(1024**2,8388608-count))
                if not block:break
                count+=len(block)
                if time.monotonic()-started>70:break
        elapsed=time.monotonic()-started
        result.update(complete=True,bytes=count,seconds=elapsed,MiB_per_second=count/2**20/elapsed)
    except Exception as e:result.update(error=type(e).__name__+': '+str(e)[:180])
    data=json.dumps(result).encode()
    with opener.open(urllib.request.Request(config['put'],data=data,method='PUT'),timeout=30):pass
    print(json.dumps(result),flush=True)
