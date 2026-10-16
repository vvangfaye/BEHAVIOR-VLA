"""One bounded observation of the exact expert-control smoke, without job mutation."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'smoke/bundle'))
import dispatch_support as dispatch
from nebula.submit_oss import OSSClient
from verify_smoke import verify_smoke


def main():
    receipt=dispatch.read(ROOT/'smoke/receipt.json')
    assert receipt['identifiers']==['240212']
    assert receipt['run_id']=='b1k-expert-control-1688977-20261016-smoke'
    client=OSSClient();prefix=client.key(receipt['output'])
    observation={'observed_at':time.time(),'status':dispatch.status('240212'),
        'cloud_read_only':True,'artifacts':{},'failures':{},'accepted':False}
    for suffix in ('','rank-0/','rank-1/'):
        page=client.bucket.list_objects(prefix=prefix+'/'+suffix,delimiter='/',max_keys=100)
        for item in page.object_list:
            name=item.key[len(prefix)+1:]
            observation['artifacts'][name]=item.size
            if name.endswith('failure.json') or name.startswith('transport_failure_'):
                if item.size>100000:raise ValueError('Unexpected failure report size')
                observation['failures'][name]=json.loads(client.read(item.key))
    if '_SMOKE_SUCCESS.json' in observation['artifacts']:
        proof=verify_smoke(client,prefix)
        dispatch.save(ROOT/'smoke/verified_gpu_smoke.json',proof)
        observation['accepted']=True
    dispatch.save(ROOT/'smoke/latest_observation.json',observation)
    print(json.dumps(observation))


if __name__=='__main__':main()
