"""Sign exact destination objects on the credential-owning host; never return keys."""
import configparser
import hashlib
import json
from pathlib import Path
import sys
import oss2


def sign(items):
    cache = Path('/Users/faye/Library/Caches/behavior-data-prep-20261013-v1')
    config = configparser.ConfigParser(interpolation=None); config.read(Path.home()/'.ossutilconfig')
    c = config['Credentials']; network = json.loads((cache/'connectivity.json').read_text())
    assert network['bucket'] == 'traffic-model'
    bucket = oss2.Bucket(oss2.Auth(c['accesskeyid'], c['accesskeysecret']), network['endpoint'], network['bucket'])
    prefix = network['prefix']
    results = []
    for item in items:
        proof = item['proof']; kind = proof['kind']; assert kind in ('subtask','full_episode')
        branch = 'full_episode/exports/' if kind == 'full_episode' else 'subtask/archives/'
        target = {'bucket':network['bucket'],'key':prefix+branch+proof['source_sha256']+'.tar',
                  'bytes':proof['target']['bytes'],'sha256':proof['target']['sha256']}
        index = item['index']; index['data_target'] = target
        data = (json.dumps(index,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        itarget = {'bucket':network['bucket'],'key':prefix+'indexes/'+proof['root_id']+'.json',
                   'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        destinations = {}
        for label,t in [('data',target),('index',itarget)]:
            headers={'Content-Type':'application/octet-stream','x-oss-forbid-overwrite':'true','x-oss-meta-sha256':t['sha256']}
            destinations[label]={'target':t,'put':bucket.sign_url('PUT',t['key'],21600,headers=headers),
                                 'get':bucket.sign_url('GET',t['key'],21600),'headers':headers}
        results.append({**item,'index_body':data.decode(),'destinations':destinations})
    return results


if __name__ == '__main__':print(json.dumps(sign(json.load(sys.stdin))))
