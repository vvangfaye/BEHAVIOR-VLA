"""Recognize complete task checkpoints without waiting for unrelated tasks."""
import hashlib
import json
from pathlib import Path


def verified_checkpoints(client,root):
    manifest=json.loads((root/'checkpoint_manifest.json').read_text())['items']
    prefix='behavior1k/serf-multitask-20261029/assets/'
    expected={prefix+i['destination']:i for i in manifest}
    path=root/'checkpoint_inventory.json'
    cached=json.loads(path.read_text()).get('uploaded',[]) if path.exists() else []
    found={i['key']:i for i in cached};cursor=None
    while True:
        args={'Bucket':'260010165','Prefix':prefix+'checkpoints/'}
        if cursor:args['ContinuationToken']=cursor
        page=client.list_objects_v2(**args)
        for obj in page.get('Contents',[]):
            key=obj['Key']
            if key in found or key not in expected:continue
            item=expected[key]
            assert obj['Size']==item['size'], 'Checkpoint size mismatch: '+item['destination']
            h=client.head_object(Bucket='260010165',Key=key);sha=h.get('Metadata',{}).get('sha256')
            assert sha and len(sha)==64
            if item.get('sha256'):assert sha==item['sha256']
            else:
                data=client.get_object(Bucket='260010165',Key=key)['Body'].read()
                assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==item['git_blob_sha1']
                assert hashlib.sha256(data).hexdigest()==sha
            found[key]={'key':key,'destination':item['destination'],'size':item['size'],'sha256':sha}
        if not page.get('IsTruncated'):break
        cursor=page['NextContinuationToken']
    result={'complete':len(found)==len(expected),'uploaded':list(found.values()),'tasks_ready':[]}
    for tid in [21,26]:
        keys={key for key,item in expected.items() if f'task-{tid:04d}/' in item['destination']}
        if keys<=set(found):result['tasks_ready'].append(tid)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
    return result
