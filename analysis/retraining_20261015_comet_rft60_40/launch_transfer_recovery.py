"""One manual, finite zero-GPU recovery for the66 frozen compact exports."""
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
PREP=HERE.parent/'retraining_20261015_round3_v2'
sys.path.insert(0,str(PREP))
os.environ['BEHAVIOR_PREP_SIGNING_SSH']='macpro'
import transfer_pipeline as old


def main():
    existing=HERE/'transfer_recovery/transfer_launch.json'
    if existing.exists():
        record=json.loads(existing.read_text())
        print(json.dumps(dict(existing_recovery_only=True,observation=old.describe(record))))
        return
    record=json.loads((PREP/'transfer_launch.json').read_text())
    old_observation=old.describe(record)
    assert old_observation['uid']=='b41c7100a14643e6ae244b693cc19596' and old_observation['state']=='FAILED'
    assert not (PREP/'STOP').exists() and not (HERE/'STOP').exists()
    frozen=json.loads((PREP/'source_archives.json').read_text())
    public=old.aoss_client();internal=old.internal_client(public)
    # Reconcile any package accepted by the stopped Mac writer before freezing the continuation.
    for key in old.listed(public,old.PREFIX+'receipts/'):
        proof=old.cloud_json(public,key)
        assert proof['accepted'] and proof['target']['bucket']=='traffic-model'
        identity=old.hashlib.sha256(proof['source_key'].encode()).hexdigest()[:20]
        old.save(PREP/'destination_receipts'/(identity+'.json'),proof)
    accepted={r['source_key'] for r in map(lambda p:json.loads(p.read_text()),(PREP/'destination_receipts').glob('*.json'))}
    proofs=[r for r in map(lambda p:json.loads(p.read_text()),(PREP/'cloud_receipts').glob('*.json')) if r['source_key'] not in accepted]
    assert 0<len(proofs)<=66 and len(proofs)+len(accepted)==82
    root=HERE/'transfer_recovery';root.mkdir(exist_ok=True)
    (root/'private').mkdir(exist_ok=True);(root/'private').chmod(0o700)
    deadline=min(json.loads((PREP/'pipeline_launch.json').read_text())['deadline_epoch'],time.time()+7200)
    prefix='behavior1k/retraining_20261015_round3_v2/batch-0/direct-transfer-g1/'
    old.save(root/'authorization.json',dict(human_request='macpro可以用了，看看怎么启动train吧',
        purpose='Finish previously authorized data preparation before training',old_task=old_observation,
        manual_recovery_generation=1,source_count=len(proofs),prior_accepted=len(accepted),source_fingerprint=frozen['fingerprint'],
        maximum_jobs=1,gpu_count=0,cpu_count=8,memory_gib=32,quota_type='SPOT',workspace='share-space',
        deadline_epoch=deadline,maximum_seconds=7200,automatic_resubmission=False,training_allowed=False,
        preserve_prior16=True,source_and_destination_objects_immutable=True))
    with ThreadPoolExecutor(max_workers=4) as pool:items=list(pool.map(lambda p:old.unsigned(p,public,internal),proofs))
    for item in items:
        item['receipt_put']=internal.generate_presigned_url('put_object',Params={'Bucket':'260010165',
            'Key':prefix+'receipts/'+item['identity']+'.json'},ExpiresIn=129600)
    items=old.remote(dict(operation='sign',items=items))
    queue=dict(fingerprint=frozen['fingerprint'],training_allowed=False,all_sources_prepared=True,items=items)
    old.save(root/'private/transfer_queue.json',queue)
    public.put_object(Bucket='260010165',Key=prefix+'queue.json',Body=json.dumps(queue).encode(),IfNoneMatch='*')
    # Reuse the reviewed CPU bootstrap and exact resource shape, with a fresh generation identity.
    source=inspect.getsource(old.launch_transfer)
    source=source.replace("'b1k-r3-fdf69560-direct-g0'","'b1k-r3-fdf69560-direct-g1'")
    source=source.replace("/batch-0/direct-transfer'","/batch-0/direct-transfer-g1'")
    source=source.replace('86400','7200')
    namespace=dict(vars(old));namespace.update(ROOT=root,PREFIX=prefix)
    exec(compile(source,str(__file__),'exec'),namespace)
    result=namespace['launch_transfer'](public,internal,deadline,frozen['fingerprint'])
    print(json.dumps({k:result[k] for k in ['name','uid','state','gpu_count','cpu_count','memory_gib','deadline_epoch']}))


if __name__=='__main__':main()
