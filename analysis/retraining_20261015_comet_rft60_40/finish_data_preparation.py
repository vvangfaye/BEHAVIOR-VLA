"""Finite receipt reconciliation/publication only. Cannot submit or train a model."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PREP=HERE.parent/'retraining_20261015_round3_v2'
sys.path.insert(0,str(PREP))
os.environ['BEHAVIOR_PREP_SIGNING_SSH']='macpro'
import transfer_pipeline as old


def publish_two(manifest):
    code='''import configparser,hashlib,json,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import oss2
d=json.load(sys.stdin)
assert d['kind']=='b1k_two_source_data_preparation' and not d['training_allowed']
assert len(d['archives'])==270 and all(r['kind']=='full_episode' and r['accepted'] for r in d['archives'])
c=configparser.ConfigParser(interpolation=None);c.read(Path.home()/'.ossutilconfig');v=c['Credentials']
b=oss2.Bucket(oss2.Auth(v['accesskeyid'],v['accesskeysecret']),'https://oss-cn-zhangjiakou.aliyuncs.com','traffic-model',connect_timeout=30)
prefix='wangfaye/embodied/datasets/behavior-1k/retrain_20261015_v2/two_source_60_40/'
def put(name,value):
 body=(json.dumps(value,indent=2,ensure_ascii=False)+'\\n').encode();digest=hashlib.sha256(body).hexdigest();key=prefix+name
 try:b.put_object(key,body,headers={'x-oss-forbid-overwrite':'true','x-oss-meta-sha256':digest})
 except oss2.exceptions.ServerError as e:
  if e.status not in (409,412):raise
 r=b.get_object(key)
 try:assert r.read()==body
 finally:r.resp.response.close()
 return dict(bucket='traffic-model',key=key,bytes=len(body),sha256=digest,full_destination_readback=True)
def verify(target):
 assert b.head_object(target['key']).content_length==target['bytes']
with ThreadPoolExecutor(max_workers=8) as pool:
 list(pool.map(verify,[target for r in d['archives'] for target in [r['target'],r['index_target']]]))
target=put('data_preparation_manifest.json',d)
marker=put('_DATA_READY.json',dict(data_ready=True,manifest=target,training_allowed=False,training_executed=False,training_integration_ready=False))
print(json.dumps(dict(data_ready=True,manifest=target,marker=marker,training_allowed=False)))
'''
    args=['ssh','-S','none','-o','BatchMode=yes','-o','ConnectTimeout=8','macpro',
          shlex.join(['/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/env/bin/python','-c',code])]
    cp=subprocess.run(args,input=json.dumps(manifest),capture_output=True,text=True,timeout=240)
    if cp.returncode:raise RuntimeError('Two-source metadata publication failed; preserve exact bytes for retry')
    return json.loads(cp.stdout)


def run():
    recovery=HERE/'transfer_recovery'
    deadline=json.loads((recovery/'authorization.json').read_text())['deadline_epoch']
    prefix='behavior1k/retraining_20261015_round3_v2/batch-0/direct-transfer-g1/'
    client=old.aoss_client();seen=set();accepted={}
    with (HERE/'data_finalizer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while time.time()<deadline:
            if (HERE/'STOP').exists() or (PREP/'STOP').exists():raise RuntimeError('Human STOP')
            for base in [old.PREFIX,prefix]:
                for key in old.listed(client,base+'receipts/'):
                    if key in seen:continue
                    proof=old.cloud_json(client,key)
                    assert proof['accepted'] and proof['target']['full_destination_readback']
                    old.checked_receipts(json.loads((PREP/'source_archives.json').read_text())['source_archives'],[proof],require_complete=False)
                    accepted[proof['source_key']]=proof;seen.add(key)
                    identity=hashlib.sha256(proof['source_key'].encode()).hexdigest()[:20]
                    old.save(PREP/'destination_receipts'/(identity+'.json'),proof)
            status=old.optional_json(client,prefix+'status.json') or {}
            old.save(HERE/'data_progress.json',dict(observed_at=time.time(),destination_new_accepted=len(accepted),
                full_training_OSS_accepted=188+len(accepted),expected_full=270,
                worker_phase=status.get('phase'),worker_error_type=status.get('error_type'),
                data_ready=False,training_started=False,deadline_epoch=deadline))
            if status.get('phase')=='bootstrap_needs_diagnostic' or status.get('error_type'):
                old.save(HERE/'data_finalizer_failure.json',dict(observed_at=time.time(),worker=status,training_started=False))
                raise RuntimeError('CPU continuation needs diagnosis; no automatic recreation')
            if len(accepted)==82:break
            time.sleep(30)
        assert len(accepted)==82,'Finite data deadline reached'
        items={}
        for path in [PREP/'private/transfer_queue.json',HERE/'private/macpro_queue.json',recovery/'private/transfer_queue.json']:
            if path.exists():items.update({x['identity']:x for x in json.loads(path.read_text())['items']})
        frozen=json.loads((PREP/'source_archives.json').read_text())
        parent_result=old.finalize(frozen,accepted,items)
        full_manifest=json.loads((PREP/'data_preparation_manifest.json').read_text())
        full=[r for r in full_manifest['archives'] if r['kind']=='full_episode'];assert len(full)==270
        selection=json.loads((HERE/'selected_original_expert.json').read_text())
        stable=dict(original_index_sha256=selection['index_sha256'],
                    collected=[dict(root_id=r['root_id'],export_sha256=r['target']['sha256'],index_sha256=r['index_target']['sha256']) for r in sorted(full,key=lambda r:r['root_id'])])
        manifest_path=HERE/'two_source_data_manifest.json'
        if manifest_path.exists():manifest=json.loads(manifest_path.read_text())
        else:
            manifest=dict(kind='b1k_two_source_data_preparation',version=1,created_at=time.time(),data_ready=True,
                training_allowed=False,training_executed=False,training_integration_ready=False,
                source_fingerprint=hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                archives=full,missing_archives=[],duplicate_collected_content=[],
                original_expert=dict(full_manifest['original_expert'],recovered_episodes=0,
                    selected_replay_episodes=15949,metadata_overlay_contains_unused_recovered_records=True),
                held_out_task_instances=full_manifest['held_out_task_instances'],
                collected_data=dict(full_episode=full_manifest['collected_data']['full_episode']),
                selected_sources=['replay','full_episode'],excluded_sources=['recovered','collected_subtask'],
                selected_episode_counts=dict(replay=15949,full_episode=270),
                selected_full_frames=2529538,selected_full_tasks=52,
                parent_accepted_manifest=parent_result['manifest'],original_selection=selection['index_sha256'],
                supervision=dict(replay='original human labels and actions',full_episode='action only;generated subtask CE zero'))
            old.save(manifest_path,manifest)
        result=publish_two(manifest)
        assert result['manifest']['sha256']==hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        old.save(HERE/'_TWO_SOURCE_DATA_READY.json',result)
        old.save(HERE/'launch_bundle/inputs/data_manifest.json',manifest)
        job=json.loads((HERE/'launch_bundle/nebula/worker_job.json').read_text())
        job.update(data_manifest_key=result['manifest']['key'],data_manifest_sha256=result['manifest']['sha256'])
        old.save(HERE/'launch_bundle/nebula/worker_job.json',job)
        old.save(HERE/'data_progress.json',dict(observed_at=time.time(),destination_new_accepted=82,
            full_training_OSS_accepted=270,expected_full=270,data_ready=True,publication=result,training_started=False,
            real_decoded_CPU_batch_verified=False,GPU_smoke_verified=False))
        handoff=json.loads((HERE/'HANDOFF.json').read_text())
        handoff.update(updated_at=time.time(),data_ready=True,training_oss_accepted_full_episodes=270,
            pending_transfer_full_episodes=0,training_integration_ready=False,training_allowed=False,
            latest_data_publication=result,existing_training_modified=False)
        old.save(HERE/'HANDOFF.json',handoff)
        print(json.dumps(dict(data_ready=True,full_successes=270,training_started=False)))


if __name__=='__main__':
    try:run()
    except BaseException as e:
        old.save(HERE/'data_finalizer_error.json',dict(error_type=type(e).__name__,observed_at=time.time(),training_started=False));raise
