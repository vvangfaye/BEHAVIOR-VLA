"""Read round3 receipts into a separate preparation snapshot; no collection edits."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time
import types

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
REPO=WORKSPACE/'BEHAVIOR-VLAs'
COLLECTION=REPO/'eval_results/sco_full_episode_collection_20261014_final1000_round3'
LEAF=COLLECTION/'round-3'
sys.path.insert(0,str(REPO/'sco'))
from video_eval_adapter import aoss_client
from restore_full_episode_collection import read_plan,prefix_for,private_json


def main():
    (ROOT/'inventory').mkdir(exist_ok=True)
    output=ROOT/'inventory'
    cloud=aoss_client()
    plan=read_plan(LEAF)
    assert plan['expected_episodes']==1000
    started=time.time()
    objects={}
    for page in cloud.get_paginator('list_objects_v2').paginate(Bucket='260010165',Prefix=prefix_for(plan)+'/episodes/'):
        objects.update({m['Key']:m for m in page.get('Contents',[])})
    receipts=[m for key,m in objects.items() if re.search(r'/attempt-1(?:-g[0-9]+)?\.json$',key)]
    pairs=[m for m in receipts if m['Key'][:-5]+'.tar' in objects]
    metadata=[dict(key=m['Key'],etag=m['ETag'],bytes=m['Size'],modified_at=m['LastModified'].timestamp()) for m in pairs]
    observation=dict(observed_at=started,rounds=[dict(root=str(LEAF),round_id=3,
        metadata=metadata,unchanged_rows=[],changed=metadata)])
    private_json(output/'statistics_observation_current.json',observation)
    for kind in ['receipts','rows']:
        target=output/'statistics_cache'/kind
        target.mkdir(parents=True,exist_ok=True)
        for f in (COLLECTION/'statistics_cache'/kind).glob('*.json'):
            if not (target/f.name).exists():
                shutil.copy2(f,target/f.name)
    helper=types.ModuleType('behavior_aoss_client');helper.client=lambda:cloud
    sys.modules['behavior_aoss_client']=helper
    spec=importlib.util.spec_from_file_location('retained_inventory',COLLECTION/'inventory_all_retained.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.CURRENT=output
    print(json.dumps(dict(phase='reading_receipt_metadata',published_pairs=len(pairs),
        receipt_bytes=sum(m['Size'] for m in pairs),archive_bytes=sum(objects[m['Key'][:-5]+'.tar']['Size'] for m in pairs),
        original_collection_untouched=True)),flush=True)
    module.main()
    full=json.loads((output/'all_retained_episode_index_current.json').read_text())
    eligible=[row for row in full['episode_rows'] if row['bc_eligible']]
    archives=[]
    for row in eligible:
        archive=objects[row['object_key']]
        archives.append(dict(id='r3_'+row['task']+'_'+str(row['instance_id']),kind='full_episode',
            key=row['object_key'],sha256=row['archive_sha256'],raw_sha256=row['raw_sha256'],
            bytes=archive['Size'],etag=archive['ETag'],frames=row['steps'],episodes=1,
            task=row['task'],instance_id=row['instance_id'],run_id=row['run_id'],sampling_round=3,
            execution_uid=row['execution_uid'],receipt_key=row['receipt_key'],
            receipt_sha256=row['receipt_sha256']))
    private_json(ROOT/'round3_successful_sources.json',dict(observed_at=started,completed_at=time.time(),
        expected_episodes=1000,published_episodes=len(pairs),bc_eligible=len(eligible),
        eligible_frames=sum(row['steps'] for row in eligible),eligible_tasks=len({row['task'] for row in eligible}),
        source_archives=archives,source_cloud_bucket='260010165',source_cloud_prefix=prefix_for(plan),
        exact_episode_identity_includes_round=True,raw_sources_deleted=False,training_started=False))
    print(json.dumps(dict(phase='round3_inventory_complete',published_episodes=len(pairs),
        bc_eligible=len(eligible),frames=sum(row['steps'] for row in eligible),
        tasks=len({row['task'] for row in eligible}))),flush=True)


if __name__=='__main__':main()
