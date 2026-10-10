from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

import pytest

from submit_public_test_shard import build_payload, validate_capacity, POOL


def fixtures():
    root=Path(__file__).resolve().parents[1]/'eval_results/sco_test2000_1688977_20261010'
    ready=json.loads((root/'preparation_ready.json').read_text())
    plan=json.loads((root/'production_plan.json').read_text())
    # Pure scheduling tests use synthetic acceptance; no real pilot success is asserted.
    pilot=dict(state='accepted',checkpoint_step=1688977,platform_state='SUCCEEDED',
        full_artifact_sha256_verified=True,source_manifest_sha256=ready['source_manifest_sha256'],
        driver_sha256='fixture',task='turning_on_radio',instance_ids=[301],
        whole_task_native_generated_subtasks=True,video_decoded_and_reviewed=True,
        canonical_result_kept_regardless_of_score=True)
    return ready,plan,pilot


def test_full_assignment_union_preserves_fixed_pilot():
    ready,plan,pilot=fixtures()
    payloads=[build_payload(ready,plan,pilot,index,'fixture','fixture') for index in range(400)]
    keys=[(p['task'],i) for p in payloads for i in p['instance_ids']]
    assert len(keys)==1999 and len(set(keys))==1999
    assert ('turning_on_radio',301) not in keys
    assert set(keys)|{('turning_on_radio',301)}=={(task,i) for task in ready['scope']['tasks'] for i in range(301,321)}


def test_expansion_requires_real_accepted_pilot():
    ready,plan,pilot=fixtures();pilot['platform_state']='STARTING'
    with pytest.raises(AssertionError):build_payload(ready,plan,pilot,0,'fixture','fixture')


def test_other_checkpoint_pilot_rejected():
    ready,plan,pilot=fixtures();pilot['checkpoint_step']=1375000
    with pytest.raises(AssertionError):build_payload(ready,plan,pilot,0,'fixture','fixture')


def test_stale_standard_and_zero_capacity_cannot_admit_spot_jobs():
    now=datetime.now(timezone.utc)
    valid=dict(resource_pool=POOL,quota_type='spot',fresh_query=True,observed_at=now.isoformat(),
        idle_gpus=1,idle_vcpu=14,idle_memory_gib=60)
    validate_capacity(valid,now=now)
    for bad in [dict(valid,idle_gpus=0),dict(valid,quota_type='reserved'),
                dict(valid,observed_at=(now-timedelta(minutes=6)).isoformat())]:
        with pytest.raises(AssertionError):validate_capacity(bad,now=now)


def test_spot_slot_count_respects_cpu_memory_and_expired_auth():
    now=datetime.now(timezone.utc)
    valid=dict(resource_pool=POOL,quota_type='spot',fresh_query=True,observed_at=now.isoformat(),
        idle_gpus=10,idle_vcpu=140,idle_memory_gib=600)
    assert validate_capacity(valid,now=now)==10
    assert validate_capacity(dict(valid,idle_vcpu=28),now=now)==2
    assert validate_capacity(dict(valid,idle_memory_gib=180),now=now)==3
    for bad in [dict(valid,idle_vcpu=13),dict(valid,idle_memory_gib=59),
                dict(valid,idle_vcpu=float('inf')),dict(valid,fresh_query=False),
                dict(valid,idle_gpus=True)]:
        with pytest.raises(AssertionError):validate_capacity(bad,now=now)
