import copy

import inventory_full_episode_campaign as campaign
from test_restore_full_episode_collection import collection


def test_stopped_or_partial_shards_keep_published_episodes(collection):
    root,plan,client=collection
    # Missing whole-shard completion must not hide its fully archived episode.
    del client.objects[campaign.prefix_for(plan)+'/shards/0001.json']
    result=campaign.scan_round(root,2,client)
    assert result['episodes']==2 and result['bc_eligible']==1
    assert all(v['sampling_round']==2 for v in result['episode_rows'])
    assert not any(v['export_readback_accepted'] for v in result['episode_rows'])


def test_truncated_success_never_enters_default_bc():
    receipt=dict(success=True,result=dict(terminated=True,truncated=True))
    assert campaign.bc_eligible(receipt) is False
    receipt['result']['truncated']=False
    assert campaign.bc_eligible(receipt) is True


def test_same_public_instance_in_different_rounds_is_retained():
    first=dict(run_id='first',sampling_round=0,task='task',instance_id=301,rollout_id=0)
    repeat=dict(first,run_id='repeat',sampling_round=1)
    assert len({campaign.episode_identity(first),campaign.episode_identity(repeat)})==2
