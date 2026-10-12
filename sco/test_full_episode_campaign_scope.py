import hashlib,json
import pytest
from full_episode_campaign_scope import effective_authorization


def test_reduction_uses_only_original_round_and_preserves_deadline(tmp_path):
    original=dict(roots=['/first','/repeat1','/repeat2'],fingerprints=['a','b','c'],deadline_epoch=123,total_episodes=6000)
    p=tmp_path/'authorization.json';p.write_text(json.dumps(original))
    scope=dict(kind='user_reduced_original_round_2000',original_authorization_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),roots=['/first'],fingerprints=['a'],deadline_epoch=123,total_episodes=2000,total_shards=401,failure_recovery_enabled=False)
    s=tmp_path/'active_scope.json';s.write_text(json.dumps(scope));auth=effective_authorization(tmp_path,original)
    assert auth['roots']==['/first'] and auth['total_episodes']==2000 and auth['total_shards']==401
    assert original['roots']==['/first','/repeat1','/repeat2']
    scope['deadline_epoch']=124;s.write_text(json.dumps(scope))
    with pytest.raises(AssertionError):effective_authorization(tmp_path,original)
