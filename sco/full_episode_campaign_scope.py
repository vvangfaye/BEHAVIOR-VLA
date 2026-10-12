"""Validate the user's explicit reduction to the original frozen 2000 episodes."""
import hashlib
import json
from pathlib import Path


def effective_authorization(campaign, original):
    path=Path(campaign)/'active_scope.json'
    if not path.exists():return original
    scope=json.loads(path.read_text())
    assert scope['kind']=='user_reduced_original_round_2000'
    assert scope['original_authorization_sha256']==hashlib.sha256((Path(campaign)/'authorization.json').read_bytes()).hexdigest()
    assert scope['roots']==original['roots'][:1] and scope['fingerprints']==original['fingerprints'][:1]
    assert scope['total_episodes']==2000 and scope['total_shards']==401
    assert scope['deadline_epoch']==original['deadline_epoch']
    assert scope['failure_recovery_enabled'] is False
    return dict(original,roots=scope['roots'],fingerprints=scope['fingerprints'],rounds=[0],
                total_episodes=2000,additional_episodes=0,total_shards=401)
