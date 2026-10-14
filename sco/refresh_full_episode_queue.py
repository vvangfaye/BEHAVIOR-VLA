"""Refresh the shared, finite five-slot Spot queue observation; never create jobs."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from full_episode_collection import save
from free_capacity_collection import pending_requests
from submit_full_collection_shard import describe, environment
from bounded_collection_admission import ACTIVE_STATES


def refresh(campaign, cli):
    with (Path(campaign) / '.queue_observation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _refresh(campaign, cli)


def _refresh(campaign, cli):
    campaign = Path(campaign)
    auth = json.loads((campaign / 'authorization.json').read_text())
    from full_episode_campaign_scope import effective_authorization
    auth = effective_authorization(campaign, auth)
    policy = json.loads((campaign / 'queue_policy.json').read_text())
    assert policy['roots'] == auth['roots']
    assert policy['queue_authorization'] == 'user_20261012_max_five_queued'
    assert policy['max_pending_tasks'] == 5
    assert policy['original_deadline_epoch'] == auth['deadline_epoch']
    observed_at = time.time()
    assert observed_at < auth['deadline_epoch']
    for root in [campaign] + [Path(r) for r in auth['roots']]:
        assert not (root / 'STOP').exists(), 'Respect STOP; do not refresh admission'
    records = {}
    for root in auth['roots']:
        for path in (Path(root) / 'submissions').glob('*.json'):
            row = json.loads(path.read_text())
            assert row.get('uid'), 'Reconcile unknown intent first'
            assert row['name'] not in records
            records[row['name']] = row
    pending = pending_requests(cli, environment())
    campaign_only = policy.get('queue_scope') == 'authorized_campaign_only'
    if campaign_only:
        pending = {name: value for name, value in pending.items() if name in records}
        # A newly created request may not yet appear in list results. Release
        # its reservation only after exact UID proof of RUNNING or termination.
        old_path = campaign / 'queue_admission.json'
        old = json.loads(old_path.read_text()) if old_path.exists() else {}
        unresolved = set(old.get('pending_requests', {})) & records.keys()
        unresolved.update(name for name, row in records.items()
                          if row.get('created_at', float('inf')) >= old.get('observed_at', 0))
        for name in sorted(unresolved - pending.keys()):
            job = describe(cli, environment(), name)
            assert job['uid'] == records[name]['uid']
            if job['state'] in ACTIVE_STATES and job['state'] != 'RUNNING':
                pending[name] = 1
            else:
                assert job['state'] in ('RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED',
                                        'STOPPED', 'DELETED'), 'Unknown state cannot release a slot'
    observation = dict(observed_at=observed_at, query_completed_at=time.time(),
        workspace='share-space', resource_pool='computing-cluster-5090-01g', quota_type='spot',
        admission_mode='bounded_spot_queue', queue_authorization=policy['queue_authorization'],
        max_pending_tasks=5, pending_requests=pending, deadline_epoch=auth['deadline_epoch'],
        monitor_interval_minutes=policy.get('controller_poll_seconds', 600) / 60, free_capacity_required=False,
        queue_scope=policy.get('queue_scope', 'owner_target_pool'),
        controller_poll_seconds=policy.get('controller_poll_seconds', 30))
    # Timestamp the start, so every creation racing this multi-state query is
    # charged by the three-round ledger, even if it is not visible in the list.
    save(campaign / 'queue_admission.json', observation)
    return observation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--cli', default='/private/tmp/behavior-sco-runtime/bin/sco')
    args = parser.parse_args()
    print(json.dumps(refresh(args.campaign, args.cli)))
