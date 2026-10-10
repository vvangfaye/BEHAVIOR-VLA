"""Fresh owner-wide standard admission for reusing the one authorized pilot GPU."""
from datetime import datetime, timezone
import json
import subprocess


OWNER = '260010165'
ACTIVE_STATES = {'WAITING', 'INIT', 'QUEUEING', 'CREATING', 'STARTING', 'RUNNING',
                 'RESTARTING', 'SUSPENDING', 'DELETING', 'RECOVERING'}


def standard_counts(jobs):
    active = []
    identities = set()
    for job in jobs:
        assert job['ownership']['user_name'] == OWNER
        assert job['uid'] not in identities, 'Pagination repeated a job; refresh instead of undercounting'
        identities.add(job['uid'])
        if job.get('state') not in ACTIVE_STATES or job.get('scheduling', {}).get('quota_type') != 'RESERVED':
            continue
        count = 0
        for role in job.get('roles', []):
            specs = role.get('resource_spec', [])
            if isinstance(specs, dict):
                specs = [specs]
            for spec in specs:
                replicas = max(1, int(spec.get('replicas', 1)))
                if len(specs) == 1:
                    replicas = max(replicas, int(role.get('total_replicas', 1)))
                count += int(spec.get('limits', {}).get('nvidia.com/gpu', 0)) * replicas
        if count:
            active.append(dict(name=job['name'], uid=job['uid'], state=job['state'], gpu_count=count))
    return dict(active_reserved=active, used_gpus=sum(x['gpu_count'] for x in active),
                latest_goal_active_gpus=sum(x['gpu_count'] for x in active
                    if x['name'].startswith('behavior-public1688977-')))


def query_standard_capacity(cli, env):
    # Historical startup scripts can exceed the service's 4 MiB message limit.
    # Query every active state with bounded pagination instead of truncating history.
    jobs = {}
    pages_read = 0
    states = ['WAITING', 'INIT', 'QUEUEING', 'CREATING', 'STARTING',
              'RESTARTING', 'RECOVERING', 'RUNNING', 'SUSPENDING', 'DELETING']
    for state in states:
        state_identities = set()
        for page in range(1, 21):
            reply = subprocess.run([cli, 'acp', 'jobs', 'list', '--workspace-name=share-space',
                '--user-name=' + OWNER, '--state=' + state, '--page-size=100',
                '--page-token=' + str(page), '-o', 'json'],
                env=env, capture_output=True, text=True, check=True, timeout=30)
            pages_read += 1
            batch = [] if reply.stdout.strip() == 'No jobs found' else json.loads(reply.stdout)
            assert isinstance(batch, list)
            for job in batch:
                assert job['state'] == state, 'State filter was not applied; no admission'
                assert job['uid'] not in state_identities, 'Repeated page; refresh before admission'
                state_identities.add(job['uid'])
                # Forward transitions may appear in two state queries; retain them once.
                jobs[job['uid']] = job
            if len(batch) < 100:
                break
        else:
            raise RuntimeError('Bounded pagination incomplete; no standard admission')
    return dict(standard_counts(list(jobs.values())), fresh_query=True, quota_type='reserved',
                owner=OWNER, observed_at=datetime.now(timezone.utc).isoformat(),
                pages_read=pages_read, queried_states=states, standard_limit=4, latest_goal_standard_limit=1)


def validate_standard_capacity(capacity, *, now):
    assert capacity['quota_type'] == 'reserved' and capacity['owner'] == OWNER
    assert capacity['fresh_query'] is True
    observed = datetime.fromisoformat(capacity['observed_at'])
    assert observed.tzinfo is not None and 0 <= (now - observed).total_seconds() <= 60
    assert capacity['standard_limit'] == 4 and capacity['latest_goal_standard_limit'] == 1
    assert 0 <= capacity['used_gpus'] <= 3, 'No standard slot is free'
    assert capacity['latest_goal_active_gpus'] == 0, 'The latest-model standard card is still occupied'
