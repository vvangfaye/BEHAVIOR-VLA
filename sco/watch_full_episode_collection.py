"""Finite local controller: accept the fixed pilot, then use fresh idle capacity.

The controller runs on the submitting machine, never sleeps on a GPU worker,
never retries an uncertain creation, and never recovers a failed episode.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from full_episode_collection import CapacityUnavailable, save, submit
from full_episode_acceptance import accept_pilot, validate_gate
from submit_full_collection_shard import describe, environment


def step(root, cli, client, capacity):
    root = Path(root)
    plan = json.loads((root / 'plan.json').read_text())
    pilot = json.loads((root / 'submissions/0000.json').read_text())
    if not pilot.get('uid'):
        return 'needs_exact_submission_reconciliation'
    if not (root / 'pilot_accepted.json').exists():
        job = describe(cli, environment(), pilot['name'])
        if job['uid'] != pilot['uid']:
            raise ValueError('Pilot UID changed')
        if job['state'] in ('FAILED', 'CANCELLED', 'STOPPED', 'DELETED'):
            return 'pilot_failed_no_production'
        if job['state'] != 'SUCCEEDED':
            return 'waiting_for_pilot'
        accept_pilot(root, cli, client)
    validate_gate(root, plan)
    for shard in plan['shards'][1:]:
        if (root / 'STOP').exists():
            return 'admission_stopped'
        receipt_path = root / 'submissions' / f"{shard['index']:04d}.json"
        if receipt_path.exists():
            if not json.loads(receipt_path.read_text()).get('uid'):
                return 'needs_exact_submission_reconciliation'
            continue
        try:
            submit(root, shard['index'], capacity, cli, client)
        except CapacityUnavailable:
            return 'waiting_for_fresh_idle_capacity'
    return 'all_shards_submitted'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--capacity', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--cli', default='/private/tmp/behavior-sco-runtime/bin/sco')
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    assert 30 <= args.poll_seconds <= 60 and time.time() < args.deadline
    from behavior_aoss_client import client
    storage = client()
    plan = json.loads((args.root / 'plan.json').read_text())
    authorization = dict(kind='native_full_episode_finite_controller',
        plan_fingerprint=plan['fingerprint'], expected_episodes=2000,
        deadline_epoch=args.deadline, max_active_gpus=None,
        admission='fresh_complete_idle_worker_capacity', automatic_failure_recovery=False)
    auth_path = args.root / 'controller_authorization.json'
    if auth_path.exists():
        assert json.loads(auth_path.read_text()) == authorization
    else:
        save(auth_path, authorization)
    with (args.root / '.controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < args.deadline and not (args.root / 'STOP').exists():
            try:
                with (args.root / '.submission.lock').open('a') as submission_lock:
                    fcntl.flock(submission_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    state = step(args.root, args.cli, storage, args.capacity)
                error_type = None
            except BlockingIOError:
                state, error_type = 'waiting_for_submission_lock', None
            except Exception as error:
                # No sensitive SDK response, URL, startup command or config is logged.
                state, error_type = 'needs_operator_inspection', type(error).__name__
            records = [json.loads(p.read_text()) for p in (args.root / 'submissions').glob('*.json')]
            status = dict(state=state, error_type=error_type, pid=os.getpid(), observed_at=time.time(),
                plan_fingerprint=plan['fingerprint'], submitted_shards=sum(bool(r.get('uid')) for r in records),
                submitted_episode_budget=sum(len(r['instance_ids']) for r in records if r.get('uid')),
                expected_episodes=2000, deadline_epoch=args.deadline,
                pilot_accepted=(args.root / 'pilot_accepted.json').exists(),
                maximum_active_gpus=None, actual_completed_episodes=None)
            save(args.root / 'controller_status.json', status)
            print(json.dumps(status), flush=True)
            if state in ('all_shards_submitted', 'pilot_failed_no_production',
                         'needs_exact_submission_reconciliation', 'needs_operator_inspection'):
                return
            time.sleep(min(args.poll_seconds, max(0, args.deadline - time.time())))
        save(args.root / 'controller_status.json', dict(state='admission_stopped',
            observed_at=time.time(), pid=os.getpid(), plan_fingerprint=plan['fingerprint']))


if __name__ == '__main__':
    main()
