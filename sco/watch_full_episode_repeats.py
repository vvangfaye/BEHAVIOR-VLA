"""One finite admission controller for the original and two seeded rounds."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from full_episode_collection import CapacityUnavailable, save, submit
from full_episode_acceptance import accept_pilot
from full_episode_repeats import read_plan, submission_context, validate_seed_gate
from submit_full_collection_shard import describe, environment
from full_episode_scope import pilot_gate_waived


def records(root):
    return [json.loads(path.read_text()) for path in (Path(root) / 'submissions').glob('*.json')]


def query_failure_can_retry(authorization):
    # Creation persists an intent before the RPC. A missing UID makes a retry
    # unsafe, even when the exception resembles an ordinary network timeout.
    return all(row.get('uid') for root in authorization['roots'] for row in records(root))


def snapshot(campaign, authorization, state, error_type=None):
    by_round = []
    for round_id, root in enumerate(authorization['roots']):
        rows = records(root)
        by_round.append(dict(sampling_round=authorization.get('rounds', list(range(len(authorization['roots']))))[round_id], root=root,
            submitted_shards=sum(bool(row.get('uid')) for row in rows),
            submitted_episode_budget=sum(len(row['instance_ids']) for row in rows if row.get('uid')),
            unknown_intents=[row['name'] for row in rows if not row.get('uid')],
            pilot_accepted=(Path(root) / 'pilot_accepted.json').exists(),
            pilot_gate_waived_by_user=pilot_gate_waived(root)))
    result = dict(state=state, error_type=error_type, pid=os.getpid(), observed_at=time.time(),
        expected_episodes=authorization['total_episodes'], expected_shards=authorization.get('total_shards', 1203), deadline_epoch=authorization['deadline_epoch'],
        maximum_active_gpus=None, automatic_failure_recovery=False, rounds=by_round,
        submitted_shards=sum(row['submitted_shards'] for row in by_round),
        submitted_episode_budget=sum(row['submitted_episode_budget'] for row in by_round),
        actual_completed_episodes=None)
    recovery_path = Path(campaign) / 'failure_recovery_authorization.json'
    if recovery_path.exists():
        recovery_auth = json.loads(recovery_path.read_text())
        recovery_rows = [r for root in authorization['roots'] for r in records(root) if r.get('failure_recovery')]
        result['explicit_failure_recovery'] = dict(authorized_failed_shards=len(recovery_auth['failures']),
            eligible_shards=sum(v['eligible'] for v in recovery_auth['failures']),
            submitted_shards=sum(bool(v.get('uid')) for v in recovery_rows),
            submitted_missing_episode_budget=sum(len(v['failure_recovery']['missing_instance_ids']) for v in recovery_rows if v.get('uid')),
            canary_accepted=(Path(campaign) / 'failure_recovery_canary_accepted.json').exists(),
            automatic_retry_of_recovery_failure=False)
    save(Path(campaign) / 'controller_status.json', result)
    print(json.dumps(result), flush=True)
    return result


def choose(authorization, cli, storage):
    candidates = []
    for round_id, root in enumerate(authorization['roots']):
        root = Path(root)
        plan = read_plan(root)
        assert plan['fingerprint'] == authorization['fingerprints'][round_id]
        rows = {row['index']: row for row in records(root)}
        if any(not row.get('uid') for row in rows.values()):
            raise ValueError('Unknown creation intent requires exact-name reconciliation')
        pilot = rows.get(0)
        if pilot is None:
            candidates.append((-1, round_id, root, 0))
            continue
        waived = pilot_gate_waived(root, plan)
        if not waived and not (root / 'pilot_accepted.json').exists():
            job = describe(cli, environment(), pilot['name'])
            assert job['uid'] == pilot['uid']
            if job['state'] == 'SUCCEEDED':
                accept_pilot(root, cli, storage)
            elif job['state'] in ('FAILED', 'CANCELLED', 'STOPPED', 'DELETED'):
                raise RuntimeError('Seeded pilot failed; automatic recovery is disabled')
            else:
                continue
        if not waived:
            validate_seed_gate(root)
        for shard in plan['shards'][1:]:
            if shard['index'] not in rows:
                candidates.append((shard['index'], round_id, root, shard['index']))
                break
    if not candidates:
        total = sum(len(records(root)) for root in authorization['roots'])
        return ('all_shards_submitted' if total == authorization.get('total_shards', 1203) else 'waiting_for_seeded_pilots'), None
    _, _, root, index = min(candidates)
    return 'candidate_ready', (root, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--cli', default='/private/tmp/behavior-sco-runtime/bin/sco')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    assert 30 <= args.poll_seconds <= 300
    campaign = args.campaign.resolve()
    raw_auth = (campaign / 'authorization.json').read_bytes()
    authorization = json.loads(raw_auth)
    paired = authorization['kind'] == 'paired_checkpoint_evaluation'
    original_repeats = authorization['kind'] == 'native_original_three_repeats_3000'
    independent = authorization['kind'] == 'native_full_episode_independent_final1000' or paired or original_repeats
    if original_repeats:
        from original_full_episode_repeat import validate_campaign
        validate_campaign(campaign, authorization)
        plan = read_plan(authorization['roots'][0])
    elif independent:
        assert len(authorization['roots']) == 1
        plan = read_plan(authorization['roots'][0])
        if paired:
            from paired_checkpoint_evaluation import validate_plan
            validate_plan(plan)
            assert authorization['total_episodes'] == plan['expected_episodes']
            assert authorization['total_shards'] == len(plan['shards'])
        else:
            from full_episode_scope import validate_final1000
            validate_final1000(plan)
            assert authorization['total_episodes'] == 1000 and authorization['total_shards'] == 201
        assert authorization['rounds'] == [plan['sampling_round']]
        assert authorization['fingerprints'] == [plan['fingerprint']]
        assert authorization['deadline_epoch'] <= authorization['inherited_deadline_epoch']
    else:
        assert authorization['kind'] == 'native_full_episode_three_round_campaign'
        assert authorization['total_episodes'] == 6000 and authorization['additional_episodes'] == 4000
        assert authorization['rounds'] == [0, 1, 2] and len(authorization['roots']) == 3
    assert authorization['max_active_gpus'] is None and authorization['automatic_failure_recovery'] is False
    assert time.time() < authorization['deadline_epoch']
    lock_roots = list(authorization['roots'])
    from full_episode_campaign_scope import effective_authorization
    authorization = effective_authorization(campaign, authorization)
    raw_scope = (campaign / 'active_scope.json').read_bytes() if (campaign / 'active_scope.json').exists() else None
    from behavior_aoss_client import client
    storage = client()
    with ExitStack() as stack:
        # Also own the legacy root lock, preventing its former controller from restarting.
        for root in [campaign] + [Path(root) for root in lock_roots]:
            lock = stack.enter_context((root / '.controller.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        next_queue_refresh = 0
        while time.time() < authorization['deadline_epoch']:
            assert (campaign / 'authorization.json').read_bytes() == raw_auth
            if raw_scope is not None:
                assert (campaign / 'active_scope.json').read_bytes() == raw_scope
            if any((root / 'STOP').exists() for root in [campaign] + [Path(r) for r in authorization['roots']]):
                snapshot(campaign, authorization, 'admission_stopped')
                return
            try:
                policy_path = campaign / 'queue_policy.json'
                if policy_path.exists():
                    policy = json.loads(policy_path.read_text())
                    if (policy.get('controller_refreshes_queue') is True
                            and time.monotonic() >= next_queue_refresh):
                        from refresh_full_episode_queue import refresh
                        refresh(campaign, args.cli)
                        if independent and (plan.get('trained_model_export') or plan.get('publish_compact_metrics')):
                            if paired:
                                from paired_checkpoint_evaluation import observe_metrics
                            elif plan.get('original_repeat_collection'):
                                from original_full_episode_repeat import observe_metrics
                            else:
                                from trained_full_episode_collection import observe_metrics
                            observe_metrics(campaign, storage)
                        next_queue_refresh = time.monotonic() + args.poll_seconds
                from full_episode_failure_recovery import choose_recovery, prepare
                explicitly_authorized_independent_recovery = independent and (campaign / 'failure_recovery_authorization.json').exists()
                recovery_candidate = (None if raw_scope is not None or (independent and not explicitly_authorized_independent_recovery)
                                      else choose_recovery(campaign, args.cli, storage))
                state, candidate = ('recovery_candidate_ready', recovery_candidate) if recovery_candidate else choose(authorization, args.cli, storage)
                if candidate is None:
                    snapshot(campaign, authorization, state)
                    if state == 'all_shards_submitted':
                        if (campaign / 'failure_recovery_authorization.json').exists():
                            if independent:
                                from full_episode_failure_recovery import monitor_final1000_coverage
                                progress = monitor_final1000_coverage(campaign, storage)
                                if progress['all_target_pairs_published']:
                                    snapshot(campaign, authorization, 'all_1000_episode_pairs_published')
                                    return
                            snapshot(campaign, authorization, 'initial_shards_submitted_monitoring_finite_recovery')
                        else:
                            if independent and (plan.get('trained_model_export') or plan.get('publish_compact_metrics')):
                                if paired:
                                    from paired_checkpoint_evaluation import observe_metrics
                                elif plan.get('original_repeat_collection'):
                                    from original_full_episode_repeat import observe_metrics
                                else:
                                    from trained_full_episode_collection import observe_metrics
                                progress = observe_metrics(campaign, storage)
                                if progress['completed_episodes'] == authorization['total_episodes']:
                                    snapshot(campaign, authorization, 'all_target_episode_metrics_published')
                                    return
                            else:
                                return
                else:
                    root, index = candidate
                    with (root / '.submission.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        with submission_context(root, authorization['roots']):
                            admission_path = campaign / 'queue_admission.json'
                            if not admission_path.exists():
                                admission_path = campaign / 'capacity.json'
                            if recovery_candidate:
                                previous, recovery = prepare(campaign, root, index, args.cli)
                                submit(root, index, admission_path, args.cli, storage, previous=previous, recovery=recovery)
                            else:
                                submit(root, index, admission_path, args.cli, storage)
                    snapshot(campaign, authorization, 'admitting_unique_shards')
                    continue
            except CapacityUnavailable:
                snapshot(campaign, authorization, 'waiting_for_queue_slot'
                    if (campaign / 'queue_admission.json').exists()
                    else 'waiting_for_fresh_idle_capacity')
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                if not query_failure_can_retry(authorization):
                    snapshot(campaign, authorization, 'needs_operator_inspection', type(error).__name__)
                    return
                # Repeat observation only when every existing intent has a UID.
                # Failed or uncertain creation is never repeated here.
                snapshot(campaign, authorization, 'waiting_for_platform_query_recovery', type(error).__name__)
            except Exception as error:
                snapshot(campaign, authorization, 'needs_operator_inspection', type(error).__name__)
                return
            time.sleep(min(args.poll_seconds, max(0, authorization['deadline_epoch'] - time.time())))
        snapshot(campaign, authorization, 'admission_deadline_reached')


if __name__ == '__main__':
    main()
