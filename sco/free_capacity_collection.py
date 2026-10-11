"""Resume the finite low50 extension only against fresh Spot capacity evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from bounded_collection_admission import ACTIVE_STATES, requested_gpus, save


class CapacityUnavailable(RuntimeError):
    pass


def available_slots(observation, ledger, pending, now):
    assert observation['workspace'] == 'share-space'
    assert observation['resource_pool'] == 'computing-cluster-5090-01g'
    assert observation['quota_type'] == 'spot'
    assert observation['gpus_per_job'] == 1 and observation['vcpus_per_job'] == 14
    assert observation['memory_gib_per_job'] == 60
    age = now - observation['observed_at']
    if not 0 <= age <= 120:
        raise CapacityUnavailable('Fresh capacity observation required')
    slots = min(int(observation['free_gpus']), int(observation['free_vcpus'] // 14),
                int(observation['free_memory_gib'] // 60))
    assert slots >= 0
    # Free capacity excludes allocated workers, but not necessarily pending
    # requests. Charge all owner pending requests and this batch's invisible
    # intents. Creations after the observation also consume its finite budget.
    charged = dict(pending)
    for name, item in ledger.items():
        if (not item.get('uid') or item['reserved_at'] >= observation['observed_at']
                or item.get('created_at', float('inf')) >= observation['observed_at']):
            charged[name] = 1
    return slots - sum(charged.values())


def pending_requests(cli, env):
    def read_state(state):
        rows, seen = [], set()
        for page in range(1, 101):
            result = subprocess.run([cli, 'acp', 'jobs', 'list', '--workspace-name=share-space',
                '--user-name=260010165', '--state=' + state, '--page-size=20',
                '--page-token=' + str(page), '-o', 'json'], env=env, capture_output=True,
                text=True, check=True, timeout=30)
            batch = [] if result.stdout.strip() in ('No jobs found', 'No jobs found.') else json.loads(result.stdout)
            assert isinstance(batch, list)
            for job in batch:
                assert job['ownership']['user_name'] == '260010165' and job['state'] == state
                assert job['uid'] not in seen
                seen.add(job['uid'])
                rows.append(job)
            if len(batch) < 20:
                return rows
        raise RuntimeError('Incomplete pending pagination')
    with ThreadPoolExecutor(max_workers=6) as pool:
        batches = list(pool.map(read_state, [s for s in ACTIVE_STATES if s != 'RUNNING']))
    jobs = {j['uid']: j for batch in batches for j in batch}
    return {j['name']: requested_gpus(j) for j in jobs.values()
            if j['resource_pool']['name'] == 'computing-cluster-5090-01g'
            and j['scheduling']['quota_type'] == 'SPOT' and requested_gpus(j)}


class CapacityGate:
    def __init__(self, policy_path, output, cli, env, query=None, clock=time.time, sleeper=time.sleep):
        self.policy_path, self.output = Path(policy_path), Path(output)
        self.policy_bytes = self.policy_path.read_bytes()
        self.policy = json.loads(self.policy_bytes)
        self.clock, self.sleep = clock, sleeper
        self.query = query or (lambda: pending_requests(cli, env))
        self.ledger_path = self.output / 'capacity_reservations.json'
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}

    def check_authorization(self):
        assert self.policy_path.read_bytes() == self.policy_bytes, 'Authorization changed'
        assert self.policy['kind'] == 'free_capacity_low50_extension_authorization'
        assert self.policy['max_active_gpus'] is None
        assert self.policy['indices'] == list(range(299, 335))
        if self.clock() >= self.policy['deadline_epoch']:
            raise TimeoutError('Original campaign deadline reached')
        for marker, sha in self.policy['cancellation_marker_sha256'].items():
            assert hashlib.sha256(Path(marker).read_bytes()).hexdigest() == sha, 'Cancellation changed'

    def wait_for_slot(self, name, already_reserved=False):
        while True:
            self.check_authorization()
            try:
                observation = json.loads(Path(self.policy['capacity_observation']).read_text())
                pending = self.query()
                slots = available_slots(observation, self.ledger, pending, self.clock())
                # Existing reservations are already included in the deduction.
                sufficient = slots >= (0 if already_reserved else 1)
                save(self.output / 'capacity_admission_status.json', dict(observed_at=self.clock(),
                     state='admitted' if sufficient else 'waiting_for_capacity', candidate=name,
                     remaining_complete_slots=slots, capacity_observed_at=observation['observed_at'],
                     pending_gpus=sum(pending.values()), max_active_gpus=None))
                if sufficient:
                    return
            except (CapacityUnavailable, subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    json.JSONDecodeError, ConnectionError, TimeoutError) as error:
                save(self.output / 'capacity_admission_status.json', dict(observed_at=self.clock(),
                     state='waiting_for_capacity_refresh', candidate=name, error_type=type(error).__name__,
                     max_active_gpus=None))
            self.sleep(3)

    def reserve(self, name):
        assert name not in self.ledger, 'Reconcile existing reservation, never resubmit blindly'
        self.wait_for_slot(name)
        self.ledger[name] = dict(reserved_at=self.clock(), creation_attempted=False)
        save(self.ledger_path, self.ledger)

    def before_copy(self, command):
        name = next(x.split('=', 1)[1] for x in command if x.startswith('--name='))
        assert name in self.ledger and not self.ledger[name]['creation_attempted']
        self.wait_for_slot(name, already_reserved=True)
        self.ledger[name].update(creation_attempted=True, created_at=self.clock())
        save(self.ledger_path, self.ledger)


class GuardedSubprocess:
    def __init__(self, gate):
        self.gate = gate

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, command, **kwargs):
        if command[1:4] == ['acp', 'jobs', 'copy']:
            self.gate.before_copy(command)
        return subprocess.run(command, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.authorization.read_text())
    root, original, output = map(Path, (policy['extension_root'], policy['original_root'], policy['output_root']))
    campaign = json.loads((root / 'campaign.json').read_text())
    assert campaign['fingerprint'] == policy['campaign_fingerprint']
    assert campaign['deadline_epoch'] == policy['deadline_epoch']
    frozen = Path(policy['frozen_helper_root'])
    for item in json.loads(Path(policy['frozen_manifest']).read_text())['files']:
        assert hashlib.sha256((frozen.parent / item['path']).read_bytes()).hexdigest() == item['sha256']
    sys.path.insert(0, str(frozen))
    sys.path.insert(0, str(frozen.parent / 'scripts'))
    module = importlib.util.spec_from_file_location('free_capacity_submit_helper', frozen / 'submit_full_collection_shard.py')
    helper = importlib.util.module_from_spec(module)
    module.loader.exec_module(helper)
    gate = CapacityGate(args.authorization, output, policy['cli'], helper.environment())
    helper.subprocess = GuardedSubprocess(gate)
    from continue_collection_extension import highest_receipt
    results = []
    with ExitStack() as stack:
        for path in (original / '.dispatcher.lock', root / 'extension.lock'):
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index in policy['indices']:
            gate.check_authorization()
            prepared = root / 'groups' / f'{index:03d}' / 'prepared.json'
            previous_path, previous = highest_receipt(root, index, json.loads(prepared.read_text()))
            if previous['generation'] > 0:
                live = helper.describe(policy['cli'], helper.environment(), previous['name'])
                assert live['uid'] == previous.get('uid'), 'Unknown intent needs reconciliation'
                results.append(dict(index=index, state='already_submitted', uid=live['uid'], name=live['name']))
                continue
            assert previous['generation'] == 0 and previous.get('uid')
            name = previous['name'].rsplit('-g', 1)[0] + '-g1'
            receipt_path = root / f'spot_{index:03d}_g1.json'
            gate.reserve(name)
            try:
                receipt = helper.submit(SimpleNamespace(receipt=str(receipt_path), prepared=str(prepared),
                    name=name, quota_type='spot', gates=policy['gates'], timeout_hours=24,
                    object_prefix=previous['object_prefix'], cli=policy['cli'],
                    aoss_client_module=policy['aoss_client_module'], recover_receipt=str(previous_path)))
            except Exception as error:
                if not receipt_path.exists():
                    del gate.ledger[name]
                    save(gate.ledger_path, gate.ledger)
                save(output / 'batch_status.json', dict(state='needs_reconciliation', index=index,
                     error_type=type(error).__name__, results=results, updated_at=time.time()))
                return 1
            gate.ledger[name].update(uid=receipt['uid'])
            save(gate.ledger_path, gate.ledger)
            results.append(dict(index=index, state=receipt['platform_state'], name=name, uid=receipt['uid']))
            save(output / 'batch_status.json', dict(state='submitting', results=results, updated_at=time.time()))
        save(output / 'batch_status.json', dict(state='submitted', results=results, updated_at=time.time(),
             max_active_gpus=None, expected_attempts=1080, shard_count=36))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps(dict(state='stopped', error_type=type(error).__name__)))
        sys.exit(1)
