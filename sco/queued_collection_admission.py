"""Recover difficult collection shards with a campaign-only pending GPU limit."""
import hashlib
import json
from pathlib import Path

from bounded_collection_admission import GPUAdmission, GPUAllowanceFull, save


def pending_gpu_count(snapshot):
    # Unknown RPC outcomes remain charged until the exact name is reconciled.
    return (sum(count for state, count in snapshot['active_job_states'].items()
                if state != 'RUNNING') + len(snapshot['unknown_or_reserved_jobs']))


class QueuedCollectionAdmission(GPUAdmission):
    def __init__(self, policy, root, specs, helper, cli, env, *, authorization, **kwargs):
        super().__init__(policy, root, specs, helper, cli, env, **kwargs)
        self.authorization_path = Path(authorization)
        self.authorization_bytes = self.authorization_path.read_bytes()
        self.queue_policy = json.loads(self.authorization_bytes)
        self.round_interval = self.queue_policy.get('recovery_round_interval_seconds', 0)
        assert self.round_interval in (0, 300)
        assert self.queue_policy.get('submission_interval_seconds', 0) == 0
        self.round_clock_path = self.authorization_path.parent / 'recovery_round.json'
        self.limit = None
        self.pending_limit = 5
        self.recovery_names = set()
        indices = []
        for spec in specs:
            group, = spec['prepared']['plan']['selected_groups']
            if group['rate'] < .5:
                indices.append(spec['index'])
                if spec['index'] in self.queue_policy['allowed_indices']:
                    self.recovery_names.update(spec['prefix'] + f'-g{g}' for g in (1, 2))
        allowed = self.queue_policy['allowed_indices']
        assert allowed and sorted(set(allowed)) == allowed and set(allowed) <= set(indices)
        self.check_authorization()

    def check_authorization(self):
        assert self.authorization_path.read_bytes() == self.authorization_bytes
        p = self.queue_policy
        assert p['kind'] == 'hard50_collection_pending5_authorization'
        assert p['max_pending_gpus'] == 5 and p['max_active_gpus'] is None
        assert p['campaign_fingerprint'] == self.policy['campaign_fingerprint']
        assert p['deadline_epoch'] == self.policy['deadline_epoch']
        assert p['cancelled_markers_sha256'] == self.policy['cancelled_markers_sha256']
        if self.clock() >= p['deadline_epoch']:
            raise TimeoutError('Original collection deadline reached')
        for root, sha in p['cancelled_markers_sha256'].items():
            assert hashlib.sha256((Path(root) / 'USER_CANCEL_NONRUNNING.json').read_bytes()).hexdigest() == sha

    def snapshot(self, current_candidate=None):
        self.check_authorization()
        value = super().snapshot(current_candidate)
        value.update(max_active_gpus=None, max_pending_gpus=5,
                     running_gpus=value['active_job_states'].get('RUNNING', 0),
                     pending_gpus=pending_gpu_count(value),
                     scope='this_finite_collection_campaign_only')
        save(self.root / 'gpu_admission_status.json', value)
        self.last = value
        return value

    def begin_pass(self):
        self.blocked_for_pass = True
        snapshot = self.snapshot()
        if snapshot['pending_gpus'] >= self.pending_limit:
            return
        if self.round_interval and self.round_clock_path.exists():
            last = json.loads(self.round_clock_path.read_text())['started_at']
            if self.clock() - last < self.round_interval:
                return
        if self.round_interval:
            save(self.round_clock_path, dict(started_at=self.clock(),
                 recovery_round_interval_seconds=self.round_interval,
                 pending_gpus_at_start=snapshot['pending_gpus'], creation_attempts=[]))
        self.blocked_for_pass = False

    def reserve(self, name):
        assert name in self.allowed
        self.check_authorization()
        if name not in self.recovery_names:
            raise GPUAllowanceFull('Only baseline success below 50 percent is authorized for recovery')
        if self.blocked_for_pass or self.snapshot()['pending_gpus'] >= self.pending_limit:
            self.blocked_for_pass = True
            raise GPUAllowanceFull('Five collection GPUs pending; defer until the next monitor pass')
        assert name not in self.reservations and name not in self.last['counted_job_names']
        self.reservations[name] = dict(reserved_at=self.clock(), reason='hard50_pending5_recovery')
        save(self.reservations_path, self.reservations)

    def before_copy(self, command):
        self.check_authorization()
        name = next(arg.split('=', 1)[1] for arg in command if arg.startswith('--name='))
        assert name in self.recovery_names and name in self.reservations
        value = self.snapshot(current_candidate=name)
        assert name not in value['counted_job_names'], 'Reconcile the existing exact name'
        if value['pending_gpus'] > self.pending_limit:
            self.blocked_for_pass = True
            raise GPUAllowanceFull('Pending capacity changed before creation')
        if self.round_interval:
            round_state = json.loads(self.round_clock_path.read_text())
            round_state['creation_attempts'].append(dict(name=name, attempted_at=self.clock()))
            save(self.round_clock_path, round_state)
