"""Durable, bounded submission controller for the checkpoint 2M video evaluation.

The adapter owns cloud access and artifact verification. Its describe(name) and
create(record) return {uid, state, name?}; inspect(record) returns None until
artifacts exist, or {verified: bool, model_success?: bool, ...}. A module may
expose these methods directly, or a factory(root: Path, plan: dict).

An intent is never submitted twice, even when a crash happened before its API
call. An uncertain intent must be reconciled by its exact saved name. Running
jobs do not use the five pending slots, but new submissions reserve their slot
for the entire current pass. Failed jobs are retried only under a persisted explicit user recovery policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any


MAX_PENDING_GPUS = 5
ARTIFACT_GRACE_SECONDS = 300
TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "STOPPED", "DELETED", "CANCELLED", "CANCELED"})
INSPECTION_FIELDS = frozenset({
    "verified", "model_success", "q_score", "artifact_manifest_path", "result_path",
    "video_path", "result_sha256", "video_sha256", "checked_at", "reason",
    "original_model_execution_complete", "artifact_recovery_audit_uid",
    "artifact_recovery_audit_state", "no_model_rerun", "batch_completed_original",
})


def accepted_execution(record):
    inspection = record.get('inspection', {})
    if inspection.get('verified') is not True:
        return False
    if record.get('state') == 'SUCCEEDED':
        return True
    if (record.get('batch_id') and record.get('state') in TERMINAL_STATES
            and inspection.get('batch_completed_original') is True):
        return True
    # Only the diagnosed pilot whose real episode finished before observer
    # packaging failed can be accepted by an independently successful CPU audit.
    return bool(record.get('index') == 0 and record.get('state') == 'FAILED'
        and inspection.get('original_model_execution_complete') is True
        and inspection.get('no_model_rerun') is True
        and inspection.get('artifact_recovery_audit_state') == 'SUCCEEDED'
        and isinstance(inspection.get('artifact_recovery_audit_uid'), str)
        and inspection['artifact_recovery_audit_uid'])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    """Commit an intent exclusively, or replace a report atomically, durably."""
    payload = canonical_bytes(value) + b"\n"
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    fsync_directory(path.parent)


def pending_gpu_count(records: list[dict]) -> int:
    """Unknown/unrecognized states, intents and stopping jobs remain charged."""
    return sum(1 for record in gpu_records(records)
               if str(record.get("state", "UNKNOWN")).upper() not in TERMINAL_STATES | {"RUNNING"})


def gpu_records(records):
    groups = {}
    for index, record in enumerate(records):
        key = record.get('physical_name') or record.get('name') or ('row', index)
        groups.setdefault(key, []).append(record)
    result = []
    for rows in groups.values():
        pending = [r for r in rows if r.get('state', 'UNKNOWN') not in TERMINAL_STATES | {'RUNNING'}]
        running = [r for r in rows if r.get('state') == 'RUNNING']
        result.append((pending or running or rows)[0])
    return result


def validate_plan(plan: dict) -> None:
    if type(plan.get("checkpoint_step")) is not int or plan["checkpoint_step"] != 2_000_000:
        raise ValueError("plan checkpoint_step must be 2000000")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,50}", plan.get("run_id", "")):
        raise ValueError("run_id must be a unique lowercase cloud-safe name of at most 51 characters")
    episodes = plan.get("episodes", [])
    if (len(episodes) != 1000 or any(type(e.get("index")) is not int or
                                    type(e.get("instance_id")) is not int for e in episodes) or
            {e.get("index") for e in episodes} != set(range(1000))):
        raise ValueError("plan must contain exactly indices 0..999")
    pairs = {(e.get("task"), e.get("instance_id")) for e in episodes}
    tasks = {e.get("task") for e in episodes}
    if (len(tasks) != 100 or len(pairs) != 1000 or
            any(not isinstance(t, str) or not t for t in tasks) or
            pairs != {(t, i) for t in tasks for i in range(301, 311)}):
        raise ValueError("plan must contain 100 unique tasks with instances 301..310")
    if any(e.get("rollout", 0) != 0 or e.get("gpu_count", 1) != 1 for e in episodes):
        raise ValueError("each episode must have rollout 0 and exactly one GPU")


class Controller:
    def __init__(self, root: Path | str, adapter: Any):
        self.root = Path(root).resolve()
        self.adapter = adapter
        self.plan = json.loads((self.root / "plan.json").read_text())
        validate_plan(self.plan)
        self.fingerprint = hashlib.sha256(canonical_bytes(self.plan)).hexdigest()
        self.episodes = {e["index"]: e for e in self.plan["episodes"]}
        self.directory = self.root / "submissions"
        policy_path = self.root / 'batch_policy.json'
        policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
        self.max_pending_gpus = policy.get('max_pending_gpus', MAX_PENDING_GPUS)
        if type(self.max_pending_gpus) is not int or self.max_pending_gpus not in (5, 10):
            raise ValueError('Unsupported pending GPU limit')
        if self.max_pending_gpus != MAX_PENDING_GPUS and not policy.get('pending_limit_user_authorization'):
            raise ValueError('Changed pending limit requires explicit user authorization')

    @contextmanager
    def lock(self):
        """The CLI keeps this lock throughout polling, including its sleeps."""
        with (self.root / "controller.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another video evaluation controller holds the lock") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _check_plan(self) -> None:
        current = json.loads((self.root / "plan.json").read_text())
        if hashlib.sha256(canonical_bytes(current)).hexdigest() != self.fingerprint:
            raise RuntimeError("frozen plan changed")
        path = self.root / "plan.lock.json"
        expected = {"plan_sha256": self.fingerprint, "run_id": self.plan["run_id"]}
        try:
            write_json(path, expected, exclusive=True)
        except FileExistsError:
            if json.loads(path.read_text()) != expected:
                raise RuntimeError("plan differs from the durable frozen fingerprint")

    def _name(self, index: int) -> str:
        return f'{self.plan["run_id"]}-e{index:04d}'

    def _path(self, index: int) -> Path:
        return self.directory / f"{index:04d}.json"

    def _load_records(self) -> dict[int, dict]:
        self.directory.mkdir(mode=0o700, exist_ok=True)
        # Recover durable reservations only; never recreate an uncertain cloud job.
        for intent in sorted((self.root / 'batch_submissions').glob('*.json')):
            batch = json.loads(intent.read_text())
            if batch['plan_fingerprint'] != self.fingerprint:
                raise RuntimeError('Batch plan identity changed')
            reserved_rows = batch['records']
            indices = [r['index'] for r in reserved_rows]
            expected_name = self.plan['run_id'] + '-b%04d-%04d' % (min(indices), max(indices))
            if (not 2 <= len(indices) <= 10 or len(set(indices)) != len(indices)
                    or any(i not in self.episodes for i in indices)
                    or len({self.episodes[i]['task'] for i in indices}) != 1
                    or intent.stem != expected_name):
                raise RuntimeError('Invalid durable batch reservation')
            for reserved in reserved_rows:
                i = reserved['index']
                if (reserved.get('name') != self._name(i) or reserved.get('batch_id') != expected_name
                        or reserved.get('physical_name') != expected_name or reserved.get('batch_indices') != indices
                        or reserved.get('task') != self.episodes[i]['task']
                        or reserved.get('instance_id') != self.episodes[i]['instance_id']
                        or reserved.get('plan_fingerprint') != self.fingerprint):
                    raise RuntimeError('Batch reservation identity changed')
            for reserved in batch['records']:
                if not self._path(reserved['index']).exists():
                    write_json(self._path(reserved['index']), reserved, exclusive=True)
        records = {}
        for path in sorted(self.directory.glob("*.json")):
            record = json.loads(path.read_text())
            index = record["index"]
            if (index not in self.episodes or index in records or path != self._path(index) or
                    record.get("plan_fingerprint") != self.fingerprint or
                    record.get("name") != self._name(index) or
                    record.get("task") != self.episodes[index]["task"] or
                    record.get("instance_id") != self.episodes[index]["instance_id"] or
                    record.get("run_id") != self.plan["run_id"] or record.get("gpu_count") != 1 or
                    record.get("checkpoint_step") != 2_000_000 or record.get("rollout") != 0 or
                    (record.get("state") != "UNKNOWN" and not record.get("uid"))):
                raise RuntimeError("submission does not match its frozen episode")
            records[index] = record
        return records

    @staticmethod
    def _apply_observation(record: dict, response: Any) -> None:
        if not isinstance(response, dict) or response.get("query_complete", True) is not True:
            raise ValueError("incomplete observation")
        if response.get("not_found") is True:
            raise LookupError("saved intent not found; never blindly recreate it")
        uid, state = response.get("uid"), response.get("state")
        if not isinstance(uid, str) or not uid or not isinstance(state, str) or not state:
            raise ValueError("missing UID or state")
        if response.get("name", record["name"]) != record["name"]:
            raise ValueError("observed name mismatch")
        if record.get("uid") and record["uid"] != uid:
            raise ValueError("observed UID changed")
        record.update(uid=uid, state=state.upper(), observed_at=utc_now(), query_complete=True)
        record.pop("query_error", None)

    def _inspect(self, record: dict) -> None:
        if record.get("inspection", {}).get("verified") is True:
            return
        try:
            result = self.adapter.inspect(dict(record))
            if result is not None:
                if not isinstance(result, dict) or not isinstance(result.get("verified"), bool):
                    raise ValueError("inspection needs an explicit verified boolean")
                record["inspection"] = {key: value for key, value in result.items()
                                        if key in INSPECTION_FIELDS}
                record["inspection"]["checked_at"] = utc_now()
                record.pop("inspection_error", None)
        except Exception as exc:
            # Exception strings and raw adapter responses can contain credentials.
            record["inspection_error"] = type(exc).__name__

    def _observe(self, record: dict) -> bool:
        complete = True
        if record.get("state") not in TERMINAL_STATES:
            try:
                self._apply_observation(record, self.adapter.describe(record["name"]))
            except Exception as exc:
                record.update(state="UNKNOWN", query_complete=False, query_error=type(exc).__name__)
                complete = False
        if record.get("state") in TERMINAL_STATES:
            record.setdefault("terminal_observed_epoch", time.time())
        # This also collects available artifacts from terminal failures without retrying.
        self._inspect(record)
        write_json(self._path(record["index"]), record)
        return complete

    def _submit(self, index: int) -> dict:
        self._check_plan()
        episode = self.episodes[index]
        record = dict(episode, index=index, run_id=self.plan["run_id"], name=self._name(index),
                      plan_fingerprint=self.fingerprint, checkpoint_step=2_000_000,
                      gpu_count=1, rollout=0, state="UNKNOWN", query_complete=False,
                      create_outcome="UNKNOWN", intent_created_at=utc_now())
        # The name and complete intent reach disk before the external API call.
        write_json(self._path(index), record, exclusive=True)
        try:
            self._apply_observation(record, self.adapter.create(dict(record)))
            record["create_outcome"] = "ACKNOWLEDGED"
            if record.get("state") in TERMINAL_STATES:
                record.setdefault("terminal_observed_epoch", time.time())
        except Exception as exc:
            record.update(state="UNKNOWN", query_complete=False, create_error=type(exc).__name__)
        write_json(self._path(index), record)
        return record

    def _group_indices(self, eligible, records):
        path = self.root / 'batch_policy.json'
        if not path.exists():
            return eligible[:1]
        policy = json.loads(path.read_text())
        if policy.get('enabled') is not True:
            return eligible[:1]
        if policy['batch_worker_sha256'] != getattr(self.adapter, 'batch_worker_sha256', None):
            raise RuntimeError('Batch worker differs from the reviewed policy')
        if policy['plan_fingerprint'] != self.fingerprint or policy['max_episodes_per_worker'] not in (5, 10):
            raise RuntimeError('Invalid batch scope')
        pilot = policy['pilot_indices']
        if (len(pilot) != 2 or len(set(pilot)) != 2 or any(i not in self.episodes or i == 0 for i in pilot)
                or len({self.episodes[i]['task'] for i in pilot}) != 1):
            raise RuntimeError('Invalid two-episode batch pilot')
        if policy.get('require_batch_pilot', True) is False:
            waiver = policy.get('pilot_gate_waiver', {})
            if waiver.get('authorized_by_user') is not True or not waiver.get('instruction'):
                raise RuntimeError('Batch pilot waiver requires explicit user authorization')
        else:
            expected_pilot = self.plan['run_id'] + '-b%04d-%04d' % (min(pilot), max(pilot))
            if not all(i in records and records[i].get('state') == 'SUCCEEDED'
                       and records[i].get('batch_id') == expected_pilot
                       and records[i].get('inspection', {}).get('batch_completed_original') is True
                       and accepted_execution(records[i]) for i in pilot):
                return pilot if all(i in eligible for i in pilot) else []
        first = eligible[0]
        task = self.episodes[first]['task']
        return [i for i in eligible if self.episodes[i]['task'] == task][:policy['max_episodes_per_worker']]

    def _submit_group(self, indices):
        if len(indices) == 1:
            return [self._submit(indices[0])]
        self._check_plan()
        name = self.plan['run_id'] + '-b%04d-%04d' % (min(indices), max(indices))
        if len(name) > 63:
            raise ValueError('Batch name too long')
        records = []
        for index in indices:
            records.append(dict(self.episodes[index], index=index, run_id=self.plan['run_id'],
                name=self._name(index), plan_fingerprint=self.fingerprint, checkpoint_step=2000000,
                gpu_count=1, rollout=0, state='UNKNOWN', query_complete=False, create_outcome='UNKNOWN',
                intent_created_at=utc_now(), physical_name=name, batch_id=name, batch_indices=indices,
                batch_worker_sha256=self.adapter.batch_worker_sha256))
        directory = self.root / 'batch_submissions'; directory.mkdir(exist_ok=True)
        path = directory / (name + '.json')
        write_json(path, dict(plan_fingerprint=self.fingerprint, records=records), exclusive=True)
        for record in records:
            write_json(self._path(record['index']), record, exclusive=True)
        try:
            response = self.adapter.create_group([dict(r) for r in records])
            for record in records:
                self._apply_observation(record, dict(response, name=record['name']))
                record['create_outcome'] = 'ACKNOWLEDGED'
        except Exception as error:
            for record in records:
                record.update(state='UNKNOWN', query_complete=False, create_error=type(error).__name__)
        for record in records:
            write_json(self._path(record['index']), record)
        write_json(path, dict(plan_fingerprint=self.fingerprint, records=records))
        return records

    def _recovery_policy(self):
        path = self.root / 'automatic_recovery.json'
        if not path.exists():
            return None
        policy = json.loads(path.read_text())
        assert policy['user_authorization'] == '可以自动重提了'
        assert policy['checkpoint_step'] == 2000000 and policy['run_id'] == self.plan['run_id']
        return policy if policy.get('enabled') is True else None

    def _recover(self, old, proof):
        assert proof['eligible'] is True and proof['old_uid'] == old['uid']
        assert proof['old_writer_stopped'] is True and proof['probe_state'] == 'SUCCEEDED'
        assert not old.get('inspection', {}).get('verified') and not proof.get('protected_completed_outputs')
        generation = int(old.get('infrastructure_generation', 0)) + 1
        history = self.root / 'production_recoveries' / ('%04d' % old['index']) / ('g%03d' % (generation - 1))
        history.mkdir(parents=True, exist_ok=False)
        write_json(history / 'submission.json', old, exclusive=True)
        write_json(history / 'proof.json', proof, exclusive=True)
        launch = self.root / 'launches' / ('%04d.json' % old['index'])
        write_json(history / 'launch.json', json.loads(launch.read_text()), exclusive=True)
        artifact = self.root / 'artifacts' / ('%04d.json' % old['index'])
        if artifact.exists():
            write_json(history / 'receipt.json', json.loads(artifact.read_text()), exclusive=True)
        launch.unlink()
        record = {k: v for k, v in old.items() if k not in (
            'uid', 'inspection', 'inspection_error', 'query_error', 'observed_at',
            'terminal_observed_epoch', 'create_error', 'approved_empty_startup_recovery', 'recovery_decision')}
        for field in ('batch_id', 'batch_indices', 'batch_worker_sha256'):
            record.pop(field, None)
        chain = list(old.get('recovery_chain', []))
        chain.append(dict(old_uid=old['uid'], old_name=proof['old_name'], old_state=proof['old_state'],
            old_writer_stopped=True, probe_uid=proof['probe_uid'], probe_state='SUCCEEDED',
            attempt_count=proof['attempt_count'], prior_assignment_root=proof['assignment_root'],
            proof_sha256=hashlib.sha256(canonical_bytes(proof)).hexdigest(), user_authorization='可以自动重提了'))
        record.update(state='UNKNOWN', query_complete=False, create_outcome='UNKNOWN',
            infrastructure_generation=generation, isolated_generation=True,
            physical_name=old['name'] + '-g%03d' % generation,
            assignment_root=self.plan['afs_output_root'] + '/episodes/%04d-g%03d' % (old['index'], generation),
            recovery_chain=chain, previous_model_attempts=int(old.get('previous_model_attempts', 0)) + proof['attempt_count'],
            intent_created_at=utc_now())
        write_json(self._path(old['index']), record)
        try:
            self._apply_observation(record, self.adapter.create(dict(record)))
            record['create_outcome'] = 'ACKNOWLEDGED'
        except Exception as exc:
            record.update(state='UNKNOWN', query_complete=False, create_error=type(exc).__name__)
        write_json(self._path(old['index']), record)
        return record

    def _run_once(self) -> dict:
        self._check_plan()
        records = self._load_records()
        started = utc_now()
        if (self.root / "STOP").exists():
            return self._status(records, started, [], False, "STOP", pending_gpu_count(list(records.values())))
        query_complete = True
        self.round_preparation_error = None
        if hasattr(self.adapter, "begin_round"):
            try:
                self.adapter.begin_round([dict(r) for r in records.values()])
            except Exception as exc:
                self.round_preparation_error = type(exc).__name__
                query_complete = False
        for record in records.values():
            query_complete = self._observe(record) and query_complete
        self.auxiliary_gpu = dict(state='NOT_REQUESTED', pending_gpus=0, query_complete=True)
        if hasattr(self.adapter, 'auxiliary_gpu_step'):
            try:
                self.auxiliary_gpu = self.adapter.auxiliary_gpu_step(allow_submit=False)
                query_complete = query_complete and self.auxiliary_gpu['query_complete']
            except Exception:
                self.auxiliary_gpu = dict(state='UNKNOWN', pending_gpus=1, query_complete=False)
                query_complete = False
        pending_before = pending_gpu_count(list(records.values())) + self.auxiliary_gpu['pending_gpus']
        available = max(0, self.max_pending_gpus - pending_before)
        if query_complete and available and self.auxiliary_gpu['state'] == 'PLANNED' and not (self.root / 'STOP').exists():
            try:
                self.auxiliary_gpu = self.adapter.auxiliary_gpu_step(allow_submit=True)
                query_complete = query_complete and self.auxiliary_gpu['query_complete']
            except Exception:
                self.auxiliary_gpu = dict(state='UNKNOWN', pending_gpus=1, query_complete=False)
                query_complete = False
            pending_before += self.auxiliary_gpu['pending_gpus']
            available = max(0, self.max_pending_gpus - pending_before)
        pilot = records.get(0)
        pilot_accepted = bool(pilot and accepted_execution(pilot))
        reason = None
        if not query_complete:
            reason = "observation_incomplete"
        elif pilot and not pilot_accepted:
            reason = "pilot_not_accepted"
        eligible = sorted(set(self.episodes) - records.keys()) if pilot_accepted else ([0] if not pilot else [])
        submitted = []
        self.recovered_this_pass = []
        policy = self._recovery_policy()
        if query_complete and pilot_accepted and policy and available and hasattr(self.adapter, 'prepare_recovery'):
            for old in sorted(records.values(), key=lambda r: r['index']):
                if old['state'] not in TERMINAL_STATES - {'SUCCEEDED'} or old.get('inspection', {}).get('verified'):
                    continue
                if time.time() - old.get('terminal_observed_epoch', time.time()) < ARTIFACT_GRACE_SECONDS:
                    continue
                if int(old.get('infrastructure_generation', 0)) >= policy['max_generation']:
                    old['recovery_decision'] = dict(eligible=False, reason='repeated_failure_requires_diagnosis')
                    write_json(self._path(old['index']), old)
                    continue
                if (self.root / 'STOP').exists():
                    reason = 'STOP'
                    break
                try:
                    proof = self.adapter.prepare_recovery(dict(old))
                    old['recovery_decision'] = dict(eligible=bool(proof and proof.get('eligible')),
                        reason=(proof or {}).get('reason', 'awaiting_recovery_proof'),
                        waiting=bool((proof or {}).get('waiting', False)))
                    write_json(self._path(old['index']), old)
                    if proof and proof.get('eligible') is True:
                        record = self._recover(old, proof)
                        records[old['index']] = record
                        submitted.append(old['index'])
                        self.recovered_this_pass.append(old['index'])
                        available -= 1
                        if record.get('query_complete') is not True:
                            query_complete = False
                            reason = 'recovery_outcome_unknown'
                            break
                        if available == 0:
                            break
                except Exception as exc:
                    old['recovery_decision'] = dict(eligible=False, reason='recovery_observation_' + type(exc).__name__, waiting=True)
                    write_json(self._path(old['index']), old)
        if query_complete:
            while eligible and available > 0:
                if (self.root / "STOP").exists():
                    reason = "STOP"
                    break
                indices = self._group_indices(eligible, records)
                if not indices:
                    reason = 'batch_pilot_pending'
                    break
                group = self._submit_group(indices)
                for record in group:
                    records[record['index']] = record
                submitted.extend(indices)
                eligible = [i for i in eligible if i not in indices]
                available -= 1
                # A new RUNNING job still consumes one of this pass's slots.
                if any(record.get("query_complete") is not True for record in group):
                    query_complete = False
                    reason = "submission_outcome_unknown"
                    break
        if not available and reason is None:
            reason = "pending_limit"
        return self._status(records, started, submitted, query_complete, reason, pending_before)

    def _status(self, records: dict[int, dict], started: str, submitted: list[int],
                query_complete: bool, reason: str | None, pending_before: int) -> dict:
        rows = list(records.values())
        states = dict(Counter(r.get("state", "UNKNOWN") for r in gpu_records(rows)))
        verified = sum(r.get("inspection", {}).get("verified") is True for r in rows)
        all_terminal = len(rows) == 1000 and all(r.get("state") in TERMINAL_STATES for r in rows)
        now = time.time()
        awaiting_artifacts = [r["index"] for r in rows
                              if r.get("state") in TERMINAL_STATES and
                              r.get("inspection", {}).get("verified") is not True and
                              now - r.get("terminal_observed_epoch", now) < ARTIFACT_GRACE_SECONDS]
        policy = self._recovery_policy()
        recovery_wait = [r['index'] for r in rows if policy and r.get('state') in TERMINAL_STATES - {'SUCCEEDED'}
            and not r.get('inspection', {}).get('verified')
            and int(r.get('infrastructure_generation', 0)) < policy['max_generation']
            and (not r.get('recovery_decision') or r['recovery_decision'].get('waiting')
                 or r['recovery_decision'].get('eligible'))]
        finished = all_terminal and not awaiting_artifacts and not recovery_wait
        clean = verified == 1000 and all(accepted_execution(r) for r in rows)
        completion_state = ("completed" if clean else "completed_with_failures") if finished else "in_progress"
        if all_terminal and awaiting_artifacts:
            completion_state = "awaiting_artifacts"
        if reason == "STOP":
            completion_state = "stopped"
        status = dict(run_id=self.plan["run_id"], plan_fingerprint=self.fingerprint,
                      pass_started_at=started, pass_finished_at=utc_now(), query_complete=query_complete,
                      planned=1000, submitted_total=len(rows), unsubmitted=1000 - len(rows),
                      states=states, pending_gpus=pending_gpu_count(rows) + getattr(self, "auxiliary_gpu", {}).get("pending_gpus", 0), pending_before=pending_before,
                      auxiliary_gpu=getattr(self, "auxiliary_gpu", None),
                      running_gpus=states.get("RUNNING", 0), max_pending_gpus=self.max_pending_gpus,
                      submitted_this_pass=submitted, reserved_this_pass=pending_before + len(gpu_records([records[i] for i in submitted])),
                      submitted_workers=len(gpu_records(rows)),
                      episode_states=dict(Counter(r.get('state', 'UNKNOWN') for r in rows)),
                      verified_episodes=verified,
                      model_successes=sum(r.get("inspection", {}).get("verified") is True and
                                          r.get("inspection", {}).get("model_success") is True for r in rows),
                      block_reason=reason, stopped=reason == "STOP",
                      finished=finished, completion_state=completion_state,
                      awaiting_terminal_artifacts=awaiting_artifacts,
                      automatic_recovery_enabled=bool(policy), awaiting_recovery=recovery_wait,
                      recovered_this_pass=getattr(self, 'recovered_this_pass', []),
                      recovery_attention=[r['index'] for r in rows if r.get('recovery_decision')
                          and r['recovery_decision'].get('eligible') is False and not r['recovery_decision'].get('waiting')],
                      artifact_grace_seconds=ARTIFACT_GRACE_SECONDS,
                      round_preparation_error=getattr(self, "round_preparation_error", None),
                      inspection_errors=[r["index"] for r in rows if r.get("inspection_error") or
                                         r.get("inspection", {}).get("verified") is False])
        write_json(self.root / "status.json", status)
        return status

    def run_once(self) -> dict:
        with self.lock():
            return self._run_once()

    def run(self, *, once: bool = False, poll_seconds: float = 45.0) -> dict:
        if poll_seconds < 1:
            raise ValueError("poll_seconds must be at least 1")
        with self.lock():
            while True:
                status = self._run_once()
                print(json.dumps(status, sort_keys=True), flush=True)
                if once or status["stopped"] or status["finished"]:
                    return status
                # Start the full delay after observation, acceptance and submission.
                until = time.monotonic() + poll_seconds
                while time.monotonic() < until and not (self.root / "STOP").exists():
                    time.sleep(min(1.0, max(0.0, until - time.monotonic())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adapter", default="video_eval_adapter")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=45.0)
    args = parser.parse_args()
    module = importlib.import_module(args.adapter)
    plan = json.loads((args.root / "plan.json").read_text())
    adapter = module.factory(args.root.resolve(), plan) if hasattr(module, "factory") else module
    Controller(args.root, adapter).run(once=args.once, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
