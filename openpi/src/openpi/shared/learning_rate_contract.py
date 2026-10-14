"""Frozen LR identity and optimizer-clock checks, independent of run length."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

FILENAME = "learning_rate_contract.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def make_contract(schedule, *, peak_limit=None):
    if type(schedule).__name__ != "CosineDecaySchedule":
        raise ValueError("Guarded fine-tuning requires an explicit CosineDecaySchedule")
    fields = dataclasses.asdict(schedule)
    if set(fields) != {"warmup_steps", "peak_lr", "decay_steps", "decay_lr"}:
        raise ValueError("Unknown cosine schedule fields")
    warmup, end = fields['warmup_steps'], fields['decay_steps']
    if type(warmup) is not int or type(end) is not int or not 0 < warmup < end:
        raise ValueError("Require 0 < warmup_steps < decay_steps")
    peak, floor = fields['peak_lr'], fields['decay_lr']
    peak_limit = peak if peak_limit is None else peak_limit
    if not all(math.isfinite(v) and v > 0 for v in (peak, floor, peak_limit)) or floor > peak > 0 or peak > peak_limit:
        raise ValueError("Learning rate must be finite, positive and below its frozen peak limit")
    value = dict(schema_version=1, kind="b1k_learning_rate_contract", schedule=fields,
                 peak_limit=peak_limit, clock="completed_optimizer_updates_in_this_run",
                 initial_lr=peak/(warmup+1), after_decay="constant_floor")
    value['sha256'] = hashlib.sha256(canonical(value).encode()).hexdigest()
    return value


def expected_rate(contract, step):
    s = contract['schedule']; w = s['warmup_steps']
    if step < w:
        return contract['initial_lr'] + (s['peak_lr']-contract['initial_lr'])*step/w
    ratio = min(max((step-w)/(s['decay_steps']-w), 0), 1)
    return s['decay_lr'] + .5*(s['peak_lr']-s['decay_lr'])*(1+math.cos(math.pi*ratio))


def check_schedule(schedule, expected_path):
    expected = json.loads(Path(expected_path).read_text())
    actual = make_contract(schedule, peak_limit=expected['peak_limit'])
    if actual != expected:
        raise ValueError("Learning-rate schedule differs from the frozen contract; budget must not stretch decay")
    # Exercise the real Optax schedule, not only the independent reconstruction.
    s = actual['schedule']
    points = sorted({0, 1, s['warmup_steps'], 1000, 5000, 10000, s['decay_steps'], s['decay_steps']*2})
    fn = schedule.create()
    for step in points:
        observed = float(fn(step))
        if not math.isfinite(observed) or not math.isclose(observed, expected_rate(actual, step), rel_tol=2e-5, abs_tol=1e-12):
            raise ValueError(f"Runtime learning rate differs at optimizer update {step}")
        if observed > actual['peak_limit'] * (1+2e-6):
            raise ValueError("Runtime learning rate exceeds the frozen peak limit")
    return actual


def save(directory, contract):
    if contract is not None:
        (Path(directory)/FILENAME).write_text(json.dumps(contract, sort_keys=True, indent=2)+'\n')


def check_resume(directory, contract):
    path = Path(directory)/FILENAME
    if contract is None:
        if path.exists():
            raise ValueError("Cannot resume a guarded checkpoint without its learning-rate contract")
        return
    if not path.exists() or json.loads(path.read_text()) != contract:
        raise ValueError("Checkpoint learning-rate contract is missing or changed")


def optimizer_schedule_step(state):
    """Find the actual schedule counter inside Optax's optimizer state."""
    found = []
    def visit(value):
        if type(value).__name__ == 'ScaleByScheduleState':
            found.append(value.count)
        elif isinstance(value, dict):
            for child in value.values(): visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value: visit(child)
    visit(state)
    if len(found) != 1:
        raise ValueError("Expected exactly one Optax schedule clock")
    return found[0]


def check_optimizer_clock(state, completed_steps):
    if int(optimizer_schedule_step(state)) != int(completed_steps):
        raise ValueError("Optimizer schedule counter differs from completed training updates")
