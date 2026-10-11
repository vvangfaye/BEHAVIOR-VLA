#!/usr/bin/env python3
"""Freeze a finite three-source schedule, optionally extending a trained prefix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openpi/src"))
from openpi.training.subtask_mixture import MixturePlan, compare_consumed_contract, consumed_contract


def make_source(kind, paths, weight, weak_groups=(), weak_multiplier=3.0):
    if not math.isfinite(weak_multiplier) or weak_multiplier <= 0:
        raise ValueError("Weak-group multiplier must be finite and positive.")
    refs = []
    available = set()
    for path in paths:
        path = Path(path).resolve()
        raw = path.read_bytes()
        index = json.loads(raw)
        records = {row["episode_index"]: row for row in index["episodes"]}
        available.update((records[row["episode_index"]]["task_name"], row["subtask"]) for row in index["intervals"])
        refs.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    result = {"kind": kind, "weight": weight, "indexes": refs}
    if weak_groups:
        result["group_weights"] = [{"task_name": task, "subtask": subtask, "weight": weak_multiplier}
                                   for task, subtask in sorted(available & set(weak_groups))]
    return result


def weak_groups_from_file(path):
    if path is None:
        return set()
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        value = value.get("requested_groups")
    if not isinstance(value, list):
        raise ValueError("Weak groups must be a list or a frozen campaign's requested_groups.")
    groups = set()
    for row in value:
        task, subtask = row.get("task_name", row.get("task")), row.get("subtask")
        if not isinstance(task, str) or not task.strip() or not isinstance(subtask, str) or not subtask.strip():
            raise ValueError("Weak groups require canonical task and subtask strings.")
        groups.add((task, subtask))
    return groups


def build_manifest(sources, *, end_step, batch_size=128, seed=42, previous_path=None, start_step=0):
    if type(end_step) is not int or type(start_step) is not int or start_step < 0 or end_step <= start_step:
        raise ValueError("Specify an absolute end_step greater than the completed start_step.")
    phases = []
    previous = None
    if previous_path is not None:
        previous = MixturePlan.load(previous_path)
        if start_step == 0 or start_step > previous.num_steps:
            raise ValueError("Continuation starts at an explicit completed step within the previous schedule.")
        if (batch_size, seed, 32) != (previous.global_batch_size, previous.seed, previous.action_horizon):
            raise ValueError("Continuation must preserve global batch, seed and action horizon.")
        for original in previous.manifest["phases"]:
            if original["start_step"] >= start_step:
                break
            phase = copy.deepcopy(original)
            phase["end_step"] = min(start_step, phase["end_step"])
            for source in phase["sources"]:
                for ref in source["indexes"]:
                    ref["path"] = str((Path(previous_path).resolve().parent / ref["path"]).resolve())
            phases.append(phase)
    elif start_step:
        raise ValueError("A nonzero start_step requires the original frozen mixture manifest.")
    phases.append({"start_step": start_step, "end_step": end_step, "sources": sources})
    manifest = {"schema_version": 1, "kind": "b1k_three_source_mixture", "seed": seed,
                "global_batch_size": batch_size, "action_horizon": 32, "phases": phases}
    plan = MixturePlan(manifest, base_dir=Path.cwd())
    if previous is not None:
        compare_consumed_contract(consumed_contract(previous, start_step), plan, start_step)
    return manifest, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in ("recovered", "collected", "replay"):
        parser.add_argument(f"--{kind}-index", action="append", type=Path, required=True,
                            help="Repeat for additional frozen indexes; weight applies to the whole source kind.")
    parser.add_argument("--recovered-weight", type=float, default=.8)
    parser.add_argument("--collected-weight", type=float, default=.1)
    parser.add_argument("--replay-weight", type=float, default=.1)
    parser.add_argument("--weak-groups", type=Path, help="Frozen 299-group campaign or [{task, subtask}].")
    parser.add_argument("--weak-group-multiplier", type=float, default=3.0,
                        help="Sampling weight for weak groups; all other groups retain weight 1.")
    parser.add_argument("--end-step", type=int, required=True, help="Absolute optimizer update budget.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--previous-mixture", type=Path)
    parser.add_argument("--start-step", type=int, default=0,
                        help="Completed optimizer step when appending a new data phase; verified again during resume.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path; frozen mixtures must not be overwritten.")
    groups = weak_groups_from_file(args.weak_groups)
    sources = [make_source(kind, getattr(args, kind + "_index"), getattr(args, kind + "_weight"),
                           groups, args.weak_group_multiplier) for kind in ("recovered", "collected", "replay")]
    manifest, plan = build_manifest(sources, end_step=args.end_step, batch_size=args.batch_size, seed=args.seed,
                                    previous_path=args.previous_mixture, start_step=args.start_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"mixture": str(args.output), "fingerprint": plan.fingerprint,
                      "global_batch_size": plan.global_batch_size, "end_step": plan.num_steps,
                      "phase_start_step": args.start_step, "sources": [
                          {"kind": source["kind"], "weight": source["weight"], "indexes": len(source["indexes"]),
                           "boosted_groups": len(source.get("group_weights", []))} for source in sources],
                      "videos_decoded": 0, "training_started": False}))


if __name__ == "__main__":
    main()
