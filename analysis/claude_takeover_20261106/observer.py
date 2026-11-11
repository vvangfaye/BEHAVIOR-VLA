#!/usr/bin/env python3
"""Observe the seven existing experiments from local mirrors; never submit or control jobs."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LOCAL = ROOT / "analysis/libero_feasibility_20261103/results"
CLOUD = ROOT / "analysis/cluster_probe_20261104/cluster_runs"
sys.path.insert(0, str(LOCAL.parent))
from paired import mcnemar_exact  # noqa: E402

TZ = dt.timezone(dt.timedelta(hours=8))
PATTERNS = {
    "libero": re.compile(r"\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)"),
    "calvin": re.compile(r"\[calvin eval\] trial_id=(\d+), completed=(\d+)"),
    "robocasa": re.compile(r"\[robocasa eval\] task=(\w+), env=(\d+), ep=(\d+), seed=\d+, success=(True|False)"),
}
ARMS = {
    "sft": "eval_sft_c10", "ppo_s1": "eval_balanced_s20_c10",
    "ppo_s2": "eval_seed2_5090_s20_c10", "ppo_s3": "eval_balanced_seed3_s20_c10",
    "subgoal_s1": "cluster_libero_seed1_subgoal_eval_s20",
    "subgoal_s2": "cluster_libero_seed2_subgoal_eval_s20",
    "subgoal_s3": "eval_ppo_balanced_seed3_subgoal_s20_c10",
    "agent_v2_s1": "cluster_libero_seed1_mgr_agent_eval_agent_s20",
    "agent_v2_s2": "cluster_libero_seed2_mgr_agent_r3_eval_agent_s20",
    "agent_v2_s3": "eval_ppo_balanced_seed3_agent_v2_s20_c10",
    "rule_s3": "eval_ppo_balanced_seed3_mgr_rule_s20_c10",
    "replay_lr_s3": "eval_ppo_balanced_seed3_mgr_replay_lr_s20_c10",
    "random_s3": "eval_ppo_balanced_seed3_mgr_random_s20_c10",
    "pirl_s20": "cluster_pirl_p1b_eval_s20",
}
JOBS = {
    "robocasa_mgr_agent": ("robocasa", ["eval_s10", "eval_s20"]),
    "robocasa_progress_ppo_r2": ("robocasa", ["eval_s10", "eval_s20"]),
    "libero_seed2_mgr_rule_random_r2": ("libero", ["eval_rule_s20", "eval_random_s20"]),
    "libero_seed2_mgr_agent_v3": ("libero", ["eval_agent_s20"]),
    "libero_seed1_mgr_agent_v3": ("libero", ["eval_agent_s20"]),
    "calvin_s1_mgr_agent_v3": ("calvin", ["eval_s20"]),
}


def stamp(path):
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {"path": str(path), "exists": True, "bytes": st.st_size,
            "modified_cst": dt.datetime.fromtimestamp(st.st_mtime, TZ).isoformat(),
            "age_seconds": round(max(0, time.time() - st.st_mtime), 1)}


def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        return {"read_error": type(exc).__name__}


def evaluate(path, benchmark):
    info = {"source": stamp(path), "benchmark": benchmark, "complete": False}
    values = {}
    if not path.exists():
        return info, values
    duplicates = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            match = PATTERNS[benchmark].search(line)
            if not match:
                continue
            groups = match.groups()
            value = int(groups[-1]) if benchmark == "calvin" else groups[-1] == "True"
            key = tuple(groups[:-1])
            if key in values:
                duplicates += 1
                if values[key] != value:
                    info.update(error="conflicting_duplicate_episode", conflict_key=key)
                    return info, {}
            values[key] = value
    expected = {"libero": 500, "calvin": 1000, "robocasa": 288}[benchmark]
    info.update(unique_episodes=len(values), expected_episodes=expected, identical_duplicates=duplicates)
    if len(values) > expected:
        info["error"] = "too_many_unique_episodes"
        return info, {}
    info["complete"] = len(values) == expected
    if benchmark == "libero" and info["complete"]:
        expected_keys = {(str(t), str(i)) for t in range(10) for i in range(50)}
        if set(values) != expected_keys:
            info.update(complete=False, error="unexpected_task_trial_identities")
            return info, {}
    if values:
        name = "avg_completed_subtasks" if benchmark == "calvin" else "success_rate"
        info[name] = sum(values.values()) / len(values)
        info["metric_scope"] = "final" if info["complete"] else "partial_evaluation"
        if benchmark == "libero":
            info["per_task_successes"] = {str(t): sum(v for k, v in values.items() if k[0] == str(t)) for t in range(10)}
    return info, values


def paired(reference, other):
    if len(reference) != 500 or set(reference) != set(other):
        return {"error": "evaluation_identities_do_not_match"}
    losses = sum(reference[k] and not other[k] for k in reference)
    gains = sum(other[k] and not reference[k] for k in reference)
    return {"pairs": 500, "gains": gains, "losses": losses,
            "delta_success_rate": (gains - losses) / 500,
            "mcnemar_exact_p": mcnemar_exact(losses, gains)}


def snapshot():
    state = {"observed_cst": dt.datetime.now(TZ).isoformat(), "read_only": True,
             "scope": "Existing local files and mirrors only; no scheduler queries, retries, or model calls.",
             "mirror_freshness_note": "File ages indicate mirror freshness, not proof of worker liveness.",
             "libero": {}, "pending": {}}
    raw = {}
    for name, tag in ARMS.items():
        state["libero"][name], raw[name] = evaluate(LOCAL / tag / "console.log", "libero")
    for name, values in raw.items():
        if name != "sft" and state["libero"][name]["complete"]:
            state["libero"][name]["vs_sft"] = paired(raw["sft"], values)
        seed = name[-2:]
        if name.startswith(("agent_", "rule_", "replay_lr_", "random_", "subgoal_")) and state["libero"][name]["complete"]:
            state["libero"][name]["vs_same_seed_ppo"] = paired(raw.get("ppo_" + seed, {}), values)
    random_eval = state["libero"]["random_s3"]
    local_log = LOCAL / "ppo_balanced_seed3_mgr_random/console.log"
    decision_file = Path("/mnt/data1/faye/agent_mgr_logs/ppo_balanced_seed3_mgr_random/am/decisions.jsonl")
    decisions = sum(bool(line.strip()) for line in decision_file.read_text().splitlines()) if decision_file.exists() else 0
    state["pending"]["local_random_s3"] = {"settled": random_eval["complete"],
        "state": "evaluation_complete" if random_eval["complete"] else "awaiting_final_evaluation",
        "manager_decisions": decisions, "training_log": stamp(local_log), "evaluation": random_eval}
    for tag, (benchmark, stages) in JOBS.items():
        directory = CLOUD / tag
        status = read_json(directory / "status.json")
        stage_status = read_json(directory / "tb/stages_status.json")
        evaluations = {}
        for stage in stages:
            info, values = evaluate(directory / "tb" / stage / "console.log", benchmark)
            if benchmark == "libero" and info["complete"]:
                info["vs_sft"] = paired(raw["sft"], values)
                seed = "s1" if "seed1" in tag else "s2"
                info["vs_same_seed_ppo"] = paired(raw["ppo_" + seed], values)
            evaluations[stage] = info
        terminal = str(status.get("state", "")).startswith("finished_rc_")
        failed = terminal and status["state"] != "finished_rc_0"
        complete = all(v["complete"] for v in evaluations.values())
        receipt = CLOUD.parent / "runs" / f"{tag}.json"
        current = stage_status.get("current")
        # The root console may still contain the previous stage's final epoch.
        console = directory / "tb" / current / "console.log" if current else directory / "console.log"
        progress = []
        if console.exists():
            progress = re.findall(r"Global Step:\s*(\d+)/(\d+)", console.read_text(errors="replace"))
        state["pending"][tag] = {"state": status.get("state", "submitted_no_run_evidence" if receipt.exists() else "no_submission_evidence"),
            "current_stage": stage_status.get("current"), "stage_results": stage_status.get("results", {}),
            "latest_logged_global_step": list(map(int, progress[-1])) if progress else None,
            "settled": complete or failed, "failed": failed,
            "status_source": stamp(directory / "status.json"), "receipt_source": stamp(receipt), "training_log": stamp(console),
            "evaluations": evaluations}
    state["all_settled"] = all(p["settled"] for p in state["pending"].values())
    return state


def meaningful(value):
    if isinstance(value, dict):
        return {k: meaningful(v) for k, v in value.items() if k not in {
            "observed_cst", "age_seconds", "modified_cst", "bytes"}}
    if isinstance(value, list):
        return [meaningful(v) for v in value]
    return value


def publish(state):
    target = HERE / "state.json"
    previous = read_json(target)
    signature = hashlib.sha256(json.dumps(meaningful(state), sort_keys=True).encode()).hexdigest()
    if previous.get("signature") != signature:
        with (HERE / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"observed_cst": state["observed_cst"], "event": "snapshot_changed",
                "signature": signature, "all_settled": state["all_settled"],
                "pending": meaningful(state["pending"])}, ensure_ascii=False) + "\n")
    state["signature"] = signature
    temporary = HERE / f"state.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--deadline-hours", type=float, default=18)
    parser.add_argument("--interval", type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.deadline_hours <= 24 or args.interval < 30:
        parser.error("deadline must be 0..24 hours, interval at least 30 seconds")
    with (HERE / "observer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An observer already holds the lock; not starting another.")
        deadline = time.monotonic() + args.deadline_hours * 3600
        while True:
            state = snapshot()
            publish(state)
            print(json.dumps({"observed_cst": state["observed_cst"], "all_settled": state["all_settled"],
                "settled": sum(p["settled"] for p in state["pending"].values()), "tracked": len(state["pending"])}), flush=True)
            if args.once or state["all_settled"] or time.monotonic() >= deadline:
                break
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
