#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def final_q_score(result: dict) -> float:
    value = result.get("q_score", 0.0)
    if isinstance(value, dict):
        value = value.get("final", 0.0)
    return float(value or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    results = []
    invalid_files = []
    for path in sorted(args.results_root.glob("pi05/*/json/*.json")):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or "task" not in value or "success" not in value:
                raise ValueError("missing task or success")
            results.append(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            invalid_files.append({"path": str(path), "error": str(exc)})

    per_task = {}
    for result in results:
        task = str(result["task"])
        stats = per_task.setdefault(task, {"rollouts": 0, "successes": 0, "q_score_sum": 0.0})
        stats["rollouts"] += 1
        stats["successes"] += int(bool(result["success"]))
        stats["q_score_sum"] += final_q_score(result)

    for stats in per_task.values():
        count = stats["rollouts"]
        stats["success_rate"] = stats["successes"] / count
        stats["mean_q_score"] = stats.pop("q_score_sum") / count

    count = len(results)
    successes = sum(bool(result["success"]) for result in results)
    summary = {
        "run_id": args.run_id,
        "profile": args.profile,
        "checkpoint": args.checkpoint,
        "expected_rollouts": args.expected_rollouts,
        "completed_rollouts": count,
        "missing_rollouts": max(args.expected_rollouts - count, 0),
        "successes": successes,
        "success_rate": successes / count if count else 0.0,
        "mean_q_score": sum(final_q_score(result) for result in results) / count if count else 0.0,
        "invalid_files": invalid_files,
        "tasks": per_task,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and (count != args.expected_rollouts or invalid_files):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
