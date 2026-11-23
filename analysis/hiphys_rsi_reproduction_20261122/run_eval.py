"""Run the stock BEHAVIOR evaluator against the radio program on a declared split."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "plan.json").read_text())
POLICY = ROOT / "radio_policy.py"
POLICY_READY_TIMEOUT = 120


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_summary(path, report):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(path)


def poll_ready(process, port):
    loopback = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    while process.poll() is None and time.monotonic() - started < POLICY_READY_TIMEOUT:
        try:
            with loopback.open(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except OSError:
            time.sleep(0.25)
    elapsed = time.monotonic() - started
    raise RuntimeError(f"Policy server did not become ready after {elapsed:.2f}s, exit={process.poll()}. Inspect policy.log.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="BEHAVIOR-1K checkout containing OmniGibson")
    parser.add_argument("--urdf", type=Path, required=True, help="Stock R1Pro URDF from the robot assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "public_test"), default="train")
    parser.add_argument("--train-ids", type=int, nargs="+")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--wall-seconds", type=int, default=7200)
    parser.add_argument("--freeze", type=Path, help="Manifest of a train-selected policy for public evaluation")
    parser.add_argument("--candidate", choices=("pipeline", "vla", "routed"), default="pipeline")
    parser.add_argument("--vla-port", type=int)
    args = parser.parse_args()

    policy_sha = sha256(POLICY)
    urdf_sha = sha256(args.urdf)
    if args.wall_seconds < 1:
        parser.error("--wall-seconds must be positive")
    if args.candidate in ("vla", "routed") and not args.vla_port:
        parser.error("VLA experiments require a running local policy server")
    if args.candidate == "routed" and args.freeze is None:
        parser.error("Use a train-selected policy card for the routed candidate")
    if args.mode == "train":
        ids = args.train_ids or PLAN["first_development_instance_ids"]
        if len(set(ids)) != len(ids) or not set(ids) <= set(PLAN["train_instance_ids"]):
            parser.error("Training IDs must be distinct members of the fixed development split")
        if args.max_steps is not None and args.max_steps < 1:
            parser.error("--max-steps must be positive")
        indices = ids
    else:
        if args.train_ids or args.max_steps is not None or args.freeze is None:
            parser.error("Public evaluation requires --freeze, all ten fixed cases and the default horizon")
        frozen = json.loads(args.freeze.read_text())
        if (frozen["task"], frozen["policy_sha256"], frozen["urdf_sha256"]) != (PLAN["task"], policy_sha, urdf_sha):
            parser.error("The current program or robot asset differs from the freeze manifest")
        training = []
        for summary_path in frozen["training_evidence"]:
            source = Path(summary_path)
            rows = json.loads(source.read_text())
            if (rows["mode"] != "train" or rows["policy_sha256"] != policy_sha or
                    rows["urdf_sha256"] != urdf_sha or not rows["complete"]):
                parser.error(f"Invalid training evidence at {source}")
            training.append(rows)
        if not training:
            parser.error("Freeze the candidate with completed train results before evaluating it on public")
        if args.candidate == "routed":
            from policy_card import build

            if len(training) != 2 or frozen["routes"] != build(*training)["routes"]:
                parser.error("Frozen routes disagree with the paired train evidence")
        ids = PLAN["public_instance_ids"]
        indices = list(range(len(ids)))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = (args.repo / "OmniGibson").resolve(strict=True)
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    server = [sys.executable, "-u", str(POLICY), "--urdf", str(args.urdf.resolve()),
              "--trace", str(output / "policy_trace.jsonl"), "--port", str(port), "--candidate", args.candidate]
    if args.vla_port:
        server.extend(["--vla-port", str(args.vla_port)])
    if args.candidate == "routed":
        server.extend(["--card", str(args.freeze.resolve())])
    command = [sys.executable, "-m", "omnigibson.eval.eval", "--task-name", PLAN["task"], "--mode", args.mode,
               "--instance-indices", *map(str, indices), "--num-rollouts", "1", "--prompt-mode", "whole_task",
               "--env-wrapper", "omnigibson.eval.wrappers.RGBDFullResWrapper", "--policy", "websocket",
               "--host", "127.0.0.1", "--port", str(port), "--write-video", "--output-dir", str(output)]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    report = {"task": PLAN["task"], "mode": args.mode, "instance_ids": ids, "policy_sha256": policy_sha,
              "urdf_sha256": urdf_sha, "max_steps_override": args.max_steps, "complete": False,
              "candidate": args.candidate, "evaluator_command": command}
    summary = output / "summary.json"
    save_summary(summary, report)
    policy_env = os.environ.copy()
    policy_env["CUDA_VISIBLE_DEVICES"] = ""
    with (output / "policy.log").open("w") as srv_log, (output / "evaluator.log").open("w") as eval_log:
        policy = subprocess.Popen(server, cwd=repo, env=policy_env, stdout=srv_log, stderr=subprocess.STDOUT)
        try:
            report["policy_ready_seconds"] = round(poll_ready(policy, port), 3)
            save_summary(summary, report)
            print("HIPHYS_POLICY_READY", json.dumps({"elapsed_seconds": report["policy_ready_seconds"]}), flush=True)
            subprocess.run(command, cwd=repo, stdout=eval_log, stderr=subprocess.STDOUT,
                           timeout=args.wall_seconds, check=True)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            report["execution_error"] = {"type": type(error).__name__, "message": str(error)}
            save_summary(summary, report)
            raise
        finally:
            if policy.poll() is None:
                policy.terminate()
                try:
                    policy.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    policy.kill()
                    policy.wait()

    trace = [json.loads(line) for line in (output / "policy_trace.jsonl").read_text().splitlines()]
    route_records = [row for row in trace if row.get("kind") == "episode_route"]
    if len(route_records) != len(ids):
        raise RuntimeError("Expected one initial routing record per scored episode")
    results = []
    for i, selection in zip(ids, route_records):
        name = f"{PLAN['task']}_{i}_0"
        metrics = json.loads((output / "json" / f"{name}.json").read_text())
        if ((metrics["task"], metrics["instance_id"], metrics["rollout_id"]) != (PLAN["task"], i, 0) or
                metrics["conditioning"]["oracle"] or metrics["conditioning"]["privileged"] or
                (output / "videos" / f"{name}.mp4").stat().st_size == 0):
            raise RuntimeError(f"Invalid artifacts for {name}")
        results.append({"instance_id": i, "q": metrics["q_score"]["final"], "success": bool(metrics["success"]),
                        "steps": metrics["steps"], "context": selection["context"], "selected": selection["candidate"],
                        "phase_transitions": dict(Counter(row["phase"] for row in trace if row["episode"] == selection["episode"]
                                                          and row.get("kind") in (None, "episode_route")))})
    report.update(complete=True, results=results, mean_q=sum(row["q"] for row in results) / len(ids),
                  full_success=sum(row["success"] for row in results) / len(ids))
    save_summary(summary, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
