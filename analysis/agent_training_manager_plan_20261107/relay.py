#!/usr/bin/env python3
"""Serve only the new PPO controller's frozen schema and run allowlist."""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
CLI_PROXY_ENV = {key: os.environ[key] for key in PROXY_KEYS if key in os.environ}
sys.path.insert(0, str(ROOT / "runtime/RLinf"))
sys.path.insert(0, str(ROOT.parent / "cluster_probe_20261104"))
import aoss_io
from rlinf.b1k_tools.agent_manager import CODEX, _codex_args, _ledger_reserve
from rlinf.b1k_tools.ppo_update_controller import HEADER, SCHEMA, atomic_json


def ask_logged(prompt, directory):
    if not _ledger_reserve():
        return {"error": "daily_cap_reached"}
    directory.mkdir(parents=True, exist_ok=True)
    schema_path = directory / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA))
    (directory / "prompt.txt").write_text(prompt)
    output_path = directory / "last.txt"
    env, restrictions = _codex_args()
    env.update(CLI_PROXY_ENV)
    command = [CODEX, "exec", "-", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
               "--output-schema", str(schema_path), "--output-last-message", str(output_path),
               "--color", "never", "--json", "--model", "gpt-6.1-sol",
               "--config", 'model_reasoning_effort="high"', "--config", 'service_tier="ultrafast"',
               *restrictions]
    started = time.time()
    with (directory / "cli_events.jsonl").open("w") as events, (directory / "cli_stderr.txt").open("w") as errors:
        try:
            result = subprocess.run(command, input=prompt, env=env, cwd=directory,
                                    stdout=events, stderr=errors, text=True, timeout=150)
        except subprocess.TimeoutExpired:
            return {"error": "cli_timeout", "_seconds": round(time.time() - started, 1)}
    atomic_json(directory / "cli_status.json", {"returncode": result.returncode,
                "output_exists": output_path.exists(), "seconds": round(time.time() - started, 1)})
    if result.returncode or not output_path.exists():
        return {"error": f"cli_exit_{result.returncode}_output_{int(output_path.exists())}"}
    try:
        answer = json.loads(output_path.read_text())
    except (ValueError, OSError):
        return {"error": "invalid_cli_output"}
    answer["_seconds"] = round(time.time() - started, 1)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    if not all(re.fullmatch(r"pc_[a-z0-9_]+", tag) for tag in args.tags):
        raise ValueError("Relay run tags must belong to this PPO-control campaign")
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)
    client = aoss_io.local()
    state_path = ROOT / "relay_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    with (ROOT / "relay.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + args.hours * 3600
        while time.monotonic() < deadline:
            heartbeat = {"time": time.time(), "tags": args.tags, "errors": []}
            for tag in args.tags:
                key = f"rlinf_runs_20261104/{tag}/relay/train/req.json"
                try:
                    request = json.loads(client.get_object(Bucket=aoss_io.BUCKET, Key=key)["Body"].read())
                except client.exceptions.NoSuchKey:
                    continue
                except Exception as error:
                    heartbeat["errors"].append({"tag": tag, "type": type(error).__name__})
                    continue
                nonce = request.get("nonce", "")
                if state.get(tag) == nonce:
                    continue
                prompt = request.get("prompt", "")
                if not re.fullmatch(r"[0-9a-f]{32}", nonce) or request.get("schema") != SCHEMA:
                    heartbeat["errors"].append({"tag": tag, "type": "InvalidRequest"})
                    continue
                age = time.time() - float(request.get("time", 0))
                if not prompt.startswith(HEADER) or len(prompt) > 20000 or not -60 <= age <= 7200:
                    continue
                directory = ROOT / "relay_calls" / tag / nonce
                answer_path = directory / "answer.json"
                if answer_path.exists():
                    answer = json.loads(answer_path.read_text())
                else:
                    atomic_json(directory / "request.json", request)
                    answer = ask_logged(prompt, directory)
                    atomic_json(answer_path, answer)
                response = {"nonce": nonce, "time": time.time(), "answer": answer}
                answer_key = key.replace("req.json", "ans.json")
                try:
                    client.put_object(Bucket=aoss_io.BUCKET, Key=answer_key, Body=json.dumps(response).encode())
                except Exception as error:
                    heartbeat["errors"].append({"tag": tag, "type": type(error).__name__})
                    continue
                state[tag] = nonce
                atomic_json(state_path, state)
                print(json.dumps({"tag": tag, "mode": answer.get("mode"), "error": answer.get("error")}), flush=True)
            atomic_json(ROOT / "relay_health.json", heartbeat)
            time.sleep(10)


if __name__ == "__main__":
    main()
