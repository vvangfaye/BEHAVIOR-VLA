#!/usr/bin/env python3
"""Local side of the AOSS relay for agent-managed cluster runs (2026-11-05).

Cluster jobs have no internet, so a manager in mode=relay (rlinf/b1k_tools/agent_manager.py) PUTs its prompt to
aoss://260010165/rlinf_runs_20261104/<tag>/relay/<stage>/req.json through a presigned URL and polls ans.json. This
daemon lists those request objects every --interval seconds, answers each new nonce once with the same Codex call as
the local manager (gpt-6.1-sol, tools/web/MCP disabled, shared daily ledger) and PUTs {nonce, answer} to ans.json.
Only manager prompts are forwarded (fixed header, size limit, fresh timestamp); they contain aggregate per-task
training counts and the public task instructions. Prompts and answers are kept under /mnt/data1/faye/agent_mgr_logs/relay.
Run with boto3: uv run --offline --with boto3 python sol_relay.py TAG [TAG...] [--interval 30] [--once]"""
import importlib.util
import inspect
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cluster_probe_20261104"))
import aoss_io  # noqa: E402

AM_PATH = Path("/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/rlinf/b1k_tools/agent_manager.py")
_spec = importlib.util.spec_from_file_location("b1k_agent_manager", AM_PATH)
AM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AM)

STATE = HERE / "sol_relay_state.json"
LOG = HERE / "sol_relay.log"
LOGROOT = Path("/mnt/data1/faye/agent_mgr_logs/relay")
HEADER = "You manage online reinforcement-learning fine-tuning"
MAX_PROMPT, MAX_AGE_S = 20000, 2 * 3600


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def valid_schema(sch):
    """Accept only the manager's own output schema (same fields as AM.SCHEMA, 1..64 tasks); None otherwise."""
    try:
        props = sch["properties"]
        if set(props) != set(AM.SCHEMA["properties"]) or sch.get("required") != AM.SCHEMA["required"]:
            return None
        n = props["task_weights"]["minItems"]
        if not (isinstance(n, int) and 1 <= n <= 64 and props["task_weights"]["maxItems"] == n
                and props["prob_regressed"]["minItems"] == n == props["prob_regressed"]["maxItems"]):
            return None
        refs = [AM.schema_version(n, v) for v in AM.VERSIONS] if hasattr(AM, "VERSIONS") else []
        ref = json.loads(json.dumps(AM.SCHEMA))
        for k in ("task_weights", "prob_regressed"):
            ref["properties"][k]["minItems"] = ref["properties"][k]["maxItems"] = n
        refs.append(ref)
        return sch if any(r == sch for r in refs) else None  # identical to a manager version's schema up to the task count
    except Exception:  # noqa: BLE001
        return None


def once(s3, tags, state):
    for tag in tags:
        prefix = f"rlinf_runs_20261104/{tag}/relay/"
        resp = s3.list_objects_v2(Bucket=aoss_io.BUCKET, Prefix=prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            m = re.fullmatch(re.escape(prefix) + r"([A-Za-z0-9_.-]{1,64})/req\.json", key)
            if not m:
                continue
            stage = m.group(1)
            try:
                req = json.loads(s3.get_object(Bucket=aoss_io.BUCKET, Key=key)["Body"].read().decode())
            except Exception as e:  # noqa: BLE001
                log(f"{tag}/{stage}: unreadable request ({type(e).__name__})")
                continue
            nonce, prompt = str(req.get("nonce", "")), str(req.get("prompt", ""))
            sid = f"{tag}/{stage}"
            if state.get(sid) == nonce:
                continue
            bad = None
            if not re.fullmatch(r"[0-9a-f]{32}", nonce):
                bad = "bad nonce"
            elif not prompt.startswith(HEADER) or len(prompt) > MAX_PROMPT:
                bad = "not a manager prompt"
            elif time.time() - float(req.get("time", 0)) > MAX_AGE_S:
                bad = "stale request"
            if bad:
                log(f"{sid}: skipped request {nonce[:8]} ({bad})")
                state[sid] = nonce
                STATE.write_text(json.dumps(state, indent=1))
                continue
            n = sum(1 for k in state if k.startswith(sid + "#")) + 1
            ev = LOGROOT / tag / stage / f"{n:03d}_{nonce[:8]}"
            t0 = time.time()
            sch = valid_schema(req.get("schema"))
            if req.get("schema") is not None and sch is None:
                log(f"{sid}: skipped request {nonce[:8]} (unexpected schema)")
                state[sid] = nonce
                STATE.write_text(json.dumps(state, indent=1))
                continue
            if sch is not None and "schema" in inspect.signature(AM.ask_agent).parameters:
                ans = AM.ask_agent(prompt, ev, schema=sch)  # sized to the run's task count (RoboCasa: 6)
            else:
                ans = AM.ask_agent(prompt, ev)
            out = {"nonce": nonce, "time": time.time(), "answer": ans}
            (ev / "relay_answer.json").write_text(json.dumps(out, indent=1))
            s3.put_object(Bucket=aoss_io.BUCKET, Key=f"{prefix}{stage}/ans.json", Body=json.dumps(out).encode())
            state[sid] = nonce
            state[f"{sid}#{n:03d}"] = nonce
            STATE.write_text(json.dumps(state, indent=1))
            summ = "error " + ans["error"] if "error" in ans else f"lr x{ans.get('lr_multiplier')} weights {ans.get('task_weights')}"
            log(f"{sid}: answered request {n} ({nonce[:8]}) in {time.time() - t0:.0f} s: {summ}")


def main():
    args = sys.argv[1:]
    interval = float(args[args.index("--interval") + 1]) if "--interval" in args else 30.0
    single = "--once" in args
    tags = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--interval")]
    assert tags, __doc__
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    s3 = aoss_io.local()
    log(f"relay up for {tags} (interval {interval:.0f} s)")
    while True:
        try:
            once(s3, tags, state)
        except Exception as e:  # noqa: BLE001  (transient AOSS errors must not kill the relay)
            log(f"poll error {type(e).__name__}: {str(e)[:200]}")
        if single:
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
