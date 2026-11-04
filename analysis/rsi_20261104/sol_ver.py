#!/usr/bin/env python3
"""External in-loop verifier (Ver) for the Physical-RSI loop (2026-11-04, user-approved scope).

Judges LIBERO episodes from frames dumped by the eval env (third-person camera, no overlays) with gpt-6.1-sol
(reasoning high, service tier ultrafast) through the local Codex CLI (ChatGPT login), the same way as
BEHAVIOR-VLAs/scripts/codex_collection_teacher.py: `codex exec -` with tools/web/MCP disabled, a JSON output
schema and the images attached. The verifier never sees simulator success; that stays with the independent
evaluator. Hard cap: 500 calls per calendar day (ledger sol_ledger.json). Evidence per call in <out>/<episode>/.

Usage: sol_ver.py judge <frames_root> <out_root> [--workers 6] [--max N]
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL, EFFORT, TIER = "gpt-6.1-sol", "high", "ultrafast"
DAILY_CAP = 500
LEDGER = HERE / "sol_ledger.json"
CODEX = "/home/faye/.local/bin/codex"
DISABLED = ("shell_tool", "apps", "plugins", "multi_agent", "multi_agent_v2", "browser_use", "browser_use_external",
            "computer_use", "image_generation", "code_mode", "code_mode_host", "view_image", "goals", "sleep_tool",
            "skill_search", "tool_suggest", "workspace_dependencies")
LIBERO10 = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    1: "put both the cream cheese box and the butter in the basket",
    2: "turn on the stove and put the moka pot on it",
    3: "put the black bowl in the bottom drawer of the cabinet and close it",
    4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    5: "pick up the book and place it in the back compartment of the caddy",
    6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    7: "put both the alphabet soup and the cream cheese box in the basket",
    8: "put both moka pots on the stove",
    9: "put the yellow and white mug in the microwave and close it",
}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["success", "progress", "reason"],
          "properties": {"success": {"type": "boolean"},
                         "progress": {"type": "number", "minimum": 0, "maximum": 1},
                         "reason": {"type": "string", "maxLength": 300}}}
PROMPT = """You are a strict evaluator of robot manipulation episodes recorded in a tabletop simulator.
The robot was given this instruction: "{instr}"
The {n} attached images are frames from a fixed third-person camera, in chronological order; the last image is the final state of the episode.
Using only the images, decide whether the robot fully completed the instruction at some point during the episode (partial completion is not success; every object named in the instruction must end up where the instruction says).
Also estimate task progress at the end of the episode, from 0 (nothing achieved) to 1 (fully achieved).
Answer with JSON only."""

_lock = threading.Lock()
_restrictions = None


def _ledger_reserve():
    """Reserve one call under the daily cap (thread lock + cross-process flock); False when the cap is reached."""
    import fcntl

    today = datetime.date.today().isoformat()
    with _lock, open(str(LEDGER) + ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
            if led.get(today, 0) >= DAILY_CAP:
                return False
            led[today] = led.get(today, 0) + 1
            LEDGER.write_text(json.dumps(led, indent=1))
            return True
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _setup():
    """Verify the ChatGPT login and build the tool/MCP restriction flags once."""
    global _restrictions
    if _restrictions is not None:
        return _restrictions
    env = {k: v for k, v in os.environ.items() if k not in {"CODEX_API_KEY", "OPENAI_API_KEY"}}
    auth = subprocess.run([CODEX, "login", "status"], env=env, capture_output=True, text=True, timeout=30)
    methods = re.findall(r"(?im)^\s*Logged in using (.+?)\s*$", (auth.stdout or "") + "\n" + (auth.stderr or ""))
    if auth.returncode != 0 or methods != ["ChatGPT"]:
        raise RuntimeError("Codex CLI must be logged in using ChatGPT")
    args = ["--config", 'web_search="disabled"']
    for f in DISABLED:
        args += ["--disable", f]
    mcp = subprocess.run([CODEX, "mcp", "list", "--json", *args], env=env, capture_output=True, text=True, timeout=30)
    for s in json.loads(mcp.stdout or "[]"):
        name = s.get("name") if isinstance(s, dict) else None
        if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,256}", name):
            args += ["--config", f"mcp_servers.{name}.enabled=false"]
    _restrictions = (env, args)
    return _restrictions


def pick_frames(ep_dir: Path, k: int = 6):
    fs = sorted(ep_dir.glob("step*.png"))
    if len(fs) <= k:
        return fs
    idx = sorted({round(i * (len(fs) - 1) / (k - 1)) for i in range(k)})
    return [fs[i] for i in idx]


def judge(ep_dir: Path, out_dir: Path, timeout: float = 240.0) -> dict:
    m = re.match(r"task(\d+)_trial(\d+)", ep_dir.name)
    tid = int(m.group(1))
    ev = out_dir / ep_dir.name
    resp = ev / "response.json"
    if resp.exists():  # idempotent: never re-query a judged episode
        return json.loads(resp.read_text())
    frames = pick_frames(ep_dir)
    if not frames:
        return {"episode": ep_dir.name, "error": "no_frames"}
    if not _ledger_reserve():
        return {"episode": ep_dir.name, "error": "daily_cap_reached"}
    env, restr = _setup()
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "schema.json").write_text(json.dumps(SCHEMA))
    prompt = PROMPT.format(instr=LIBERO10[tid], n=len(frames))
    (ev / "prompt.txt").write_text(prompt)
    (ev / "images.json").write_text(json.dumps([{"file": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                                                for f in frames], indent=1))
    argv = [CODEX, "exec", "-", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
            "--output-schema", str(ev / "schema.json"), "--output-last-message", str(ev / "last.txt"),
            "--color", "never", "--json", "--model", MODEL,
            "--config", "model_reasoning_effort=" + json.dumps(EFFORT), "--config", "service_tier=" + json.dumps(TIER),
            *restr]
    for f in frames:
        argv += ["--image", str(f)]
    t0 = time.time()
    try:
        p = subprocess.run(argv, input=prompt, env=env, cwd=ev, capture_output=True, text=True, timeout=timeout)
        rc, err = p.returncode, (p.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        rc, err = -9, "timeout"
    out = {"episode": ep_dir.name, "task_id": tid, "model": MODEL, "effort": EFFORT, "tier": TIER,
           "n_frames": len(frames), "seconds": round(time.time() - t0, 1), "rc": rc}
    try:
        ans = json.loads((ev / "last.txt").read_text())
        out.update(success=bool(ans["success"]), progress=float(ans["progress"]), reason=str(ans["reason"])[:300])
    except Exception as e:
        out.update(error=f"parse_failed: {type(e).__name__}", stderr_tail=err)
    if "error" not in out:
        resp.write_text(json.dumps(out, indent=1))
    else:
        (ev / "failed.json").write_text(json.dumps(out, indent=1))
    return out


def judge_all(frames_root: Path, out_root: Path, workers: int = 6, max_n: int | None = None):
    eps = sorted(d for d in Path(frames_root).iterdir() if d.is_dir() and re.match(r"task\d+_trial\d+", d.name))
    if max_n:
        eps = eps[:max_n]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda d: judge(d, Path(out_root)), eps))
    ok = [r for r in res if "error" not in r]
    summary = {"n": len(res), "n_ok": len(ok), "success_rate": (sum(r["success"] for r in ok) / len(ok)) if ok else None,
               "progress_mean": (sum(r["progress"] for r in ok) / len(ok)) if ok else None,
               "median_seconds": sorted(r["seconds"] for r in ok)[len(ok) // 2] if ok else None,
               "errors": sorted({r.get("error") for r in res if "error" in r})}
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "summary.json").write_text(json.dumps({"summary": summary, "episodes": res}, indent=1))
    return summary, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge"])
    ap.add_argument("frames_root")
    ap.add_argument("out_root")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()
    s, _ = judge_all(Path(a.frames_root), Path(a.out_root), a.workers, a.max)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
