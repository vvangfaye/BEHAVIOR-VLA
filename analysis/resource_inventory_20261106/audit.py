"""Read-only resource inventory. Never persists job commands or environments."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
STATES = ["WAITING", "INIT", "QUEUEING", "CREATING", "STARTING", "RESTARTING",
          "RECOVERING", "RUNNING", "SUSPENDING", "DELETING"]


def call(arguments):
    result = subprocess.run(["sco", *arguments], capture_output=True, text=True, timeout=60)
    return {"arguments": arguments, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "queried_at_utc": datetime.now(timezone.utc).isoformat()}


def brief(job):
    roles = []
    for role in job.get("roles", []):
        specs = role.get("resource_spec", [])
        if isinstance(specs, dict):
            specs = [specs]
        roles.append({"name": role.get("name"), "total_replicas": role.get("total_replicas"),
                      "resource_spec": specs})
    gpus = sum(int(spec.get("limits", {}).get("nvidia.com/gpu", 0)) *
               max(1, int(spec.get("replicas") or 1), int(role.get("total_replicas") or 1))
               for role in roles for spec in role["resource_spec"])
    return {"name": job.get("name"), "display_name": job.get("display_name"), "uid": job.get("uid"),
            "state": job.get("state"), "owner": job.get("ownership", {}).get("user_name"),
            "quota_type": job.get("scheduling", {}).get("quota_type"),
            "pool": job.get("resource_pool", {}).get("name"), "gpus": gpus, "roles": roles,
            "create_time": job.get("create_time"), "start_time": job.get("start_time")}


def active_state(state):
    jobs, seen = [], set()
    for page in range(1, 21):
        row = call(["acp", "jobs", "list", "--workspace-name=share-space", "--user-name=260010165",
                    "--state="+state, "--page-size=20", "--page-token="+str(page), "-o", "json"])
        if row["returncode"]:
            return {"state": state, "complete": False, "error": row["stderr"][-800:], "jobs": jobs}
        payload = row["stdout"].strip()
        batch = [] if payload in ("No jobs found", "No jobs found.") else json.loads(payload)
        if not isinstance(batch, list):
            raise ValueError("Unexpected job-list schema")
        for j in batch:
            if j["uid"] in seen or j["state"] != state or j["ownership"]["user_name"] != "260010165":
                raise ValueError("Repeated page or mismatched owner/state")
            seen.add(j["uid"])
            jobs.append(brief(j))
        if len(batch) < 20:
            return {"state": state, "complete": True, "jobs": jobs}
    return {"state": state, "complete": False, "error": "pagination limit", "jobs": jobs}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(active_state, STATES))
    jobs = {j["uid"]: j for r in results for j in r["jobs"]}
    snapshot = {"queried_at_utc": datetime.now(timezone.utc).isoformat(), "owner": "260010165",
                "workspace": "share-space", "complete": all(r["complete"] for r in results),
                "states": results, "jobs": list(jobs.values())}
    snapshot["active_reserved_gpus"] = sum(j["gpus"] for j in jobs.values() if j["quota_type"] == "RESERVED")
    snapshot["active_spot_gpus"] = sum(j["gpus"] for j in jobs.values() if j["quota_type"] == "SPOT")
    (ROOT / "active_jobs.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in snapshot.items() if k != "states"}, ensure_ascii=False))
    commands = [
        ["aec2", "clusters", "list", "--wide"],
        ["aec2", "clusters", "usage", "--name=share-cluster"],
        ["aec2", "clusters", "usage", "--name=computing-cluster-5090-01g"],
        ["aec2", "clusters", "list-workerspec", "--workspace-name=share-space", "--aec2-name=share-cluster"],
        ["aec2", "clusters", "list-workerspec", "--workspace-name=share-space", "--aec2-name=computing-cluster-5090-01g"],
        ["ws", "--help"],
        ["afs", "dir-quota", "list", "--volumeID=01a04263-91e5-7603-bc01-c67e503da6b5", "--dirPath=/260010165"],
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        inventory = list(executor.map(call, commands))
    (ROOT / "cluster_queries.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+"\n")
    for row in inventory:
        print(json.dumps({**row, "stdout": row["stdout"][:10000], "stderr": row["stderr"][-800:]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
