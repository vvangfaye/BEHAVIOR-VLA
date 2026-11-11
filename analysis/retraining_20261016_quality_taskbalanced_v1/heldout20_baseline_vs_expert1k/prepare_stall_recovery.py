"""Prepare one unfinished episode after the confirmed batch-01 simulator stall."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import zlib

import controller

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "recovery_g1"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    if DEST.exists():
        raise FileExistsError("Inspect existing recovery; do not prepare a duplicate")
    plan = controller.load(ROOT / "plan.json")
    controller.verify_local(plan)
    old = controller.load(ROOT / "intents/batch-01.json")
    if old["uid"] != "e668d901e3dc4498a833a3b118f10b1b":
        raise ValueError("Unexpected original writer")
    group = next(g for g in plan["groups"] if g["id"] == "batch-01")
    pending, preserved = [], {}
    for item in group["members"]:
        identity = item["role"] + "-" + f"{item['index']:02d}"
        path = ROOT / "reports" / (identity + ".json")
        if path.exists() and controller.load(path).get("state") == "accepted":
            preserved[identity] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            pending.append(item)
    if pending != [{"role": "candidate", "index": 1}] or len(preserved) != 3:
        raise ValueError("Recovery must contain only the one unfinished episode")
    evidence = controller.load(ROOT / "candidate01_stall_evidence.json")
    if len(evidence["observations"]) != 2 or len({x["stack_sha256"] for x in evidence["observations"]}) != 1:
        raise ValueError("Repeated stall evidence is missing")
    DEST.mkdir()
    (DEST / "private").mkdir(mode=0o700)
    recovery = copy.deepcopy(plan)
    recovery["afs_root"] = plan["afs_root"] + "/recovery_g1"
    recovery["groups"] = [{"id": "recovery-01", "name": "b1k-heldout20-1017-r01-c01",
                           "members": pending}]
    data, sources = controller.make_startups(controller.transport.aoss_client(), recovery)
    source = sources["recovery-01"]
    original_rule = "len(results) == 4"
    if source.count(original_rule) != 1:
        raise ValueError("Unexpected outer batch runner")
    # The outer driver handles a single-member recovery. The inner worker,
    # evaluator, policy server and all their source hashes remain unchanged.
    source = source.replace(original_rule, "len(results) == len(config['members'])")
    call = ast.parse(source).body[-1].exc.args[0]
    config = ast.literal_eval(call.args[0])
    if len(config["members"]) != 1 or config["members"][0]["id"] != "candidate-01":
        raise ValueError("Recovery driver contains an extra episode")
    member_source = zlib.decompress(base64.b64decode(config["members"][0]["source"])).decode()
    supervision = ast.literal_eval(ast.parse(member_source).body[-1].exc.args[0].args[0])
    child = zlib.decompress(base64.b64decode(supervision["child_source"])).decode()
    assignments = {node.targets[0].id: node.value for node in ast.parse(child).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    def unpack(expression):
        encoded = expression.args[0].args[0].args[0].value
        return json.loads(zlib.decompress(base64.b64decode(encoded)))
    payload, files = unpack(assignments["payload"]), unpack(assignments["files"])
    expected = controller.load(ROOT / "frozen_sha256.json")
    if {n: hashlib.sha256(v.encode()).hexdigest() for n, v in files.items()} != expected:
        raise ValueError("Inner scientific worker changed")
    if payload["pair"] != plan["pairs"][1] or payload["role"] != "candidate":
        raise ValueError("Recovered model or instance changed")
    private = DEST / "private/startup.py"
    private.write_text(source)
    private.chmod(0o600)
    manifest = {"id": "recovery-01", "name": recovery["groups"][0]["name"],
        "old_id": "batch-01", "old_name": old["name"], "old_uid": old["uid"],
        "members": pending, "preserved_reports": preserved,
        "original_plan_sha256": hashlib.sha256((ROOT / "plan.json").read_bytes()).hexdigest(),
        "worker_bundle_sha256": data["bundle_hash"],
        "startup_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "startup_path": plan["afs_root"] + "/launch/recovery-01.py",
        "recovery_afs_root": recovery["afs_root"], "job_walltime_seconds": plan["member_walltime_seconds"],
        "gpus": 1, "maximum_concurrent_gpus": 10, "generation": 1,
        "old_worker_absent_required": True,
        "driver_change": "Accept exactly the supplied one-member recovery; inner frozen files unchanged."}
    save(DEST / "manifest.json", manifest)
    save(DEST / "validation.json", {"passed": True, "episodes": 1,
        "preserved_accepted_episodes": 3, "inner_worker_files_unchanged": True,
        "protocol_unchanged": payload["plan"]["protocol_sha256"] == plan["protocol_sha256"],
        "new_output_namespace": supervision["episode_root"].startswith(recovery["afs_root"] + "/"),
        "cloud_jobs_created": 0})
    print(json.dumps({"prepared": True, "members": pending, "preserved": list(preserved), "cloud_jobs_created": 0}))


if __name__ == "__main__":
    main()
