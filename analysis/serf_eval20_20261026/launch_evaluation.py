"""Start exactly one of two fixed shards after the geometry gate passes."""
import argparse
import json
import io
import tarfile
import time
from cluster_helpers import HERE, OLD, aoss_client, signed, submit, describe
from recovery_archives import completed_archives

arg=argparse.ArgumentParser()
arg.add_argument("--rank",type=int,choices=[0,1],required=True)
arg.add_argument("--recover",action="store_true")
args=arg.parse_args()
plan=json.loads((HERE/"plan.json").read_text())
gate=json.loads((HERE/"gate_result.json").read_text())
assert gate["complete"] and gate["gate"]["passed"]
gater=json.loads((HERE/"gate_submission.json").read_text())
actual=describe(gater["name"])
assert actual["uid"]==gater["uid"] and actual["state"]=="SUCCEEDED"
maps=json.loads((HERE/"asset_upload.json").read_text())
assert maps["complete"] and len(maps["uploaded"])==20
source=json.loads((HERE/"source_upload.json").read_text())
if source["source_sha256"]!=gate["source_sha256"]:
    compatibility=json.loads((HERE/"gate_compatibility.json").read_text())
    assert compatibility["gate_source_sha256"]==gate["source_sha256"]
    assert compatibility["new_source_sha256"]==source["source_sha256"] and compatibility["passed"]
checkpoints=json.loads((OLD/"checkpoint_upload_progress.json").read_text())
assert checkpoints["complete"] and len(checkpoints["uploaded"])==32
tracker=json.loads((OLD/"cotracker_upload.json").read_text())
legacy=json.loads((OLD/"legacy_stage01_result.json").read_text())
deps=json.loads((OLD/"policydeps03_result.json").read_text())
c=aoss_client()
try:
    generation=1
    seconds=plan["evaluation"]["max_seconds_per_worker"]
    resume=[]
    completed={}
    current=HERE/f"eval_worker{args.rank}_submission.json"
    old=None
    if args.recover:
        old=json.loads(current.read_text())
        actual=describe(old["name"])
        assert actual["uid"]==old["uid"] and actual["state"] in {"FAILED","SUSPENDED","STOPPED","TERMINATED","CANCELED"}
        generation=old.get("generation",1)+1
        assert generation<=plan["evaluation"].get("max_generation_by_rank",{}).get(str(args.rank),plan["evaluation"]["max_generation_per_worker"]),"Recovery limit reached"
        started=old.get("submitted_at_unix",current.stat().st_mtime)
        seconds=int(old["max_seconds"]-(time.time()-started))
        assert seconds>=900,"No remaining per-shard runtime budget for a recovery"
        chain=[old]
        for previous_generation in range(generation-2,0,-1):
            path=HERE/f"eval_worker{args.rank}_generation{previous_generation}.json"
            if path.exists():chain.append(json.loads(path.read_text()))
        cases=[(plan['instance_ids'][i],condition)
               for i in plan['evaluation']['indices_by_rank'][str(args.rank)]
               for condition in ['image','serf']]
        completed,resume=completed_archives(c,'260010165',chain,cases,
                                           lambda key:signed(c,'get_object',key))
    key=f"behavior1k/serf-eval20-20261026/eval{generation:02d}/worker{args.rank}/"
    cfg=dict(output=f"/tmp/b1k-serf20-eval{generation:02d}-worker{args.rank}",
        result_key=key+"status.json", mode="full", stage=legacy["root"],
        checks_file=legacy["root"]+"/stage_result.json",maps_root="resolved-on-node",
        maps=[dict(x,url=signed(c,"get_object",x["key"])) for x in maps["uploaded"]],
        gate_file=gater["output"]+"/status.json",
        indices=plan["evaluation"]["indices_by_rank"][str(args.rank)],instance_ids=plan["instance_ids"],
        source_get=signed(c,"get_object",source["key"]),source_sha256=source["source_sha256"],
        cotracker_get=signed(c,"get_object",tracker["key"]),cotracker_sha256=tracker["sha256"],
        site_path="/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/dev01/site",
        policy_dependency_path=deps["site"],check_serf_construction=True,
        background_assets=json.loads((OLD/"background_stage_result.json").read_text())["verified_background"],
        checkpoints=[dict(relative=x["key"].split("/checkpoints/",1)[1],size=x["size"],sha256=x["sha256"],url=signed(c,"get_object",x["key"])) for x in checkpoints["uploaded"]],
        checkpoint_folders={"image":"pi_behavior_b1k_fast--50t_lora--task-0022","serf":"pi_serf_behavior_b1k_fast--4d_env_robot_feat_map--50t_lora--task-0022"},
        episode_puts={f"{plan['instance_ids'][i]}_{condition}":signed(c,"put_object",key+f"episodes/{plan['instance_ids'][i]}_{condition}.tar.gz")
                      for i in plan["evaluation"]["indices_by_rank"][str(args.rank)] for condition in ["image","serf"]},
        instance_states=json.loads((HERE/"instance_states20_manifest.json").read_text())["files"],
        resume_archives=resume,
        result_put=signed(c,"put_object",key+"status.json"),artifacts_put=signed(c,"put_object",key+"artifacts.tar.gz"))
    if old:
        cfg["conditions_by_index"]={str(i):[condition for condition in ["image","serf"]
            if f"{plan['instance_ids'][i]}_{condition}" not in completed]
            for i in plan["evaluation"]["indices_by_rank"][str(args.rank)]}
        current.rename(HERE/f"eval_worker{args.rank}_generation{generation-1}.json")
        old_result=HERE/f"eval_worker{args.rank}_result.json"
        if old_result.exists():old_result.rename(HERE/f"eval_worker{args.rank}_generation{generation-1}_result.json")
    receipt=submit(f"b1k-serf20-eval-1026-w{args.rank}-a{generation:02d}",HERE/"gpu_worker.py",cfg,current,seconds,True)
    receipt.update(generation=generation,preserved_episodes=list(completed),previous_uid=old["uid"] if old else None)
    from cluster_helpers import write
    write(current,receipt)
finally:c.close()
