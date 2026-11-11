"""One bounded controller: gate, two shards, collection; never duplicate a job."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from cluster_helpers import HERE, aoss_client, describe, write


def main():
    lock=(HERE/"monitor.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    state=dict(phase="waiting_for_gate",complete=False,completed_episodes=0,required_episodes=40,
               pid=os.getpid(),started_at=started,deadline=started+60*3600)
    client=aoss_client()
    def get(key):
        try:return json.loads(client.get_object(Bucket="260010165",Key=key)["Body"].read())
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] in {"404","NoSuchKey"}:return None
            raise
    try:
        while time.time()<state["deadline"]:
            gate=get("behavior1k/serf-eval20-20261026/gate01/status.json")
            stage=json.loads((HERE/"asset_upload.json").read_text())
            for name,value in [("gate",gate)]:
                if value is not None:
                    write(HERE/f"{name}_result.json",value)
                    if value.get("error"):raise RuntimeError(name+" failed: "+value["error"])
            state.update(updated_at=time.time(),gate_phase=(gate or {}).get("phase"),stage_phase="node_local_maps_from_aoss")
            if gate and gate.get("complete") and stage and stage.get("complete"):
                assert gate["gate"]["passed"]
                receipt=json.loads((HERE/"gate_submission.json").read_text())
                live=describe(receipt["name"]);assert live["uid"]==receipt["uid"]
                if live["state"]=="SUCCEEDED":
                    state["gate_passed"]=True
                    for rank in [0,1]:
                        receipt_file=HERE/f"eval_worker{rank}_submission.json"
                        if not receipt_file.exists():
                            state["phase"]="launching_worker"+str(rank);write(HERE/"status.json",state)
                            with (HERE/f"worker{rank}_launch.log").open("w") as log:
                                subprocess.run([sys.executable,str(HERE/"launch_evaluation.py"),"--rank",str(rank)],
                                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
                    state["phase"]="evaluating"
            reports={}
            target_source=json.loads((HERE/"source_upload.json").read_text())["source_sha256"]
            for rank in [0,1]:
                f=HERE/f"eval_worker{rank}_submission.json"
                if not f.exists():continue
                receipt=json.loads(f.read_text())
                if receipt.get("source_sha256")!=target_source:
                    first=reports.get(0,{})
                    progress={}
                    try:progress=json.loads(first.get("heartbeat",{}).get("progress.json","{}"))
                    except (ValueError,TypeError):pass
                    if rank==1 and (progress.get("steps",0)>=100 or first.get("conditions")):
                        state["phase"]="launching_repaired_worker1";write(HERE/"status.json",state)
                        with (HERE/"worker1_repair_launch.log").open("w") as log:
                            subprocess.run([sys.executable,str(HERE/"launch_evaluation.py"),"--rank","1","--recover"],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                        state["repair_serial_gate_passed"]=True
                    else:
                        state["phase"]="waiting_for_first_repaired_closed_loop"
                    continue
                report=get(receipt["result_key"])
                if report:
                    write(HERE/f"eval_worker{rank}_result.json",report);reports[rank]=report
                    if report.get("error") and not report.get("termination_signal"):
                        raise RuntimeError("Worker "+str(rank)+": "+report["error"])
            state["completed_episodes"]=sum(len(v.get("conditions",{})) for v in reports.values())
            state["workers"]={str(k):dict(phase=v.get("phase"),complete=v.get("complete"),
                                          completed=len(v.get("conditions",{})),heartbeat=v.get("heartbeat")) for k,v in reports.items()}
            if state["completed_episodes"]==40 and len(reports)==2 and all(v.get("complete") for v in reports.values()):
                ready=True
                for rank in [0,1]:
                    receipt=json.loads((HERE/f"eval_worker{rank}_submission.json").read_text())
                    try:client.head_object(Bucket="260010165",Key=receipt["result_key"].replace("status.json","artifacts.tar.gz"))
                    except client.exceptions.ClientError as exc:
                        if exc.response["Error"]["Code"] in {"404","NoSuchKey"}:ready=False
                        else:raise
                if ready:
                    state["phase"]="collecting";write(HERE/"status.json",state)
                    with (HERE/"collection.log").open("w") as log:
                        subprocess.run([sys.executable,str(HERE/"collect_and_compare.py"),"--artifacts"],
                                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
                    result=json.loads((HERE/"comparison.json").read_text());assert result["complete"] and result["artifact_checks_passed"]
                    state.update(phase="complete",complete=True,comparison="comparison.json")
                    write(HERE/"status.json",state);return
            # Check exact identities and stop on terminal failures, even when the
            # worker was killed before it could upload its last heartbeat.
            if int((time.time()-started)/30)%4==0:
                files=[HERE/"gate_submission.json"]+[HERE/f"eval_worker{r}_submission.json" for r in [0,1]]
                for f in files:
                    if not f.exists():continue
                    d=json.loads(f.read_text());actual=describe(d["name"]);assert actual["uid"]==d["uid"]
                    if f.name.startswith("eval_worker") and d.get("source_sha256")!=target_source:continue
                    if actual["state"] in {"FAILED","SUSPENDED","STOPPED","TERMINATED","CANCELED"}:
                        if f.name.startswith("eval_worker") and d.get("generation",1)<json.loads((HERE/"plan.json").read_text())["evaluation"].get("max_generation_by_rank",{}).get(f.name.removeprefix("eval_worker").split("_")[0],json.loads((HERE/"plan.json").read_text())["evaluation"]["max_generation_per_worker"]):
                            workers=subprocess.run(["/home/faye/.local/bin/sco","acp","jobs","get-workers",d["name"],"--workspace-name=share-space"],capture_output=True,text=True,timeout=45)
                            if "Running" in workers.stdout:
                                continue
                            rank=int(f.name.removeprefix("eval_worker").split("_")[0])
                            state["phase"]="recovering_worker"+str(rank);write(HERE/"status.json",state)
                            with (HERE/f"worker{rank}_recovery.log").open("w") as log:
                                subprocess.run([sys.executable,str(HERE/"launch_evaluation.py"),"--rank",str(rank),"--recover"],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                            state["phase"]="evaluating"
                        else:
                            raise RuntimeError(d["name"]+" is "+actual["state"]+"; bounded recovery exhausted")
            write(HERE/"status.json",state)
            time.sleep(30)
        raise TimeoutError("Controller deadline reached")
    except BaseException:
        state.update(phase="needs_attention",error=traceback.format_exc(),updated_at=time.time())
        write(HERE/"status.json",state)
        raise
    finally:client.close()


if __name__=="__main__":main()
