"""Collect the new fixed cohort only; report paired uncertainty and raw cases."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from cluster_helpers import HERE, aoss_client, write


def main():
    p=argparse.ArgumentParser();p.add_argument("--artifacts",action="store_true");args=p.parse_args()
    plan=json.loads((HERE/"plan.json").read_text());records={};c=aoss_client()
    try:
        for rank in [0,1]:
            receipt=json.loads((HERE/f"eval_worker{rank}_submission.json").read_text())
            report=json.loads(c.get_object(Bucket="260010165",Key=receipt["result_key"])["Body"].read())
            write(HERE/f"eval_worker{rank}_result.json",report)
            for case,value in report["conditions"].items():
                key=(value["instance"],value["condition"])
                assert key not in records
                records[key]=value
            if args.artifacts:
                root=HERE/"retrieved"/f"worker{rank}";root.mkdir(parents=True,exist_ok=True)
                archive=root.parent/f"worker{rank}.tar.gz"
                c.download_file("260010165",receipt["result_key"].replace("status.json","artifacts.tar.gz"),str(archive))
                with tarfile.open(archive) as tar:
                    for member in tar:
                        path=Path(member.name)
                        assert member.isfile() and not path.is_absolute() and ".." not in path.parts
                        dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True)
                        with tar.extractfile(member) as src,dest.open("wb") as dst:
                            import shutil;shutil.copyfileobj(src,dst)
    finally:c.close()
    rows=[]
    for index,instance in enumerate(plan["instance_ids"]):
        row={"instance":instance,"index":index,"image":records.get((instance,"image")),"serf":records.get((instance,"serf"))}
        row["score_delta"]=row["serf"]["score"]-row["image"]["score"] if row["image"] and row["serf"] else None
        if args.artifacts and row["score_delta"] is not None:
            rank=index%2;root=HERE/"retrieved"/f"worker{rank}"
            initials=[json.loads((root/f"{instance}_{condition}"/"initial.json").read_text()) for condition in ["image","serf"]]
            assert initials[0]["initial_state_sha256"]==initials[1]["initial_state_sha256"]
            assert all(x["max_initial_state_delta"]<1e-4 for x in initials)
            state_file=root/f"shared_initial_{index}"/"initial_state.npz"
            assert hashlib.sha256(state_file.read_bytes()).hexdigest()==initials[0]["initial_state_sha256"]
            for condition in ["image","serf"]:
                log=(root/f"{instance}_{condition}"/"server.log").read_text()
                assert "POLICY_STEP_ALIGNMENT_PASSED" in log and "Policy action counter mismatch" not in log
                assert "keepalive ping failed" not in (root/f"{instance}_{condition}"/"evaluation.log").read_text()
        rows.append(row)
    matched=[r for r in rows if r["score_delta"] is not None]
    result=dict(required_episodes=40,completed_episodes=len(records),paired_instances=len(matched),
                complete=len(records)==40,artifact_checks_passed=args.artifacts,rows=rows)
    if matched:
        delta=np.array([r["score_delta"] for r in matched]);rng=np.random.default_rng(20261026)
        means=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
        result.update(image_mean_score=float(np.mean([r["image"]["score"] for r in matched])),
            serf_mean_score=float(np.mean([r["serf"]["score"] for r in matched])),mean_score_delta=float(delta.mean()),
            paired_bootstrap_95_ci=np.percentile(means,[2.5,97.5]).tolist(),
            wins=int((delta>1e-8).sum()),ties=int((np.abs(delta)<=1e-8).sum()),losses=int((delta<-1e-8).sum()),
            image_successes=sum(r["image"]["success"] for r in matched),serf_successes=sum(r["serf"]["success"] for r in matched))
    write(HERE/"comparison.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))


if __name__=="__main__":main()
