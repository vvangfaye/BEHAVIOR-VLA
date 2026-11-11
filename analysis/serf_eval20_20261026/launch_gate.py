from cluster_helpers import HERE, OLD, aoss_client, signed, submit, write
import hashlib
import json

c=aoss_client()
try:
    source=json.loads((HERE/"source_receipt.json").read_text())
    path=HERE/"source.tar.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source["source_sha256"]
    source_key="behavior1k/serf-eval20-20261026/source-"+source["source_sha256"]+".tar.gz"
    c.upload_file(str(path),"260010165",source_key,ExtraArgs={"Metadata":{"sha256":source["source_sha256"]}})
    assert c.head_object(Bucket="260010165",Key=source_key)["ContentLength"]==source["bytes"]
    write(HERE/"source_upload.json",dict(source,key=source_key))
    legacy=json.loads((OLD/"legacy_stage01_result.json").read_text())
    tracker=json.loads((OLD/"cotracker_upload.json").read_text())
    result_key="behavior1k/serf-eval20-20261026/gate01/status.json"
    cfg=dict(output="/mnt/afs/260010165/behavior_1k/serf_eval20_20261026/gate01",result_key=result_key,
        stage=legacy["root"],checks_file=legacy["root"]+"/stage_result.json",
        maps_root=legacy["root"]+"/assets/maps", mode="gate",
        source_sha256=source["source_sha256"],source_get=signed(c,"get_object",source_key),
        cotracker_get=signed(c,"get_object",tracker["key"]),cotracker_sha256=tracker["sha256"],
        site_path="/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/dev01/site",
        background_assets=json.loads((OLD/"background_stage_result.json").read_text())["verified_background"],
        gate_actions="/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/eval05/worker0/256_serf/repro/actions/putting_shoes_on_rack/256/0",
        gate_initial="/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/eval05/worker0/shared_initial_2",
        result_put=signed(c,"put_object",result_key),
        artifacts_put=signed(c,"put_object","behavior1k/serf-eval20-20261026/gate01/artifacts.tar.gz"))
    submit("b1k-serf20-gate-1026-a01",HERE/"gpu_worker.py",cfg,HERE/"gate_submission.json",3600,True)
finally:c.close()
