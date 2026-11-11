from cluster_helpers import HERE, aoss_client, signed, submit, describe
import json

assets=json.loads((HERE/"asset_upload.json").read_text())
assert assets["complete"] and len(assets["uploaded"])==20
c=aoss_client()
try:
    previous=HERE/"stage02_submission.json"
    if previous.exists():
        old=json.loads(previous.read_text());actual=describe(old["name"])
        assert actual["uid"]==old["uid"] and actual["state"] in {"FAILED","SUCCEEDED","STOPPED","TERMINATED","CANCELED"}
    key="behavior1k/serf-eval20-20261026/assets03/status.json"
    config=dict(output="/mnt/afs/260010165/behavior_1k/serf_eval20_20261026/assets03",result_key=key,
                result_put=signed(c,"put_object",key),
                maps=[dict(item,url=signed(c,"get_object",item["key"])) for item in assets["uploaded"]])
    submit("b1k-serf20-assets-1026-a03",HERE/"stage_maps_worker.py",config,HERE/"stage_submission.json",1800,False)
finally:c.close()
