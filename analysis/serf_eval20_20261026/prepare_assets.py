"""Fetch only the fixed twenty map files, using the already-pinned public release."""
from pathlib import Path
import concurrent.futures
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "serf_reproduction_20261025"
spec = importlib.util.spec_from_file_location("old_download", OLD / "download_assets.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)
sys.path.insert(0, str(HERE.parents[1] / "BEHAVIOR-VLAs/sco"))
from video_eval_adapter import aoss_client


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    public = json.loads((OLD / "public_asset_manifest.json").read_text())
    revision = json.loads((OLD / "download_manifest.json").read_text())["map_revision"]
    records = {r["path"]: r for r in public["points_task22"]["data"] if r["type"] == "file"}
    items = []
    for instance in plan["instance_ids"]:
        path = f"exported_neural_points/task-0022/eval/scene_{instance}.hdf5"
        record = records[path]
        lfs = record.get("lfs") or {}
        digest = lfs.get("sha256", lfs.get("oid"))
        assert digest
        items.append(dict(instance=instance, destination="maps/"+path,
            url=f"https://huggingface.co/datasets/suk063/SERF/resolve/{revision}/{path}",
            size=record["size"], sha256=digest, git_blob_sha1=None))
    manifest = dict(revision=revision, items=items, total_bytes=sum(i["size"] for i in items))
    (HERE / "map_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    root = HERE / "assets"
    root.mkdir(exist_ok=True)
    for item in items:
        existing = OLD / "assets" / item["destination"]
        target = root / item["destination"]
        if download.digest_file(existing, item) and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(existing.resolve())
    def fetch(item):
        target = root / item["destination"]
        if target.is_symlink():
            assert download.digest_file(target, item)
            return dict(path=item["destination"], size=item["size"], status="verified-reused")
        return download.fetch(root, item)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, items):
            print(json.dumps(result), flush=True)
    client = aoss_client()
    uploaded = []
    try:
        for item in items:
            path = root / item["destination"]
            key = f"behavior1k/serf-eval20-20261026/maps/scene_{item['instance']}.hdf5"
            try:
                head = client.head_object(Bucket="260010165", Key=key)
                exists = head["ContentLength"] == item["size"] and head.get("Metadata", {}).get("sha256") == item["sha256"]
            except client.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                    raise
                exists = False
            if not exists:
                client.upload_file(str(path), "260010165", key, ExtraArgs={"Metadata": {"sha256": item["sha256"]}})
            check = client.head_object(Bucket="260010165", Key=key)
            assert check["ContentLength"] == item["size"] and check["Metadata"]["sha256"] == item["sha256"]
            uploaded.append(dict(instance=item["instance"], key=key, size=item["size"], sha256=item["sha256"]))
            (HERE / "asset_upload.json").write_text(json.dumps(dict(complete=len(uploaded)==20, uploaded=uploaded),indent=2)+"\n")
    finally:
        client.close()
    plan["storage"]["additional_map_bytes"] = manifest["total_bytes"]
    (HERE / "plan.json").write_text(json.dumps(plan, ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(complete=True, files=len(uploaded), bytes=manifest["total_bytes"])), flush=True)


if __name__ == "__main__":
    main()
