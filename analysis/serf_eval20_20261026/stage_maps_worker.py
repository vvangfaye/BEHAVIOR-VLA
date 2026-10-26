"""Stage twenty checked maps in the user's own AFS, without touching old data."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import traceback
import urllib.request


def run(config):
    report = {"phase":"checking_storage", "complete":False}
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        # Parallel per-directory sizes form a conservative upper bound when
        # hardlinks cross directories; this can overcount, never undercount.
        children=list(Path("/mnt/afs/260010165").iterdir())
        def size(child):
            result=subprocess.run(["du","-s","-B1",str(child)],timeout=900,text=True,capture_output=True)
            record={"path":str(child),"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}
            (out/("scan_"+hashlib.sha256(str(child).encode()).hexdigest()[:12]+".json")).write_text(json.dumps(record,indent=2))
            lines=[x for x in result.stderr.splitlines() if x.strip()]
            transient_missing=bool(lines) and all("No such file or directory" in x for x in lines)
            if result.returncode and not transient_missing:
                raise RuntimeError("Storage scan failed: "+json.dumps(record))
            # Concurrent checkpoint cleanup can unlink a path while du scans.
            # Reserve an extra 32 GiB instead of treating this as zero usage.
            return int(result.stdout.split()[0]) + (32*1024**3 if transient_missing else 0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            sizes=list(pool.map(size,children))
        used=sum(sizes)
        report["usage_method"]="sum of parallel top-level du, conservative for cross-directory hardlinks"
        report["owner_usage_bytes"] = used
        assert used + 8*1024**3 < 2_000_000_000_000, "Insufficient personal AFS headroom for the frozen run"
        maps = out / "maps"
        target = maps/"exported_neural_points/task-0022/eval"
        target.mkdir(parents=True)
        old = Path("/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/legacy01/assets/maps")
        (maps/"map_models").symlink_to(old/"map_models",target_is_directory=True)
        (maps/"robot").symlink_to(old/"robot",target_is_directory=True)
        def fetch(item):
            dest = target/f"scene_{item['instance']}.hdf5"
            h = hashlib.sha256()
            with opener.open(item["url"],timeout=120) as r, dest.open("xb") as f:
                for block in iter(lambda:r.read(8*1024**2),b""):
                    f.write(block);h.update(block)
            assert dest.stat().st_size == item["size"] and h.hexdigest() == item["sha256"]
            return {k:item[k] for k in ["instance","size","sha256"]}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            report["verified_maps"] = list(pool.map(fetch, config["maps"]))
        report.update(phase="complete", complete=True, maps_root=str(maps))
    except BaseException:
        report["error"] = traceback.format_exc()
    (out/"status.json").write_text(json.dumps(report,indent=2))
    with opener.open(urllib.request.Request(config["result_put"],data=json.dumps(report).encode(),method="PUT"),timeout=90):pass
    print(json.dumps(report),flush=True)
    if not report["complete"]:raise SystemExit(1)
