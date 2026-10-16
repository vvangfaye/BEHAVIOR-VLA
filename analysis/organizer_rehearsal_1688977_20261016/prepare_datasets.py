"""Download pinned official archives, verify hashes, then unpack on the data disk."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
from range_download import download

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / "runtime/downloads"
DATA = ROOT / "runtime/datasets"
DOWNLOADS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


def prepare(item):
    archive = DOWNLOADS / item["filename"]
    receipt = DATA / (item["destination"] + ".verified.json")
    if receipt.exists():
        return json.loads(receipt.read_text())
    if not archive.exists() or archive.stat().st_size != item["bytes"]:
        print("download", item["filename"], item["bytes"], flush=True)
        if item["bytes"] > 1024 ** 3:
            download(item["url"], archive, item["bytes"])
        else:
            result = subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "4", "--connect-timeout", "30", "--speed-limit", "1024",
            "--speed-time", "60", "--continue-at", "-", "--output", str(archive), item["url"],
            ])
            if result.returncode:
                raise RuntimeError("Dataset transfer failed: " + item["filename"])
    h = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    if archive.stat().st_size != item["bytes"] or h.hexdigest() != item["sha256"]:
        raise RuntimeError("Official archive hash mismatch: " + item["filename"])
    print("verified", item["filename"], flush=True)
    dest = DATA / item["destination"]
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path = Path(entry.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe archive path")
            bundle.extract(entry, dest)
        count = len(bundle.infolist())
    result = {**{k: v for k, v in item.items() if k != "url"},
              "verified_epoch": time.time(), "entries": count, "path": str(dest)}
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, receipt)
    print("ready", item["destination"], count, flush=True)
    return result


if __name__ == "__main__":
    items = json.loads((ROOT / "OFFICIAL_DATASETS.json").read_text())["items"]
    mirrors = ROOT / "AOSS_DOWNLOADS.private.json"
    if mirrors.exists():
        sources = json.loads(mirrors.read_text())
        for item in items:
            if item["filename"] in sources:
                item["url"] = sources[item["filename"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(prepare, items))
    (ROOT / "DATASETS_READY.json").write_text(json.dumps(results, indent=2) + "\n")
