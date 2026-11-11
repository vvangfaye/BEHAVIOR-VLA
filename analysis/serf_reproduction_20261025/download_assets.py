"""Download only pinned manifest files; verify size and content before promotion.

No credentials are needed. This script is resumable at the file level and does
not replace any existing file whose verified contents match the manifest.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def digest_file(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    algorithm = "sha256" if item["sha256"] else "sha1"
    h = hashlib.new(algorithm)
    if algorithm == "sha1":
        h.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest() == (item["sha256"] or item["git_blob_sha1"])


def fetch(root, item):
    target = root / item["destination"]
    assert target.resolve().is_relative_to(root.resolve())
    if digest_file(target, item):
        return {"path": item["destination"], "size": item["size"], "status": "verified-existing"}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(item["url"], timeout=90) as source, partial.open("wb") as sink:
                count = 0
                for chunk in iter(lambda: source.read(8 * 1024**2), b""):
                    count += len(chunk)
                    if count > item["size"]:
                        raise ValueError("download exceeds manifest size")
                    sink.write(chunk)
            if not digest_file(partial, item):
                raise ValueError("content hash or size differs from pinned release")
            os.replace(partial, target)
            return {"path": item["destination"], "size": item["size"], "status": "downloaded-verified"}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path(__file__).with_name("download_manifest.json"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--metadata-only", action="store_true")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    items = manifest["items"]
    if args.metadata_only:
        items = [x for x in items if x["destination"].startswith("instances/")]
    assert 1 <= args.workers <= 8
    args.root.mkdir(parents=True, exist_ok=True)
    receipt = args.root / "download_receipt.jsonl"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool, receipt.open("a") as log:
        futures = [pool.submit(fetch, args.root, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(json.dumps(result), flush=True)
    print(json.dumps({"verified_files": len(items), "verified_bytes": sum(x["size"] for x in items)}))


if __name__ == "__main__":
    main()
