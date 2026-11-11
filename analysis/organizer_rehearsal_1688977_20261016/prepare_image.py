"""Fetch the immutable public policy image as a verified OCI layout; no GPU use."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from range_download import download

ROOT = Path(__file__).resolve().parent
LAYOUT = ROOT / "runtime/image/oci"
BLOBS = LAYOUT / "blobs/sha256"
REPO = "faye0138/behavior-pi05-1688977"
DIGEST = "sha256:469ddc6a55ee7e2f1a9486c3d366b1de7c44469750562bc628b099c4f88c2dea"
BASE = "https://registry-1.docker.io/v2/" + REPO


def token():
    with urllib.request.urlopen("https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + REPO + ":pull", timeout=40) as r:
        return json.load(r)["token"]


def fetch_blob(desc):
    name = desc["digest"].split(":")[1]
    final = BLOBS / name
    partial = BLOBS / (name + ".partial")
    if final.exists():
        h = hashlib.file_digest(final.open("rb"), "sha256").hexdigest()
        if h == name and final.stat().st_size == desc["size"]:
            return desc
        raise RuntimeError("Previously downloaded blob changed: " + name)
    for attempt in range(5):
        try:
            if desc["size"] > 512 * 1024 * 1024:
                download(BASE + "/blobs/" + desc["digest"], partial, desc["size"], workers=4,
                         headers=lambda: {"Authorization": "Bearer " + token()}, use_proxy=True)
                with partial.open("rb") as source:
                    h = hashlib.file_digest(source, "sha256").hexdigest()
                if h != name:
                    raise RuntimeError("Blob hash mismatch: " + name)
                partial.replace(final)
                print("verified_blob", name[:12], desc["size"], flush=True)
                return desc
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Authorization": "Bearer " + token()}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            req = urllib.request.Request(BASE + "/blobs/" + desc["digest"], headers=headers)
            with urllib.request.urlopen(req, timeout=120) as stream:
                if offset and stream.status != 206:
                    offset = 0
                with partial.open("ab" if offset else "wb") as output:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        output.write(chunk)
            h = hashlib.file_digest(partial.open("rb"), "sha256").hexdigest()
            if h != name or partial.stat().st_size != desc["size"]:
                raise RuntimeError("Blob hash or size mismatch: " + name)
            partial.replace(final)
            print("verified_blob", name[:12], desc["size"], flush=True)
            return desc
        except Exception as exc:
            print("blob_retry", name[:12], attempt + 1, type(exc).__name__, flush=True)
            if attempt == 4:
                raise RuntimeError("Blob download failed: " + name) from None
            time.sleep(2)


def main():
    BLOBS.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(BASE + "/manifests/" + DIGEST, headers={
        "Authorization": "Bearer " + token(),
        "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json",
    })
    with urllib.request.urlopen(req, timeout=40) as r:
        raw = r.read()
        media_type = r.headers.get("Content-Type").split(";")[0]
    if "sha256:" + hashlib.sha256(raw).hexdigest() != DIGEST:
        raise RuntimeError("Submitted image manifest digest mismatch")
    manifest = json.loads(raw)
    (BLOBS / DIGEST.split(":")[1]).write_bytes(raw)
    descriptors = [manifest["config"]] + manifest["layers"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch_blob, descriptors))
    (LAYOUT / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}\n')
    (LAYOUT / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [{
        "mediaType": media_type, "digest": DIGEST, "size": len(raw),
        "annotations": {"org.opencontainers.image.ref.name": "1688977"},
    }]}, indent=2) + "\n")
    report = {"digest": DIGEST, "layout": str(LAYOUT), "blob_count": len(descriptors),
              "compressed_bytes": sum(d["size"] for d in descriptors), "verified_epoch": time.time()}
    (ROOT / "IMAGE_READY.json").write_text(json.dumps(report, indent=2) + "\n")
    print("image_ready", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
