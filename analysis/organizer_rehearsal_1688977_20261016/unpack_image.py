"""Unpack verified OCI layers using GNU tar, compatible with this NTFS volume."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
BLOBS = RUNTIME / "image/oci/blobs/sha256"
DIGEST = "469ddc6a55ee7e2f1a9486c3d366b1de7c44469750562bc628b099c4f88c2dea"


def main():
    if not (ROOT / "IMAGE_READY.json").exists():
        raise SystemExit("Verified image is not ready")
    bundle = RUNTIME / "policy-bundle"
    rootfs = bundle / "rootfs"
    rootfs.mkdir(parents=True, exist_ok=True)
    marker = bundle / "ROOTFS_READY.json"
    if marker.exists():
        return
    readiness = json.loads((ROOT / "IMAGE_READY.json").read_text())
    manifest = json.loads((BLOBS / DIGEST).read_text())
    config = json.loads((BLOBS / manifest["config"]["digest"].split(":")[1]).read_text())
    docker_save = readiness.get("unpacked_archive")
    if docker_save:
        archive_root = Path(docker_save)
        docker_manifest = json.loads((archive_root / "manifest.json").read_text())[0]
        layer_paths = [archive_root / p for p in docker_manifest["Layers"]]
    else:
        layer_paths = [BLOBS / layer["digest"].split(":")[1] for layer in manifest["layers"]]
    verified = []
    for number, (layer, diff_id) in enumerate(zip(manifest["layers"], config["rootfs"]["diff_ids"], strict=True)):
        receipt = bundle / f"layer-{number:02d}.json"
        if receipt.exists():
            verified.append(json.loads(receipt.read_text()))
            continue
        path = layer_paths[number]
        with path.open("rb") as stream:
            compressed = stream.read(2) == b"\x1f\x8b"
        open_stream = gzip.open if compressed else open
        with open_stream(path, "rb") as stream:
            actual = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != diff_id:
            raise RuntimeError("Uncompressed layer digest mismatch")
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Unsafe layer path")
                if not relative.name.startswith(".wh."):
                    continue
                parent = rootfs / relative.parent
                if not parent.resolve().is_relative_to(rootfs.resolve()):
                    raise RuntimeError("Whiteout traverses a symlink outside the image")
                targets = list(parent.iterdir()) if relative.name == ".wh..wh..opq" else [parent / relative.name[4:]]
                for target in targets:
                    if target.is_symlink() or target.is_file():
                        target.unlink()
                    elif target.is_dir():
                        shutil.rmtree(target)
        subprocess.run(["tar", "--no-same-owner", "--delay-directory-restore", "--exclude=dev/*",
                        "--exclude=./dev/*", "--exclude=*/.wh.*", "--exclude=.wh.*",
                        "-xpf", str(path), "-C", str(rootfs)], check=True)
        result = {"layer": layer["digest"], "diff_id": actual}
        receipt.write_text(json.dumps(result) + "\n")
        verified.append(result)
        print("unpacked_layer", number, flush=True)
    marker.write_text(json.dumps({"original_manifest_digest": "sha256:" + DIGEST,
                                  "layers": verified, "ownership": "rootless single-uid mapping"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
