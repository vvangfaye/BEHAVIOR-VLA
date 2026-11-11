"""Download the submitted docker-save archive from its existing private link."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from range_download import download

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
EXPECTED_CONFIG = "ae53492e21d915a832cbd68d129c9ca9013e029a18838f1f01651bad3db38a9e"


def main():
    source = json.loads((ROOT / "AOSS_IMAGE_SOURCE.private.json").read_text())
    archive = RUNTIME / "image/policy-image1688977.docker.tar.gz"
    if not archive.exists() or archive.stat().st_size != source["bytes"]:
        download(source["url"], archive, source["bytes"], workers=8)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source["sha256"]:
        raise RuntimeError("Submitted archive SHA256 mismatch")
    dest = RUNTIME / "image/docker-save"
    if not (dest / "_UNPACKED").exists():
        dest.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                p = Path(member.name)
                if p.is_absolute() or ".." in p.parts:
                    raise RuntimeError("Unsafe archive member")
        subprocess.run(["tar", "--no-same-owner", "-xzf", str(archive), "-C", str(dest)], check=True)
        (dest / "_UNPACKED").touch()
    manifest = json.loads((dest / "manifest.json").read_text())
    if len(manifest) != 1:
        raise RuntimeError("Expected exactly one submitted image")
    config = dest / manifest[0]["Config"]
    if hashlib.sha256(config.read_bytes()).hexdigest() != EXPECTED_CONFIG:
        raise RuntimeError("Archive config does not match the public submitted image")
    result = {"format": "docker-save", "archive": str(archive), "unpacked_archive": str(dest),
              "archive_sha256": digest, "config_digest": "sha256:" + EXPECTED_CONFIG,
              "digest": "sha256:469ddc6a55ee7e2f1a9486c3d366b1de7c44469750562bc628b099c4f88c2dea",
              "provenance": "Original submitted archive; original registry manifest retained separately"}
    (ROOT / "IMAGE_READY.json").write_text(json.dumps(result, indent=2) + "\n")
    print("submitted_image_archive_ready", flush=True)


if __name__ == "__main__":
    main()
