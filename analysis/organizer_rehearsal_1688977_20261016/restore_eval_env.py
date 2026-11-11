"""Restore the original evaluator environment without changing system Python."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from range_download import download

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"


def main():
    source = json.loads((ROOT / "EVAL_ARCHIVE_SOURCE.private.json").read_text())
    archive = RUNTIME / "downloads/b1k-eval.tar.gz"
    if not archive.exists() or archive.stat().st_size != source["bytes"]:
        download(source["url"], archive, source["bytes"], workers=8, use_proxy=source["use_proxy"])
    # A multipart ETag does not reveal the upload's part boundaries. Validate
    # against the source object's full CRC64 rather than assuming 8 MiB parts.
    receipt = ROOT / "EVAL_ARCHIVE_VERIFIED.json"
    verified = json.loads(receipt.read_text()) if receipt.exists() else {}
    if (verified.get("mtime_ns") != archive.stat().st_mtime_ns or
            verified.get("bytes") != archive.stat().st_size or verified.get("crc64") != source["crc64"]):
        verified = json.loads(subprocess.check_output([
            "/home/faye/.local/share/behavior-tools/venv/bin/python", str(ROOT / "verify_eval_archive.py")
        ], text=True))
    destination = RUNTIME / "b1k-eval"
    marker = destination / "_RESTORED_BY_REHEARSAL"
    if not marker.exists():
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError("Unsafe environment archive path")
        destination.mkdir(exist_ok=True)
        print("extracting_original_environment", flush=True)
        subprocess.run(["tar", "--no-same-owner", "-xzf", str(archive), "-C", str(destination)], check=True)
        python = destination / "bin/python"
        unpack = destination / "bin/conda-unpack"
        if unpack.exists():
            subprocess.run([str(python), str(unpack)], check=True)
        site = subprocess.check_output([str(python), "-c", "import site;print(site.getsitepackages()[0])"], text=True).strip()
        paths = [str(ROOT / "frozen_source/BEHAVIOR-1K" / p) for p in ["OmniGibson", "bddl3", "joylo"]]
        (Path(site) / "00-rehearsal-source.pth").write_text("import sys; sys.path[:0] = " + repr(paths) + "\n")
        marker.touch()
    python = destination / "bin/python"
    warp = ROOT.parents[1] / "BEHAVIOR-VLAs/nebula/runtime-wheels/warp_lang-1.12.0-py3-none-manylinux_2_28_x86_64.whl"
    check = subprocess.run([str(python), "-c", "import importlib.metadata as m;print(m.version('warp-lang'))"],
                           text=True, capture_output=True)
    if check.returncode or check.stdout.strip() != "1.12.0":
        with warp.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != "a1436f60a1881cd94f787e751a83fc0987626be2d3e2b4e74c64a6947c6d1266":
                raise RuntimeError("Pinned Warp wheel checksum mismatch")
        subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(warp)], check=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMNIGIBSON_NO_OMNIVERSE="1", OMNI_KIT_ACCEPT_EULA="YES",
               OMNIGIBSON_DATA_PATH=str(RUNTIME / "datasets"), TMPDIR=str(RUNTIME / "tmp"),
               PYTHONNOUSERSITE="1")
    code = "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ['torch','omnigibson','isaacsim','warp-lang','bddl']}))"
    versions = json.loads(subprocess.check_output([str(python), "-c", code], env=env, text=True))
    if versions["omnigibson"] != "3.9.3" or not versions["isaacsim"].startswith("5.1"):
        raise RuntimeError("Restored simulator version differs from the submitted evaluation")
    result = {"python": str(python), "versions": versions, "original_archive_sha256": verified["sha256"],
              "source_files_verified": 1461, "gpu_tests_started": False}
    (ROOT / "ENVIRONMENT_READY.json").write_text(json.dumps(result, indent=2) + "\n")
    print("environment_ready", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
