"""Stage the exact Isaac Sim wheels listed by the frozen official setup script."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "runtime/isaac-wheels"
DEST.mkdir(exist_ok=True)


def download(item):
    path = DEST / item["filename"]
    if not path.exists():
        partial = path.with_suffix(".partial")
        subprocess.run(["curl", "-fLsS", "--retry", "4", "--connect-timeout", "30",
                        "--speed-limit", "1024", "--speed-time", "60", "-C", "-",
                        "-o", str(partial), item["url"]], check=True)
        partial.replace(path)
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise RuntimeError("Corrupt wheel " + path.name)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print("wheel_ready", path.name, path.stat().st_size, flush=True)
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, json.loads((ROOT / "ISAAC_WHEELS.json").read_text())))
    (ROOT / "ISAAC_WHEELS_READY.json").write_text(json.dumps(results, indent=2) + "\n")
