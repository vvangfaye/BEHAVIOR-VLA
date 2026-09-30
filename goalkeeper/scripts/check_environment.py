"""Read-only environment inventory; does not start OmniGibson or submit jobs."""
import argparse
import importlib.util
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def run(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


parser = argparse.ArgumentParser(description=__doc__)
for name in ("behavior-root", "openpi-root", "checkpoint", "data-root"):
    parser.add_argument("--" + name)
args = parser.parse_args()
report = {"python": sys.version, "platform": platform.platform(),
          "modules": {name: importlib.util.find_spec(name) is not None
                      for name in ("torch", "openpi", "openpi_client", "omnigibson", "transformers")},
          "paths": {name: {"path": value, "exists": Path(value).exists()} for name, value in vars(args).items() if value}}
if shutil.which("nvidia-smi"):
    report["gpu"] = run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"])
for name in ("behavior_root", "openpi_root"):
    location = getattr(args, name)
    if location and Path(location).is_dir() and shutil.which("git"):
        report[name + "_revision"] = run(["git", "rev-parse", "HEAD"], location)
report["scope"] = "Inventory only. GPU rendering, checkpoint compatibility and evaluation require an actual smoke rollout."
print(json.dumps(report, indent=2))
