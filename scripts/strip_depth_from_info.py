#!/usr/bin/env python
"""Drop unused video keys from a local LeRobot dataset's meta/info.json.

The π0.5 b1k pipeline consumes only the 3 RGB cameras
(openpi/src/openpi/configs/robots/b1k.py), but lerobot decodes and existence-
checks *every* feature whose dtype is "video" - including the 3 depth cameras.
Removing the depth entries makes lerobot ignore them, which:
  * lets the local (RGB-only) copy pass the cache-sufficiency check instead of
    falling back to the Hugging Face Hub, and
  * halves per-sample video decoding.

The original file is backed up next to it as info.json.orig once.

Usage:
  ./scripts/strip_depth_from_info.py /var/tmp/behavior-1k/2026-challenge-demos
"""
import json
import shutil
import sys
from pathlib import Path


def main(dataset_root: str) -> None:
    info_path = Path(dataset_root) / "meta" / "info.json"
    backup = info_path.with_suffix(".json.orig")

    info = json.loads(info_path.read_text())
    features = info["features"]

    dropped = [
        k
        for k, v in features.items()
        if v.get("dtype") == "video" and (v.get("info") or {}).get("is_depth_map")
    ]
    if not dropped:
        print("No depth video features to remove; nothing to do.")
        return

    if not backup.exists():
        shutil.copy2(info_path, backup)
        print(f"Backed up original to {backup}")

    for k in dropped:
        features.pop(k)

    info_path.write_text(json.dumps(info, indent=4))
    print(f"Removed {len(dropped)} depth video keys:")
    for k in dropped:
        print(f"  - {k}")
    remaining = [k for k, v in features.items() if v.get("dtype") == "video"]
    print(f"Remaining video keys: {remaining}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
