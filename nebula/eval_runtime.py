#!/usr/bin/env python3
"""Fetch and verify the pinned Linux Warp wheel required by OmniGibson 3.9.3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

VERSION = "1.12.0"
FILENAME = "warp_lang-1.12.0-py3-none-manylinux_2_28_x86_64.whl"
SHA256 = "a1436f60a1881cd94f787e751a83fc0987626be2d3e2b4e74c64a6947c6d1266"
PYPI_METADATA = f"https://pypi.org/pypi/warp-lang/{VERSION}/json"


def verify(directory):
    wheel = Path(directory) / FILENAME
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError("Warp wheel does not match the pinned official SHA256")
    return {"version": VERSION, "filename": FILENAME, "sha256": SHA256,
            "bytes": wheel.stat().st_size}


def fetch(directory):
    directory = Path(directory)
    if (directory / FILENAME).exists():
        return verify(directory)
    with urllib.request.urlopen(PYPI_METADATA, timeout=30) as response:
        metadata = json.load(response)
    item = next(item for item in metadata["urls"] if item["filename"] == FILENAME)
    if item["digests"]["sha256"] != SHA256:
        raise ValueError("PyPI metadata differs from the pinned official SHA256")
    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / (FILENAME + ".partial")
    digest, size = hashlib.sha256(), 0
    # An interrupted earlier download must be inspected rather than overwritten.
    with partial.open("xb") as output:
        try:
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                for block in iter(lambda: response.read(4 * 1024**2), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
            if size != item["size"] or digest.hexdigest() != SHA256:
                raise ValueError("Downloaded Warp wheel is incomplete or corrupt")
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    # link() publishes without overwriting a concurrently prepared wheel.
    os.link(partial, directory / FILENAME)
    partial.unlink()
    return verify(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "verify"))
    parser.add_argument("--wheel-dir", type=Path, default=Path(__file__).parent / "runtime-wheels")
    args = parser.parse_args()
    try:
        report = fetch(args.wheel_dir) if args.command == "fetch" else verify(args.wheel_dir)
    except (OSError, ValueError, StopIteration) as error:
        parser.exit(1, f"Eval runtime dependency check failed: {error}\n"
                    "Prepare the pinned wheel with: python3 nebula/eval_runtime.py fetch\n")
    print(json.dumps({"warp_wheel": "verified", **report}))


if __name__ == "__main__":
    main()
