#!/usr/bin/env python3
"""Fetch a cluster run's mirrored outputs from AOSS into cluster_runs/<tag>/ (console.log, status.json, tensorboard)."""
import sys, tarfile, io
from pathlib import Path
import aoss_io
tag = sys.argv[1]
out = Path(__file__).resolve().parent / "cluster_runs" / tag
out.mkdir(parents=True, exist_ok=True)
s3 = aoss_io.local()
names = ["status.json", "console.log", "tensorboard.tgz"] + (["frames.tgz"] if "--frames" in sys.argv else [])
for name in names:
    try:
        body = s3.get_object(Bucket=aoss_io.BUCKET, Key=f"rlinf_runs_20261104/{tag}/{name}")["Body"].read()
    except Exception as e:
        print(f"{name}: not yet ({type(e).__name__})"); continue
    (out / name).write_bytes(body)
    if name == "tensorboard.tgz":
        with tarfile.open(fileobj=io.BytesIO(body)) as t:
            t.extractall(out / "tb", filter="data")
    if name == "frames.tgz":
        with tarfile.open(fileobj=io.BytesIO(body)) as t:
            t.extractall(out / "frames", filter="data")
    print(f"{name}: {len(body)} bytes")
print(out)
