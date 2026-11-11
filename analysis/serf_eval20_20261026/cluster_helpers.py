"""Bounded SCO submission helpers; never print commands containing signed URLs."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "serf_reproduction_20261025"
CLI = "/home/faye/.local/bin/sco"
sys.path.insert(0, str(HERE.parents[1] / "BEHAVIOR-VLAs/sco"))
from video_eval_adapter import aoss_client
from public_test_admission import query_standard_capacity


def write(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    tmp.replace(path)


def describe(name):
    p = subprocess.run([CLI, "acp", "jobs", "describe", name, "--workspace-name=share-space", "-o", "json"],
                       capture_output=True, text=True, timeout=45, check=True)
    d = json.loads(p.stdout)
    assert d["ownership"]["user_name"] == "260010165"
    return d


def signed(client, method, key):
    return client.generate_presigned_url(method, Params={"Bucket":"260010165", "Key":key}, ExpiresIn=604800)


def submit(name, worker_file, config, receipt_path, max_seconds, gpu):
    assert not receipt_path.exists(), "Resolve the existing submission receipt; do not duplicate"
    template = "b1k-repeat1688-1019-batch-00" if gpu else "b1k-storage-inv-1024"
    source = describe(template)
    assert source["resource_pool"]["name"] == ("computing-cluster-5090-01g" if gpu else "share-cluster")
    if gpu:
        capacity = query_standard_capacity(CLI, dict(os.environ))
        write(receipt_path.with_name(receipt_path.stem+"_capacity.json"), capacity)
        quota = "reserved" if capacity["used_gpus"] < capacity["standard_limit"] else "spot"
    else:
        quota = "reserved"
    code = worker_file.read_text()+"\nrun("+repr(config)+")\n"
    packed = base64.b64encode(zlib.compress(code.encode())).decode()
    command = f"timeout --signal=TERM --kill-after=60s {max_seconds}s python3 -u -c " + shlex.quote(
        "import base64,zlib;exec(zlib.decompress(base64.b64decode("+repr(packed)+")))")
    intent = dict(name=name, max_seconds=max_seconds, gpu_count=int(gpu), quota_type=quota,
                  output=config["output"], result_key=config["result_key"],
                  worker_sha256=hashlib.sha256(worker_file.read_bytes()).hexdigest(),
                  source_sha256=config.get("source_sha256"), submission_attempted=True,
                  submitted_at_unix=time.time())
    write(receipt_path, intent)
    reply = subprocess.run([CLI,"acp","jobs","copy","--workspace-name=share-space", "--copy-job-name="+template,
        "--name="+name,"--job-name="+name,"--worker-spec="+("n12lp.nn.i10a.1" if gpu else "n6ls.iu.i40.2c8g"),
        "--worker-nodes=1","--quota-type="+quota,"--retry-times=0","--enable-fault-tolerance=false",
        "--enable-anomaly-detection=false","--command="+command],capture_output=True,text=True,timeout=120)
    actual = describe(name)
    intent.update(uid=actual["uid"],state=actual["state"],returncode=reply.returncode)
    write(receipt_path, intent)
    print(json.dumps(intent), flush=True)
    return intent
