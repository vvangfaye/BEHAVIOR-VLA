"""AOSS helpers for cluster jobs (reuses the project's existing clients; never prints credentials)."""
import sys
from pathlib import Path

ROOT = Path("/mnt/data1/faye/code/Behavior-1k Challenge")
sys.path.insert(0, str(ROOT / "BEHAVIOR-VLAs/sco"))
sys.path.insert(0, str(ROOT / "analysis/rlinf_adaptation_20261020/multitask_validation_20261029/capacity_repair_20261030"))
BUCKET = "260010165"


def local():
    from video_eval_adapter import aoss_client
    return aoss_client()


def internal():
    from internal_storage import internal_aoss_client
    return internal_aoss_client()


def presign(method, key, seconds=2 * 86400):
    return internal().generate_presigned_url(method, Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=seconds)


def get_text(key):
    return local().get_object(Bucket=BUCKET, Key=key)["Body"].read().decode(errors="replace")
