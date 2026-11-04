import sys, time, os, aoss_io
from boto3.s3.transfer import TransferConfig
src, key = sys.argv[1], sys.argv[2]
cfg = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=128 * 2**20, max_concurrency=24)
t = time.time()
aoss_io.local().upload_file(src, aoss_io.BUCKET, key, Config=cfg)
size = os.path.getsize(src)
head = aoss_io.local().head_object(Bucket=aoss_io.BUCKET, Key=key)
assert head["ContentLength"] == size, (head["ContentLength"], size)
print(f"  {key}: {size / 2**30:.2f} GiB in {time.time() - t:.0f}s ({size / 2**20 / max(time.time() - t, 1e-3):.1f} MiB/s), size verified", flush=True)
