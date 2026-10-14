"""Executed through the existing macpro SSH session. No creation or training edits."""
import hashlib
import json
import os
import time
from nebula.submit_oss import OSSClient

RUN = 'b1k-four-source-finetune-20261014_080449-20k'
ROOT = 'wangfaye/result/behavior-challenge/four_source_finetune/' + RUN
CKPT = ROOT + '/checkpoints/pi05_b1k/' + RUN + '/5000'
c = OSSClient()


def read_consumed_mixture(client, key, max_bytes=128 * 1024**2):
    # The checkpoint includes the complete episode indexes (~75 MB). The
    # generic OSSClient.read intentionally permits only small metadata.
    size = client.head(key)
    if not size or size > max_bytes:
        raise ValueError('Consumed mixture asset missing or exceeds128MiB')
    body = client.bucket.get_object(key)
    try:
        raw = body.read(size + 1)
    finally:
        body.resp.response.close()
    if len(raw) != size:
        raise ValueError('Consumed mixture asset changed or truncated')
    value = json.loads(raw)
    if value.get('kind') != 'b1k_consumed_mixture_contract' or value.get('schema_version') != 1:
        raise ValueError('Unsupported consumed mixture contract')
    expected = value.get('fingerprint')
    content = {k: v for k, v in value.items() if k != 'fingerprint'}
    canonical = json.dumps(content, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != expected:
        raise ValueError('Consumed mixture fingerprint differs')
    return value, hashlib.sha256(raw).hexdigest()


def inference_objects(objects, root):
    selected = {}
    for key, size in objects.items():
        if not (key.startswith(root + '/params/') or key.startswith(root + '/assets/')
                or key == root + '/_CHECKPOINT_METADATA'):
            continue
        # The Nebula OSS mount publishes POSIX directory markers. They are
        # never parameter files and cannot be downloaded onto directory paths.
        if key.endswith('/') and size == 0:
            continue
        if size <= 0:
            raise ValueError('Empty inference file')
        selected[key] = size
    return selected


if os.environ.get('PAIRED_TRANSFER_PROBE') == '1':
    parent = 'wangfaye/embodied/datasets/behavior-1k/retrain_20261013_v1/checkpoints/pi05_b1k_full100_joint_1688977_850e3248c733'
    files = []
    for relative in ('params/_METADATA','assets/b1k_full100_joint_subtask_v1_drop_episode_s1/norm_stats.json'):
        key = parent + '/' + relative
        raw = c.read(key)
        head = c.bucket.head_object(key)
        files.append(dict(relative_path=relative,bytes=len(raw),etag=head.etag.strip('"'),
                          sha256=hashlib.sha256(raw).hexdigest(),get_url=c.bucket.sign_url('GET',key,172800)))
    print(json.dumps(dict(state='read_only_transfer_probe',files=files,observed_at=time.time())))
    raise SystemExit(0)
metrics = ROOT + '/attempts/stage20000-g0/training_metrics.jsonl'
result = {'observed_at': time.time(), 'state': 'waiting_checkpoint5000', 'run_id': RUN}
if c.head(metrics):
    rows = c.bucket.get_object(metrics).read().decode().splitlines()
    updates = [json.loads(row) for row in rows if row.strip()]
    if updates:
        result['completed_updates'] = updates[-1].get('completed_updates')
if c.head(CKPT + '/_CHECKPOINT_METADATA'):
    metadata_raw = c.read(CKPT + '/_CHECKPOINT_METADATA')
    metadata = json.loads(metadata_raw)
    # Orbax publishes this only after finalizing all item futures. An initial
    # timestamp, directory existence or progress counter is not sufficient.
    commit = metadata.get('commit_timestamp_nsecs')
    if type(commit) is not int or commit <= 0:
        raise ValueError('Checkpoint has not committed')
    contract = json.loads(c.read(CKPT + '/assets/formal_run_contract.json'))
    consumed, consumed_sha = read_consumed_mixture(c, CKPT + '/assets/subtask_mixture.json')
    if contract['run_id'] != RUN or contract['max_authorized_updates'] != 20000:
        raise ValueError('Wrong formal training identity')
    if consumed['step'] != 5000 or consumed['consumed_samples'] != 5000 * 192:
        raise ValueError('Checkpoint progress differs from5000 completed updates')
    objects = c.inventory(CKPT)
    selected = inference_objects(objects, CKPT)
    if selected.get(CKPT + '/params/_METADATA', 0) <= 0:
        raise ValueError('Inference parameter metadata missing')
    files = []
    for key, size in sorted(selected.items()):
        head = c.bucket.head_object(key)
        if head.content_length != size or size <= 0:
            raise ValueError('Checkpoint listing/head changed')
        relative = key[len(CKPT) + 1:]
        if 'train_state' in relative.split('/'):
            raise ValueError('Never export the optimizer')
        files.append({'relative_path': relative, 'bytes': size,
                      'etag': head.etag.strip('"'), 'source_key': key,
                      'get_url': c.bucket.sign_url('GET', key, 172800),
                      'sha256': consumed_sha if relative == 'assets/subtask_mixture.json' else
                      hashlib.sha256(c.read(key)).hexdigest() if size <= 4*1024**2 else None})
    if c.read(CKPT + '/_CHECKPOINT_METADATA') != metadata_raw or c.inventory(CKPT) != objects:
        raise ValueError('Committed checkpoint changed during snapshot')
    result.update(state='committed_checkpoint5000', metadata=metadata,
                  source_checkpoint=CKPT, formal_contract=contract, consumed_step=consumed['step'], files=files)
print(json.dumps(result))
