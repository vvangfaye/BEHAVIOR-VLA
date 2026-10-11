"""Stream complete native episodes to private storage before publishing their JSON."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

from collection_checkpoint import file_hash, request, upload_json


def checkpoint_episode(path, result, manifest, *, attempt_id):
    # A multipart session belongs to exactly one first attempt. Unknown uploads
    # are reconciled using their receipt instead of being silently overwritten.
    from full_episode_trajectory import inspect_full_episode_trajectory
    from full_episode_multipart import MultipartArchive
    path = Path(path)
    info = inspect_full_episode_trajectory(path, require_success=False)
    if info['result'] != result or info['metadata']['attempt_id'] != attempt_id:
        raise ValueError('Object checkpoint differs from the actual completed episode')
    key = f"{result['task']}:{result['instance_id']}:{result['rollout_id']}"
    storage = manifest['object_storage'][key]
    try:
        request(storage['receipt']['get'])
    except FileNotFoundError:
        pass
    else:
        raise ValueError('Episode already has an object receipt; reconcile rather than overwrite')
    sink = MultipartArchive(storage['archive'], request)
    result_bytes = (json.dumps(result, sort_keys=True, allow_nan=False) + '\n').encode()
    with tarfile.open(fileobj=sink, mode='w|') as archive:
        archive.add(path, arcname='episode.hdf5', recursive=False)
        member = tarfile.TarInfo('result.json')
        member.size = len(result_bytes)
        archive.addfile(member, io.BytesIO(result_bytes))
    stored = sink.finish()
    receipt = dict(kind='native_full_episode_object_checkpoint', run_id=manifest['run_id'],
        source_manifest_sha256=manifest['source_manifest_sha256'], attempt_id=attempt_id,
        task=result['task'], instance_id=result['instance_id'], rollout_id=result['rollout_id'],
        steps=info['steps'], success=result['success'], trajectory_sha256=file_hash(path),
        content_sha256=info['content_sha256'], subtask_sha256=info['subtask_sha256'],
        archive=stored, result=json.loads(json.dumps(result, allow_nan=False)))
    upload_json(receipt, storage['receipt'])
    return receipt
