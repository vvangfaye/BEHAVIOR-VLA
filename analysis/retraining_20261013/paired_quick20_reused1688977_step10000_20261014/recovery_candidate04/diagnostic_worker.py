"""Read only the failed episode's persisted evidence; never start a policy."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/mnt/afs/260010165/behavior_1k/paired_quick20_reused1688977_step10000_20261014')
FILES = [
    'workers/candidate-04/worker.log',
    'workers/candidate-04/report.json',
    'episodes/candidate-04/policy.log',
    'episodes/candidate-04/eval.log',
    'episodes/candidate-04/seed_ack.json',
    'episodes/candidate-04/policy_metadata.json',
    'episodes/candidate-04/acceptance.json',
    'episodes/candidate-04/results/json/putting_up_Christmas_decorations_inside_301_0.json',
    'checkpoint10000/_READY',
]


def main():
    result = dict(observed_at=time.time(), read_only=True, files={},
                  old_uid='96a9f2f9f1c5470d850f886c37c1c7c4',
                  training_modified=False, policy_started=False)
    for relative in FILES:
        path = ROOT / relative
        record = {'exists': path.is_file()}
        if record['exists']:
            stat = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4*1024**2), b''):
                    digest.update(block)
            record.update(bytes=stat.st_size, mtime=stat.st_mtime, sha256=digest.hexdigest())
            with path.open('rb') as stream:
                head = stream.read(min(stat.st_size, 64*1024))
                stream.seek(max(0, stat.st_size - 768*1024))
                tail = stream.read(768*1024)
            record.update(head=head.decode(errors='replace'), tail=tail.decode(errors='replace'))
        result['files'][relative] = record
    video = ROOT / 'episodes/candidate-04/results/videos/putting_up_Christmas_decorations_inside_301_0.mp4'
    result['video'] = {'exists': video.is_file(), 'bytes': video.stat().st_size if video.is_file() else None}
    return result
