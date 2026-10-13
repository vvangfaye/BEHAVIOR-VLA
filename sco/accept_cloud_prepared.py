"""Read back a source-cloud training export and transfer it to the final OSS."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from prepare_retraining_data import Preparation, safe_extract, save, sha, verify_dataset


def run(cache, output, key, prepare_only=False):
    identity = hashlib.sha256(key.encode()).hexdigest()[:20]
    # Concurrent packages have disjoint input/work/output keys. The same package
    # still has an exclusive lock, and finalization runs only after all finish.
    prep = Preparation(cache, output, lock_name='accept-' + identity + '.lock')
    source = next(r for r in prep.inputs['source_archives'] if r['key'] == key)
    receipt_path = Path(output) / 'receipts' / (identity + '.json')
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get('accepted') and old.get('index_target') and old['source_sha256'] == source['sha256']:
            print(json.dumps({'already_accepted': True, 'id': identity})); return
    work = Path(cache) / 'prepared_transfers' / identity
    work.mkdir(parents=True, exist_ok=True)
    if prepare_only:
        print(json.dumps({'already_accepted': False, 'id': identity, 'work': str(work)})); return
    proof = json.loads((work / 'cloud_receipt.json').read_text())
    assert proof['accepted'] and proof['execution_location'] == 'source_cloud_cpu'
    assert proof['source_key'] == key and proof['source_sha256'] == source['sha256']
    assert proof['acceptance']['real_lerobot_reader'] and proof['target']['full_destination_readback']
    archive = work / 'data.tar'
    assert archive.stat().st_size == proof['target']['bytes'] and sha(archive) == proof['target']['sha256']
    index_file = work / 'cloud_index.json'
    assert index_file.stat().st_size == proof['index_target']['bytes'] and sha(index_file) == proof['index_target']['sha256']
    portable = json.loads(index_file.read_text())
    assert portable['raw_source']['key'] == key and portable['raw_source']['sha256'] == source['sha256']
    unpacked = work / 'unpacked'
    if unpacked.exists():
        shutil.rmtree(unpacked)
    safe_extract(archive, unpacked, max_bytes=prep.auth['maximum_scratch_bytes'])
    reports = list(unpacked.rglob('export_report.json'))
    assert len(reports) == 1
    dataset = reports[0].parent
    export = json.loads(reports[0].read_text())
    readback = verify_dataset(dataset, export['repo_id'], permit_empty_labels=source['kind'] == 'full_episode')
    assert readback['episodes'] == proof['acceptance']['episodes'] == source['episodes']
    assert readback['frames'] == proof['acceptance']['frames'] == source['frames']
    if portable['index'] is not None:
        from openpi.training.recovery_index import validate_index
        validate_index(portable['index'])
    branch = 'full_episode/exports/' if source['kind'] == 'full_episode' else 'subtask/archives/'
    target = prep.upload(archive, prep.prefix + branch + source['sha256'] + '.tar')
    portable['data_target'] = target
    local_index = Path(output) / 'indexes' / (proof['root_id'] + '.json')
    save(local_index, portable)
    index_target = prep.upload(local_index, prep.prefix + 'indexes/' + proof['root_id'] + '.json')
    final = {**proof, 'target': target, 'index_target': index_target,
        'source_cloud_target': proof['target'], 'source_cloud_acceptance': proof['acceptance'],
        'destination_reader_acceptance': readback, 'verified_at': time.time(), 'training_executed': False}
    save(receipt_path, final)
    shutil.rmtree(work)
    print(json.dumps({'accepted': True, 'id': identity, 'episodes': readback['episodes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args(); run(a.cache, a.output, a.key, a.prepare_only)
