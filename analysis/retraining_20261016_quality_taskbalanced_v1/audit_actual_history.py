"""Read and verify every published metric row of the existing239707 run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from audit_training_progress import BUNDLE, ROOT, check_record, expected_sources


REMOTE = r'''
import importlib.util,json,time
from pathlib import Path
root=Path('/Users/faye/Library/Caches/behavior-quality-taskbalanced-20261016')
spec=importlib.util.spec_from_file_location('dispatch',root/'dispatch_support.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
client=module.module('history_oss',root/'formal_stage/bundle/nebula/submit_oss.py').OSSClient()
key='wangfaye/result/behavior-challenge/quality_taskbalanced/b1k-quality-taskbalanced-1688977-20261016/attempts/stage5000-g0/training_metrics.jsonl'
size=client.head(key)
if not size or size>4*1024**2:raise ValueError('Published metrics absent or exceed bounded read')
response=client.bucket.get_object(key,byte_range=(0,size-1))
try:raw=response.read(size+1).decode()
finally:response.resp.response.close()
if len(raw.encode())!=size:raise ValueError('Metric snapshot truncated')
print(json.dumps({'observed_at':time.time(),'source_key':key,'raw':raw}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verified-ipv6', action='store_true',
                        help='Use the legacy Mac-local IPv6 route only from that host.')
    args = parser.parse_args()
    receipt = json.loads((ROOT / 'formal_stage/receipt.mirrored.json').read_text())
    if receipt['identifiers'] != ['239707']:
        raise ValueError('Only the existing239707 job may be read')
    command = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=8', '-oStrictHostKeyChecking=yes']
    if args.verified_ipv6:
        command += ['-oHostKeyAlias=MBP16.local', '-oHostName=fe80::100e:8e6c:1f72:2de%%en7']
    command += ['macpro', 'cd "/Users/faye/code/Behavior-1k Challenge/BEHAVIOR-VLAs" && '
               'source nebula/config.sh >/dev/null 2>&1 && '
               '/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/env/bin/python -']
    result = subprocess.run(command, input=REMOTE, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError('Read-only metric history fetch failed')
    snapshot = json.loads(result.stdout)
    lines = snapshot['raw'].splitlines()
    records = []
    incomplete_tail = False
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if i != len(lines) - 1:
                raise
            incomplete_tail = True
            continue
        if row.get('kind') != 'training_update':
            raise ValueError('Unexpected metric record type')
        records.append(row)
    if not records:
        raise ValueError('No actual optimizer update records')
    last = records[-1]['completed_updates']
    steps = list(range(1, min(last, 200) + 1)) + list(range(210, last + 1, 10))
    if [r['completed_updates'] for r in records] != steps:
        raise ValueError('Published metric history has missing or duplicate rows')
    expected = expected_sources()
    contract = json.loads((BUNDLE / 'inputs/compiled/learning_rate_contract.json').read_text())
    checked = [check_record(row, expected, contract) for row in records]
    if any(row['metrics']['subtask_loss/full_episode'] != 0 for row in records):
        raise ValueError('Full-episode text loss was not zero')
    (ROOT / 'training_metrics.snapshot.jsonl').write_text(snapshot['raw'])
    audit = {'kind': 'complete_published_training_metric_history_audit',
             'observed_at': snapshot['observed_at'], 'read_only': True, 'job': '239707',
             'last_completed_update': last, 'record_count': len(records),
             'published_rows_contiguous': True, 'incomplete_tail_ignored': incomplete_tail,
             'all_recorded_source_prefixes_match': True, 'all_recorded_lr_and_clocks_match': True,
             'all_recorded_metrics_finite': True, 'all_recorded_full_text_loss_zero': True,
             'source_totals': checked[-1]['actual_source_totals'],
             'last_learning_rate': checked[-1]['actual_learning_rate'],
             'max_recorded_learning_rate': max(r['metrics']['learning_rate'] for r in records),
             'source_sha256': hashlib.sha256(snapshot['raw'].encode()).hexdigest(),
             'source_key': snapshot['source_key'], 'model_improvement_proven': False}
    (ROOT / 'actual_history_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
