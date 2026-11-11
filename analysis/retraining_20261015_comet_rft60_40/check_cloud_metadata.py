"""Read existing preparation receipts and indexes. Never submit or transfer raw data."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
sys.path.insert(0, str(WORKSPACE / 'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
import boto3
from botocore.config import Config


def main():
    configured = aoss_client()
    # A single bounded attempt per small metadata request; no background retry loop.
    credentials = configured._request_signer._credentials.get_frozen_credentials()
    client = boto3.client('s3', endpoint_url=configured.meta.endpoint_url, region_name='cn-sh-01b',
                         aws_access_key_id=credentials.access_key,
                         aws_secret_access_key=credentials.secret_key,
                         aws_session_token=credentials.token,
                         config=Config(signature_version='s3v4', s3={'addressing_style':'path'},
                                       proxies={}, connect_timeout=6, read_timeout=12,
                                       retries={'total_max_attempts':1}))
    bucket = '260010165'
    prep = WORKSPACE / 'analysis/retraining_20261015_round3_v2'
    frozen = json.loads((prep / 'source_archives.json').read_text())
    expected = {row['key']: row for row in frozen['source_archives']}
    evidence = HERE / 'evidence'
    evidence.mkdir(exist_ok=True)

    def read_json(key):
        response = client.get_object(Bucket=bucket, Key=key)
        with response['Body'] as stream:
            raw = stream.read(16 * 1024**2 + 1)
        if len(raw) > 16 * 1024**2:
            raise ValueError('Metadata exceeds the bounded size limit')
        return raw, json.loads(raw)

    def inspect(proof_path):
        proof = json.loads(proof_path.read_text())
        assert proof['source_key'] in expected
        target = proof['index_target']
        raw, index = read_json(target['key'])
        assert hashlib.sha256(raw).hexdigest() == target['sha256']
        assert index['root_id'] == proof['root_id']
        assert index['supervision_policy'] == 'action_only; policy_subtask_text_is_weak_not_human'
        selected = index['action_only_episodes']
        assert len(selected) == 1 and selected[0]['language_loss_enabled'] is False
        source = expected[proof['source_key']]
        assert selected[0]['content_sha256'] == source['content_sha256']
        assert selected[0]['length'] == source['frames']
        head = client.head_object(Bucket=bucket, Key=proof['target']['key'])
        assert head['ContentLength'] == proof['target']['bytes']
        path = evidence / 'new_full_indexes' / (proof['root_id'] + '.json')
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        return dict(source_key=proof['source_key'], root_id=proof['root_id'],
                    index_sha256=target['sha256'], export_bytes=head['ContentLength'],
                    export_head_verified=True, index_original_bytes_verified=True,
                    prior_export_full_sha_and_reader_accepted=proof['accepted'],
                    export_rehashed_this_check=False)

    paths = sorted((prep / 'cloud_receipts').glob('*.json'))
    assert len(paths) == len(expected) == 82
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(inspect, paths))

    prefix = 'behavior1k/retraining_20261015_round3_v2/batch-0/direct-transfer/'
    receipt_keys = []
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix+'receipts/'):
        receipt_keys.extend(o['Key'] for o in page.get('Contents', []) if o['Key'].endswith('.json'))
    def read_receipt(key):
        raw, receipt = read_json(key)
        assert receipt['source_key'] in expected
        assert receipt['accepted'] and receipt['target']['bucket'] == 'traffic-model'
        assert receipt['target']['full_destination_readback']
        (evidence / 'destination_receipts').mkdir(exist_ok=True)
        (evidence / 'destination_receipts' / key.rsplit('/', 1)[1]).write_bytes(raw)
        return receipt
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(read_receipt, receipt_keys))
    _, status = read_json(prefix+'status.json')
    keys = {r['source_key'] for r in receipts}
    assert len(keys) == len(receipts)
    output = dict(observed_at=time.time(), scope='existing 82 frozen sources only',
                  new_jobs_created=0, training_started=False, cloud_prepared_indexes_verified=len(results),
                  cloud_prepared_bytes=sum(r['export_bytes'] for r in results),
                  destination_accepted_receipts=len(receipts), destination_frames=sum(expected[k]['frames'] for k in keys),
                  destination_receipts_are_prior_byte_readbacks_not_fresh_ali_oss_head=True,
                  direct_status_summary={k:status.get(k) for k in ['state','accepted','completed','pending','failed','updated_at','training_executed']},
                  results=results)
    (evidence/'cloud_metadata_check.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='results'},ensure_ascii=False))


if __name__ == '__main__':
    main()
