"""Receive ephemeral signed URLs on stdin and download only verified prepared files."""
import json
from pathlib import Path
import ssl
import sys
import urllib.request

from prepare_retraining_data import download, error_detail, save


def run(payload):
    root = Path('/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/prepared_transfers')
    identity = payload['identity']
    if len(identity) != 20 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('Invalid prepared package identity')
    work = root / identity
    work.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    assert set(payload['targets']) == {'data.tar', 'cloud_index.json'}
    for filename, row in payload['targets'].items():
        download(row, work / filename, opener, workers=4)
    save(work / 'cloud_receipt.json', payload['proof'])
    return {'downloaded': True, 'identity': identity, 'training_executed': False}


if __name__ == '__main__':
    try:
        print(json.dumps(run(json.load(sys.stdin))))
    except BaseException as error:
        print(json.dumps(error_detail(error)))
        raise SystemExit(1)
