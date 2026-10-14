import io
import json
from types import SimpleNamespace

import pytest
from relay_retraining_stream import collect_new
from accept_cloud_prepared import run


@pytest.mark.parametrize('stdout', [
    '{\n  "data_ready": true\n}\n',
    '{"phase": "checking"}\n{"data_ready": true}\n',
])
def test_remote_result_accepts_indented_json_and_progress(stdout, monkeypatch):
    import relay_retraining_stream as module
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=stdout, stderr=''))
    assert module.command_remote('finalize_retraining_data.py', []) == {'data_ready': True}


def row():
    return {'key': 'one', 'sha256': 's', 'bytes': 1, 'frames': 10, 'episodes': 1}


def receipt(key='one'):
    return {'source_key': key, 'source_sha256': 's', 'accepted': True,
            'target': {'full_destination_readback': True}, 'index_target': {'full_destination_readback': True},
            'acceptance': {'real_lerobot_reader': True, 'frames': 10, 'episodes': 1}}


class Store:
    def __init__(self, value): self.value = value; self.gets = 0
    def get_paginator(self, name): return self
    def paginate(self, **kwargs):
        assert kwargs['Prefix'] == 'root/receipts/'
        return [{'Contents': [{'Key': 'root/receipts/one.json'}]}]
    def get_object(self, **kwargs):
        self.gets += 1
        return {'Body': io.BytesIO(json.dumps(self.value).encode())}


def test_stream_accepts_verified_partial_receipts_and_does_not_reread():
    store = Store(receipt()); state = {'by_key': {}, 'seen_objects': set()}
    plan = {'shards': [{'manifest_key': 'root/cloud.json', 'source_keys': ['one']}]}
    frozen = {'source_archives': [row()]}
    assert list(collect_new(store, plan, frozen, state)) == ['one']
    collect_new(store, plan, frozen, state)
    assert store.gets == 1


@pytest.mark.parametrize('damage', ['scope', 'sha', 'frames', 'readback', 'conflict'])
def test_stream_rejects_invalid_or_conflicting_receipts(damage):
    r = receipt()
    state = {'by_key': {}, 'seen_objects': set()}
    if damage == 'scope': r['source_key'] = 'outside'
    if damage == 'sha': r['source_sha256'] = 'different'
    if damage == 'frames': r['acceptance']['frames'] = 9
    if damage == 'readback': r['target']['full_destination_readback'] = False
    if damage == 'conflict': state['by_key']['one'] = dict(r, different=True)
    with pytest.raises(ValueError):
        collect_new(Store(r), {'shards': [{'manifest_key': 'root/cloud.json', 'source_keys': ['one']}]},
                    {'source_archives': [row()]}, state)


def test_destination_locks_are_per_source_and_acceptance_still_required(tmp_path, monkeypatch):
    import accept_cloud_prepared as module
    seen = []
    def prep(cache, public, *, lock_name):
        seen.append(lock_name)
        return SimpleNamespace(inputs={'source_archives': [row(), dict(row(), key='two')]})
    monkeypatch.setattr(module, 'Preparation', prep)
    run(tmp_path, tmp_path, 'one', prepare_only=True)
    run(tmp_path, tmp_path, 'two', prepare_only=True)
    run(tmp_path, tmp_path, 'one', prepare_only=True)
    assert seen[0] == seen[2] and seen[0] != seen[1]
    assert not list(tmp_path.glob('receipts/*.json'))


def test_direct_download_checks_input_identity_and_does_not_accept(tmp_path, monkeypatch):
    import download_prepared_targets as module
    with pytest.raises(ValueError):
        module.run({'identity': '../outside'})
