"""Resume contract tests; fake video subprocesses never prove real readback."""
import copy
import io
import json
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_codex_original_resume as resume
from test_prepare_codex_original_data import payload, replay_inputs, signed_payload, demo_process, stage


@pytest.fixture
def partial(signed_payload, monkeypatch, tmp_path):
    payload, contents = signed_payload
    calls = []
    def download(request, **kwargs):
        if len(calls) == 2:
            raise TimeoutError('synthetic third-file timeout')
        index = int(request.full_url.split('?')[0].rsplit('/', 1)[1])
        calls.append(index)
        response = io.BytesIO(contents[index])
        response.headers = {'Content-Length': str(len(contents[index])), 'ETag': f'"fixture-{index}"'}
        return response
    monkeypatch.setattr(stage.urllib.request, 'urlopen', download)
    with pytest.raises(TimeoutError):
        stage.run(payload)
    old = Path(payload['output_root'])
    refreshed = copy.deepcopy(payload)
    refreshed['output_root'] = str(tmp_path / 'next-output')
    for row in refreshed['signed_files']:
        row['url'] = row['url'].replace('DO_NOT_PERSIST', 'FRESH_PRIVATE_TOKEN')
    return old, refreshed, contents


def prepare(partial, tmp_path):
    old, refreshed, _ = partial
    return resume.prepare_resume(old, refreshed, previous_job=dict(name='exact-owned-job', uid='exact-owned-uid', state='FAILED'),
        reuse_root=tmp_path / 'verified-reuse', payload_path=tmp_path / 'private-payload.json')


def test_verified_completed_files_reused_and_only_remaining_files_downloaded(partial, tmp_path, monkeypatch):
    old, refreshed, contents = partial
    before = {p.relative_to(old).as_posix(): stage.sha(p) for p in old.rglob('*') if p.is_file()}
    result = prepare(partial, tmp_path)
    assert result['reused_files'] == result['remaining_files'] == 2
    assert result['reused_bytes'] + result['remaining_download_bytes'] == sum(map(len, contents))
    private = tmp_path / 'private-payload.json'
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert 'FRESH_PRIVATE_TOKEN' not in json.dumps(result)
    calls = []
    def download(request, **kwargs):
        index = int(request.full_url.split('?')[0].rsplit('/', 1)[1]); calls.append(index)
        response = io.BytesIO(contents[index])
        response.headers = {'Content-Length': str(len(contents[index])), 'ETag': f'"fixture-{index}"'}
        return response
    monkeypatch.setattr(stage.urllib.request, 'urlopen', download)
    monkeypatch.setattr(stage.subprocess, 'run', demo_process)
    completed = stage.run(json.loads(private.read_text()))
    assert completed['state'] == 'complete' and len(calls) == 2
    assert before == {p.relative_to(old).as_posix(): stage.sha(p) for p in old.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault', ['running', 'weights', 'metadata', 'identity', 'labels', 'scope', 'phase'])
def test_changed_or_live_inputs_rejected_without_creating_resume(partial, tmp_path, fault):
    old, refreshed, _ = partial
    state = json.loads((old / 'status.json').read_text())
    if fault == 'running': (old / '_RUNNING').touch()
    elif fault == 'weights': (old / state['transfers'][0]['path']).write_bytes(b'corrupted')
    elif fault == 'metadata': (old / 'meta/info.json').write_text('{}')
    elif fault == 'identity': refreshed['signed_files'][0]['etag'] = 'changed'
    elif fault == 'labels': Path(refreshed['segments_path']).write_bytes(b'changed')
    elif fault == 'scope': refreshed['demonstrations'][0]['raw_path'] = '/other/raw.hdf5'
    else:
        state['failure_phase'] = 'real_reader_and_demonstrations'
        (old / 'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError): prepare(partial, tmp_path)
    assert not (tmp_path / 'verified-reuse').exists()
    assert not (tmp_path / 'private-payload.json').exists()


def test_unreceipted_final_or_partial_file_is_not_reused(partial, tmp_path):
    old, _, _ = partial
    state = json.loads((old / 'status.json').read_text())
    remaining = sorted(set(state['required_files']) - {r['path'] for r in state['transfers']})
    unexpected = old / remaining[0]; unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_bytes(b'not verified')
    unexpected.with_suffix(unexpected.suffix + '.partial').write_bytes(b'incomplete')
    result = prepare(partial, tmp_path)
    assert result['reused_files'] == 2
    assert not (tmp_path / 'verified-reuse' / remaining[0]).exists()


def test_nonterminal_cloud_job_is_not_accepted(partial, tmp_path):
    old, refreshed, _ = partial
    with pytest.raises(ValueError, match='terminal'):
        resume.prepare_resume(old, refreshed, previous_job=dict(name='job', uid='uid', state='RUNNING'),
            reuse_root=tmp_path / 'verified-reuse', payload_path=tmp_path / 'private-payload.json')
