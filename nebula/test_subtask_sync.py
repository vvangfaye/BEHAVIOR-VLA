"""Regression coverage of result publication without an installed rsync."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subtask_eval as evaluation


def put(root, name, data=b'content'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_sync_publishes_outputs_and_excludes_live_or_unvalidated_files(tmp_path):
    local, output = tmp_path / 'local', tmp_path / 'output'
    allowed = ['manifest.json', 'status/worker-0.json', 'logs/worker.log',
               'results/pi05/task/json/result.json', 'results/pi05/task/videos/result.mp4']
    excluded = ['queue.sqlite3', 'queue.sqlite3-wal', 'queue.sqlite3-shm',
                '_SUCCESS', '_FAILED', '_RUNNING', '.supervisor.lock',
                'attempts/task/attempt-01/result.json', 'status/.worker.json.abc.tmp']
    for name in allowed + excluded:
        put(local, name)
    evaluation.sync_results(local, output)
    assert {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == set(allowed)


def test_incremental_sync_skips_unchanged_large_files_and_updates_same_size_changes(tmp_path, monkeypatch):
    local, output = tmp_path / 'local', tmp_path / 'output'
    source = put(local, 'results/video.mp4', b'old content')
    copied, calls = {}, []
    original = evaluation.shutil.copyfile
    def copy(src, dst):
        calls.append(src)
        return original(src, dst)
    monkeypatch.setattr(evaluation.shutil, 'copyfile', copy)
    evaluation.sync_results(local, output, copied)
    evaluation.sync_results(local, output, copied)
    assert len(calls) == 1
    stamp = source.stat().st_mtime_ns
    source.write_bytes(b'new content')
    os.utime(source, ns=(stamp + 1000000, stamp + 1000000))
    evaluation.sync_results(local, output, copied)
    assert len(calls) == 2 and (output / 'results/video.mp4').read_bytes() == b'new content'
    (output / 'results/video.mp4').unlink()
    evaluation.sync_results(local, output, copied)
    assert len(calls) == 3  # A missing destination is restored even with a cached identity.


def test_failed_copy_keeps_previous_destination_and_removes_partial(tmp_path, monkeypatch):
    local, output = tmp_path / 'local', tmp_path / 'output'
    put(local, 'result.json', b'new complete result')
    put(output, 'result.json', b'previous result')
    def interrupted(src, dst):
        Path(dst).write_bytes(b'partial')
        raise OSError('simulated write failure')
    monkeypatch.setattr(evaluation.shutil, 'copyfile', interrupted)
    copied = {}
    with pytest.raises(OSError, match='write failure'):
        evaluation.sync_results(local, output, copied)
    assert (output / 'result.json').read_bytes() == b'previous result'
    assert list(output.iterdir()) == [output / 'result.json'] and not copied


def test_incomplete_copy_cannot_replace_published_file(tmp_path, monkeypatch):
    local, output = tmp_path / 'local', tmp_path / 'output'
    put(local, 'video.mp4', b'complete video')
    put(output, 'video.mp4', b'previous video')
    monkeypatch.setattr(evaluation.shutil, 'copyfile', lambda src, dst: Path(dst).write_bytes(b'half'))
    with pytest.raises(OSError, match='Incomplete result copy'):
        evaluation.sync_results(local, output)
    assert (output / 'video.mp4').read_bytes() == b'previous video'
    assert not list(output.glob('*.tmp'))


def test_file_changed_during_copy_is_deferred_until_stable(tmp_path, monkeypatch):
    local, output = tmp_path / 'local', tmp_path / 'output'
    source = put(local, 'logs/worker.log', b'before')
    put(output, 'logs/worker.log', b'published')
    original = evaluation.shutil.copyfile
    def growing(src, dst):
        original(src, dst)
        source.write_bytes(b'after append')
    monkeypatch.setattr(evaluation.shutil, 'copyfile', growing)
    copied = {}
    evaluation.sync_results(local, output, copied)
    assert (output / 'logs/worker.log').read_bytes() == b'published' and not copied
    monkeypatch.setattr(evaluation.shutil, 'copyfile', original)
    evaluation.sync_results(local, output, copied)
    assert (output / 'logs/worker.log').read_bytes() == b'after append'
    assert not list(output.rglob('*.tmp'))


@pytest.mark.parametrize('directory', [False, True])
def test_result_symlinks_cannot_escape_output_tree(tmp_path, directory):
    local, output = tmp_path / 'local', tmp_path / 'output'
    local.mkdir()
    target = tmp_path / 'outside'
    target.mkdir() if directory else target.write_bytes(b'outside data')
    (local / 'link').symlink_to(target, target_is_directory=directory)
    with pytest.raises(ValueError, match='symlinks'):
        evaluation.sync_results(local, output)
    assert not output.exists()


def test_supervisor_completes_with_no_external_commands_on_path(tmp_path, monkeypatch):
    local, output = tmp_path / 'local', tmp_path / 'output'
    script = tmp_path / 'producer.py'
    script.write_text('''from pathlib import Path
import sys,time
root=Path(sys.argv[1]);root.mkdir()
(root/'result.json').write_text('{"complete":true}')
(root/'_SUCCESS').write_text('local completion marker')
time.sleep(0.1)
''')
    monkeypatch.setenv('PATH', '')
    evaluation.supervise_results([sys.executable, str(script), str(local)], local, output, poll_interval=0.01)
    assert json.loads((output / 'result.json').read_text()) == {'complete': True}
    assert not (output / '_SUCCESS').exists()


def test_sync_failure_terminates_tracked_child_and_preserves_original_error(tmp_path, monkeypatch, capsys):
    local, output = tmp_path / 'local', tmp_path / 'output'
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    monkeypatch.setattr(evaluation.subprocess, 'Popen', lambda *a, **k: child)
    errors = iter(['primary sync failure', 'secondary final sync failure'])
    def fail(*a, **k):
        raise OSError(next(errors))
    monkeypatch.setattr(evaluation, 'sync_results', fail)
    try:
        with pytest.raises(OSError, match='primary sync failure'):
            evaluation.supervise_results(['unused'], local, output, poll_interval=0.01)
        assert child.poll() is not None
        assert 'secondary final sync failure' in capsys.readouterr().err
    finally:
        if child.poll() is None:
            child.kill();child.wait()


def test_failed_child_is_not_masked_by_final_sync_error(tmp_path, monkeypatch, capsys):
    child = SimpleNamespace(returncode=7, poll=lambda: 7)
    monkeypatch.setattr(evaluation.subprocess, 'Popen', lambda *a, **k: child)
    monkeypatch.setattr(evaluation, 'sync_results', lambda *a, **k: (_ for _ in ()).throw(OSError('final sync failure')))
    with pytest.raises(RuntimeError, match='exit code 7'):
        evaluation.supervise_results(['unused'], tmp_path / 'local', tmp_path / 'output')
    assert 'final sync failure' in capsys.readouterr().err


@pytest.mark.parametrize('marker,complete', [(False, True), (True, False), (True, True)])
def test_success_marker_requires_strict_child_completion(tmp_path, monkeypatch, marker, complete):
    local, output, checkpoint = tmp_path / 'local', tmp_path / 'output', tmp_path / 'checkpoint'
    checkpoint.mkdir();(checkpoint / '_READY').write_text('{"step":1375000}')
    panel = put(tmp_path, 'panel.json', b'{}')
    inventory = put(tmp_path, 'inventory.json', b'{}')
    args = argparse.Namespace(panel=panel, inventory=inventory, run_root=local, persistent_output=output,
        raw_source=tmp_path / 'source', raw_root=tmp_path / 'raw', run_id='test', checkpoint=checkpoint,
        checkpoint_source=str(checkpoint), repo_id='repo', workers=4, base_port=8300, sample_max_steps=2048,
        eval_env=tmp_path / 'eval', pi05_env=tmp_path / 'policy', asset_path=tmp_path / 'assets')
    monkeypatch.setattr(evaluation, 'load_inputs', lambda *a: ({'fingerprint':'frozen', 'selection':{'tasks':['task']},
        'episodes':[{}], 'samples':[{}], 'exclusions':[]}, {}))
    monkeypatch.setattr(evaluation, 'stage_raw', lambda *a: tmp_path / 'raw')
    def produce(*a):
        local.mkdir();(local / 'summary.json').write_text(json.dumps({'complete':complete}))
        if marker:(local / '_SUCCESS').write_text('strict completion')
        evaluation.sync_results(local, output)
    monkeypatch.setattr(evaluation, 'supervise_results', produce)
    monkeypatch.setattr(evaluation, 'write_reports', lambda *a: None)
    if marker and complete:
        evaluation.run(args)
        assert (output / '_SUCCESS').read_text() == 'strict completion'
    else:
        with pytest.raises(RuntimeError, match='strict successful completion'):
            evaluation.run(args)
        assert not (output / '_SUCCESS').exists()
