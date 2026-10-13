import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
from unittest.mock import patch

import pytest
from video_eval_adapter import Adapter, build_startup, fetch_bootstrap, install_explicit_observer_launcher
from video_eval_worker import PROGRESS_HOOK


def test_bootstrap_uses_direct_opener_and_reaches_exact_payload(monkeypatch):
    payload = b'{"checkpoint_step":2000000}'
    installed, handlers = [], []

    class Opener:
        def open(self, url, timeout):
            assert timeout == 30
            return io.BytesIO(payload)

    def direct(handler):
        handlers.append(handler)
        return Opener()

    monkeypatch.setattr('urllib.request.build_opener', direct)
    monkeypatch.setattr('urllib.request.install_opener', installed.append)
    monkeypatch.setenv('ALL_PROXY', 'http://unavailable.invalid')
    source = b'seen=None\ndef run(payload):\n global seen\n seen=payload\n'
    scope = {}
    exec(build_startup('https://test.invalid', hashlib.sha256(payload).hexdigest(), source), scope)
    assert handlers[0].proxies == {} and len(installed) == 1
    assert scope['ns']['seen'] == {'checkpoint_step': 2000000}


def test_corrupt_bootstrap_is_not_retried(monkeypatch):
    class Opener:
        def open(self, *_a, **_kw):
            return io.BytesIO(b'changed')
    monkeypatch.setattr('urllib.request.build_opener', lambda *_: Opener())
    monkeypatch.setattr('urllib.request.install_opener', lambda *_: None)
    with pytest.raises(ValueError):
        fetch_bootstrap('https://test.invalid', 'a' * 64)


def test_read_only_transport_has_four_bounded_attempts(monkeypatch):
    calls = []
    class Opener:
        def open(self, *_a, **_kw):
            calls.append(1)
            raise urllib.error.URLError('unavailable')
    monkeypatch.setattr('urllib.request.build_opener', lambda *_: Opener())
    monkeypatch.setattr('urllib.request.install_opener', lambda *_: None)
    monkeypatch.setattr('time.sleep', lambda *_: None)
    with pytest.raises(RuntimeError, match='four bounded attempts'):
        fetch_bootstrap('https://test.invalid', 'a' * 64)
    assert len(calls) == 4


def test_approved_physical_retry_keeps_episode_identity(tmp_path):
    adapter = Adapter.__new__(Adapter)
    adapter.root, adapter.plan = tmp_path, {'run_id': 'fixed-run'}
    (tmp_path / 'launches').mkdir()
    (tmp_path / 'launches/0000.json').write_text(json.dumps(
        {'name': 'fixed-run-e0000', 'physical_name': 'fixed-run-e0000-g001'}))
    job = {'name': 'fixed-run-e0000-g001', 'uid': 'new-uid', 'state': 'RUNNING',
           'ownership': {'user_name': '260010165'},
           'resource_pool': {'name': 'computing-cluster-5090-01g'}}
    with patch('video_eval_adapter.cli_json', return_value=job) as describe:
        response = adapter._describe_live('fixed-run-e0000')
    assert describe.call_args.args[0][3] == 'fixed-run-e0000-g001'
    assert response['name'] == 'fixed-run-e0000' and response['uid'] == 'new-uid'


def test_explicit_hook_survives_activation_clearing_pythonpath(tmp_path):
    actual = tmp_path / 'runtime/envs/b1k-eval/bin'
    actual.mkdir(parents=True)
    (actual / 'python').symlink_to(sys.executable)
    package = tmp_path / 'packages/omnigibson/eval'
    package.mkdir(parents=True)
    (package / 'evaluator.py').write_text('class Evaluator:\n'
        ' def __init__(self): self._rollout_steps=0\n'
        ' def step(self):\n  self._rollout_steps+=1\n  return True,False\n')
    code = tmp_path / 'code/scripts'
    code.mkdir(parents=True)
    script = code / 'eval_b1k_queue.py'
    script.write_text('from omnigibson.eval.evaluator import Evaluator\n'
        'e=Evaluator()\nassert e.step()==(True,False)\nassert e._rollout_steps==1\n')
    root, cache, hook = tmp_path / 'root', tmp_path / 'cache', tmp_path / 'hook'
    for path in (root, cache, hook):
        path.mkdir()
    (hook / 'sitecustomize.py').write_text(PROGRESS_HOOK)
    progress = cache / 'native_progress.json'
    env = dict(os.environ, PYTHONPATH=str(tmp_path / 'packages'),
               VIDEO_EVAL_PROGRESS=str(progress), VIDEO_EVAL_RUN_ID='fixed-episode')
    ns = {'_environment': lambda *_: (dict(env), ['test GPU'])}
    install_explicit_observer_launcher(ns)
    payload = {'runtime': str(tmp_path / 'runtime'), 'code': str(tmp_path / 'code')}
    adapted, _ = ns['_environment'](payload, {}, root, cache, hook)
    result = subprocess.run([str(Path(adapted['EVAL_ENV']) / 'bin/python'), str(script)],
                            env=adapted, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    measured = json.loads(progress.read_text())
    assert measured['run_id'] == 'fixed-episode' and measured['steps'] == 1 and measured['terminal']
    assert (root / 'logs/explicit_progress_python.sh').exists()
