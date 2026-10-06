import hashlib
import io
import json
from pathlib import Path

import pytest

from nebula import eval_runtime as runtime


def fixture_wheel(tmp_path, monkeypatch):
    data = b"verified-wheel-content"
    monkeypatch.setattr(runtime, "SHA256", hashlib.sha256(data).hexdigest())
    return tmp_path / runtime.FILENAME, data


def test_missing_wheel_blocks_preflight(tmp_path):
    with pytest.raises(FileNotFoundError):
        runtime.verify(tmp_path)


def test_corrupt_wheel_is_rejected(tmp_path, monkeypatch):
    path, data = fixture_wheel(tmp_path, monkeypatch)
    path.write_bytes(data)
    assert runtime.verify(tmp_path)["sha256"] == hashlib.sha256(data).hexdigest()
    path.write_bytes(data + b"corruption")
    with pytest.raises(ValueError, match="pinned official SHA256"):
        runtime.verify(tmp_path)


def test_existing_verified_wheel_reuses_without_network(tmp_path, monkeypatch):
    path, data = fixture_wheel(tmp_path, monkeypatch)
    path.write_bytes(data)
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k: pytest.fail("No network expected"))
    assert runtime.fetch(tmp_path)["bytes"] == len(data)


def metadata(data, digest):
    return {"urls": [{"filename": runtime.FILENAME, "digests": {"sha256": digest},
                      "url": "https://files.pythonhosted.org/fixture.whl", "size": len(data)}]}


def test_changed_publisher_metadata_is_rejected_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k:
                        io.BytesIO(json.dumps(metadata(b"data", "changed")).encode()))
    with pytest.raises(ValueError, match="PyPI metadata differs"):
        runtime.fetch(tmp_path)
    assert not (tmp_path / runtime.FILENAME).exists()


def test_complete_download_is_verified_and_published(tmp_path, monkeypatch):
    path, data = fixture_wheel(tmp_path, monkeypatch)
    responses = iter([json.dumps(metadata(data, runtime.SHA256)).encode(), data])
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(next(responses)))
    report = runtime.fetch(tmp_path)
    assert report["bytes"] == len(data) and path.read_bytes() == data
    assert not (tmp_path / (runtime.FILENAME + ".partial")).exists()


def test_incomplete_download_never_publishes_wheel(tmp_path, monkeypatch):
    path, data = fixture_wheel(tmp_path, monkeypatch)
    responses = iter([json.dumps(metadata(data, runtime.SHA256)).encode(), data[:3]])
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(next(responses)))
    with pytest.raises(ValueError, match="incomplete or corrupt"):
        runtime.fetch(tmp_path)
    assert not path.exists() and not (tmp_path / (runtime.FILENAME + ".partial")).exists()


def test_interrupted_download_is_not_overwritten(tmp_path, monkeypatch):
    path, data = fixture_wheel(tmp_path, monkeypatch)
    partial = tmp_path / (runtime.FILENAME + ".partial")
    partial.write_bytes(b"previous attempt")
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k:
                        io.BytesIO(json.dumps(metadata(data, runtime.SHA256)).encode()))
    with pytest.raises(FileExistsError):
        runtime.fetch(tmp_path)
    assert partial.read_bytes() == b"previous attempt" and not path.exists()


def test_pin_matches_bundled_omnigibson_and_worker_checks_before_staging():
    root = Path(__file__).resolve().parents[1]
    setup = (root / "BEHAVIOR-1K/OmniGibson/setup.py").read_text()
    requirement = (root / "nebula/requirements-eval-runtime.txt").read_text()
    assert f"warp-lang=={runtime.VERSION}" in setup and f"warp-lang=={runtime.VERSION}" in requirement
    assert "--hash=sha256:" + runtime.SHA256 in requirement
    worker = (root / "nebula/eval_pi05_worker.sh").read_text()
    assert worker.index('"$SCRIPT_DIR/check_eval_runtime.py"') < worker.index('log "restoring OpenPI cache"')
    assert worker.index('"$SCRIPT_DIR/check_eval_runtime.py"') < worker.index('log "copying inference checkpoint from OSS"')
    assert "--no-index --no-deps" in worker and "--require-hashes" in worker
    assert "import bddl, isaacsim, omnigibson, omnigibson.eval.eval, torch" in worker
    submit = (root / "nebula/submit_pi05_eval.sh").read_text()
    assert submit.index('"$SCRIPT_DIR/eval_runtime.py" verify') < submit.index('--destination "$OSS_RECEIPT_PREFIX.claim.json"')


@pytest.mark.parametrize("scenario", ["wrong_version", "no_cuda", "bad_kernel"])
def test_runtime_rejects_wrong_version_cpu_fallback_and_bad_kernel(tmp_path, monkeypatch, scenario):
    import importlib.util
    import sys
    from types import SimpleNamespace
    import numpy as np

    wp = SimpleNamespace(__version__="old" if scenario == "wrong_version" else runtime.VERSION,
        kernel=lambda fn: fn, array=lambda **kwargs: object, float32=object,
        config=SimpleNamespace(), init=lambda: None,
        get_device=lambda name: SimpleNamespace(is_cuda=scenario != "no_cuda"),
        zeros=lambda *args, **kwargs: SimpleNamespace(numpy=lambda: np.zeros(4, dtype=np.float32)),
        launch=lambda *args, **kwargs: None, synchronize_device=lambda device: None)
    monkeypatch.setitem(sys.modules, "warp", wp)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "eval_runtime", runtime)
    path = Path(__file__).with_name("check_eval_runtime.py")
    spec = importlib.util.spec_from_file_location("runtime_checker_fixture", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    error = AssertionError if scenario == "bad_kernel" else RuntimeError
    with pytest.raises(error):
        checker.check(tmp_path / "kernel-cache")


def test_single_gpu_subtask_import_check_uses_visible_device_zero(tmp_path):
    """Execute the actual import-check block with a one-device runtime stand-in."""
    import os
    import subprocess
    source=(Path(__file__).with_name('eval_pi05_worker.sh')).read_text()
    start=source.index('IMPORT_CHECK_RUNTIME=')
    end=source.index('LD_LIBRARY_PATH=',source.index("torch.cuda.get_device_name())'",start))
    block=source[start:end]
    env_root=tmp_path/'env';(env_root/'bin').mkdir(parents=True)
    python=env_root/'bin/python'
    python.write_text('#!/bin/bash\n[[ "$CUDA_VISIBLE_DEVICES" == 0 ]] || exit 91\nprintf "visible_device=%s\\n" "$CUDA_VISIBLE_DEVICES"\n')
    python.chmod(0o755)
    env=os.environ.copy();env.update(EVAL_PROFILE='subtasks100',EVAL_WORKERS='1',PI05_MODEL_FORMAT='jax',
        RUNTIME_ROOT=str(tmp_path/'runtime'),EVAL_ENV=str(env_root),OMNIGIBSON_DATA_PATH=str(tmp_path/'assets'))
    result=subprocess.run(['/bin/bash','-eu','-c',block],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='visible_device=0'
