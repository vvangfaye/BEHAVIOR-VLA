import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nebula.submit_oss import OSSClient, PreflightError, object_key, preflight, validate_manifest

BUCKET = 'test-bucket'
MOUNT = '/data/oss_bucket_0'


class FakeClient:
    def __init__(self, objects):
        self.objects = objects
        self.reads = []

    def key(self, path):
        return object_key(path, BUCKET)

    def head(self, key):
        return len(self.objects[key]) if key in self.objects else None

    def inventory(self, root):
        return {k: len(v) for k, v in self.objects.items() if k.startswith(root + '/')}

    def read(self, key):
        self.reads.append(key)
        return self.objects[key]

    def unused(self, key, directory=True):
        prefix = key + ('/' if directory else '.')
        if key in self.objects or any(k.startswith(prefix) for k in self.objects):
            raise PreflightError('Output or submission receipt already exists')


def manifest(names):
    return ''.join('0' * 64 + '  ' + name + '\n' for name in names).encode()


@pytest.fixture
def inputs(tmp_path):
    assets = ['behavior-1k-assets-3.9.0.zip', 'omnigibson-robot-assets-3.8.2.zip',
              '2026-challenge-task-instances.zip', 'omnigibson.key']
    model = ['params/_METADATA', 'params/weights/chunk', 'assets/repo/norm_stats.json']
    objects = {**{'assets/' + name: b'asset' for name in assets},
               **{'checkpoint/' + name: b'model' for name in model},
               'archives/eval.tar.gz': b'archive',
               'checkpoint/_CHECKPOINT_METADATA': b'{}',
               'assets/SHA256SUMS': manifest(assets),
               'checkpoint/SHA256SUMS.inference': manifest(model)}
    objects['checkpoint/_READY'] = json.dumps({'manifest_sha256': hashlib.sha256(
        objects['checkpoint/SHA256SUMS.inference']).hexdigest()}).encode()
    args = SimpleNamespace(checkpoint='oss://test-bucket/checkpoint',
                           assets=MOUNT + '/assets', archive=[MOUNT + '/archives/eval.tar.gz'],
                           repo_id='repo', model_format='jax', staged=True,
                           output=MOUNT + '/results/new-run',
                           receipt_prefix=MOUNT + '/results/.submission_receipts/new-run',
                           ready_out=str(tmp_path / 'ready.json'))
    return FakeClient(objects), args


def test_preflight_reads_only_small_metadata_and_caches_ready(inputs):
    client, args = inputs
    preflight(client, args)
    assert client.reads == ['assets/SHA256SUMS', 'checkpoint/SHA256SUMS.inference', 'checkpoint/_READY']
    assert json.loads(Path(args.ready_out).read_text())['manifest_sha256']


@pytest.mark.parametrize('path', ['oss://other-bucket/model', '/tmp/model',
                                  MOUNT + '/a/../model', 'oss://test-bucket/model?query=x'])
def test_invalid_remote_paths_fail(path):
    with pytest.raises(PreflightError):
        object_key(path, BUCKET)


def test_uri_and_mount_address_same_object():
    assert object_key('oss://test-bucket/a/model', BUCKET) == object_key(MOUNT + '/a/model', BUCKET)


@pytest.mark.parametrize('key', ['archives/eval.tar.gz', 'checkpoint/params/weights/chunk',
                                'assets/omnigibson.key', 'checkpoint/_READY'])
def test_missing_remote_input_blocks_submission(inputs, key):
    client, args = inputs
    client.objects.pop(key)
    with pytest.raises(PreflightError, match='missing or empty'):
        preflight(client, args)
    assert not Path(args.ready_out).exists()


def test_empty_model_chunk_blocks_submission(inputs):
    client, args = inputs
    client.objects['checkpoint/params/weights/chunk'] = b''
    with pytest.raises(PreflightError, match='missing or empty'):
        preflight(client, args)


def test_ready_manifest_mismatch_blocks_submission(inputs):
    client, args = inputs
    client.objects['checkpoint/_READY'] = b'{"manifest_sha256":"wrong"}'
    with pytest.raises(PreflightError, match='does not match'):
        preflight(client, args)


@pytest.mark.parametrize('key', ['results/new-run/manifest.json',
                                'results/.submission_receipts/new-run.claim.json',
                                'results/.submission_receipts/new-run.stdout.log'])
def test_existing_output_or_receipt_blocks_reuse(inputs, key):
    client, args = inputs
    client.objects[key] = b'prior run'
    with pytest.raises(PreflightError, match='already exists'):
        preflight(client, args)


def test_wrong_checkpoint_format_blocks_submission(inputs):
    client, args = inputs
    client.objects['checkpoint/model.safetensors'] = b'other format'
    with pytest.raises(PreflightError, match='both JAX and Torch'):
        preflight(client, args)


@pytest.mark.parametrize('data', [b'bad line', manifest(['../escape']), manifest(['/absolute']),
                                manifest(['file', './file'])])
def test_invalid_or_escaping_manifest_fails(data):
    with pytest.raises(PreflightError):
        validate_manifest(data, 'root', {'root/file': 1})


def test_head_permission_error_is_not_treated_as_absence():
    class NotFound(Exception):
        pass

    client = OSSClient.__new__(OSSClient)
    client.oss2 = SimpleNamespace(exceptions=SimpleNamespace(NotFound=NotFound))

    class Bucket:
        def head_object(self, key):
            raise PermissionError('403')

    client.bucket = Bucket()
    with pytest.raises(PermissionError):
        client.head('object')


def test_reservation_and_receipt_upload_never_overwrite_existing_object(tmp_path):
    uploads = {}

    class Bucket:
        def put_object(self, key, data, headers):
            assert headers['x-oss-forbid-overwrite'] == 'true'
            if key in uploads:
                raise FileExistsError(key)
            uploads[key] = data

    client = OSSClient.__new__(OSSClient)
    client.bucket = Bucket()
    source = tmp_path / 'claim.json'
    source.write_bytes(b'first submission')
    client.put_new('run.claim.json', source)
    source.write_bytes(b'second submission')
    with pytest.raises(FileExistsError):
        client.put_new('run.claim.json', source)
    assert uploads['run.claim.json'] == b'first submission'

@pytest.mark.parametrize('scenario', ['dryrun', 'success', 'reserve_failure', 'sync_failure'])
def test_shell_submission_without_mount_preserves_receipts_and_never_retries(tmp_path, scenario):
    import os
    import shlex
    import shutil
    import subprocess
    import sys

    source = Path(__file__).resolve().parent
    repo = tmp_path / 'repo'
    nebula = repo / 'nebula'
    nebula.mkdir(parents=True)
    shutil.copy(source / 'submit_pi05_eval.sh', nebula)
    for folder in ('openpi/src', 'openpi/packages/openpi-client/src',
                   'BEHAVIOR-1K/OmniGibson/omnigibson', 'BEHAVIOR-1K/bddl3/bddl',
                   'BEHAVIOR-1K/joylo/gello'):
        directory = repo / folder
        directory.mkdir(parents=True)
        (directory / 'fixture.py').write_text('# source fixture\n')
    for filename in ('openpi/scripts/b1k/serve_b1k.py', 'openpi/scripts/install_transformers_patch.py',
                     'BEHAVIOR-1K/docs/assets/OmniGibson_logo.png', 'scripts/eval_b1k.sh',
                     'nebula/eval_pi05.py', 'nebula/eval_pi05_worker.sh', 'nebula/summarize_pi05_eval.py', 'nebula/eval_storage.py'):
        path = repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# source fixture\n')
    for filename in ('b1k_tasks.txt', 'b1k_tasks_reduced.txt'):
        shutil.copy(source.parent / 'scripts' / filename, repo / 'scripts')
    shutil.copy(source / 'multihost_cluster_4gpu.json', nebula)
    # The real runtime verifier has separate integrity tests. This fixture tests
    # submission/receipt control flow without bundling a 136 MB binary per case.
    (nebula / 'eval_runtime.py').write_text('print("fixture wheel verification passed")\n')
    (nebula / 'check_eval_runtime.py').write_text('# GPU runtime fixture\n')
    (nebula / 'requirements-eval-runtime.txt').write_text('# pinned runtime fixture\n')
    wheels = nebula / 'runtime-wheels'
    wheels.mkdir()
    (wheels / 'warp_lang-1.12.0-py3-none-manylinux_2_28_x86_64.whl').write_bytes(b'fixture wheel')

    # Fake credentials only; tests never read the developer's config.sh.
    (nebula / 'config.sh').write_text('export AI_HUB_TOKEN=fake AI_HUB_NAMESPACE=fake '
                                     'OSS_ACCESS_ID=fake OSS_ACCESS_KEY=fake '
                                     'OSS_BUCKET=test-bucket OSS_ENDPOINT=http://test.invalid\n')
    commands = tmp_path / 'commands'
    commands.mkdir()
    git = commands / 'git'
    git.write_text('#!/bin/sh\nif [ "$1" = rev-parse ]; then printf "fixture-commit\\n"; fi\n')
    git.chmod(0o755)
    ai_hub = commands / 'ai-hub-cli'
    ai_hub.write_text('#!/bin/sh\nprintf invoked >> "$FAKE_JOB_CALLS"\n'
                      'printf \'{"instance_id":"654321", "iid":654321}\\n\'\n')
    ai_hub.chmod(0o755)
    driver = tmp_path / 'sdk_driver.py'
    driver.write_text('''import json, os, pathlib, sys
args = sys.argv[1:]
if args[0] == '-c':
    sys.exit(0)
operation = args[1]
with pathlib.Path(os.environ['FAKE_SDK_CALLS']).open('a') as stream:
    stream.write(operation + '\\n')
if os.environ['FAKE_SCENARIO'] == operation + '_failure':
    sys.exit(2)
if operation == 'preflight':
    pathlib.Path(args[args.index('--ready-out')+1]).write_text(
        json.dumps({'manifest_sha256':'fixture-manifest-digest'}))
if operation == 'sync':
    assert pathlib.Path(args[args.index('--receipt')+1]).is_file()
    assert pathlib.Path(args[args.index('--stdout')+1]).is_file()
''')
    interpreter = commands / 'submit-python'
    interpreter.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(driver)) + ' "$@"\n')
    interpreter.chmod(0o755)
    sdk_calls = tmp_path / 'sdk-calls'
    job_calls = tmp_path / 'job-calls'
    receipts = tmp_path / 'receipts'
    env = os.environ.copy()
    env.update(PATH=str(commands) + os.pathsep + env['PATH'],
               NEBULA_SUBMIT_PYTHON=str(interpreter), AI_HUB_CLI=str(ai_hub), SUBMIT_STORAGE_MODE='oss',
               EVAL_PROFILE='full4', CONFIRM_FULL='1', RUN_ID='fixture-run',
               CHECKPOINT_SOURCE='oss://test-bucket/checkpoint', PI05_REPO_ID='repo',
               RECEIPT_ROOT=str(receipts), FAKE_SCENARIO=scenario,
               FAKE_SDK_CALLS=str(sdk_calls), FAKE_JOB_CALLS=str(job_calls),
               BUNDLE_PARENT=str(tmp_path), DRY_RUN='1' if scenario == 'dryrun' else '0',
               KEEP_BUNDLE='0')
    for name in ('EXTERNAL_SUBTASK_SELECTOR', 'EXTERNAL_SUBTASK', 'CLUSTER_FILE',
                 'SUBTASK_INTERVAL', 'SUBTASK_VOTE_WINDOW', 'SUBTASK_VOTE_THRESHOLD'):
        env.pop(name, None)
    result = subprocess.run(['/bin/bash', str(nebula / 'submit_pi05_eval.sh')],
                            cwd=repo, env=env, capture_output=True, text=True)
    calls = sdk_calls.read_text().splitlines()
    if scenario == 'dryrun':
        assert result.returncode == 0, result.stderr
        assert calls == ['preflight']
        assert not job_calls.exists() and not receipts.exists()
    elif scenario == 'reserve_failure':
        assert result.returncode != 0
        assert calls == ['preflight', 'reserve']
        assert not job_calls.exists()
        assert (receipts / 'fixture-run.claim.json').is_file()
    else:
        assert result.returncode == (1 if scenario == 'sync_failure' else 0), result.stderr
        assert calls == ['preflight', 'reserve', 'sync']
        assert job_calls.read_text() == 'invoked'
        receipt = json.loads((receipts / 'fixture-run.json').read_text())
        assert receipt['checkpoint_source'] == MOUNT + '/checkpoint'
        assert receipt['checkpoint_manifest_sha256'] == 'fixture-manifest-digest'
        assert receipt['job_identifiers'] == ['654321']
        assert receipt['eval_workers'] == 4
