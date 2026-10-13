"""CPU-only export of the exact validated policy environment and frozen source."""


def run(payload):
    import hashlib
    import io
    import json
    import os
    from pathlib import Path
    import shutil
    import tarfile
    import time
    import urllib.request

    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    runtime, code, destination = (Path(payload[k]) for k in ('runtime', 'code', 'destination'))
    assert destination.parent == Path('/mnt/afs/260010165/behavior_1k/serving_bundles')
    assert payload['checkpoint_step'] == 2000000
    destination.mkdir(parents=True, exist_ok=False)

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(block)
        return h.hexdigest()

    def encoded(value):
        return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()

    def publish(value):
        value.update(checkpoint_step=2000000, run_id=payload['run_id'], observed_at=time.time())
        raw = encoded(value)
        temporary = destination / 'STATUS.tmp'
        temporary.write_bytes(raw)
        os.replace(temporary, destination / 'STATUS.json')
        request = urllib.request.Request(payload['report_put'], data=raw, method='PUT')
        with opener.open(request, timeout=60) as response:
            response.read()
        print('STATUS_JSON=' + json.dumps({k: v for k, v in value.items() if k not in ('parts',)}), flush=True)

    stage = 'preflight'
    try:
        assert digest(code / '_PUBLIC_TEST_SOURCE_READY') == payload['source_manifest_sha256']
        source = json.loads((code / '_PUBLIC_TEST_SOURCE_READY').read_bytes())
        assert source['checkpoint_step'] == 2000000
        assert digest(runtime / '_RUNTIME_READY') == payload['runtime_ready_sha256']
        tokenizer = runtime / 'openpi-cache/big_vision/paligemma_tokenizer.model'
        assert digest(tokenizer) == payload['tokenizer_sha256']
        environment = runtime / 'envs/b1k-pi05'
        assert environment.is_dir() and (environment / 'bin/python').resolve().is_file()
        assert shutil.disk_usage(destination).free >= payload['required_free_bytes']
        publish(dict(state='exporting', stage=stage, gpu_count=0, no_model_rollout=True))
        rows = []
        total_bytes = 0

        class HashingReader:
            def __init__(self, stream):
                self.stream, self.hash = stream, hashlib.sha256()

            def read(self, count=-1):
                value = self.stream.read(count)
                self.hash.update(value)
                return value

        archive = destination / 'serving-context.tar.gz'
        stage = 'archive'
        with tarfile.open(archive, 'w:gz', compresslevel=1, dereference=False) as output:
            def add(path, relative, expected=None):
                nonlocal total_bytes
                assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
                before = path.lstat()
                info = output.gettarinfo(str(path), arcname=relative)
                # Conda may hardlink two installed paths; preserve both file bytes.
                if info.islnk():
                    info.type, info.size, info.linkname = tarfile.REGTYPE, before.st_size, ''
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                if info.isfile():
                    with path.open('rb') as stream:
                        reader = HashingReader(stream)
                        output.addfile(info, reader)
                    sha = reader.hash.hexdigest()
                    after = path.stat()
                    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                    if expected:
                        assert sha == expected, 'Frozen policy source differs: ' + relative
                    rows.append(dict(path=relative, bytes=before.st_size, sha256=sha, mode=info.mode))
                    total_bytes += before.st_size
                elif info.issym():
                    target = path.resolve()
                    allowed = (environment.resolve(), code.resolve(), Path('/lib'), Path('/lib64'),
                               Path('/usr/lib'), Path('/usr/local/cuda'))
                    assert any(target == root or root in target.parents for root in allowed), 'External environment symlink: ' + relative
                    output.addfile(info)
                    rows.append(dict(path=relative, symlink=os.readlink(path), mode=info.mode))
                elif info.isdir():
                    output.addfile(info)
                else:
                    raise ValueError('Nonregular exported entry: ' + relative)
                if len(rows) and len(rows) % 2000 == 0:
                    publish(dict(state='exporting', stage=stage, file_count=len(rows), source_bytes=total_bytes,
                                 gpu_count=0, no_model_rollout=True))

            for directory, children, filenames in os.walk(environment, followlinks=False):
                children[:] = [name for name in children if name != '__pycache__']
                base = Path(directory)
                add(base, 'runtime/' + base.relative_to(runtime).as_posix())
                for name in sorted(children):
                    child = base / name
                    if child.is_symlink():
                        add(child, 'runtime/' + child.relative_to(runtime).as_posix())
                for name in sorted(filenames):
                    if name.endswith(('.pyc', '.pyo')):
                        continue
                    child = base / name
                    add(child, 'runtime/' + child.relative_to(runtime).as_posix())
            add(tokenizer, 'runtime/openpi-cache/big_vision/paligemma_tokenizer.model', payload['tokenizer_sha256'])
            add(runtime / '_RUNTIME_READY', 'runtime/_RUNTIME_READY', payload['runtime_ready_sha256'])
            for relative, expected in sorted(source['files_sha256'].items()):
                add(code / relative, 'code/' + relative, expected)
            for relative, target in source['symlinks'].items():
                assert os.readlink(code / relative) == target
                add(code / relative, 'code/' + relative)
            add(code / '_PUBLIC_TEST_SOURCE_READY', 'code/_PUBLIC_TEST_SOURCE_READY', payload['source_manifest_sha256'])
            manifest = dict(schema_version=1, checkpoint_step=2000000, run_id=payload['run_id'],
                source_manifest_sha256=payload['source_manifest_sha256'], runtime_ready_sha256=payload['runtime_ready_sha256'],
                tokenizer_sha256=payload['tokenizer_sha256'], files=rows, source_bytes=total_bytes,
                include_only_validated_pi05_environment_frozen_source_and_tokenizer=True,
                checkpoint_excluded_reuse_local_verified_bytes=True, gpu_count=0, no_model_rollout=True)
            raw = encoded(manifest)
            (destination / 'EXPORT_MANIFEST.json').write_bytes(raw)
            entry = tarfile.TarInfo('EXPORT_MANIFEST.json')
            entry.size, entry.mode = len(raw), 0o644
            output.addfile(entry, io.BytesIO(raw))
        archive_sha = digest(archive)
        size = archive.stat().st_size
        assert size <= payload['part_bytes'] * len(payload['part_puts'])
        parts = []
        stage = 'upload'
        with archive.open('rb') as stream:
            while True:
                block = stream.read(payload['part_bytes'])
                if not block:
                    break
                index = len(parts)
                request = urllib.request.Request(payload['part_puts'][index], data=block,
                    headers={'Content-Type': 'application/octet-stream'}, method='PUT')
                with opener.open(request, timeout=600) as response:
                    response.read()
                parts.append(dict(index=index, bytes=len(block), sha256=hashlib.sha256(block).hexdigest()))
                publish(dict(state='exporting', stage=stage, uploaded_parts=len(parts), archive_bytes=size,
                             gpu_count=0, no_model_rollout=True))
        report = dict(state='exported', gpu_count=0, no_model_rollout=True, archive_bytes=size,
            archive_sha256=archive_sha, manifest_sha256=hashlib.sha256(raw).hexdigest(), parts=parts,
            file_count=len(rows), source_bytes=total_bytes, source_manifest_sha256=payload['source_manifest_sha256'],
            runtime_ready_sha256=payload['runtime_ready_sha256'], tokenizer_sha256=payload['tokenizer_sha256'],
            checkpoint_not_downloaded=True, upload_complete=True)
        publish(report)
        return report
    except BaseException as error:
        publish(dict(state='failed', stage=stage, error_type=type(error).__name__, gpu_count=0, no_model_rollout=True))
        raise RuntimeError('Policy environment export failed at ' + stage) from None
