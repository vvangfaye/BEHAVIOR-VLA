"""Finish a diagnosed CPU preparation without re-downloading verified files."""


def finish_existing(p):
    import hashlib
    import json
    import os
    from pathlib import Path
    import urllib.request

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda: f.read(8 * 1024**2), b''):
                h.update(b)
        return h.hexdigest()

    def encoded(value):
        return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()

    checkpoint, code, runtime, oldcode = (Path(p[k]) for k in ('checkpoint', 'code', 'runtime', 'oldcode'))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def publish(value):
        (checkpoint / 'prepare_status.json').write_bytes(encoded(value))
        print('STATUS_JSON=' + json.dumps({k: v for k, v in value.items()
              if k not in ('source_manifest', 'checkpoint_ready', 'scope')}), flush=True)
        request = urllib.request.Request(p['report_put'], data=encoded(value), method='PUT')
        with opener.open(request, timeout=60) as response:
            response.read()

    stage = 'preflight'
    try:
        assert not (checkpoint / '_READY').exists(), 'Existing ready checkpoint must be investigated'
        assert checkpoint.is_dir() and code.is_dir()
        assert sha(oldcode / '_PUBLIC_TEST_SOURCE_READY') == p['old_source_sha256']
        old = json.loads((oldcode / '_PUBLIC_TEST_SOURCE_READY').read_bytes())
        assert sha(oldcode / '_CODE_SNAPSHOT_READY') == old['base_source_sha256']
        base = json.loads((oldcode / '_CODE_SNAPSHOT_READY').read_bytes())
        original_files = dict(base['overlay_files_sha256'])
        original_files.update(old['files_sha256'])
        for relative, expected in original_files.items():
            assert sha(oldcode / relative) == expected, 'Base source changed'
        assert sha(runtime / 'graphics-runtime/_READY') == base['graphics_runtime_manifest_sha256']
        assert (runtime / '_RUNTIME_READY').is_file()
        stage = 'checkpoint_readback'
        rows = []
        for item in p['items']:
            relative = Path(item['relative_path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            path = checkpoint / relative
            assert not path.is_symlink() and checkpoint.resolve() in path.resolve().parents
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
            rows.append(dict(relative_path=str(relative), bytes=item['bytes'], sha256=item['sha256'],
                             full_local_readback=True))
            publish(dict(state='preparing', stage=stage, checkpoint_step=2000000,
                         run_id=p['run_id'], verified_files=len(rows), total_files=24))
        assert len(rows) == 24 and sum(r['bytes'] for r in rows) == p['total_bytes']
        assert json.loads((checkpoint / 'assets/subtask_conditioning.json').read_bytes())['action_conditioning'] == 'task_and_subtask'
        stage = 'source_freeze'
        files, symlinks, runtime_links = {}, {}, {}
        for path in sorted(code.rglob('*')):
            relative = path.relative_to(code).as_posix()
            if relative == '_PUBLIC_TEST_SOURCE_READY' or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_symlink():
                symlinks[relative] = os.readlink(path)
                if not code.resolve() in path.resolve().parents:
                    # This is the one intended shared, already prepared environment.
                    assert relative == 'openpi/.venv'
                    assert path.resolve() == (runtime / 'envs/b1k-pi05').resolve()
                    assert path.is_dir()
                    runtime_links[relative] = dict(target=str(path.resolve()),
                        runtime_ready_sha256=sha(runtime / '_RUNTIME_READY'))
                    continue
            if path.is_file():
                files[relative] = sha(path)
                if relative in p['overlay_files_sha256']:
                    assert files[relative] == p['overlay_files_sha256'][relative]
                else:
                    original = oldcode / relative
                    assert original.is_file() and sha(original) == files[relative], 'Source copy differs from frozen base'
        assert set(p['overlay_files_sha256']) <= set(files)
        assert all(f in files for f in ('scripts/eval_b1k.sh', 'scripts/pi05_eval_queue.py', 'sco/video_eval_worker.py'))
        library = old['supplemental_library']
        libroot = Path(library['root'])
        assert sha(libroot / '_READY') == library['ready_manifest_sha256']
        libready = json.loads((libroot / '_READY').read_bytes())
        for relative, expected in libready['libraries_sha256'].items():
            assert sha(Path(libready['library_path']) / relative) == expected
        provenance = dict(checkpoint_step=2000000, run_id=p['run_id'],
            base_source_sha256=old['base_source_sha256'], previous_source_manifest_sha256=p['old_source_sha256'],
            files_sha256=files, symlinks=symlinks, shared_runtime_links=runtime_links,
            checkpoint_files=rows, supplemental_library=library, overlay_sha256=p['overlay_sha256'],
            code=str(code), runtime=str(runtime), scope=p['scope'],
            source_policy='independent frozen source; explicit validated shared runtime environment',
            preparation_recovery=p['preparation_recovery'])
        source_bytes = encoded(provenance)
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        (code / '_PUBLIC_TEST_SOURCE_READY').write_bytes(source_bytes)
        ready = dict(checkpoint_step=2000000, files=24, total_bytes=p['total_bytes'],
            full_readback_sha256=True, source_manifest_sha256=source_sha, run_id=p['run_id'])
        ready_bytes = encoded(ready)
        (checkpoint / '_READY').write_bytes(ready_bytes)
        publish(dict(state='ready', checkpoint_step=2000000, run_id=p['run_id'], files=24,
            checkpoint=str(checkpoint), code=str(code), runtime=str(runtime), scope=p['scope'],
            source_manifest=provenance, checkpoint_ready=ready, source_manifest_sha256=source_sha,
            checkpoint_ready_sha256=hashlib.sha256(ready_bytes).hexdigest(), worker_sha256=files['sco/video_eval_worker.py'],
            checkpoint_full_sha256_readback=True, plan_sha256=p['plan_sha256']))
    except BaseException as error:
        publish(dict(state='failed', stage=stage, checkpoint_step=2000000, run_id=p['run_id'],
                     error_type=type(error).__name__))
        raise RuntimeError('CPU preparation completion failed at ' + stage) from None
