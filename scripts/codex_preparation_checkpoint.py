"""Durable, non-pickle preparation state and exact qualification reuse.

Only the named simulator dump and its native grasp counters are saved. PhysX
solver internals not exposed by that dump are not claimed to be serialized.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

CONTRACT = 'codex_prepared_full_scene_v1'
CACHE_FIELDS = ('history_identity_grasps', 'history_recovery_protocol', 'robot_reference_anchors')
MAX_BYTES = 2 * 1024**3
TRANSPORT_CHANGES = {'scripts/codex_collection_teacher.py',
    'scripts/collect_b1k_codex_subtask.py', 'sco/run_codex_subtask_collection.py'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _pack(value, arrays):
    if type(value) is float and not math.isfinite(value):
        return ['float', value.hex()]
    if value is None or type(value) in (str, int, float, bool):
        return ['value', value]
    if isinstance(value, dict):
        return ['dict', [[_pack(k, arrays), _pack(v, arrays)] for k, v in value.items()]]
    if isinstance(value, (list, tuple)):
        return ['tuple' if isinstance(value, tuple) else 'list', [_pack(v, arrays) for v in value]]
    kind = 'numpy'
    if type(value).__module__.split('.')[0] == 'torch':
        import torch
        if not isinstance(value, torch.Tensor):
            raise TypeError('Only dense tensors may enter a preparation checkpoint')
        value = value.detach().cpu().numpy()
        kind = 'torch'
    elif isinstance(value, np.generic):
        value = np.asarray(value)
        kind = 'numpy_scalar'
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError('Object and structured arrays are not checkpoint data')
        name = f'arrays/{len(arrays):08d}.npy'
        arrays[name] = np.array(value, copy=True, order='C')
        return [kind, name]
    raise TypeError('Unsupported preparation value: ' + type(value).__name__)


def _unpack(tree, arrays):
    kind, value = tree
    if kind == 'float':
        return float.fromhex(value)
    if kind == 'value':
        if value is not None and type(value) not in (str, int, float, bool):
            raise ValueError('Invalid checkpoint primitive')
        return value
    if kind == 'dict':
        pairs = [(_unpack(k, arrays), _unpack(v, arrays)) for k, v in value]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError('Duplicate checkpoint dictionary key')
        return result
    if kind in ('list', 'tuple'):
        result = [_unpack(v, arrays) for v in value]
        return tuple(result) if kind == 'tuple' else result
    if kind in ('numpy', 'numpy_scalar', 'torch'):
        array = arrays[value].copy()
        if kind == 'numpy_scalar':
            return array[()]
        if kind == 'torch':
            import torch
            return torch.from_numpy(array)
        return array
    raise ValueError('Unknown checkpoint value type')


def _content_hash(tree, arrays):
    h = hashlib.sha256(json.dumps(tree, sort_keys=True, allow_nan=False, separators=(',', ':')).encode())
    for name, array in arrays.items():
        h.update(name.encode())
        h.update(array.dtype.str.encode())
        h.update(json.dumps(array.shape).encode())
        h.update(array.tobytes(order='C'))
    return h.hexdigest()


def save_preparation(directory, plan, states, cache, *, source_reuse=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if plan['sample']['sample_id'] not in states:
        raise ValueError('Preparation lacks the requested starting state')
    payload = dict(states=states, cache={k: cache[k] for k in CACHE_FIELDS if k in cache})
    arrays = {}
    tree = _pack(payload, arrays)
    expected = _content_hash(tree, arrays)
    archive = directory / 'full-scene-state.npz'
    with archive.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
            output.writestr('tree.json', json.dumps(tree, allow_nan=False, separators=(',', ':')))
            for name, array in arrays.items():
                with output.open(name, 'w') as destination:
                    np.save(destination, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    if archive.stat().st_size > MAX_BYTES:
        raise ValueError('Preparation archive exceeds its storage budget')
    manifest = dict(contract=CONTRACT, plan_fingerprint=plan['fingerprint'],
        checkpoint_identity=plan['checkpoint_identity'], policy_seed=plan['policy_seed'],
        sample_id=plan['sample']['sample_id'], raw_sha256=cache['sha256'],
        archive=archive.name, archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
        content_sha256=expected, full_named_scene_state=True, native_grasp_counters_saved=True,
        states=list(states), source_qualification_reuse=source_reuse,
        runtime_signature=cache.get('preparation_runtime_signature', {}),
        physics_solver_internal_caches_serialized=False)
    with (directory / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    # READY is written only after reading and validating the actual AFS bytes.
    loaded, restored_cache, evidence = load_preparation(directory, plan, cache['sha256'],
        runtime_signature=cache.get('preparation_runtime_signature', {}), require_ready=False)
    with (directory / 'READY.json').open('x') as stream:
        json.dump(evidence, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return loaded, restored_cache, evidence


def load_preparation(directory, plan, raw_sha256, *, runtime_signature=None, require_ready=True):
    directory = Path(directory)
    if require_ready and not (directory / 'READY.json').is_file():
        raise ValueError('Preparation has not passed durable readback')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['contract'] != CONTRACT or manifest['plan_fingerprint'] != plan['fingerprint']
            or manifest['checkpoint_identity'] != plan['checkpoint_identity']
            or manifest['policy_seed'] != plan['policy_seed']
            or manifest['sample_id'] != plan['sample']['sample_id']
            or manifest['raw_sha256'] != raw_sha256
            or manifest.get('runtime_signature', {}) != (runtime_signature or {})
            or manifest['archive'] != 'full-scene-state.npz'):
        raise ValueError('Preparation belongs to another frozen experiment')
    archive = directory / manifest['archive']
    if (archive.is_symlink() or archive.stat().st_size != manifest['archive_bytes']
            or not 0 < archive.stat().st_size <= MAX_BYTES or sha(archive) != manifest['archive_sha256']):
        raise ValueError('Preparation bytes changed')
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        if (len(infos) > 200000 or sum(x.file_size for x in infos) > MAX_BYTES
                or len({x.filename for x in infos}) != len(infos)
                or any(x.filename != 'tree.json' and not (
                    x.filename.startswith('arrays/') and x.filename.endswith('.npy')
                    and len(Path(x.filename).parts) == 2) for x in infos)):
            raise ValueError('Invalid preparation archive inventory')
        tree = json.loads(source.read('tree.json'))
        arrays = {x.filename: np.load(io.BytesIO(source.read(x)), allow_pickle=False)
                  for x in infos if x.filename != 'tree.json'}
    if _content_hash(tree, arrays) != manifest['content_sha256']:
        raise ValueError('Preparation contents changed')
    payload = _unpack(tree, arrays)
    if set(payload) != {'states', 'cache'} or plan['sample']['sample_id'] not in payload['states']:
        raise ValueError('Invalid preparation root')
    evidence = dict(contract=CONTRACT, path=str(directory.resolve()),
        manifest_sha256=sha(directory / 'manifest.json'), archive_sha256=manifest['archive_sha256'],
        content_sha256=manifest['content_sha256'], readback_verified=True,
        plan_fingerprint=plan['fingerprint'], initialization_replay_is_training_data=False)
    if require_ready and json.loads((directory / 'READY.json').read_text()) != evidence:
        raise ValueError('Preparation READY receipt differs from its verified bytes')
    return payload['states'], payload['cache'], evidence


def reuse_qualification(source_root, destination, plan, code_root):
    """Copy an unchanged, already qualified baseline; never call an evaluator."""
    from codex_subtask_qualification import verify_qualification
    source_root, destination, code_root = map(Path, (source_root, destination, code_root))
    status = json.loads((source_root / 'status.json').read_text())
    if (status.get('state') != 'failed' or status.get('plan_fingerprint') != plan['fingerprint']
            or status.get('checkpoint_identity') != plan['checkpoint_identity']):
        raise ValueError('Qualification source is not the exact retired experiment')
    if json.loads((source_root / 'attempt/failed.json').read_text()).get('steps') != 0:
        raise ValueError('This recovery only resumes a failed zero-action attempt')
    unchanged = status['source_files_sha256']
    for name, digest in unchanged.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid frozen source path')
        if name in TRANSPORT_CHANGES:
            continue
        p = code_root / name
        if not p.is_file() or sha(p) != digest:
            raise ValueError('Qualification physics or policy source changed: ' + name)
    source = source_root / 'qualification'
    verified = verify_qualification(source / 'qualification.json', plan)
    qualification = json.loads((source / 'qualification.json').read_text())
    if qualification['baseline']['steps'] != plan['budgets']['max_steps']:
        raise ValueError('Cached baseline did not exhaust the same step budget')
    destination.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for file in sorted(source.rglob('*')):
        if file.is_symlink():
            raise ValueError('Qualification contains a symlink')
        if file.is_file():
            relative = file.relative_to(source)
            if len(inventory) >= 1000 or sum(x['bytes'] for x in inventory.values()) + file.stat().st_size > MAX_BYTES:
                raise ValueError('Qualification exceeds its copy budget')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with file.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst); dst.flush(); os.fsync(dst.fileno())
            digest = sha(file)
            if sha(target) != digest:
                raise ValueError('Qualification readback differs')
            inventory[str(relative)] = dict(bytes=file.stat().st_size, sha256=digest)
    proof = dict(contract='exact_failed_attempt_qualification_reuse_v1', source_root=str(source_root.resolve()),
        plan_fingerprint=plan['fingerprint'], source_qualification_sha256=verified['sha256'],
        calibration_controls_executed=0, baseline_controls_executed=0,
        source_baseline_steps=qualification['baseline']['steps'], source_files_verified=True,
        verified_source_files_sha256={k:v for k,v in unchanged.items() if k not in TRANSPORT_CHANGES},
        files=inventory)
    verify_qualification(destination / 'qualification.json', plan)
    with (destination / 'reuse.json').open('x') as stream:
        json.dump(proof, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return proof
