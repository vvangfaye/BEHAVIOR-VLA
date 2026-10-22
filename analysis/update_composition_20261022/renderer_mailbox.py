"""Bounded shared-AFS transport; model inference stays on the training GPU.

Only arrays and JSON are serialized. A request is published by atomic rename;
there is one in-flight request per renderer. This does not change simulation
steps, policy inputs, seeds or rewards.
"""
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

MAX_BYTES = 64 * 1024**2


def encode(value):
    arrays = {}
    def visit(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError('Object arrays cannot cross the renderer boundary')
            key = 'array_' + str(len(arrays))
            arrays[key] = item
            return {'__array__': key}
        if isinstance(item, np.generic):
            return visit(item.item())
        if isinstance(item, dict):
            if not all(isinstance(k, str) for k in item):
                raise TypeError('String dictionary keys required')
            return {k: visit(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [visit(v) for v in item]
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        raise TypeError('Unsupported renderer value: ' + type(item).__name__)
    header = json.dumps(visit(value), allow_nan=False).encode()
    arrays['header'] = np.frombuffer(header, dtype=np.uint8)
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    blob = output.getvalue()
    if len(blob) > MAX_BYTES:
        raise ValueError('Renderer message exceeds size limit')
    return blob


def decode(blob):
    if len(blob) > MAX_BYTES:
        raise ValueError('Renderer message exceeds size limit')
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        if sum(x.file_size for x in archive.infolist()) > MAX_BYTES:
            raise ValueError('Expanded renderer message exceeds size limit')
    with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
        header = json.loads(arrays['header'].tobytes())
        def visit(item):
            if isinstance(item, dict):
                if set(item) == {'__array__'}:
                    return arrays[item['__array__']].copy()
                return {k: visit(v) for k, v in item.items()}
            if isinstance(item, list):
                return [visit(v) for v in item]
            return item
        return visit(header)


def publish(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to overwrite a renderer message')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('xb') as output:
        output.write(encode(value))
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


class Client:
    def __init__(self, directory, session):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = session
        self.sequence = 0

    def request(self, command, timeout):
        number = self.sequence
        self.sequence += 1
        request = self.root / f'{number:08d}.request.npz'
        response = self.root / f'{number:08d}.response.npz'
        publish(request, dict(session=self.session, sequence=number, command=command))
        deadline = time.monotonic() + timeout
        while not response.exists():
            status_path = self.root / 'status.json'
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status.get('session') != self.session:
                    raise ValueError('Renderer session mismatch')
                if status.get('terminal'):
                    raise RuntimeError('Renderer exited before responding: ' + str(status))
            if time.monotonic() > deadline:
                raise TimeoutError('Renderer response deadline exceeded')
            time.sleep(.1)
        value = decode(response.read_bytes())
        if value.get('sequence') != number or value.get('session') != self.session:
            raise ValueError('Renderer response identity mismatch')
        # The response is durable before the server proceeds to its next request.
        request.unlink()
        response.unlink()
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value['result']


class RemoteSimulator:
    def __init__(self, client, plan, output, instance, mode, max_steps):
        self.client = client
        self.closed = False
        self.current = client.request(dict(op='open', output=str(output), instance=instance,
            mode=mode, max_steps=max_steps, task=plan['task'], seed=plan['seed']),
            timeout=plan['startup_timeout_seconds'] + 180)

    def step(self, actions, subtask):
        self.current = self.client.request(dict(op='step', actions=np.asarray(actions),
                                               subtask=subtask), timeout=240)
        return self.current

    def close(self):
        if not self.closed:
            self.closed = True
            self.client.request(dict(op='close'), timeout=120)
