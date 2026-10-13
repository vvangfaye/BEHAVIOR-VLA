"""Finite policy-image RPC and memory probe, separate from scored rollouts.

Run with the verified image's Python inside a single-GPU container. This emits
raw evidence, not IMAGE_GPU_VERIFIED.json. In particular, measuring a 32-GiB GPU
does not by itself establish successful execution on a physical 24-GiB GPU.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import subprocess
import threading
import time


IMAGE_PATHS = (
    '/mnt/afs/260010165/behavior_1k/code_snapshots/b1k-video2m-20261013',
    '/mnt/afs/260010165/behavior_1k/runtime_20261007_1306',
    '/mnt/afs/260010165/behavior_1k/checkpoints/b1k-video2m-20261013',
    '/opt/behavior',
)


def assert_unshadowed_image_paths(mountinfo):
    """Reject host volumes that would replace any part of the image under test."""
    protected = [PurePosixPath(p) for p in IMAGE_PATHS]
    root_seen = False
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 10 or '-' not in fields:
            raise ValueError('Cannot establish container mount provenance')
        decoded = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), fields[4])
        mount = PurePosixPath(decoded)
        if not mount.is_absolute() or '..' in mount.parts:
            raise ValueError('Invalid mount path')
        if str(mount) == '/':
            root_seen = True
            continue
        if any(mount == path or mount in path.parents or path in mount.parents for path in protected):
            raise ValueError('Mounted volume shadows policy image: ' + str(mount))
    if not root_seen:
        raise ValueError('Container root mount was not observed')
    return True


def gpu_snapshot():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != 1:
        raise ValueError('Probe requires exactly one visible GPU')
    index, uuid, name, total, used = [x.strip() for x in rows[0]]
    return dict(index=int(index), uuid=uuid, name=name,
                total_mib=int(total), used_mib=int(used))


def assert_action(result, action_dim, np):
    action = np.asarray(result['action'])
    if action.shape not in ((action_dim,), (1, action_dim)) or not np.isfinite(action).all():
        raise ValueError('Policy returned invalid action shape or nonfinite values')
    return hashlib.sha256(action.tobytes()).hexdigest()


def run(args):
    mountinfo = Path('/proc/self/mountinfo').read_text()
    assert_unshadowed_image_paths(mountinfo)
    if not Path('/opt/behavior/serve.sh').is_file():
        raise ValueError('Use the actual policy image, not the pre-existing AFS environment')
    import numpy as np
    import openpi
    from openpi.configs.robots import ROBOT_REGISTRY
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect

    openpi_path = Path(openpi.__file__).resolve()
    expected_code = Path(IMAGE_PATHS[0]).resolve()
    if expected_code not in openpi_path.parents:
        raise ValueError('Policy import resolved outside the image code tree')
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    first = gpu_snapshot()
    if first['used_mib'] > 1024:
        raise ValueError('GPU has pre-existing allocations; cannot attribute memory use')
    report = dict(kind='policy_image_gpu_probe_v1', state='running',
                  image_reference=args.image_reference, image_id=args.image_id,
                  source_manifest_sha256=args.source_manifest_sha256,
                  checkpoint_ready_sha256=args.checkpoint_ready_sha256,
                  checkpoint_step=2000000, task=args.task,
                  image_paths_unshadowed=True, openpi_module_path=str(openpi_path),
                  mountinfo_sha256=hashlib.sha256(mountinfo.encode()).hexdigest(),
                  synthetic_inputs=True, simulator_started=False,
                  scored_episodes_created=0, gpu=first,
                  memory_scope='whole visible device, sampled every 0.2 seconds',
                  physical_24gib_or_smaller_device=first['total_mib'] <= 24576,
                  samples=[], rpc_results=[])

    budget = None
    if args.memory_budget_mib is not None:
        from video_eval_gpu_memory_budget import DeviceMemoryBudget
        budget = DeviceMemoryBudget(args.memory_budget_mib)
        try:
            report['memory_budget'] = budget.enter()
        except Exception:
            budget.close()
            raise

    def save():
        temp = destination / '.probe.json.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(destination / 'probe.json')

    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                snap = gpu_snapshot()
                report['samples'].append(dict(epoch=time.time(), used_mib=snap['used_mib']))
            except Exception as error:
                report.setdefault('memory_sampling_errors', []).append(type(error).__name__)
            stop.wait(.2)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    environment = os.environ.copy()
    environment.update(TASK=args.task, PORT=str(args.port))
    server = None
    save()
    (destination / 'mountinfo.txt').write_text(mountinfo)
    try:
        with (destination / 'server.log').open('w') as log:
            server = subprocess.Popen(['/opt/behavior/serve.sh'], env=environment,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report['server_pid'] = server.pid
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Policy image server exited before accepting RPC')
                try:
                    with socket.create_connection(('127.0.0.1', args.port), timeout=2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Policy image initialization timeout')
                    time.sleep(2)
            robot = ROBOT_REGISTRY['b1k/R1Pro']
            state_dim = 1 + max(i for field in robot.proprio for i in field.indices)
            rng = np.random.default_rng(2000000)
            packer = msgpack_numpy.Packer()
            with connect('ws://127.0.0.1:' + str(args.port), compression=None,
                         max_size=None, open_timeout=20, ping_interval=None) as connection:
                metadata = msgpack_numpy.unpackb(connection.recv(timeout=60))
                report['server_metadata'] = metadata
                if metadata.get('action_horizon') != 16:
                    raise ValueError('Server action horizon differs from scored production')
                for index in range(args.requests):
                    # Reset ensures every request actually exercises model inference,
                    # instead of consuming the cached 16-step action chunk.
                    connection.send(packer.pack({'reset': True}))
                    observation = {robot.name + '::proprio': np.zeros(state_dim, dtype=np.float32)}
                    for camera in robot.observations.values():
                        size = (240, 224, 480)[index % 3]
                        observation[camera.obs_key] = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
                    started = time.monotonic()
                    connection.send(packer.pack(observation))
                    result = msgpack_numpy.unpackb(connection.recv(timeout=900))
                    digest = assert_action(result, robot.action_dim, np)
                    report['rpc_results'].append(dict(index=index,
                        elapsed_seconds=time.monotonic() - started, action_sha256=digest,
                        generated_subtask=bool(result.get('subtasks'))))
                    save()
                connection.send(packer.pack({'reset': True}))
            report.update(state='completed', policy_rpc_verified=True,
                          successful_model_requests=len(report['rpc_results']))
    except Exception as error:
        report.update(state='failed', error_type=type(error).__name__, error=str(error)[:300])
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        stop.set()
        monitor.join(timeout=12)
        report['peak_observed_device_memory_mib'] = max(
            [first['used_mib']] + [x['used_mib'] for x in report['samples']])
        report['measured_peak_below_24gib'] = report['peak_observed_device_memory_mib'] <= 24576
        if budget is not None:
            report['peak_memory_excluding_reservation_mib'] = report['peak_observed_device_memory_mib'] - budget.held_bytes / 1024**2
            report['memory_budget']['held_through_all_model_requests'] = True
            budget.close()
        report['finished_epoch'] = time.time()
        report['verification_requires_source_and_platform_crosscheck'] = True
        save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--image-reference', required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--checkpoint-ready-sha256', required=True)
    parser.add_argument('--task', default='turning_on_radio')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--memory-budget-mib', type=int)
    arguments = parser.parse_args()
    if arguments.requests < 32:
        parser.error('At least 32 actual model requests are required')
    run(arguments)
