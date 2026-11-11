"""Read only selected existing videos and stream fixed-anchor RGB to a memmap.

No credentials or signed URLs are stored. Decoded intermediate video frames
are discarded; only the precommitted anchor images enter the training cache.
"""
import hashlib
import json
import io
import argparse
from pathlib import Path
import sys
import time

import av
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'analysis/training_boundary_audit_20261019'),
               str(ROOT / 'BEHAVIOR-VLAs/rlinf_adapter')]
from prepare_probe_cache import ObjectReader, TransferBudget, open_source_bucket, sha_file
from b1k_rlinf.io import resize_image

CAMERAS = ('observation.rgb.zed_link_camera_0', 'observation.rgb.left_realsense_link_camera_0',
           'observation.rgb.right_realsense_link_camera_0')


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution-plan', type=Path)
    args = parser.parse_args()
    output = HERE / 'vla_pilot'
    plan_path = output / 'data_plan.json'
    plan = json.loads(plan_path.read_text())
    execution_path = args.execution_plan or output / 'image_execution_plan.json'
    execution = json.loads(execution_path.read_text())
    assert sha_file(plan_path) == execution['data_plan_sha256']
    assert sha_file(Path(__file__)) == execution['script_sha256']
    count = len(plan['samples'])
    progress_path = output / 'image_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        'data_plan_sha256': sha_file(plan_path), 'completed_video_objects': [],
        'transfer_bytes_reserved': 0, 'transfer_requests': 0, 'status': 'running'}
    assert progress['data_plan_sha256'] == sha_file(plan_path)
    image_path = output / 'rgb.npy'
    shape = (count, 3, 224, 224, 3)
    if image_path.exists():
        images = np.load(image_path, mmap_mode='r+')
        assert images.shape == shape and images.dtype == np.uint8
    else:
        assert not progress['completed_video_objects']
        images = np.lib.format.open_memmap(image_path, mode='w+', dtype=np.uint8, shape=shape)
    original_requests = progress['transfer_requests']
    class RecordedBudget(TransferBudget):
        def reserve(self, amount):
            super().reserve(amount)
            progress['transfer_bytes_reserved'] = self.used_bytes
            progress['transfer_requests'] = original_requests + self.requests
            atomic(progress_path, progress)
    budget = RecordedBudget(execution['maximum_bytes'], execution['maximum_seconds'],
                            used_bytes=progress['transfer_bytes_reserved'])
    bucket = open_source_bucket(Path.home() / '.ossutilconfig', plan['source_bucket'])
    objects = {r['relative_path']: r for r in plan['objects']}
    requests = {}
    for index, sample in enumerate(plan['samples']):
        for channel, camera in enumerate(CAMERAS):
            video = sample['videos'][camera]
            timestamp = video['from_timestamp'] + sample['frame_index'] / plan['fps']
            requests.setdefault(video['relative_path'], []).append((timestamp, index, channel, video['shape']))
    start = time.monotonic()
    try:
        for relative, wanted in sorted(requests.items()):
            if relative in progress['completed_video_objects']:
                continue
            wanted.sort()
            next_index = 0
            tolerance = plan['image_tolerance_seconds']
            reader = io.BufferedReader(ObjectReader(bucket, objects[relative], budget),
                                       buffer_size=execution['reader_buffer_bytes'])
            with reader, av.open(reader, mode='r') as container:
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'
                stream.thread_count = execution['decoder_threads']
                container.seek(int(wanted[0][0] * av.time_base), backward=True)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    current = float(frame.pts * stream.time_base)
                    resized = None
                    while next_index < len(wanted) and wanted[next_index][0] <= current + tolerance:
                        query, index, channel, expected_shape = wanted[next_index]
                        if abs(query - current) >= tolerance:
                            raise ValueError(f'No matching PTS for anchor {index}: requested={query}, found={current}')
                        if resized is None:
                            raw = frame.to_ndarray(format='rgb24')
                            assert list(raw.shape) == expected_shape and raw.dtype == np.uint8
                            resized = resize_image(raw)
                        images[index, channel] = resized
                        next_index += 1
                    if next_index == len(wanted):
                        break
            if next_index != len(wanted):
                raise ValueError(f'Missing {len(wanted) - next_index} fixed frames')
            images.flush()
            progress['completed_video_objects'].append(relative)
            progress['completed_images'] = sum(len(requests[k]) for k in progress['completed_video_objects'])
            progress['elapsed_seconds_this_attempt'] = time.monotonic() - start
            atomic(progress_path, progress)
            print(json.dumps({'completed_objects': len(progress['completed_video_objects']),
                              'total_objects': len(requests), 'completed_images': progress['completed_images'],
                              'transfer_bytes_reserved': budget.used_bytes,
                              'seconds': round(time.monotonic() - start, 1)}), flush=True)
        assert progress['completed_images'] == count * 3
        receipt = {'complete': True, 'data_plan_sha256': sha_file(plan_path),
                   'image_file': image_path.name, 'image_sha256': sha_file(image_path),
                   'image_bytes': image_path.stat().st_size, 'shape': list(shape),
                   'camera_order': CAMERAS, 'source_video_objects': len(requests),
                   'transfer_bytes_reserved': budget.used_bytes,
                   'scope': 'Existing fixed expert observations only; no new demonstrations, labels or model inference'}
        atomic(output / 'image_receipt.json', receipt)
        progress['status'] = 'complete'
        atomic(progress_path, progress)
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        progress['status'] = 'failed'
        progress['error_type'] = type(error).__name__
        atomic(progress_path, progress)
        raise


if __name__ == '__main__':
    main()
