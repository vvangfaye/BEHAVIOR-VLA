#!/usr/bin/env python3
"""Read-only OSS range audit of every scheduled raw start; never a score gate.

Sentinel presence is a format capability signal, not proof of a correct grasp.
Only selected rows are inspected, and whole-file SHA256 is explicitly unverified.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import threading

import h5py
import numpy as np
from subtask_sample_panel import _raw_episode_structure

KIND = 'selected_raw_start_grasp_capability_audit_v1'
MAGIC = np.float32(100123456)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _etag(value):
    return str(value).strip('"')


class RangeBudget:
    def __init__(self, maximum=1024**3):
        self.maximum, self.claimed = maximum, 0
        self.lock = threading.Lock()

    def claim(self, size):
        with self.lock:
            if self.claimed + size > self.maximum:
                raise ValueError('Panel range budget exceeded')
            self.claimed += size


class FrozenRanges(io.RawIOBase):
    """Bounded seekable metadata reader plus exact dataset ranges, pinned by ETag."""
    def __init__(self, bucket, item, *, limit=16 * 1024**2, budget=None):
        super().__init__()
        self.bucket, self.item, self.limit = bucket, item, limit
        self.position = self.bytes_read = 0
        self.budget = budget or RangeBudget(limit)
        self.receipts, self.cache = [], OrderedDict()
        self._check_head()

    def _check_head(self):
        head = self.bucket.head_object(self.item['key'])
        if head.content_length != self.item['bytes'] or _etag(head.etag) != _etag(self.item['etag']):
            raise ValueError('Raw source changed against frozen inventory')

    def finish(self):
        self._check_head()
        return {'etag': self.item['etag'], 'object_bytes': self.item['bytes'],
                'downloaded_bytes': self.bytes_read, 'range_receipts': self.receipts,
                'whole_file_sha256_verified': False, 'source_reheaded_unchanged': True}

    def exact(self, offset, size):
        if not isinstance(offset, int) or not isinstance(size, int) or offset < 0 or size < 0 or offset + size > self.item['bytes']:
            raise ValueError('Out-of-object byte range')
        if size == 0:
            return b''
        if self.bytes_read + size > self.limit:
            raise ValueError('Per-episode range budget exceeded')
        self.budget.claim(size)
        response = self.bucket.get_object(self.item['key'], byte_range=(offset, offset + size - 1),
                                          headers={'If-Match': self.item['etag']})
        data = response.read(size + 1)
        if len(data) != size or _etag(response.headers.get('ETag', '')) != _etag(self.item['etag']):
            raise ValueError('Range length or ETag differs')
        self.bytes_read += size
        self.receipts.append({'offset': offset, 'bytes': size, 'sha256': digest(data)})
        return data

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        target = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.item['bytes'] + offset if whence == io.SEEK_END else -1
        if target < 0 or target > self.item['bytes']:
            raise ValueError('Invalid seek')
        self.position = target
        return target

    def readinto(self, buffer):
        size = min(len(buffer), self.item['bytes'] - self.position)
        output = bytearray()
        while len(output) < size:
            block = self.position // 65536
            if block not in self.cache:
                begin = block * 65536
                self.cache[block] = self.exact(begin, min(65536, self.item['bytes'] - begin))
                if len(self.cache) > 16:
                    self.cache.popitem(last=False)
            data = self.cache[block]
            self.cache.move_to_end(block)
            begin = self.position % 65536
            count = min(size - len(output), len(data) - begin)
            output.extend(data[begin:begin + count])
            self.position += count
        buffer[:size] = output
        return size


def _layout(dataset, *, rank, kind):
    if dataset.ndim != rank or dataset.dtype.kind != kind or dataset.chunks is not None or dataset.compression is not None:
        raise ValueError('Unsupported raw dataset layout; direct offsets are unsafe')
    offset = dataset.id.get_offset()
    if offset is None:
        raise ValueError('Dataset has no allocated contiguous storage')
    return {'shape': list(dataset.shape), 'dtype': dataset.dtype.str, 'offset': int(offset)}


def audit_episode(bucket, item, samples, *, budget=None):
    source = FrozenRanges(bucket, item, budget=budget)
    with h5py.File(source, 'r') as file:
        data = file['data']
        details, _, _, _ = _raw_episode_structure(file, Path(item['relative_path']), int(samples[0]['episode_length']))
        group_name = details['demo_group']
        group = data[group_name]
        config = json.loads(data.attrs['config'])
        scene = json.loads(data.attrs['scene_file'])
        state = _layout(group['state'], rank=2, kind='f')
        sizes = _layout(group['state_size'], rank=1, kind='i')
        action = _layout(group['action'], rank=2, kind='f')
        transitions = json.loads(group.attrs['transitions'])
    first = samples[0]
    n = int(first['episode_length'])
    if state['shape'][0] not in (n, n + 1) or sizes['shape'] != [state['shape'][0]] or action['shape'] != [n, 23]:
        raise ValueError('Raw shapes differ from frozen panel')
    if config.get('task', {}).get('activity_name') != first['task']:
        raise ValueError('Recorded task differs from frozen panel')
    size_dtype = np.dtype(sizes['dtype'])
    widths = np.frombuffer(source.exact(sizes['offset'], state['shape'][0] * size_dtype.itemsize), dtype=size_dtype)
    registry = scene['state']['registry']['object_registry']
    robots = []
    for robot in config.get('robots', []):
        row = registry.get(robot['name'], {})
        if not row:
            raise ValueError('Recorded named robot state missing')
        params = row.get('ag_obj_constraint_params', {})
        robots.append({'name': robot['name'], 'grasping_mode': robot.get('grasping_mode', 'physical'),
                       'legacy_controller_layout': 'controllers' in row and 'controller_groups' not in row,
                       'named_initial_grasp_payloads': {arm: bool(value) for arm, value in params.items()}})
    if not robots:
        raise ValueError('Recorded robot metadata missing')
    legacy_assisted = any(r['legacy_controller_layout'] and r['grasping_mode'] in ('assisted', 'sticky') for r in robots)
    rows = []
    state_dtype, act_dtype = np.dtype(state['dtype']), np.dtype(action['dtype'])
    row_bytes = state['shape'][1] * state_dtype.itemsize
    for sample in samples:
        start = int(sample['start'])
        if start < 0 or start >= n or not 0 < int(widths[start]) <= state['shape'][1]:
            raise ValueError('Selected raw state index or width invalid')
        raw = source.exact(state['offset'] + start * row_bytes, int(widths[start]) * state_dtype.itemsize)
        values = np.frombuffer(raw, dtype=state_dtype)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite selected raw state')
        count = min(32, n - start)
        actions = np.frombuffer(source.exact(action['offset'] + start * 23 * act_dtype.itemsize,
                                             count * 23 * act_dtype.itemsize), dtype=act_dtype).reshape(count, 23)
        if not np.isfinite(actions).all():
            raise ValueError('Nonfinite recorded actions')
        sentinel = bool(np.any(values == MAGIC))
        rows.append({'sample_id': sample['sample_id'], 'source_start': start, 'state_width': int(widths[start]),
                     'selected_state_sha256': digest(raw), 'ag_magic_present': sentinel,
                     'legacy_assisted_without_selected_row_grasp_signal': legacy_assisted and not sentinel,
                     'reconstruction_65_frame_history_available': start >= 64,
                     'reconstruction_32_closing_commands_both_arms': count == 32 and bool(np.all(actions[:, [14, 22]] < 1)),
                     'runtime_grasp_restoration_verified': False, 'metric_eligible': False})
    receipt = source.finish()
    infos = scene['objects_info']['init_info']
    return {'relative_path': item['relative_path'], 'task': first['task'], 'episode_index': first['episode_index'],
            'raw_episode_id': first['raw_episode_id'], 'task_instance_id': first['task_instance_id'],
            'status': 'audited', 'robots': robots, 'samples': rows, 'selected_starts': len(rows),
            'demo_group': group_name, 'state_layout': details['state_layout'],
            'recorded_omnigibson_git_hash': details['omnigibson_git_hash'],
            'recorded_objects': len(infos), 'recorded_fixed_base_objects': sum(bool(v['args'].get('fixed_base', False)) for v in infos.values()),
            'transition_entries': len(transitions), 'receipt': receipt,
            'exact_legacy_grasp_restoration_verified': False}


def audit_panel(bucket, panel, inventory, output, *, workers=4):
    if workers not in range(1, 5) or len({s['sample_id'] for s in panel['samples']}) != len(panel['samples']):
        raise ValueError('Worker count or sample identities invalid')
    if inventory['panel_fingerprint'] != panel['fingerprint']:
        raise ValueError('Raw inventory and panel differ')
    by_path = {}
    for sample in panel['samples']:
        path = f'task-{int(sample["task_index"]):04d}/episode_{int(sample["raw_episode_id"]):08d}.hdf5'
        by_path.setdefault(path, []).append(sample)
    items = {v['relative_path']: v for v in inventory['episodes']}
    if set(items) != set(by_path) or len(items) != len(inventory['episodes']):
        raise ValueError('Inventory does not exactly cover scheduled episodes')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Audit output must be fresh')
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / 'episodes'
    records_dir.mkdir(exist_ok=True)
    results = []
    budget = RangeBudget()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(audit_episode, bucket, items[path], samples, budget=budget): path for path, samples in by_path.items()}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # SDK error text may contain signed URLs; record only the type.
                result = {'relative_path': path, 'status': 'audit_failed', 'error_type': type(exc).__name__,
                          'scheduled_starts': len(by_path[path]), 'runtime_grasp_restoration_verified': False}
            results.append(result)
            (records_dir / (path.replace('/', '__') + '.json')).write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
            print(f'RAW_GRASP_AUDIT completed={len(results)}/{len(futures)} status={result["status"]} path={path}', flush=True)
    samples = [v for r in results for v in r.get('samples', [])]
    report = {'kind': KIND, 'panel_fingerprint': panel['fingerprint'],
              'panel_canonical_sha256': digest(json.dumps(panel, sort_keys=True, separators=(',', ':')).encode()),
              'inventory_canonical_sha256': digest(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()),
              'scheduled_tasks': len({s['task'] for s in panel['samples']}),
              'scheduled_episodes': len(items), 'scheduled_starts': len(panel['samples']),
              'panel_predicate_exclusions': len(panel['exclusions']),
              'audited_episodes': sum(r['status'] == 'audited' for r in results), 'audited_starts': len(samples),
              'failed_episodes': sum(r['status'] != 'audited' for r in results),
              'selected_rows_with_ag_magic': sum(s['ag_magic_present'] for s in samples),
              'legacy_assisted_rows_without_grasp_signal': sum(s['legacy_assisted_without_selected_row_grasp_signal'] for s in samples),
              'starts_without_65_frame_history': sum(not s['reconstruction_65_frame_history_available'] for s in samples),
              'starts_without_32_closing_commands_both_arms': sum(not s['reconstruction_32_closing_commands_both_arms'] for s in samples),
              'current_two_arm_reconstruction_preconditions_met': sum(s['reconstruction_65_frame_history_available'] and s['reconstruction_32_closing_commands_both_arms'] for s in samples),
              'downloaded_bytes': sum(r.get('receipt', {}).get('downloaded_bytes', 0) for r in results),
              'requested_byte_budget': budget.maximum, 'requested_bytes': budget.claimed,
              'complete_selected_row_audit': len(samples) == len(panel['samples']) and all(r['status'] == 'audited' for r in results),
              'model_action_requests': 0, 'runtime_grasp_restoration_verified': False, 'metric_eligible': False,
              'limitations': ['Only scheduled rows, not every historical frame, were inspected.',
                              'Sentinel presence is not proof of correct robot grasp decoding.',
                              'Empty named initial grasps do not prove later raw rows had free hands.',
                              'Reconstruction preconditions do not prove coupling, contact or physical retention.',
                              'Range hashes and pinned ETags do not verify a whole-file SHA256.'],
              'episodes': sorted(results, key=lambda v: v['relative_path'])}
    report['task_coverage'] = []
    for task in sorted({s['task'] for s in panel['samples']}):
        scheduled = [s for s in panel['samples'] if s['task'] == task]
        episode_ids = {s['episode_index'] for s in scheduled}
        audited = [r for r in results if r.get('task') == task and r['status'] == 'audited']
        audited_rows = [s for r in audited for s in r['samples']]
        report['task_coverage'].append({'task': task, 'scheduled_episodes': len(episode_ids),
                                      'audited_episodes': len(audited), 'scheduled_starts': len(scheduled),
                                      'audited_starts': len(audited_rows),
                                      'selected_rows_with_ag_magic': sum(s['ag_magic_present'] for s in audited_rows)})
    (output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    return report


def main():
    import oss2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bucket', default='traffic-model')
    parser.add_argument('--endpoint', default='https://oss-cn-shanghai.aliyuncs.com')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    token = os.environ.get('OSS_SESSION_TOKEN')
    auth = oss2.StsAuth(os.environ['OSS_ACCESS_KEY_ID'], os.environ['OSS_ACCESS_KEY_SECRET'], token) if token else oss2.Auth(os.environ['OSS_ACCESS_KEY_ID'], os.environ['OSS_ACCESS_KEY_SECRET'])
    bucket = oss2.Bucket(auth, args.endpoint, args.bucket)
    report = audit_panel(bucket, json.loads(args.panel.read_text()), json.loads(args.inventory.read_text()), args.output, workers=args.workers)
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}, sort_keys=True))
    return 0 if report['complete_selected_row_audit'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
