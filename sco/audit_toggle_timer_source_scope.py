"""Inventory recorded toggle timers across every original modern H5, without simulation."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def decode_toggle_prefix(cache, name, frame):
    """Read only a first ToggledOn field using the recorded rigid-body layout.

    Later non-kinematic fields need not be decoded to locate the leading toggle.
    This inventory helper does not replace the scoring fixture validator.
    """
    from recorded_native_state import MAGIC, _recorded_object_topology, uuid
    scene = cache['scene_file']
    initial = scene['state']['registry']['object_registry'][name]
    info = scene['objects_info']['init_info'].get(name)
    infos, _ = _recorded_object_topology(cache, frame)
    root_keys = set(initial.get('root_link', {}))
    if (info is None or infos.get(name) != info or 'controllers' in initial or 'controller_groups' in initial
            or root_keys not in ({'pos', 'ori'}, {'pos', 'ori', 'lin_vel', 'ang_vel'})):
        raise ValueError('Unsupported recorded toggle-prefix rigid-body schema')
    non_kin = initial.get('non_kin', {})
    if not non_kin or next(iter(non_kin)) != 'ToggledOn' or set(non_kin['ToggledOn']) != {'value', 'hand_in_marker_steps'}:
        raise ValueError('ToggledOn is not the first recorded non-kinematic field')
    joints = initial.get('joint_pos', [])
    if len(joints) != len(initial.get('joint_vel', [])):
        raise ValueError('Inconsistent recorded toggle-prefix joint schema')
    offset = (8 if root_keys == {'pos', 'ori'} else 14) + 2 * len(joints)
    object_uuid = uuid(name)
    if sum(uuid(n) == object_uuid for n in infos) != 1:
        raise ValueError('Ambiguous recorded toggle-prefix UUID')
    indices = cache.setdefault('fixture_packet_candidate_frames', {})
    key = (cache['sha256'], name)
    if key not in indices:
        indices[key] = np.flatnonzero(np.any(cache['states'] == np.float32(object_uuid), axis=1))
    for prior in reversed(indices[key][indices[key] <= frame]):
        row = cache['states'][prior, :int(cache['state_sizes'][prior])]
        boundary = np.flatnonzero(row == np.float32(uuid('object_registry')))
        if len(boundary) != 1:
            raise ValueError('Ambiguous recorded object-registry boundary')
        begin = int(boundary[0]) + 2
        references = {int(p) + 2 for p in np.flatnonzero(row == MAGIC)}
        hits = [int(p) + begin for p in np.flatnonzero(row[begin:] == np.float32(object_uuid)) if int(p) + begin not in references]
        if not hits:
            continue
        if len(hits) != 1:
            raise ValueError('Ambiguous recorded toggle-prefix packet')
        start = hits[0] + 1
        packet = row[start:start + offset + 2]
        if (row.dtype != np.dtype('float32') or len(packet) != offset + 2 or not np.isfinite(packet).all()
                or packet[0] not in (0, 1) or packet[offset] not in (0, 1) or packet[offset + 1] < 0
                or abs(float(np.linalg.norm(packet[4:8])) - 1) > 1e-4):
            raise ValueError('Invalid recorded toggle-prefix numeric fields')
        return {'timer': float(packet[offset + 1]), 'value': bool(packet[offset]), 'last_written_frame': int(prior),
                'source_toggle_prefix_sha256': hashlib.sha256(row[start-1:start + offset + 2].tobytes()).hexdigest(),
                'prefix_only': True}
    # Sparse playback starts from the recorded scene. A field with no prior
    # object packet retains that explicit initial value, not an invented zero.
    initial_toggle = non_kin['ToggledOn']
    timer = initial_toggle['hand_in_marker_steps']
    if type(initial_toggle['value']) is not bool or isinstance(timer, bool) or not np.isfinite(timer) or timer < 0:
        raise ValueError('Invalid literal initial-scene toggle state')
    return {'timer': float(timer), 'value': initial_toggle['value'], 'last_written_frame': None,
            'source_initial_toggle_sha256': hashlib.sha256(json.dumps(initial_toggle, sort_keys=True).encode()).hexdigest(),
            'source_kind': 'recorded_initial_scene_without_prior_sparse_update', 'prefix_only': True}


def audit(config):
    code = Path(config['code_root'])
    assert hashlib.sha256((code / '_CODE_SNAPSHOT_READY').read_bytes()).hexdigest() == config['source_manifest_sha256']
    sys.path[:0] = [str(code / 'scripts'), str(code / 'sco')]
    import eval_b1k_subtask_samples as runner
    from recorded_native_state import is_native_recording
    from recorded_fixture_state import decode_fixture_fields
    import full100_shards as shards

    original = Path(config['original_run_root'])
    panel = runner.load_panel(original / 'panel.json')
    assignments = json.loads((original / 'assignments.json').read_text())
    shards.verify_assignments(assignments, panel)
    assert len(panel['samples']) == 7223 and len(panel['episodes']) == 400
    selected = [a for a in assignments['assignments'] if a['index'] in config['modern_indices']]
    assert len(selected) == 50
    samples = {s['sample_id']: s for s in panel['samples']}
    nonzero, unresolved, episodes = [], [], []
    checked_fields = 0
    initial_scene_fields = 0
    for assignment in selected:
        for task, instance in assignment['episode_keys']:
            rows = [samples[i] for i in assignment['sample_ids'] if samples[i]['task_instance_id'] == instance]
            cache = runner.load_episode_cache(runner.raw_hdf5_path(config['raw_root'], rows[0]), rows)
            assert is_native_recording(cache)
            objects = cache['scene_file']['state']['registry']['object_registry']
            toggle_names = sorted(n for n, s in objects.items() if 'ToggledOn' in s.get('non_kin', {}))
            seen = {}
            for sample in rows:
                for name in toggle_names:
                    key = name, sample['start']
                    if key not in seen:
                        try:
                            fields = decode_fixture_fields(cache, name, sample['start'])
                            seen[key] = {'timer': fields['non_kin']['ToggledOn']['hand_in_marker_steps'],
                                         'value': fields['non_kin']['ToggledOn']['value'],
                                         'source_packet_sha256': fields['source_packet_sha256'],
                                         'last_written_frame': fields['last_written_frame']}
                        except (ValueError, KeyError, TypeError):
                            try:
                                seen[key] = decode_toggle_prefix(cache, name, sample['start'])
                            except (ValueError, KeyError, TypeError) as error:
                                seen[key] = {'unresolved': str(error)}
                    evidence = seen[key]
                    identity = {'index': assignment['index'], 'task': task, 'instance_id': instance,
                                'sample_id': sample['sample_id'], 'start': sample['start'],
                                'object': name, 'raw_sha256': cache['sha256']}
                    if 'unresolved' in evidence:
                        unresolved.append({**identity, **evidence})
                    else:
                        checked_fields += 1
                        initial_scene_fields += evidence.get('source_kind') == 'recorded_initial_scene_without_prior_sparse_update'
                        if evidence['timer'] != 0:
                            nonzero.append({**identity, **evidence})
            episodes.append({'index': assignment['index'], 'task': task, 'instance_id': instance,
                             'raw_sha256': cache['sha256'], 'starts': len(rows), 'toggle_objects': len(toggle_names)})
    assert len(episodes) == 200 and sum(e['starts'] for e in episodes) == 3105
    return {'kind': 'all_modern_original_start_toggle_timer_source_inventory', 'complete_full100': False,
            'source_only': True, 'simulation_steps': 0, 'original_run_root': str(original),
            'source_manifest_sha256': config['source_manifest_sha256'], 'panel_fingerprint': panel['fingerprint'],
            'modern_h5': len(episodes), 'modern_starts': sum(e['starts'] for e in episodes),
            'checked_toggle_fields': checked_fields, 'nonzero_timer_fields': len(nonzero),
            'literal_initial_scene_fields_without_prior_sparse_update': initial_scene_fields,
            'nonzero_timer_samples': len({e['sample_id'] for e in nonzero}),
            'nonzero_timer_task_indices': sorted({e['index'] for e in nonzero}),
            'unresolved_fields': len(unresolved), 'unresolved_task_indices': sorted({e['index'] for e in unresolved}),
            'unresolved_reasons': dict(collections.Counter(e['unresolved'] for e in unresolved)),
            'nonzero': nonzero, 'unresolved': unresolved, 'episodes': episodes, 'observed_at': time.time(),
            'object_scope': 'Initial scene objects whose recorded state contains ToggledOn',
            'note': 'Nonzero recorded timers identify possible impact, not proof of a changed model outcome or runtime mismatch.'}
