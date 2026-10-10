"""Literal fixture fields and annotation identities from the recorded scene.

Only the native root/joint layout and recorded ToggledOn/AttachedTo widths are
supported. A demonstration field change identifies a fixture; its desired
predicate value never participates in choosing that identity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

import numpy as np

FIXTURE_BINDING = 'recorded_scene_and_demonstration_fixture_identity_v3'
METHOD = 'unique_recorded_demonstration_fixture_event'
ROOT_KEYS = {'pos', 'ori', 'lin_vel', 'ang_vel'}
STATE_KEYS = {'ToggledOn': {'value', 'hand_in_marker_steps'}, 'AttachedTo': {'attached_obj_uuid'}}
STATE_WIDTHS = {'ToggledOn': 2, 'AttachedTo': 1}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def decode_fixture_fields(cache, name, frame):
    """Decode a named original object's last explicit sparse native packet."""
    from recorded_native_state import MAGIC, _recorded_object_topology, uuid

    scene = cache['scene_file']
    info = scene['objects_info']['init_info'].get(name)
    initial = scene['state']['registry']['object_registry'].get(name)
    infos, _ = _recorded_object_topology(cache, frame)
    if (info is None or initial is None or infos.get(name) != info
            or 'controllers' in initial or 'controller_groups' in initial
            or set(initial.get('root_link', {})) != ROOT_KEYS
            or type(frame) is not int or not 0 <= frame < len(cache['states'])):
        raise ValueError('Recorded fixture identity or native schema is unsupported')
    q = initial.get('joint_pos', [])
    velocity = initial.get('joint_vel', [])
    if len(q) != len(velocity):
        raise ValueError('Recorded fixture joint schema is inconsistent')
    non_kin = initial.get('non_kin', {})
    if (not isinstance(non_kin, dict) or not set(non_kin) <= set(STATE_WIDTHS)
            or any(set(v) != STATE_KEYS[k] for k, v in non_kin.items())):
        raise ValueError('Recorded fixture non-kinematic schema is unsupported')
    width = 14 + 2 * len(q) + sum(STATE_WIDTHS[k] for k in non_kin)
    object_uuid = uuid(name)
    if sum(uuid(n) == object_uuid for n in infos) != 1:
        raise ValueError('Recorded fixture UUID is ambiguous')
    # Source arrays are episode recordings, not simulator tensors. Index UUID
    # occurrences once, then inspect only candidate sparse rows in reverse.
    indices = cache.setdefault('fixture_packet_candidate_frames', {})
    key = (cache['sha256'], name)
    if key not in indices:
        indices[key] = np.flatnonzero(np.any(cache['states'] == np.float32(object_uuid), axis=1))
    for previous in reversed(indices[key][indices[key] <= frame]):
        prior = int(previous)
        size = int(cache['state_sizes'][prior])
        if not 0 <= size <= cache['states'].shape[1]:
            raise ValueError('Recorded fixture row length is invalid')
        row = np.asarray(cache['states'][prior, :size])
        boundary = np.flatnonzero(row == np.float32(uuid('object_registry')))
        if len(boundary) != 1:
            raise ValueError('Recorded fixture registry boundary is ambiguous')
        begin = int(boundary[0]) + 2
        # A grasp packet references an object's UUID; that occurrence is not its
        # object-registry header. Duplicate remaining occurrences are rejected.
        references = {int(p) + 2 for p in np.flatnonzero(row == MAGIC)}
        hits = [int(p) + begin for p in np.flatnonzero(row[begin:] == np.float32(object_uuid))
                if int(p) + begin not in references]
        if not hits:
            continue
        if len(hits) != 1:
            raise ValueError('Recorded fixture packet boundary is ambiguous')
        offset = hits[0]
        if (row.dtype != np.dtype('float32') or offset + 1 + width > len(row)
                or row[offset + 1] not in (0, 1)):
            raise ValueError('Recorded fixture packet numeric schema differs')
        packet = row[offset + 1:offset + 1 + width]
        if (not np.isfinite(packet).all()
                or abs(float(np.linalg.norm(packet[4:8])) - 1) > 1e-4):
            raise ValueError('Recorded fixture packet contains invalid native fields')
        cursor = 14 + 2 * len(q)
        states = {}
        for key in non_kin:
            if key == 'ToggledOn':
                if packet[cursor] not in (0, 1) or packet[cursor + 1] < 0:
                    raise ValueError('Recorded fixture toggle fields are invalid')
                states[key] = {'value': bool(packet[cursor]), 'hand_in_marker_steps': float(packet[cursor + 1])}
            else:
                value = float(packet[cursor])
                if value != int(value):
                    raise ValueError('Recorded fixture attachment UUID is invalid')
                states[key] = {'attached_obj_uuid': int(value)}
            cursor += STATE_WIDTHS[key]
        assert cursor == width
        selected = np.asarray(cache['states'][frame, :int(cache['state_sizes'][frame])])
        return {'object': name, 'object_uuid': object_uuid, 'category': info['args'].get('category'),
                'raw_sha256': cache['sha256'], 'source_frame': frame, 'last_written_frame': prior,
                'selected_state_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
                'source_packet_sha256': hashlib.sha256(row[offset:offset + 1 + width].tobytes()).hexdigest(),
                'source_float_offset': offset, 'serialized_width': width,
                'init_info_sha256': _digest(info), 'state_schema_sha256': _digest(initial),
                'root_position': packet[1:4].tolist(), 'root_orientation': packet[4:8].tolist(),
                'joint_positions': packet[14:14 + len(q)].tolist(),
                'joint_velocities': packet[14 + len(q):14 + 2 * len(q)].tolist(),
                'recorded_non_kin_order': list(non_kin), 'non_kin': states,
                'unrecorded_physics_caches_or_timers_verified': False}
    raise ValueError('No explicit recorded fixture packet before source frame')


def demonstration_fixture_identity(cache, proof, predicate, identifier):
    """Use exactly one literal changed fixture, including recorded scene objects."""
    from recorded_native_state import _recorded_object_topology

    samples = cache.get('source_samples', {})
    sample = samples.get(proof['sample_id'])
    if (not sample or sample['start'] != proof['source_start'] or sample['predicate'] != predicate
            or predicate['operation'] not in ('near', 'state')):
        return None
    event = sample
    if predicate['operation'] == 'near':
        following = [s for s in samples.values() if s['start'] >= sample['end']]
        if not following:
            return None
        event = min(following, key=lambda s: (s['start'], s['end'], s['sample_id']))
        if (event['start'] != sample['end'] or event['predicate']['operation'] != 'state'
                or event['predicate'].get('targets') != predicate.get('targets')):
            return None
    field = event['predicate'].get('state')
    frame = event['end'] - 1
    if field not in ('ToggledOn', 'Open') or not sample['start'] <= frame < len(cache['states']):
        return None
    infos, _ = _recorded_object_topology(cache, sample['start'])
    candidates = sorted(name for name, info in infos.items()
                        if identifier in (name, re.sub(r'_\d+$', '', name), info['args'].get('category')))
    if not candidates:
        return None
    changes, changed = [], []
    chosen_fields = {}
    try:
        for name in candidates:
            before = decode_fixture_fields(cache, name, sample['start'])
            after = decode_fixture_fields(cache, name, frame)
            if before['init_info_sha256'] != after['init_info_sha256']:
                return None
            if field == 'ToggledOn':
                first, last = before['non_kin']['ToggledOn']['value'], after['non_kin']['ToggledOn']['value']
                different = first != last
            else:
                first, last = before['joint_positions'], after['joint_positions']
                if not first or len(first) != len(last):
                    return None
                different = bool(np.any(np.abs(np.asarray(first) - np.asarray(last)) > 1e-4))
            changes.append({'object': name, 'before': first, 'after': last,
                            'initial_last_written_frame': before['last_written_frame'],
                            'event_last_written_frame': after['last_written_frame'],
                            'initial_packet_sha256': before['source_packet_sha256'],
                            'event_packet_sha256': after['source_packet_sha256'], 'changed': different})
            if different:
                changed.append(name); chosen_fields[name] = (before, after)
    except (KeyError, TypeError, ValueError):
        return None  # Unverifiable competing candidates cannot establish uniqueness.
    if len(changed) != 1:
        return None
    name = changed[0]
    before, after = chosen_fields[name]
    return name, {'source_sample_id': sample['sample_id'], 'event_sample_id': event['sample_id'],
                  'event_annotation_start': event['start'], 'event_annotation_end': event['end'],
                  'event_frame': frame, 'field': field, 'desired_predicate_value_used': False,
                  'candidates': changes, 'unique_changed_object': name,
                  'initial_native_fields': before, 'event_native_fields': after}


def validate_runtime_fixture_roles(binding, evaluator):
    """Compare the resolved fixture's saved fields before any policy action."""
    entries = [e for e in binding['bindings'] if e.get('method') == METHOD]
    if not entries:
        return binding
    verified = []
    for entry in entries:
        source = entry['source_fixture_event']['initial_native_fields']
        obj = evaluator.env.scene.object_registry('name', entry['object'])
        if obj is None or obj.name != source['object'] or int(obj.uuid) != source['object_uuid']:
            raise ValueError('Runtime fixture identity differs from its original source')
        actual = obj.dump_state(serialized=False)
        def numbers(value):
            return value.tolist() if hasattr(value, 'tolist') else list(value)
        position = np.asarray(numbers(actual['root_link']['pos']))
        orientation = np.asarray(numbers(actual['root_link']['ori']))
        expected = np.asarray(source['root_orientation'])
        errors = {'position': float(np.max(np.abs(position - source['root_position']))),
                  'orientation': float(min(np.max(np.abs(orientation - expected)), np.max(np.abs(orientation + expected))))}
        joints = numbers(actual.get('joint_pos', []))
        velocities = numbers(actual.get('joint_vel', []))
        if (any(not np.isfinite(v) or v > 1e-5 for v in errors.values())
                or joints != source['joint_positions'] or velocities != source['joint_velocities']):
            raise ValueError('Runtime fixture pose or joints differ from saved source fields')
        states = {key: actual['non_kin'][key] for key in source['recorded_non_kin_order']}
        if states != source['non_kin']:
            difference = {'object': obj.name, 'source_frame': source['source_frame'],
                          'last_written_frame': source['last_written_frame'],
                          'expected': source['non_kin'], 'actual': states}
            raise ValueError('Runtime fixture non-kinematic fields differ from saved source fields: '
                             + json.dumps(difference, sort_keys=True))
        verified.append({'object': obj.name, 'object_uuid': int(obj.uuid), 'root_pose_errors': errors,
                         'actual_joint_positions': joints, 'actual_joint_velocities': velocities,
                         'actual_non_kin': copy.deepcopy(states),
                         'verified_fields': ['root_pose', 'joint_positions', 'joint_velocities', 'recorded_non_kin']})
    return {**binding, 'runtime_fixture_fields_verified_before_policy': True,
            'runtime_fixture_field_attestations': verified}


def validate_fixture_result_binding(binding, sample, native_proof):
    from recorded_native_state import uuid

    entries = [e for e in binding['bindings'] if e.get('method') == METHOD]
    if not entries or binding.get('runtime_fixture_fields_verified_before_policy') is not True:
        raise ValueError('Recorded fixture binding lacks before-policy native field verification')
    attestations = binding.get('runtime_fixture_field_attestations', [])
    if len(attestations) != len(entries):
        raise ValueError('Recorded fixture binding lacks complete runtime attestations')
    for entry, attestation in zip(entries, attestations):
        event = entry.get('source_fixture_event', {})
        before, after = event.get('initial_native_fields', {}), event.get('event_native_fields', {})
        frame, end = event.get('event_frame'), event.get('event_annotation_end')
        if (entry.get('role') != 'targets' or type(frame) is not int or type(end) is not int or frame != end - 1
                or not sample['start'] <= frame < sample['episode_length']
                or event.get('source_sample_id') != sample['sample_id']
                or event.get('desired_predicate_value_used') is not False
                or event.get('field') not in ('ToggledOn', 'Open')
                or event.get('unique_changed_object') != entry['object']
                or [x.get('object') for x in event.get('candidates', []) if x.get('changed')] != [entry['object']]):
            raise ValueError('Recorded fixture binding changed its literal annotation event')
        for fields, expected_frame in ((before, sample['start']), (after, frame)):
            if (fields.get('object') != entry['object'] or fields.get('object_uuid') != uuid(entry['object'])
                    or fields.get('source_frame') != expected_frame
                    or fields.get('raw_sha256') != native_proof['raw_sha256']
                    or type(fields.get('last_written_frame')) is not int
                    or not 0 <= fields['last_written_frame'] <= expected_frame
                    or any(not re.fullmatch('[0-9a-f]{64}', fields.get(k, '')) for k in
                           ('selected_state_sha256', 'source_packet_sha256', 'init_info_sha256', 'state_schema_sha256'))):
                raise ValueError('Recorded fixture binding lacks literal source field identities')
        if (before['selected_state_sha256'] != native_proof['selected_state_sha256']
                or entry.get('source_scene_object') != {k: before[k] for k in ('object', 'object_uuid', 'init_info_sha256')}
                or attestation.get('object') != entry['object'] or attestation.get('object_uuid') != before['object_uuid']
                or attestation.get('actual_joint_positions') != before['joint_positions']
                or attestation.get('actual_joint_velocities') != before['joint_velocities']
                or attestation.get('actual_non_kin') != before['non_kin']
                or attestation.get('verified_fields') != ['root_pose', 'joint_positions', 'joint_velocities', 'recorded_non_kin']
                or set(attestation.get('root_pose_errors', {})) != {'position', 'orientation'}
                or any(not np.isfinite(v) or not 0 <= v <= 1e-5 for v in attestation['root_pose_errors'].values())):
            raise ValueError('Recorded fixture runtime attestation differs from source fields')
