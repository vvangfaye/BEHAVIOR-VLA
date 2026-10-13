"""Optional privileged target observations for the offline collection teacher.

Only this teacher receives scene truth. Policy observations, controls and the
independent success monitor are unchanged. Reads never advance physics.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from codex_robot_guidance import pose, vector, quaternion, rotation, rotation_error, multiply, conjugate

CONTRACT = 'privileged_target_geometry_v1'


def validate_selection(selection, sample):
    if (not isinstance(selection, dict) or set(selection) != {'contract', 'targets'}
            or selection['contract'] != CONTRACT
            or sample.get('task') != 'attach_a_camera_to_a_tripod'
            or sample.get('predicate', {}).get('operation') != 'grasped'
            or selection['targets'] != sample['predicate'].get('targets')
            or selection['targets'] != ['digital_camera_87']):
        raise ValueError('GT teacher requires the frozen camera grasp target')


def _corners(lower, upper):
    lo, hi = vector(lower, 3), vector(upper, 3)
    if np.any(hi < lo):
        raise ValueError('Invalid target world AABB')
    return np.asarray(list(itertools.product(*zip(lo, hi))))


def _transform(world, base):
    bp, bq = vector(base['position_m'], 3), quaternion(base['quaternion_xyzw'])
    return dict(position_m=(rotation(bq).T @ (vector(world['position_m'], 3)-bp)).tolist(),
                quaternion_xyzw=quaternion(multiply(conjugate(bq),
                    quaternion(world['quaternion_xyzw']))).tolist())


def capture(evaluator, plan, request):
    """Take live object/robot reads at the same paused, pre-action observation."""
    from omnigibson.controllers import IsGraspingState

    selection = plan['teacher_scene_ground_truth']
    validate_selection(selection, plan['sample'])
    robot = evaluator.robot
    base = pose(*robot.get_position_orientation())
    bp, inv = vector(base['position_m'], 3), rotation(base['quaternion_xyzw']).T
    for arm in ('left', 'right'):
        wp, wq = robot.eef_links[arm].get_position_orientation()
        actual = request['robot_control_state']['arms'][arm]
        if (np.linalg.norm(inv @ (vector(wp, 3)-bp)-vector(actual['eef_position_m'], 3)) > 1e-3
                or np.linalg.norm(rotation_error(multiply(conjugate(quaternion(base['quaternion_xyzw'])),
                                                         quaternion(wq)), actual['eef_quaternion_xyzw'])) > 1e-3):
            raise ValueError('Live GT base frame disagrees with the actual correction EEF frame')
    objects = []
    for name in selection['targets']:
        target = evaluator.env.scene.object_registry('name', name)
        if target is None or target.name != name:
            raise ValueError('Frozen GT target is absent from the live scene')
        world = pose(*target.get_position_orientation())
        lo, hi = (vector(v, 3) for v in target.aabb)
        center = inv @ ((lo+hi)/2-bp)
        target_links = set(target.link_prim_paths)
        arms = {}
        for arm in ('left', 'right'):
            _, contacts = robot._find_gripper_contacts(arm=arm)
            touched = set().union(*(links for obj, links in contacts.items()
                                   if obj in target_links)) if contacts else set()
            fingers = list(robot.finger_links[arm])
            if len(fingers) != 2:
                raise ValueError('GT contact requires two actual robot fingers')
            grasp = robot.is_grasping(arm=arm, candidate_obj=target)
            if grasp not in (IsGraspingState.TRUE, IsGraspingState.FALSE):
                raise ValueError('Native target grasp truth is unobservable')
            wp = [vector(link.get_position_orientation()[0], 3).tolist() for link in fingers]
            arms[arm] = dict(target_finger_contact=[link.prim_path in touched for link in fingers],
                target_raycast_hit=bool(target_links & robot._find_gripper_raycast_collisions(arm=arm)),
                native_target_grasped=grasp == IsGraspingState.TRUE,
                finger_link_world_positions_m=wp,
                finger_link_positions_in_robot_base_m=[(inv @ (vector(p, 3)-bp)).tolist() for p in wp],
                eef_to_aabb_center_m=(center-vector(request['robot_control_state']['arms'][arm]
                                                   ['eef_position_m'], 3)).tolist())
        objects.append(dict(name=name, world_pose=world, pose_in_robot_base=_transform(world, base),
            world_aabb=dict(lower_m=lo.tolist(), upper_m=hi.tolist()),
            world_aabb_corners_in_robot_base_m=((_corners(lo, hi)-bp) @ inv.T).tolist(),
            aabb_center_in_robot_base_m=center.tolist(), arms=arms))
    result = dict(contract=CONTRACT, privileged=True, policy_input=False,
        frame='robot_base_footprint', step=request['step'], decision_index=request['decision_index'],
        sample_id=plan['sample']['sample_id'], attempt_id=plan['attempt_id'],
        plan_fingerprint=plan['fingerprint'], current_base_world_pose=base, objects=objects)
    validate_request(dict(request, scene_ground_truth_protocol=CONTRACT, scene_ground_truth=result),
                     selection=selection, plan_fingerprint=plan['fingerprint'])
    return result


def _equal(actual, expected):
    return np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                       atol=1e-8, rtol=0)


def validate_request(request, *, selection=None, plan_fingerprint=None):
    fields = {'scene_ground_truth_protocol', 'scene_ground_truth'}
    if not fields.intersection(request):
        if selection is not None:
            raise ValueError('Planned GT teacher observation is missing')
        return
    if (not fields <= request.keys() or request['scene_ground_truth_protocol'] != CONTRACT
            or request['task'] != 'attach_a_camera_to_a_tripod'):
        raise ValueError('Unsupported GT teacher request')
    gt = request['scene_ground_truth']
    if (set(gt) != {'contract','privileged','policy_input','frame','step','decision_index',
                   'sample_id','attempt_id','plan_fingerprint','current_base_world_pose','objects'}
            or gt['contract'] != CONTRACT or gt['privileged'] is not True or gt['policy_input'] is not False
            or gt['frame'] != 'robot_base_footprint'
            or not isinstance(gt['plan_fingerprint'], str) or len(gt['plan_fingerprint']) != 64
            or any(c not in '0123456789abcdef' for c in gt['plan_fingerprint'])
            or any(gt[k] != request[k] for k in ('step','decision_index','sample_id','attempt_id'))
            or plan_fingerprint is not None and gt['plan_fingerprint'] != plan_fingerprint
            or [o['name'] for o in gt['objects']] != (selection or {'targets':['digital_camera_87']})['targets']):
        raise ValueError('GT provenance or target differs from the current training request')
    base = gt['current_base_world_pose']
    if set(base) != {'position_m', 'quaternion_xyzw'}:
        raise ValueError('Invalid GT robot base pose')
    bp, inv = vector(base['position_m'], 3), rotation(base['quaternion_xyzw']).T
    guidance = request.get('robot_reference_guidance')
    if guidance is not None and guidance['current_base_world_pose'] != base:
        raise ValueError('GT and robot reference use different live base poses')
    control = request['robot_control_state']
    if control['frame'] != gt['frame']:
        raise ValueError('GT and actual control frames differ')
    for obj in gt['objects']:
        if set(obj) != {'name','world_pose','pose_in_robot_base','world_aabb',
                       'world_aabb_corners_in_robot_base_m','aabb_center_in_robot_base_m','arms'}:
            raise ValueError('Unexpected target GT fields')
        for p in (obj['world_pose'], obj['pose_in_robot_base']):
            if set(p) != {'position_m','quaternion_xyzw'}:
                raise ValueError('Invalid GT object pose')
            pose(p['position_m'], p['quaternion_xyzw'])
        expected = _transform(obj['world_pose'], base)
        if (not _equal(obj['pose_in_robot_base']['position_m'], expected['position_m'])
                or not _equal(rotation(obj['pose_in_robot_base']['quaternion_xyzw']),
                              rotation(expected['quaternion_xyzw']))):
            raise ValueError('GT world-to-control object pose transform differs')
        box = obj['world_aabb']
        if set(box) != {'lower_m','upper_m'}:
            raise ValueError('Invalid GT AABB fields')
        corners = _corners(box['lower_m'], box['upper_m'])
        converted = obj['world_aabb_corners_in_robot_base_m']
        if (np.shape(converted) != (8, 3) or not _equal(converted, (corners-bp) @ inv.T)):
            raise ValueError('GT bounding-box corner transform differs')
        center = inv @ ((vector(box['lower_m'],3)+vector(box['upper_m'],3))/2-bp)
        if not _equal(vector(obj['aabb_center_in_robot_base_m'],3), center):
            raise ValueError('GT bounding-box center transform differs')
        if set(obj['arms']) != {'left','right'}:
            raise ValueError('GT requires both robot arms')
        for arm, value in obj['arms'].items():
            if (set(value) != {'target_finger_contact','target_raycast_hit','native_target_grasped',
                              'finger_link_world_positions_m','finger_link_positions_in_robot_base_m',
                              'eef_to_aabb_center_m'}
                    or len(value['target_finger_contact']) != 2
                    or any(type(v) is not bool for v in value['target_finger_contact'])
                    or type(value['target_raycast_hit']) is not bool
                    or type(value['native_target_grasped']) is not bool):
                raise ValueError('Invalid target-specific contact or grasp truth')
            wp, rp = value['finger_link_world_positions_m'], value['finger_link_positions_in_robot_base_m']
            if (np.shape(wp) != (2,3) or np.shape(rp) != (2,3)
                    or not np.isfinite(np.asarray(wp, dtype=float)).all()
                    or not np.isfinite(np.asarray(rp, dtype=float)).all()
                    or not _equal(rp, (np.asarray(wp, dtype=float)-bp) @ inv.T)
                    or not _equal(vector(value['eef_to_aabb_center_m'],3),
                                  center-vector(control['arms'][arm]['eef_position_m'],3))):
                raise ValueError('GT finger positions or current EEF error differ')
            tactile = request.get('gripper_execution', {}).get('arms', {}).get(arm)
            if tactile is not None and any(t and not env for t, env in zip(
                    value['target_finger_contact'], tactile['finger_environment_contact'])):
                raise ValueError('Target contact contradicts live robot tactile feedback')


def audit_attempt(directory, plan, metadata, result):
    directory = Path(directory)
    selection = plan.get('teacher_scene_ground_truth')
    path = directory/'scene_ground_truth.jsonl'
    folders = sorted((directory/'teacher').glob('step-*'))
    if selection is None:
        if path.exists() or any('teacher_scene_ground_truth' in x for x in (metadata, result)):
            raise ValueError('Unplanned privileged teacher observations')
        for folder in folders:
            req = json.loads((folder/'request.json').read_text())
            if {'scene_ground_truth_protocol','scene_ground_truth'}.intersection(req):
                raise ValueError('Unplanned GT in a teacher request')
        return
    from codex_subtask_collection import file_sha256
    if (metadata.get('teacher_scene_ground_truth') != selection
            or result.get('teacher_scene_ground_truth') != selection
            or metadata.get('policy_observation_includes_ground_truth') is not False
            or result.get('policy_observation_includes_ground_truth') is not False
            or file_sha256(path) != result.get('scene_ground_truth_sha256')):
        raise ValueError('GT collection provenance or log hash differs')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != len(folders):
        raise ValueError('Every teacher request requires its frozen GT snapshot')
    for row, folder in zip(rows, folders):
        req = json.loads((folder/'request.json').read_text())
        validate_request(req, selection=selection, plan_fingerprint=plan['fingerprint'])
        if row != req['scene_ground_truth']:
            raise ValueError('GT snapshot differs from the frozen teacher request')
