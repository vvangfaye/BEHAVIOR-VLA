"""An observation-only search, approach and press program for turning_on_radio."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


HEAD_KEY = "robot_r1::robot_r1:zed_link:Camera:0"
PROPRIO_KEY = "robot_r1::proprio"
CAMERA_POSES_KEY = "robot_r1::cam_rel_poses"
ARM_STATE = {"left": slice(3, 10), "right": slice(28, 35)}
ARM_ACTION = {"left": slice(7, 14), "right": slice(15, 22)}
GRIPPER_ACTION = {"left": 14, "right": 22}
TRUNK_STATE = slice(53, 57)
TRUNK_VELOCITY = slice(57, 61)
TRUNK_ACTION = slice(3, 7)
PRESS_TRUNK = np.array([1.18, -1.69, -0.75, 0.0])
PRESS_STANCES = (PRESS_TRUNK, np.array([1.18, -1.82, -0.91, 0.0]))


class HeadCamera:
    def __init__(self, obs):
        self.rgb = np.asarray(obs[HEAD_KEY + "::rgb"])[..., :3]
        self.depth = np.asarray(obs[HEAD_KEY + "::depth_linear"], dtype=float).squeeze()
        if self.depth.ndim != 2 or self.rgb.shape[:2] != self.depth.shape:
            raise ValueError("The head RGB and linear depth must be aligned")
        # The default R1Pro evaluator supplies camera relative poses in left/right/head order.
        pose = np.asarray(obs[CAMERA_POSES_KEY], dtype=float).reshape(3, 7)[2]
        self.position = pose[:3]
        self.rotation = Rotation.from_quat(pose[3:])
        height, width = self.depth.shape
        self.center = np.array([width / 2, height / 2])
        self.focal = np.array([306 * width / 720, 306 * height / 720])

    def backproject(self, uv, depth):
        uv = np.asarray(uv)
        x, y = np.moveaxis((uv - self.center) / self.focal, -1, 0)
        points = np.stack((x * depth, -y * depth, -depth), axis=-1)
        return self.rotation.apply(points) + self.position

    def normal_at(self, u, v, depth):
        height, width = self.depth.shape
        x = np.arange(max(0, u - 9), min(width, u + 10), 2)
        y = np.arange(max(0, v - 9), min(height, v + 10), 2)
        x, y = (coords.ravel() for coords in np.meshgrid(x, y))
        z = self.depth[y, x]
        valid = np.isfinite(z) & (np.abs(z - depth) <= 0.035)
        if valid.sum() < 9:
            return np.array([0.0, 0.0, 1.0])
        points = self.backproject(np.column_stack((x[valid], y[valid])), z[valid])
        center = points.mean(axis=0)
        _, spread, axes = np.linalg.svd(points - center, full_matrices=False)
        normal = axes[-1]
        if spread[1] < 0.004:
            return np.array([0.0, 0.0, 1.0])
        return normal if normal @ (self.position - center) >= 0 else -normal

    def clearance(self, direction=(1.0, 0.0)):
        height, width = self.depth.shape
        u, v = (x.ravel() for x in np.meshgrid(np.arange(0, width, 12), np.arange(height // 3, height, 12)))
        depths = self.depth[v, u]
        valid = np.isfinite(depths) & (0.0 < depths) & (depths < 2.5)
        if not valid.any():
            return 0.0
        xyz = self.backproject(np.column_stack((u[valid], v[valid])), depths[valid])
        normal = np.asarray(direction, dtype=float)
        normal = normal / np.linalg.norm(normal)
        ahead = xyz[:, :2] @ normal
        lateral = xyz[:, :2] @ np.array([-normal[1], normal[0]])
        obstacles = ahead[(ahead > 0.055) & (np.abs(lateral) < 0.29) & (xyz[:, 2] > 0.16) & (xyz[:, 2] < 0.68)]
        return float(np.percentile(obstacles, 12)) if len(obstacles) >= 4 else 2.5


@dataclass
class Button:
    position: np.ndarray
    normal: np.ndarray
    on: bool
    uv: tuple
    confidence: float


@dataclass
class RadioHousing:
    position: np.ndarray
    bounds: tuple


def find_housing(camera):
    """Find the red radio casing at its physical scale, including from the side."""
    red, green, blue = np.moveaxis(camera.rgb.astype(np.int16), -1, 0)
    other = np.maximum(green, blue)
    mask = ((red > 85) & (red > 1.55 * other) & (red - other > 45)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_area = 0
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        if area < max(50, 0.28 * width * height) or area <= best_area:
            continue
        region = labels[y:y + height, x:x + width] == i
        z = camera.depth[y:y + height, x:x + width][region]
        z = z[np.isfinite(z) & (z > 0.16) & (z < 4.0)]
        if len(z) < 0.7 * area:
            continue
        depth = float(np.median(z))
        scale = depth / camera.focal[0]
        if not (0.045 < width * scale < 0.60 and 0.105 < height * scale < 0.43):
            continue
        point = camera.backproject(centers[i], depth)
        if not (point[0] > 0.18 and 0.30 < point[2] < 1.03):
            continue
        best = RadioHousing(point, (int(x), int(y), int(width), int(height)))
        best_area = area
    return best


def find_button(camera, near=None, housing=None):
    """Ground a small colored device indicator in metric depth from the same frame."""
    red, green, blue = np.moveaxis(camera.rgb.astype(np.int16), -1, 0)
    choices = []
    for on, intensity, contrast in ((False, red, red - np.maximum(green, blue)),
                                     (True, green, green - np.maximum(red, blue))):
        if on and near is None:
            continue
        mask = ((intensity > 65) & (contrast > 38)).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, count):
            x, y, width, height, area = stats[i]
            if not (1 <= area <= 520 and 0.40 <= width / height <= 2.5):
                continue
            if housing is not None:
                left, top, body_width, body_height = housing.bounds
                u, v = centers[i]
                pad = max(4, 0.25 * body_width)
                if not (left - pad <= u <= left + body_width + pad and
                        top + 0.12 * body_height <= v <= top + 0.93 * body_height):
                    continue
            region = labels[y:y + height, x:x + width] == i
            z = camera.depth[y:y + height, x:x + width][region]
            z = z[np.isfinite(z) & (z > 0.18) & (z < 4.0)]
            if len(z) == 0:
                continue
            d = float(np.median(z))
            size = max(width, height) * d / camera.focal[0]
            if not 0.004 <= size <= 0.038:
                continue
            position = camera.backproject(centers[i], d)
            if position[0] < 0.15 or not 0.28 <= position[2] <= 1.24:
                continue
            if housing is not None and np.linalg.norm(position - housing.position) > 0.24:
                continue
            displacement = 0.0 if near is None else float(np.linalg.norm(position - near))
            if near is not None and displacement > 0.12:
                continue
            score = (float(contrast[y:y + height, x:x + width][region].mean()) / 150
                     - abs(np.log(size / 0.012)) * 0.30 - d * 0.15 - displacement * 4)
            u, v = np.round(centers[i]).astype(int)
            choices.append((score, (u, v), d, position, on))
    if not choices:
        return None
    score, (u, v), depth, position, on = max(choices, key=lambda item: item[0])
    return Button(position, camera.normal_at(u, v, depth), on, (int(u), int(v)), float(score))


@dataclass
class Joint:
    name: str
    kind: str
    origin: np.ndarray
    rotation: np.ndarray
    axis: np.ndarray
    limits: tuple


def load_chain(urdf, side):
    xml = ET.parse(urdf)
    children = {j.find("child").attrib["link"]: j for j in xml.findall("joint")}
    chain = []
    link = f"{side}_gripper_finger_link1"
    while link != "base_link":
        j = children[link]
        origin = j.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        axis = j.find("axis")
        axis = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
        limit = j.find("limit")
        limits = (float(limit.get("lower")), float(limit.get("upper"))) if limit is not None else (0.0, 0.0)
        chain.append(Joint(j.attrib["name"], j.attrib["type"], xyz, Rotation.from_euler("xyz", rpy).as_matrix(),
                           axis, limits))
        link = j.find("parent").attrib["link"]
    return list(reversed(chain))


class ArmKinematics:
    # From the distal end of the stock R1Pro finger mesh, in that finger link's frame.
    CONTACT_OFFSET = np.array([0.005, -0.010, -0.035])

    def __init__(self, urdf, side):
        self.chain = load_chain(urdf, side)
        self.arm_indices = {f"{side}_arm_joint{i + 1}": i for i in range(7)}
        self.trunk_indices = {f"torso_joint{i + 1}": i for i in range(4)}
        limits = {j.name: j.limits for j in self.chain}
        self.low, self.high = np.array([limits[name] for name in self.arm_indices]).T

    def forward(self, trunk, arm):
        point = np.zeros(3)
        axes = np.eye(3)
        origins = np.zeros((7, 3))
        angular = np.zeros((7, 3))
        for joint in self.chain:
            point += axes @ joint.origin
            axes = axes @ joint.rotation
            if joint.kind != "revolute":
                # The pinch command closes the finger, corresponding to zero prismatic travel.
                continue
            if joint.name in self.arm_indices:
                idx = self.arm_indices[joint.name]
                origins[idx] = point
                angular[idx] = axes @ joint.axis
                q = arm[idx]
            else:
                q = trunk[self.trunk_indices[joint.name]]
            axes = axes @ Rotation.from_rotvec(joint.axis * q).as_matrix()
        point += axes @ self.CONTACT_OFFSET
        linear = np.cross(angular, point - origins).T
        return point, axes[:, 2], linear, angular.T

    def solve(self, trunk, initial, goal, normal, *, weight=0.23, alignment=0.85):
        q = np.clip(initial, self.low, self.high)
        for _ in range(86):
            point, hand_axis, linear, angular = self.forward(trunk, q)
            delta = goal - point
            error = float(np.linalg.norm(delta))
            aligned = float(hand_axis @ normal)
            if error < 0.008 and aligned >= alignment:
                return q, error
            orient = np.cross(hand_axis, normal)
            if aligned < -0.98:
                orient = np.cross(hand_axis, [0.0, 1.0, 0.0])
            task_matrix = np.vstack((linear, weight * angular))
            update = task_matrix.T @ np.linalg.solve(task_matrix @ task_matrix.T + 0.029 ** 2 * np.eye(6),
                                                     np.r_[delta, weight * orient])
            q = np.clip(q + np.clip(update, -0.17, 0.17), self.low, self.high)
        return None, error


class RadioPipeline:
    """Closed-loop navigation plus bounded visual contact and recovery phases."""

    def __init__(self, urdf, trace=None):
        self.arms = {side: ArmKinematics(urdf, side) for side in ARM_STATE}
        self.trace = trace
        self.episode = 0
        self.reset()

    def reset(self):
        self.episode += 1
        self.steps = 0
        self.phase = "search"
        self.phase_start = 0
        self.housing = None
        self.target = None
        self.side = None
        self.command = None
        self.precontact = None
        self.contact = None
        self.lift = None
        self.anchor = None
        self.attempts = 0
        self.sightings = 0
        self.orbit_direction = -1
        self.orbit_until = 0
        self.blocked_steps = 0
        self.contact_dwell = 0
        self.max_reach = 0.68
        self.did_press = False
        self.last_housing_step = -100
        self.last_marker_step = -100
        self.marker_offset = None
        self.last_bearing = 0.0
        self.trunk_origin = None
        self.trunk_goal = None
        self.arm_origin = None
        self.press_point = None
        self.press_normal = None

    def transition(self, phase, **fields):
        if phase == self.phase and not fields:
            return
        self.phase = phase
        self.phase_start = self.steps
        self.record(**fields)

    def record(self, **fields):
        if self.trace:
            row = {"episode": self.episode, "step": self.steps, "phase": self.phase, **fields}
            with self.trace.open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    @staticmethod
    def hold(state):
        action = np.zeros(23, dtype=np.float32)
        action[TRUNK_ACTION] = state[TRUNK_STATE]
        for side, action_slice in ARM_ACTION.items():
            action[action_slice] = state[ARM_STATE[side]]
            action[GRIPPER_ACTION[side]] = 1.0
        return action

    def plan_press(self, state):
        point = self.target.position.copy()
        toward_base = -point[:2] / np.linalg.norm(point[:2])
        surface = self.target.normal[:2]
        if np.dot(toward_base, surface) > 0.35:
            toward_base += surface
            toward_base /= np.linalg.norm(toward_base)
        normal = np.r_[0.91 * toward_base, 0.415]
        point[1] += (0.0, -0.006, 0.006)[self.attempts]
        for stance in PRESS_STANCES:
            solution = self.solve_press(state, stance, point, normal)
            if solution is None:
                continue
            self.side, self.lift, self.precontact, self.contact, error = solution
            self.anchor = self.target.position.copy()
            self.press_point, self.press_normal = point, normal
            self.trunk_origin = state[TRUNK_STATE].copy()
            self.trunk_goal = stance
            self.did_press = False
            self.transition("stance", arm=self.side, uv=self.target.uv, point=point.tolist(), ik_error=error,
                            trunk_start=self.trunk_origin.tolist(), trunk_goal=stance.tolist())
            return True
        return False

    def solve_press(self, state, trunk, point, normal):
        candidates = sorted(self.arms, key=lambda side: abs(point[1] - (0.21 if side == "left" else -0.21)))
        if self.attempts % 2:
            candidates.reverse()
        hover = point + 0.055 * normal + np.array([0.0, 0.0, 0.032])
        for side in candidates:
            arm = self.arms[side]
            q = state[ARM_STATE[side]]
            # Clear the tabletop lip before extending toward the marker.
            lift_point = np.array([0.24, 0.22 if side == "left" else -0.22, point[2] + 0.085])
            lift, _ = arm.solve(trunk, q, lift_point, normal, weight=0.04, alignment=-1.0)
            if lift is None:
                continue
            ahead, _ = arm.solve(trunk, lift, hover, normal, weight=0.10, alignment=0.55)
            if ahead is None:
                continue
            contact, error = arm.solve(trunk, ahead, point - normal * 0.010, normal,
                                       weight=0.10, alignment=0.55)
            if contact is not None:
                return side, lift, ahead, contact, error
        return None

    def act(self, obs):
        state = np.asarray(obs[PROPRIO_KEY], dtype=float).reshape(61)
        if not np.isfinite(state).all():
            raise ValueError("Invalid R1Pro proprioception")
        action = self.hold(state)
        self.steps += 1
        if self.phase in ("complete", "failed"):
            return action
        if self.phase in ("stance", "lift", "precontact", "press", "retreat"):
            self.control_arm(state, action, obs)
            return action

        camera = HeadCamera(obs)
        if self.phase == "verify":
            housing = find_housing(camera)
            self.target = find_button(camera, self.anchor, housing) if housing is not None else None
            self.verify_press()
            return action
        if self.steps % 5 == 1:
            housing = find_housing(camera)
            if housing is not None:
                self.housing = housing
                self.last_housing_step = self.steps
                self.last_bearing = np.arctan2(housing.position[1], housing.position[0])
            elif self.steps - self.last_housing_step > 12:
                self.housing = None
            observed = find_button(camera, housing=housing) if housing is not None else None
            if observed is None:
                self.sightings = 0
            elif self.target is None or np.linalg.norm(self.target.position - observed.position) > 0.14:
                self.sightings = 1
            else:
                self.sightings += 1
            self.target = observed
            if observed is not None:
                self.last_marker_step = self.steps
                self.marker_offset = observed.position - housing.position
        if self.steps % 60 == 1:
            self.record(kind="view", housing=None if self.housing is None else self.housing.position.tolist(),
                        marker=None if self.target is None else self.target.position.tolist(),
                        front_clearance=round(camera.clearance(), 3))
        if self.housing is None:
            self.search(camera, action)
            return action
        recently_seen = self.marker_offset is not None and self.steps - self.last_marker_step <= 42
        if (self.target is None and not recently_seen) or self.steps < self.orbit_until:
            self.orbit(camera, action)
            return action
        if self.sightings < 2 and not recently_seen:
            if self.phase != "confirm":
                self.transition("confirm", uv=self.target.uv)
            return action

        point = self.target.position if self.target is not None else self.housing.position + self.marker_offset
        # Keep the marker inside a shoulder's individual reach as the base comes to a stop.
        lateral = (-0.20 if self.attempts % 2 == 0 else 0.20) * np.clip((1.22 - point[0]) / 0.48, 0.0, 1.0)
        stand_off = self.max_reach - 0.055
        heading = np.arctan2(point[1], point[0]) - np.arctan2(lateral, stand_off)
        in_reach = 0.43 < point[0] < self.max_reach and abs(point[1]) < 0.237
        if in_reach and self.sightings >= 2 and self.steps % 5 == 1 and self.plan_press(state):
            self.control_arm(state, action, obs)
            return action
        if self.phase != "approach":
            self.transition("approach", uv=None if self.target is None else self.target.uv,
                            point=point.tolist(), front_clearance=camera.clearance())
        action[2] = np.clip(heading * 1.6, -0.40, 0.40)
        if abs(heading) < 0.45:
            motion = np.array([np.clip((point[0] - stand_off) * 0.8, -0.07, 0.29),
                               np.clip((point[1] - lateral) * 0.45, -0.14, 0.14)])
            self.drive(camera, action, motion)
            blocked = np.linalg.norm(motion) > 0.04 and np.linalg.norm(action[:2]) < 0.023
            self.blocked_steps = self.blocked_steps + 1 if blocked else 0
            if self.blocked_steps >= 55:
                self.orbit_until = self.steps + 145
                self.blocked_steps = 0
                self.transition("orbit", reason="approach corridor blocked", marker=point.tolist())
        return action

    @staticmethod
    def drive(camera, action, motion):
        if np.linalg.norm(motion) < 0.015:
            return
        # RGB-D points in the swept front of the base also include low table surfaces.
        clearance = camera.clearance(motion)
        action[:2] = motion * np.clip((clearance - 0.29) / 0.15, 0.0, 1.0)

    def orbit(self, camera, action):
        if self.phase != "orbit":
            self.transition("orbit", housing=self.housing.position.tolist())
        elif self.steps - self.phase_start >= 540:
            self.orbit_direction *= -1
            self.transition("orbit", direction=self.orbit_direction)
        point = self.housing.position[:2]
        radius = np.linalg.norm(point)
        unit = point / radius
        heading = np.arctan2(point[1], point[0])
        action[2] = np.clip(1.8 * heading, -0.42, 0.42)
        if abs(heading) > 0.72:
            return
        radial = np.clip(0.60 * (radius - 1.02), -0.14, 0.14) * unit
        lateral = self.orbit_direction * 0.34 * np.array([-unit[1], unit[0]])
        self.drive(camera, action, radial + lateral)
        if np.linalg.norm(action[:2]) < 0.035:
            self.drive(camera, action, -0.12 * unit)

    def search(self, camera, action):
        if self.phase == "scan_advance":
            if self.steps - self.phase_start < 62 and camera.clearance() > 0.58:
                action[0] = 0.20
                return
            self.transition("search")
        elif self.phase != "search":
            self.transition("search")
        if self.steps - self.phase_start >= 640 and camera.clearance() > 0.58:
            self.transition("scan_advance")
            action[0] = 0.20
        elif self.steps - self.last_housing_step <= 60 and abs(self.last_bearing) > 0.06:
            action[2] = np.clip(1.6 * self.last_bearing, -0.32, 0.32)
        else:
            action[2] = 0.30

    @staticmethod
    def ramp(origin, goal, elapsed, speed):
        delta = goal - origin
        pace = min(1.0, speed * (elapsed + 1) / max(1e-6, np.max(np.abs(delta))))
        return origin + pace * delta

    def move_arm(self, phase, state, goal, **fields):
        self.arm_origin = state[ARM_STATE[self.side]].copy()
        self.command = goal
        self.transition(phase, **fields)

    def control_arm(self, state, action, obs):
        trunk = state[TRUNK_STATE]
        elapsed = self.steps - self.phase_start
        action[TRUNK_ACTION] = self.trunk_goal
        if self.phase == "stance":
            action[TRUNK_ACTION] = self.ramp(self.trunk_origin, self.trunk_goal, elapsed, 0.016)
            speed = np.max(np.abs(state[TRUNK_VELOCITY]))
            if elapsed % 30 == 0:
                self.record(kind="trunk_tracking", measured=trunk.tolist(), command=action[TRUNK_ACTION].tolist(),
                            speed=float(speed))
            if elapsed >= 60 and elapsed % 15 == 0 and (speed < 0.10 or elapsed >= 150):
                camera = HeadCamera(obs)
                housing = find_housing(camera)
                marker = find_button(camera, self.anchor, housing) if housing is not None else None
                point = self.press_point if marker is None else self.press_point + marker.position - self.anchor
                solution = self.solve_press(state, trunk, point, self.press_normal)
                if solution is not None:
                    self.side, self.lift, self.precontact, self.contact, error = solution
                    self.press_point = point
                    self.move_arm("lift", state, self.lift, measured_trunk=trunk.tolist(), ik_error=error)
                    return
            if elapsed >= 240:
                self.max_reach = max(0.51, min(self.max_reach, self.press_point[0] - 0.035))
                self.transition("approach", reason="no IK solution with the measured pressing stance",
                                trunk=trunk.tolist(), revised_reach=self.max_reach)
            return
        q = state[ARM_STATE[self.side]]
        action[ARM_ACTION[self.side]] = self.ramp(self.arm_origin, self.command, elapsed, 0.040)
        action[GRIPPER_ACTION[self.side]] = -1.0
        error = float(np.max(np.abs(self.command - q)))
        reached = error < 0.052
        if self.phase == "lift" and (reached or elapsed >= 200):
            if reached:
                self.move_arm("precontact", state, self.precontact)
            else:
                self.transition("verify", reason="lift did not settle", joint_error=error)
        elif self.phase == "precontact" and (reached or elapsed >= 200):
            goal = self.press_point - 0.010 * self.press_normal
            contact, _ = self.arms[self.side].solve(trunk, q, goal, self.press_normal, weight=0.10, alignment=0.55)
            self.contact_dwell = 0
            if reached and contact is not None:
                self.did_press = True
                self.move_arm("press", state, contact)
            else:
                self.move_arm("retreat", state, self.lift, reached=bool(reached), joint_error=error)
        elif self.phase == "press":
            self.contact_dwell = self.contact_dwell + 1 if reached else 0
            if self.contact_dwell >= 14 or elapsed >= 180:
                tip, *_ = self.arms[self.side].forward(trunk, q)
                self.move_arm("retreat", state, self.precontact, reached=bool(reached), settled_steps=self.contact_dwell,
                              fingertip=tip.tolist(), observed_marker=self.anchor.tolist(), joint_error=error)
        elif self.phase == "retreat" and (reached or elapsed >= 160):
            self.transition("verify")

    def verify_press(self):
        if self.target is not None and self.target.on:
            self.transition("complete", uv=self.target.uv)
        elif self.steps - self.phase_start >= 18:
            self.attempts += int(self.did_press)
            if self.attempts >= 3:
                self.transition("failed", reason="indicator did not turn green after three attempts")
            else:
                self.sightings = 0
                self.target = None
                self.marker_offset = None
                self.transition("search", completed_press_attempts=self.attempts)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--urdf", type=Path, required=True)
    args.add_argument("--trace", type=Path)
    args.add_argument("--port", type=int, default=18910)
    args.add_argument("--candidate", choices=("pipeline", "vla", "routed"), default="pipeline")
    args.add_argument("--vla-port", type=int)
    args.add_argument("--card", type=Path)
    config = args.parse_args()
    if config.candidate in ("vla", "routed") and not config.vla_port:
        args.error("The VLA candidate requires the port of a local BEHAVIOR policy server")
    if config.candidate == "routed" and not config.card:
        args.error("Routing requires a policy card selected from train rollouts")

    import torch
    from omnigibson.eval.utils.network_utils import WebsocketClientPolicy, WebsocketPolicyServer

    pipeline = RadioPipeline(config.urdf, config.trace)
    vla = WebsocketClientPolicy(host="127.0.0.1", port=config.vla_port, server_wait_timeout=120) if config.vla_port else None
    routes = json.loads(config.card.read_text())["routes"] if config.card else {}

    class Policy:
        selected = None

        def reset(self):
            self.selected = None
            pipeline.reset()
            if vla is not None:
                vla.reset()

        def act(self, obs):
            if self.selected is None:
                camera = HeadCamera(obs)
                housing = find_housing(camera)
                marker = find_button(camera, housing=housing) if housing is not None else None
                context = "indicator_visible" if marker is not None else "indicator_not_visible"
                self.selected = routes[context] if config.candidate == "routed" else config.candidate
                pipeline.transition("search", kind="episode_route", context=context, candidate=self.selected)
            if self.selected == "pipeline":
                return torch.from_numpy(pipeline.act(obs))
            rgb_only = {}
            for key, value in obs.items():
                if key.endswith("::rgb"):
                    rgb_only[key] = cv2.resize(np.asarray(value), (224, 224), interpolation=cv2.INTER_AREA)
                elif not key.endswith("::depth_linear"):
                    rgb_only[key] = value
            return vla.act(rgb_only)

    WebsocketPolicyServer(Policy(), host="127.0.0.1", port=config.port).serve_forever()


if __name__ == "__main__":
    main()
