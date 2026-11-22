"""Collect a few development views with train-only annotations from OmniGibson."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def project(point, pose, width, height):
    """Project a base-frame point through the evaluator's calibrated head camera."""
    x, y, z = Rotation.from_quat(pose[3:]).inv().apply(point - pose[:3])
    depth = -z
    if depth <= 0:
        return None
    u = width / 2 + (306 * width / 720) * x / depth
    v = height / 2 - (306 * height / 720) * y / depth
    if not (0 <= u < width and 0 <= v < height):
        return None
    return [float(u), float(v), float(depth)]


class SweepPolicy:
    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0

    def forward(self, obs):
        state = obs["robot_r1::proprio"]
        action = torch.zeros(23, dtype=torch.float32)
        action[3:7] = state[53:57]
        action[7:14] = state[3:10]
        action[15:22] = state[28:35]
        action[14] = 1.0
        action[22] = 1.0
        action[2] = 0.35 if self.steps < 48 else -0.35
        self.steps += 1
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.instances)) != len(args.instances) or not all(0 <= i < 300 for i in args.instances):
        parser.error("Choose distinct BEHAVIOR training instance IDs from 0 to 299")

    from omnigibson.eval import eval as entry
    from omnigibson.object_states import ToggledOn

    args.output.mkdir(parents=True, exist_ok=False)
    captured = []

    class CaptureEvaluator(entry.Evaluator):
        def load_policy(self):
            return SweepPolicy()

        def load_task_instance(self, instance_id):
            super().load_task_instance(instance_id)
            self.capture_id = instance_id

        def step(self):
            if self._rollout_steps in (0, 24, 48, 72, 96, 120, 143):
                self.capture()
            return super().step()

        def capture(self):
            camera_names = list(self.robot_camera_names)
            camera = self.robot_camera_names["head"]
            poses = self.obs["robot_r1::cam_rel_poses"].cpu().numpy().reshape(-1, 7)
            pose = poses[camera_names.index("head")]
            rgb = self.obs[camera + "::rgb"].cpu().numpy()[..., :3]
            depth = self.obs[camera + "::depth_linear"].cpu().numpy().squeeze()
            radio = self.env.task.object_scope["radio_receiver.n.01_1"]
            marker = radio.states[ToggledOn].marker
            robot_position, robot_quat = (value.cpu().numpy() for value in self.robot.get_position_orientation())
            world_position = marker.get_position_orientation()[0].cpu().numpy()
            point = Rotation.from_quat(robot_quat).inv().apply(world_position - robot_position)
            height, width = depth.shape
            projection = project(point, pose, width, height)
            observed_depth = None
            if projection is not None:
                u, v, _ = projection
                sampled = float(depth[int(v), int(u)])
                if np.isfinite(sampled):
                    observed_depth = sampled
            stem = f"train-{self.capture_id:03d}-step-{self._rollout_steps:03d}"
            target = args.output / stem
            if not cv2.imwrite(str(target.with_suffix(".jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not save {target}")
            np.savez_compressed(target.with_suffix(".npz"), depth=depth, camera_pose=pose,
                                proprio=self.obs["robot_r1::proprio"].cpu().numpy())
            record = {"id": self.capture_id, "step": self._rollout_steps, "stem": stem,
                      "projected_marker_uv_depth": projection, "sampled_depth": observed_depth}
            captured.append(record)
            with (args.output / "frames.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print("TRAIN_FRAME", json.dumps(record), flush=True)

    entry.Evaluator = CaptureEvaluator
    sys.argv = ["train_capture", "--task-name", "turning_on_radio", "--mode", "train",
                "--instance-indices", *map(str, args.instances), "--policy", "local",
                "--env-wrapper", "omnigibson.eval.wrappers.RGBDFullResWrapper",
                "--max-steps", "144", "--write-video", "--output-dir", str(args.output / "diagnostics")]
    entry.main()
    (args.output / "capture_complete.json").write_text(json.dumps({"frames": len(captured), "instances": args.instances}) + "\n")


if __name__ == "__main__":
    main()
