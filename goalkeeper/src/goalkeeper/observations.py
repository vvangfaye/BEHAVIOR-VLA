"""A narrow boundary between the simulator message and model inputs."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SensorSchema:
    robot: str = "robot_r1"
    head: str = "robot_r1::robot_r1:zed_link:Camera:0"
    left: str = "robot_r1::robot_r1:left_realsense_link:Camera:0"
    right: str = "robot_r1::robot_r1:right_realsense_link:Camera:0"

    @property
    def cameras(self): return (self.head, self.left, self.right)


@dataclass(frozen=True)
class RobotObservation:
    rgb: tuple[np.ndarray, ...]
    proprio: np.ndarray


def validate_observation(payload: dict, schema: SensorSchema, expected_task_id: int,
                         proprio_dim: int) -> RobotObservation:
    allowed = {schema.robot + "::proprio", schema.robot + "::cam_rel_poses", "task_id", "need_new_action"}
    allowed |= {c + "::" + modality for c in schema.cameras for modality in ["rgb", "depth_linear"]}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unexpected simulator fields; do not expose these to the policy: {sorted(unknown)}")
    # The server is configured for ONE public task; a reset never changes it.
    if "task_id" in payload:
        ids = np.asarray(payload["task_id"])
        if ids.size != 1 or ids.dtype.kind not in "iu" or int(ids.item()) != expected_task_id:
            raise ValueError("Task ID does not match this server's public task definition")
    key = schema.robot + "::proprio"
    state = np.asarray(payload[key])
    if state.shape != (proprio_dim,) or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
        raise ValueError("Invalid proprioception shape or values; inspect the actual robot config")
    images = []
    for camera in schema.cameras:
        x = np.asarray(payload[camera + "::rgb"])
        if x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] not in (3, 4) or min(x.shape[:2]) < 1:
            raise ValueError("Expected HWC uint8 RGB/RGBA camera image")
        images.append(np.ascontiguousarray(x[..., :3]))
    # Neither optional depth nor relative camera poses are needed by this RGB-v1
    # head. They are dropped, not passed through a generic simulator dictionary.
    return RobotObservation(tuple(images), state.astype(np.float32, copy=True))
