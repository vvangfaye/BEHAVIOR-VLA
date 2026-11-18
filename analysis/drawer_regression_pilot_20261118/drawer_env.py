"""Independent seeded LIBERO-Long drawer resets and checked action-prefix replay.

This module takes already postprocessed pi0.5 actions. It returns policy observations
in RLinf's directly-batchable format. All metrics are simulator measurements.
"""
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.utils.transform_utils import quat2axisangle


TASK_STEM = "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
INSTRUCTION = "put the black bowl in the bottom drawer of the cabinet and close it"


def array_hash(*arrays):
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str((arr.dtype.str, arr.shape)).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


class DrawerEnv:
    def __init__(self, cameras=True):
        self.use_cameras = cameras
        self.env = None
        self.bddl = Path(get_libero_path("bddl_files")) / "libero_10" / (TASK_STEM + ".bddl")
        root = Path(get_libero_path("init_states")) / "libero_10"
        sources = [torch.load(root / (TASK_STEM + ext), weights_only=False, map_location="cpu")
                   for ext in (".init", ".pruned_init")]
        self.official_states = np.concatenate(sources)

    def reset(self, seed):
        self.close()
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.env = ControlEnv(
            str(self.bddl), use_camera_obs=self.use_cameras, has_renderer=False,
            has_offscreen_renderer=self.use_cameras, camera_heights=256, camera_widths=256,
            hard_reset=False,
        )
        # Reseed after model/renderer construction so reset samples have an explicit seed.
        self.env.seed(self.seed)
        self._raw_obs = self.env.reset()
        initial_state = np.asarray(self.env.get_sim_state(), dtype=np.float64)
        if np.any(np.all(np.isclose(self.official_states, initial_state, atol=1e-10, rtol=0), axis=1)):
            raise ValueError("Fresh reset coincides with a supplied LIBERO initial state")
        self.initial_hash = array_hash(initial_state)
        self._actions = []
        self.success_once = False
        self.step_count = 0
        model = self.env.sim.model
        self.bowl_id = model.body_name2id("akita_black_bowl_1_main")
        self.region_id = model.site_name2id("white_cabinet_1_bottom_region")
        self.slide_id = model.joint_name2id("white_cabinet_1_bottom_level")
        self.slide_addr = model.jnt_qposadr[self.slide_id]
        self.closed_predicate, self.inside_predicate = self.env.env.parsed_problem["goal_state"]
        assert self.closed_predicate[0] == "close" and self.inside_predicate[0] == "in"
        settle = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
        for _ in range(15):
            self._raw_obs, _, _, _ = self.env.step(settle)
        return self.observe()

    def observe(self):
        o = self._raw_obs
        state = np.concatenate([o["robot0_eef_pos"], quat2axisangle(o["robot0_eef_quat"].copy()),
                                o["robot0_gripper_qpos"]])
        obs = {"states": state.astype(np.float32), "task_descriptions": INSTRUCTION}
        if self.use_cameras:
            obs["main_images"] = np.ascontiguousarray(o["agentview_image"][::-1, ::-1])
            obs["wrist_images"] = np.ascontiguousarray(o["robot0_eye_in_hand_image"][::-1, ::-1])
        return obs

    def metrics(self):
        data = self.env.sim.data
        # Joint range [-.16,.01]: its positive world axis points IN from the drawer front.
        signed_depth = float(np.dot(data.body_xpos[self.bowl_id] - data.site_xpos[self.region_id],
                                    data.xaxis[self.slide_id]))
        dom = self.env.env
        inside = bool(dom._eval_predicate(self.inside_predicate))
        closed = bool(dom._eval_predicate(self.closed_predicate))
        return {"bowl_in": inside, "drawer_closed": closed, "success": inside and closed,
                "bowl_depth_m": signed_depth, "signed_front_m": -signed_depth,
                "drawer_open_m": -float(data.qpos[self.slide_addr]), "step": self.step_count}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected one finite 7-D LIBERO action")
        self._raw_obs, _, _, _ = self.env.step(action)
        self.step_count += 1
        self._actions.append(action.tolist())
        m = self.metrics()
        self.success_once |= m["success"]
        return self.observe(), m

    def state_hash(self):
        sim = self.env.sim
        # Fixed fixture poses may live in the model in addition to the dynamic qpos/qvel.
        return array_hash(sim.get_state().flatten(), sim.model.body_pos, sim.model.body_quat)

    def observation_hash(self):
        obs = self.observe()
        return array_hash(*(obs[k] for k in ("states", "main_images", "wrist_images") if k in obs))

    def snapshot(self):
        return {"seed": self.seed, "actions": list(self._actions), "step": self.step_count,
                "initial_hash": self.initial_hash, "state_hash": self.state_hash(),
                "observation_hash": self.observation_hash(), "success_once": self.success_once}

    def restore(self, snap):
        self.reset(snap["seed"])
        if self.initial_hash != snap["initial_hash"]:
            raise RuntimeError("Seeded initial state did not reproduce")
        for action in snap["actions"]:
            self.step(action)
        if (self.step_count != snap["step"] or self.state_hash() != snap["state_hash"]
                or self.observation_hash() != snap["observation_hash"]
                or self.success_once != snap["success_once"]):
            raise RuntimeError("Action-prefix replay did not restore snapshot and observations exactly")
        return self.observe()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
