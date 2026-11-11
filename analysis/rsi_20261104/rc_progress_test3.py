import sys, importlib.util, numpy as np
sys.path.insert(0, "/mnt/data1/faye/code/libero_rlinf_20261103/RLinf")
spec = importlib.util.spec_from_file_location("rcvenv", "/home/faye/.claude/jobs/ba6d1e39/tmp/rc_test/RLinf/rlinf/envs/sim/robocasa/venv.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import robocasa  # noqa
from robocasa.utils.env_utils import create_env
from robocasa.utils import object_utils as OU
env = create_env(env_name="PickPlaceCounterToStove", robots="PandaOmron", camera_names=["robot0_agentview_left"], camera_widths=64, camera_heights=64, seed=0, render_onscreen=False, split="pretrain")
env.reset()
cpos = env.sim.data.body_xpos[env.obj_body_id["container"]].copy()
jn = env.objects["obj"].joints[0]
for dz in (0.01, 0.03, 0.06):
    q = env.sim.data.get_joint_qpos(jn).copy(); q[:3] = cpos + np.array([0, 0, dz]); q[3:7] = [1, 0, 0, 0]
    env.sim.data.set_joint_qpos(jn, q); env.sim.data.set_joint_qvel(jn, np.zeros(6)); env.sim.forward()
    print(f"dz {dz}: in_receptacle {OU.check_obj_in_receptacle(env, 'obj', 'container', th=0.07)}, gripper_far {OU.gripper_obj_far(env)}, success {env._check_success()}, Phi {m._rc_progress(env)}")
