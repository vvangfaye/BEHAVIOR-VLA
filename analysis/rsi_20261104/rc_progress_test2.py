import sys, importlib.util, inspect, numpy as np
sys.path.insert(0, "/mnt/data1/faye/code/libero_rlinf_20261103/RLinf")
spec = importlib.util.spec_from_file_location("rcvenv", "/home/faye/.claude/jobs/ba6d1e39/tmp/rc_test/RLinf/rlinf/envs/sim/robocasa/venv.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import robocasa  # noqa
from robocasa.utils.env_utils import create_env
mk = lambda t: create_env(env_name=t, robots="PandaOmron", camera_names=["robot0_agentview_left"], camera_widths=64, camera_heights=64, seed=0, render_onscreen=False, split="pretrain")
# articulation: set the fixture joint through the fixture API and read Phi back
for t, attr, key in (("SlideDishwasherRack", "dishwasher", "rack"), ("OpenStandMixerHead", "stand_mixer", "head")):
    env = mk(t); env.reset(); fx = getattr(env, attr)
    print(t, "set_joint_state signature:", inspect.signature(fx.set_joint_state))
    jn = [j for j in fx._joint_names.values() if key in j] if hasattr(fx, "_joint_names") and isinstance(fx._joint_names, dict) else None
    out = []
    for val in (0.5, 1.0) if not getattr(env, "should_pull", True) is False else (0.5, 0.0):
        try:
            fx.set_joint_state(env=env, min=val, max=val, joint_names=jn) if jn else fx.set_joint_state(env=env, min=val, max=val)
            env.sim.forward(); fx.update_state(env) if hasattr(fx, "update_state") else None
            out.append((val, round(float(fx.get_state(env)[key]), 3), round(m._rc_progress(env), 3), bool(env._check_success())))
        except Exception as e:
            out.append((val, "err", repr(e)[:120]))
    print(t, "joint names", jn, "->", out); env.close()
# pick-place: teleport the object into the receptacle (gripper stays far) -> success path -> Phi 1.0
from robocasa.utils import object_utils as OU
env = mk("PickPlaceCounterToStove"); env.reset()
cpos = env.sim.data.body_xpos[env.obj_body_id["container"]].copy()
jn = env.objects["obj"].joints[0]
q = env.sim.data.get_joint_qpos(jn).copy(); q[:3] = cpos + np.array([0, 0, 0.05]); env.sim.data.set_joint_qpos(jn, q); env.sim.forward()
for _ in range(30): env.sim.step()
env.sim.forward()
print("PickPlaceCounterToStove teleport: in_receptacle", OU.check_obj_in_receptacle(env, "obj", "container", th=0.07), "success", env._check_success(), "Phi", m._rc_progress(env))
