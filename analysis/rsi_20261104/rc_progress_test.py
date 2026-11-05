import os, sys, time, importlib.util, numpy as np
sys.path.insert(0, "/mnt/data1/faye/code/libero_rlinf_20261103/RLinf")
spec = importlib.util.spec_from_file_location("rcvenv", "/home/faye/.claude/jobs/ba6d1e39/tmp/rc_test/RLinf/rlinf/envs/sim/robocasa/venv.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import robocasa  # noqa
from robocasa.utils.env_utils import create_env
tasks = sys.argv[1:] or ["SlideDishwasherRack", "OpenStandMixerHead", "TurnOnElectricKettle", "TurnOnSinkFaucet", "PickPlaceCounterToStove", "CoffeeSetupMug"]
for t in tasks:
    t0 = time.time()
    env = create_env(env_name=t, robots="PandaOmron", camera_names=["robot0_agentview_left"], camera_widths=64, camera_heights=64,
                     seed=0, render_onscreen=False, split="pretrain")
    env.reset()
    p0 = m._rc_progress(env)
    ps = []
    low, high = env.action_spec
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.step(rng.uniform(low, high) * 0.2)
        ps.append(m._rc_progress(env))
    extra = ""
    if t == "SlideDishwasherRack":
        extra = f"should_pull={env.should_pull} rack={env.dishwasher.get_state(env)['rack']:.3f}"
    elif t == "OpenStandMixerHead":
        extra = f"head={env.stand_mixer.get_state(env)['head']:.3f}"
    elif t == "TurnOnElectricKettle":
        extra = f"state={ {k: (round(v, 3) if isinstance(v, float) else v) for k, v in env.electric_kettle.get_state(env).items()} }"
    elif t == "TurnOnSinkFaucet":
        extra = f"handle={env.sink.get_handle_state(env=env).get('handle_joint'):.3f}"
    print(f"{t}: Phi at reset {p0}, after 10 steps {ps[-1]}, types ok {all(isinstance(x, float) for x in [p0] + ps)} | {extra} | {time.time()-t0:.0f}s", flush=True)
    env.close()
