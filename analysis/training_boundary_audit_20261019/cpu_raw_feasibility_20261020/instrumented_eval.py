"""Local-only timing wrapper; the caller supplies a finite subprocess timeout."""
import asyncio
import faulthandler
import functools
import logging
import runpy
import time
from pathlib import Path


class PeriodicWakeupPolicy(asyncio.DefaultEventLoopPolicy):
    """Local workaround for missing cross-thread wakeups in this sandbox.

    A metadata-only control verified that an ordinary TensorStore Future can
    finish synchronously while the default asyncio wait stalls. A timer makes
    the loop revisit its ready queue; it does not change the checkpoint data.
    """

    def new_event_loop(self):
        loop = super().new_event_loop()

        def tick():
            if not loop.is_closed():
                loop.call_later(0.05, tick)

        loop.call_later(0.05, tick)
        return loop


asyncio.set_event_loop_policy(PeriodicWakeupPolicy())
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)


def timed(function, name):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        print(f"STAGE start {name}", flush=True)
        try:
            return function(*args, **kwargs)
        finally:
            print(f"STAGE end {name} elapsed={time.monotonic()-start:.3f}", flush=True)
    return wrapper


print("STAGE start imports", flush=True)
from openpi.models import model
from openpi.policies import policy_config

model.restore_params = timed(model.restore_params, "restore_params")
model.BaseModelConfig.load = timed(model.BaseModelConfig.load, "BaseModelConfig.load")
original_create = policy_config.create_trained_policy


@functools.wraps(original_create)
def create_with_infer_timer(*args, **kwargs):
    policy = timed(original_create, "create_trained_policy")(*args, **kwargs)
    policy.infer = timed(policy.infer, "policy.infer")
    return policy


policy_config.create_trained_policy = create_with_infer_timer
print("STAGE end imports", flush=True)
runpy.run_path(str(Path(__file__).resolve().parents[1] / "eval_cached_probe.py"), run_name="__main__")
