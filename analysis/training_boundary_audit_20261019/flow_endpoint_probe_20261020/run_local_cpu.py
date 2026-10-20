#!/usr/bin/env python3
"""Launch only this folder's probe with the existing local asyncio workaround.

Use an external timeout and taskset to bound wall time and CPU affinity. The
50 ms timer is process-local; it changes no checkpoint, model, or stored data.
It reproduces the already used CPU workaround without executing the old eval.
All command-line arguments are forwarded to flow_endpoint_probe.py.
"""
import asyncio
from pathlib import Path
import runpy
import sys


class PeriodicWakeupPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        loop = super().new_event_loop()

        def tick():
            if not loop.is_closed():
                loop.call_later(0.05, tick)

        loop.call_later(0.05, tick)
        return loop


if __name__ == "__main__":
    asyncio.set_event_loop_policy(PeriodicWakeupPolicy())
    target = Path(__file__).resolve().with_name("flow_endpoint_probe.py")
    sys.argv = [str(target), *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")
