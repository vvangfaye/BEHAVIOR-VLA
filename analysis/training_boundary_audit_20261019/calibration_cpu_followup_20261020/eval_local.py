"""Unchanged cached evaluator under the established local CPU asset safeguards."""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flow_endpoint_screen_20261020'))
from flow_endpoint_screen import offline_assets, cached
from run_local_cpu import PeriodicWakeupPolicy

if __name__=='__main__':
    asyncio.set_event_loop_policy(PeriodicWakeupPolicy())
    with offline_assets():cached.main()
