import asyncio
from pathlib import Path
import runpy
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flow_endpoint_screen_20261020'))
from run_local_cpu import PeriodicWakeupPolicy

if __name__=='__main__':
    asyncio.set_event_loop_policy(PeriodicWakeupPolicy())
    script=Path(__file__).resolve().with_name('probe.py')
    sys.argv=[str(script),*sys.argv[1:]]
    runpy.run_path(str(script),run_name='__main__')
