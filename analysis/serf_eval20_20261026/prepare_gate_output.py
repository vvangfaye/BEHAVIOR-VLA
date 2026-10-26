import base64,json
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/serf_eval20_20261026/gate01')
assert root.is_dir(),'Wait for gate worker output initialization'
(root/'gate/metrics').mkdir(parents=True,exist_ok=True)
r={'gate_output_directory_ready':True,'path':str(root/'gate/metrics')}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
