import base64,json
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/omnigibson-data/2026-challenge-task-instances/metadata')
tasks={'picking_up_trash','bringing_water','hanging_pictures','installing_a_scanner','installing_a_fax_machine'}
rows=[json.loads(s) for s in (root/'task.jsonl').read_text().splitlines() if s.strip()]
out=[r for r in rows if r.get('task_name') in tasks]
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
