"""Read only reusable assets and exact modem task definition from owned AFS."""
import base64
import hashlib
import json
import shutil
from pathlib import Path

runtime = Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
code = Path('/mnt/afs/260010165/behavior_1k/code_snapshots/b1k-video1688977-20261013')
task = 'installing_a_modem'
bddl = code / 'BEHAVIOR-1K/bddl3/bddl/activity_definitions' / task / 'problem0.bddl'
definition = bddl.read_text()
assert all(s in definition for s in ('modem.n.01_1', 'cabinet.n.01', 'television_receiver.n.01_1', 'nextto', 'toggled_on'))
root = runtime / 'omnigibson-data/2026-challenge-task-instances'
instances = {}
for split, sub, ids in [('train', 'scenes', range(1, 17)), ('public_test', 'scene_test/public', range(301, 311))]:
    for i in ids:
        files = list((root/sub).glob(f'*/json/*{task}*/*{task}_0_{i}_template-tro_state.json'))
        assert len(files) == 1, (split, i, len(files))
        file = files[0]
        instances[f'{split}:{i}'] = {'path': str(file), 'bytes': file.stat().st_size}
out = {'passed': True, 'task': task, 'bddl_sha256': hashlib.sha256(bddl.read_bytes()).hexdigest(),
       'bddl_definition': definition, 'instances': instances,
       'free_filesystem_bytes': shutil.disk_usage(runtime).free,
       'original_actor_and_critic_reused_from_radio': False}
for rel in ['BEHAVIOR-1K/OmniGibson/omnigibson/object_states/next_to.py',
            'BEHAVIOR-1K/OmniGibson/omnigibson/object_states/on_top.py',
            'BEHAVIOR-1K/OmniGibson/omnigibson/object_states/toggle.py']:
    source = code / rel
    assert source.is_file()
    out[rel] = hashlib.sha256(source.read_bytes()).hexdigest()
print('B1KPROBE_BEGIN' + base64.b64encode(json.dumps(out).encode()).decode() + 'B1KPROBE_END')
