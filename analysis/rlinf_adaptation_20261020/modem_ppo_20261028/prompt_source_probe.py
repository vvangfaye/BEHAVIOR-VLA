import ast
import base64
import hashlib
import json
from pathlib import Path

root=Path('/mnt/afs/260010165/behavior_1k/code_snapshots/b1k-video1688977-20261013')
path=root/'openpi/src/openpi/configs/tasks/b1k.py'
tree=ast.parse(path.read_text())
node=next(n for n in tree.body if isinstance(n,ast.Assign) and
          any(isinstance(t,ast.Name) and t.id=='TASKS' for t in n.targets))
prompt=ast.literal_eval(node.value)['installing_a_modem']
assert prompt=='Install the modem by placing it on the cabinet and turning it on.'
output={'passed':True,'prompt':prompt,'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'radio_fallback_not_used':True,'cases':{}}
base=Path('/mnt/afs/260010165/behavior_1k/rlinf_modem_20261028/pilot01/results')
for i in (1,2):
    folder=base/'baseline'/f'i{i}_s0'
    row={name:json.loads((folder/name).read_text()) for name in
         ('initialization.json','python_startup_failure.json','goal_contract.json') if (folder/name).is_file()}
    row['completed_result_exists']=(folder/'result.json').is_file()
    row['actions_exists']=(folder/'actions.json').is_file()
    output['cases'][str(i)]=row
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(output).encode()).decode()+'B1KPROBE_END')
