import pathlib,json,base64,shutil
root=pathlib.Path('/mnt/afs/260010165/behavior_1k')
base=root/'runtime_20261007_1306/omnigibson-data/2026-challenge-task-instances/scenes'
instances={}
for i in [1,2,3,4]:
 matches=list(base.glob(f'*/json/*turning_on_radio*/*turning_on_radio_0_{i}_template-tro_state.json'))
 instances[str(i)]=[str(p) for p in matches]
old=root/'rlinf_multiround_20261023/pilot01_reserved/results'
result={'training_instances':instances,'free_bytes':shutil.disk_usage(root).free,'previous_files':{str(p.relative_to(old)):p.stat().st_size for p in [old/'checkpoints/round_05/trainable.pt']+[old/'training'/f'round_{i:02d}'/'rollout.pt' for i in range(1,6)]}}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KPROBE_END',flush=True)
