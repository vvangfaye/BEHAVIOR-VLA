import base64,csv,json,os
from pathlib import Path
runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
data=runtime/'omnigibson-data/2026-challenge-task-instances'
tasks=['picking_up_trash','bringing_water','hanging_pictures','installing_a_scanner','installing_a_fax_machine']
out={'metadata':{},'tasks':{},'metadata_files':[]}
for p in (data/'metadata').iterdir():
    out['metadata_files'].append({'name':p.name,'size':p.stat().st_size})
    if p.suffix=='.csv' and p.stat().st_size<10000000:
        rows=list(csv.DictReader(p.open()))
        out['metadata'][p.name]=[r for r in rows if any(v in tasks for v in r.values())]
for task in tasks:
    out['tasks'][task]={}
    for mode,folder,ids in [('train','scenes',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]),('test','scene_test/public',list(range(301,321)))]:
        root=data/folder
        found={}
        for p in root.glob(f'*/json/*{task}*/*{task}_0_*_template-tro_state.json'):
            suffix=p.name.split(task+'_0_')[-1].split('_')[0]
            if suffix.isdigit() and int(suffix) in ids:found[int(suffix)]=str(p)
        out['tasks'][task][mode]=found
stat=os.statvfs('/mnt/afs/260010165/behavior_1k')
out['filesystem_free_bytes']=stat.f_bavail*stat.f_frsize
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
