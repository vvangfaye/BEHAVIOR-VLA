import base64,json,pathlib
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/checkpoints/original1688977')
partial=root.with_name(root.name+'.partial-restore01')
p=root if root.exists() else partial
files=[{'name':str(f.relative_to(p)),'bytes':f.stat().st_size} for f in p.rglob('*') if f.is_file()] if p.exists() else []
x={'complete_directory':root.exists(),'file_count':len(files),'written_bytes':sum(f['bytes'] for f in files)}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(x).encode()).decode()+'B1KPROBE_END',flush=True)
