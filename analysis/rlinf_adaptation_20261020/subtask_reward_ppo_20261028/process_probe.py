import base64,json,os
from pathlib import Path
base=Path('/mnt/afs/260010165/behavior_1k/rlinf_midreward_20261028/pilot01')
out={'afs':Path('/mnt/afs/260010165').exists(),'root':base.exists(),'job_dirs':[str(x) for x in (base/'jobs').glob('*')],'processes':[]}
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  raw=path.read_bytes();args=raw.split(b'\0');name=os.path.basename(args[0].decode(errors='replace'))
  if name in ('python','python3','bash','sh','tini'):
   out['processes'].append({'pid':int(path.parent.name),'name':name,'bytes':len(raw),'argc':len(args),'encoded_bootstrap':b'exec(zlib.decompress' in raw})
 except (PermissionError,FileNotFoundError,ProcessLookupError):pass
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
