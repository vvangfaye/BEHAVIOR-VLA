import pathlib,json,time,base64
out={'observed_at':time.time(),'tasks':{}}
for task in ['trash','water']:
 root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_multitask_20261029')/task/'results'
 cases=[]
 for p in sorted((root/'baseline').glob('i*_s*')):
  item={'case':p.name}
  for name in ['progress.json','result.json','initialization.json']:
   f=p/name
   if f.is_file():
    try:item[name]={'mtime':f.stat().st_mtime,'data':json.loads(f.read_text())}
    except Exception as e:item[name]=str(e)
  cases.append(item)
 out['tasks'][task]=cases
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
