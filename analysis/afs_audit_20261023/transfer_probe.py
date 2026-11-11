import pathlib,json,base64,time
rows=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  for f in (p/'fd').iterdir():
   try:
    target=str(f.readlink())
    if '/mnt/afs/260010165/behavior_1k/' in target and '/checkpoint' in target:
     info=(p/'fdinfo'/f.name).read_text();pos=next((line.split(':')[1].strip() for line in info.splitlines() if line.startswith('pos:')),None)
     rows.append({'pid':p.name,'file':target,'pos':pos})
   except OSError:pass
 except OSError:pass
result={'time':time.time(),'fds':rows,'network':pathlib.Path('/proc/net/dev').read_text()}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
