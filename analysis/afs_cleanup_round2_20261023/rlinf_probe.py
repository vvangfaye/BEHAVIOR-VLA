import pathlib,os,re,json,base64
out=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  text=(p/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')
  paths=sorted(set(re.findall(r'/mnt/afs/260010165/[A-Za-z0-9_./-]+',text)))
  if not paths:continue
  refs=[]
  for f in (p/'fd').iterdir():
   try:
    v=os.readlink(f)
    if v.startswith('/mnt/afs/260010165'):refs.append(v)
   except OSError:pass
  out.append({'pid':p.name,'paths':paths,'fd_paths':sorted(set(refs))[:200]})
 except OSError:pass
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps({'processes':out}).encode()).decode()+'B1KRUNTIMEEND',flush=True)
