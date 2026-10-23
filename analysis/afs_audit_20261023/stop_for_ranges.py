import pathlib,os,signal,json,base64,zlib,re,time
found=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if len(args)<3 or args[1]!=b'-c':continue
 command=args[2].decode(errors='ignore')
 match=re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)",command)
 if not match:continue
 try:source=zlib.decompress(base64.b64decode(match.group(1))).decode()
 except Exception:continue
 if '20261023-checkpoints-1792736315' in source and "'action': 'archive'" in source and 'result=run' in source:
  found.append(int(p.name))
for pid in found:os.kill(pid,signal.SIGTERM)
result={'stopped_own_archive_pids':found,'time':time.time(),'source_deleted':False}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
