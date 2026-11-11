import pathlib,os,json,base64,subprocess,concurrent.futures,time
root=pathlib.Path('/mnt/afs/260010165')
attrs={}
for p in [root,root.parent]:
 try:
  attrs[str(p)]={name:os.getxattr(p,name).decode(errors='replace')[:2048] for name in os.listxattr(p) if any(s in name.lower() for s in ['quota','usage','used','limit','capacity'])}
 except OSError as e:attrs[str(p)]={'error':str(e)}
paths=list(root.iterdir());
def scan(p):
 q=subprocess.run(['timeout','35','du','-x','-s','-B1',str(p)],capture_output=True,text=True)
 return {'path':str(p),'code':q.returncode,'output':q.stdout,'error':q.stderr[:1500]}
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:rows=list(pool.map(scan,paths))
r={'time':time.time(),'root':str(root),'xattrs':attrs,'rows':rows,'statvfs':dict(zip(['block_size','fragment_size','blocks','blocks_free','blocks_available','files','files_free','files_available'],list(os.statvfs(root))[:8]))}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KRUNTIMEEND',flush=True)
