import pathlib,os,signal,json,base64,time
pid=5002
p=pathlib.Path('/proc')/str(pid)
expected='/mnt/afs/260010165/behavior_1k/quality_taskbalanced_fixed20_20261016/checkpoint5000/'
matched=[]
for f in (p/'fd').iterdir():
 try:
  if str(f.readlink()).startswith(expected):matched.append(str(f.readlink()))
 except OSError:pass
assert matched and (p/'comm').read_text().strip().startswith('python')
os.kill(pid,signal.SIGTERM)
result={'stopped_storage_pid':pid,'matched_source':expected,'time':time.time(),'reason':'Reconfigure only our archive worker for independent concurrent uploads; source unchanged'}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
