import pathlib,json,base64
r={}
for pid in [2506, 3783]:
 p=pathlib.Path("/proc")/str(pid)
 r[str(pid)]={"exists":p.exists()}
 if p.exists():
  r[str(pid)]["stat"]=p.joinpath("stat").read_text()
  r[str(pid)]["fds"]=[str(x)+" "+str(x.resolve()) for x in p.joinpath("fd").iterdir()]
print("B1KRUNTIMEBEGIN"+base64.b64encode(json.dumps(r).encode()).decode()+"B1KRUNTIMEEND",flush=True)
