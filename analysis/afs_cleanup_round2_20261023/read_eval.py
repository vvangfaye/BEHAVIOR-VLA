import pathlib,json,base64
p=pathlib.Path("/tmp/b1k_afs_round2_eval.json")
r=json.loads(p.read_text()) if p.exists() else {"missing":True}
print("B1KRUNTIMEBEGIN"+base64.b64encode(json.dumps(r).encode()).decode()+"B1KRUNTIMEEND",flush=True)
