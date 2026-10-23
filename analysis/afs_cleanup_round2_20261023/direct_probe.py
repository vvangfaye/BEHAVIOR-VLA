import pathlib,os,json,base64,stat
base=pathlib.Path('/mnt/afs/260010165/behavior_1k/eval_results')
roots=[p/'local-results-sim-runtime' for p in base.iterdir() if (p/'local-results-sim-runtime').is_dir()]
rows=[]
for root in roots:
 entries=[]
 for p in root.iterdir():
  entries.append({'name':p.name,'dir':p.is_dir(),'link':os.readlink(p) if p.is_symlink() else None,'size':p.lstat().st_size})
 rows.append({'root':str(root),'entries':entries})
result={'rows':rows}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
