import pathlib,os,json,base64
base=pathlib.Path('/mnt/afs/260010165/behavior_1k');result={}
for rel in ['eval_results/b1k-video1688977-20261013','runtime_20261007_1306/inputs','eval_results/b1k-video2m-20261013','eval_results/behavior-legacy-history-eval-20261008-140818']:
 rows=[]
 for p in (base/rel).iterdir():
  s=p.lstat();rows.append({'name':p.name,'dir':p.is_dir(),'link':p.is_symlink(),'size':s.st_size})
 result[rel]=rows
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
