import os,pathlib,json,base64,subprocess,concurrent.futures
base=pathlib.Path('/mnt/afs/260010165/behavior_1k')
rels=['runtime_20261007_1306/inputs','runtime_20261007_1306/omnigibson-data','runtime_20261007_1306/envs','eval_results/b1k-video1688977-20261013','eval_results/native-full2000-1688977-20261012-v2','code_snapshots','codex_pilot_20261011']
def one(rel):
 p=base/rel
 q=subprocess.run(['timeout','35','du','-x','-B1','--max-depth=2',str(p)],capture_output=True,text=True)
 return {'path':str(p),'returncode':q.returncode,'du':q.stdout,'error':q.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:r=list(pool.map(one,rels))
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps({'rows':r}).encode()).decode()+'B1KRUNTIMEEND',flush=True)
