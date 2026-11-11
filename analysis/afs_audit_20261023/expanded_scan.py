import base64,json,pathlib,subprocess,concurrent.futures,shutil,os
root=pathlib.Path('/mnt/afs/260010165'); p=root/'behavior_1k'
targets=[p/'retraining_20261013',p/'runtime_20261007_1306',p/'eval_results',root/'cache',root/'models',root/'behavior-train-local']
def scan(t):
 q=subprocess.run(['timeout','25','du','-x','-B1','--max-depth=1',str(t)],capture_output=True,text=True)
 return {'path':str(t),'code':q.returncode,'du':q.stdout,'errors':q.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(scan,targets))
meta=[]
for n in ['expert_fixed20_step1000_20261016/checkpoint1000','paired_quick20_reused1688977_step5000_20261014/checkpoint5000']:
 t=p/n; files=[]
 for d,ds,fs in os.walk(t):
  for name in fs:
   f=pathlib.Path(d)/name;s=f.lstat();files.append({'path':str(f.relative_to(t)),'bytes':s.st_size,'nlink':s.st_nlink})
 meta.append({'root':str(t),'files':files})
result={'rows':rows,'checkpoint_samples':meta,'tools':{n:shutil.which(n) for n in ['ads-cli','quota','getfattr','quarkfs','afs','curl']},'mounts':[l for l in pathlib.Path('/proc/mounts').read_text().splitlines() if '/mnt/afs' in l]}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
