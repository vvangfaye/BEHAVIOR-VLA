import subprocess,json,base64,time,os
root='/mnt/afs/260010165'
p=subprocess.run(['timeout','240','du','-x','-B1','--max-depth=1',root],capture_output=True,text=True)
s=os.statvfs(root)
r={'time':time.time(),'root':root,'returncode':p.returncode,'du':p.stdout,'errors':p.stderr,'shared_volume':{'total':s.f_blocks*s.f_frsize,'available':s.f_bavail*s.f_frsize}}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KRUNTIMEEND',flush=True)
