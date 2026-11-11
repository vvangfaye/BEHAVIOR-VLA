import subprocess,sys,json,base64,pathlib,time
result_path='/tmp/b1k_afs_usage_20261023.json'
code="""import subprocess,json,time,os,pathlib
p=subprocess.run(['timeout','300','du','-x','-B1','--max-depth=1','/mnt/afs/260010165'],capture_output=True,text=True)
s=os.statvfs('/mnt/afs/260010165');r={'time':time.time(),'returncode':p.returncode,'du':p.stdout,'errors':p.stderr,'shared_total':s.f_blocks*s.f_frsize,'shared_available':s.f_bavail*s.f_frsize}
pathlib.Path('/tmp/b1k_afs_usage_20261023.json').write_text(json.dumps(r))
"""
p=subprocess.Popen([sys.executable,'-c',code],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
r={'pid':p.pid,'result_path':result_path}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KRUNTIMEEND',flush=True)
