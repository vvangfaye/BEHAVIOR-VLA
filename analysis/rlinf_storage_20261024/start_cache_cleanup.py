import pathlib,sys,json,subprocess,time,base64,zlib,shlex,hashlib
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
CLI='/home/faye/.local/bin/sco';NAME='b1k-storage-clear-1024';PREFIX='behavior1k/storage-reclaim-20261024/cache-clear01/'
# Verify all potentially writing RLinf jobs, by exact identity.
ids={'b1k-ppo5-1023-r01':'0a97d630ec844e61b6bec78401f3bf6a','b1k-ppoc2-1024-a01':'9b71a258461e47fcb40e1b54e81e5269','b1k-ppoc2-1024-a02':'5bd69f460a70452bb2da906d313af78c','b1k-rlinf-1023-r07':'cc2919cc6b5045d3ad07009a842762d7'}
audit=[]
for name,uid in ids.items():
 x=json.loads(subprocess.run([CLI,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=40).stdout)
 assert x['uid']==uid and x['ownership']['user_name']=='260010165' and x['state'] in ['FAILED','SUCCEEDED','SUSPENDED','TERMINATED','CANCELED']
 audit.append({k:x.get(k) for k in ['name','uid','state','complete_time']})
audit_raw=json.dumps(audit,sort_keys=True).encode();(HERE/'cleanup_dependencies.json').write_bytes(audit_raw)
roots=['/mnt/afs/260010165/behavior_1k/'+n for n in ['rlinf_multiround_20261023','rlinf_adapter_20261023','rlinf_conservative_20261024']]
plan={'roots':roots,'cutoff_ns':time.time_ns()-600*10**9,'max_allocated_bytes':300*1024**3,'deadline_seconds':1500,'active_dependency_audit_sha256':hashlib.sha256(audit_raw).hexdigest(),'delete_only':'regular single-link files under cache, warp or appdata in stopped RLinf runs','preserve_all_research_artifacts':True,'cache_reconstruction':'Retained simulator runtime recompiles these files when needed','content_backup':False,'user_context':'User requests relief of exhausted storage caused by these RLinf experiments; only regenerable caches are cleared.'}
(HERE/'cache_cleanup_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
client=aoss_client()
try:
 def signed(method,key):return client.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=14400)
 config={**plan,'manifest_key':PREFIX+'MANIFEST.json.gz','manifest_put':signed('put_object',PREFIX+'MANIFEST.json.gz'),'manifest_get':signed('get_object',PREFIX+'MANIFEST.json.gz'),'status_put':signed('put_object',PREFIX+'STATUS.json')}
 helper=(HERE/'archive_helpers.py').read_text();source=(HERE/'delete_regenerable_cache.py').read_text().replace('from archive_helpers import request,encode\n','')
 source=helper+'\n'+source+'\ntry:\n run(CONFIG)\nexcept BaseException as error:\n request(CONFIG["status_put"],"PUT",payload=encode({"state":"failed","error_type":type(error).__name__,"error":str(error) if "://" not in str(error) else "network error; URL omitted"}))\n raise\n'
 source_key=PREFIX+'worker-'+hashlib.sha256(source.encode()).hexdigest()+'.py';config_key=PREFIX+'private-config.json'
 client.put_object(Bucket='260010165',Key=source_key,Body=source.encode());client.put_object(Bucket='260010165',Key=config_key,Body=json.dumps(config).encode())
 bootstrap='import json,urllib.request;op=urllib.request.build_opener(urllib.request.ProxyHandler({}));ns={"CONFIG":json.load(op.open('+repr(signed('get_object',config_key))+'))};exec(compile(op.open('+repr(signed('get_object',source_key))+').read(),"bounded_cache_cleanup","exec"),ns)'
 command='timeout --signal=TERM --kill-after=15s 1800s python3 -u -c '+shlex.quote(bootstrap)
 receipt={'job':NAME,'state':'submitting','gpu_count':0,'cpu':2,'memory_gib':8,'max_seconds':1800,'status_key':PREFIX+'STATUS.json','manifest_key':PREFIX+'MANIFEST.json.gz','source_sha256':hashlib.sha256(source.encode()).hexdigest()}
 with (HERE/'cache_cleanup_submission.json').open('x') as f:json.dump(receipt,f,indent=2)
 p=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=pt-i8qy0bow','--name='+NAME,'--job-name='+NAME,'--worker-spec=n6ls.iu.i40.2c8g','--worker-nodes=1','--quota-type=reserved','--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--container-image-url=registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634','--command='+command],capture_output=True,text=True,timeout=90)
 receipt['returncode']=p.returncode
 x=json.loads(subprocess.run([CLI,'acp','jobs','describe',NAME,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=40).stdout)
 receipt.update(uid=x['uid'],state=x['state'])
 (HERE/'cache_cleanup_submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
finally:client.close()
