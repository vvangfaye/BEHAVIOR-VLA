import pathlib,sys,hashlib,base64,zlib,json,subprocess,tempfile,os,re
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
import yaml
CLI='/home/faye/.local/bin/sco';NAME='b1k-storage-cpu-1024';KEY='behavior1k/storage-reclaim-20261024/inventory01.json'
client=aoss_client()
listing=subprocess.run([CLI,'cci','apps','list','--workspace-name=share-space','--keyword=260010165','--page-size=100'],capture_output=True,text=True,check=True,timeout=40).stdout
assert not any(x in listing for x in ['RUNNING','STARTING','CREATING','PENDING','UPDATING','RECOVERING'])
source=(HERE/'inventory.py').read_text()
config={'put':client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':KEY},ExpiresIn=14400)}
code=source+'\nmain('+repr(config)+')\n';packed=base64.b64encode(zlib.compress(code.encode())).decode()
spec={'display_name':NAME,'resource_pool':{'name':'debug-mig-cluster','available_zone':'cn-sh-01g','vpc_id':'0198a254-1d71-7211-9a41-d5783abfe9aa'},'replicas':1,'template':{'containers':[{'name':'inventory','image_path':'registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630','resource_request':{'cpu':'2','memory':'8GiB','nvidia.com/mig-3g.40gb':'0'},'volume_mounts':[{'type':'PV_AFS','id':'01a04263-91e5-7603-bc01-c67e503da6b5','mount_path':'/mnt/afs','subdir':'/','zone':'cn-sh-01g'}],'command':['timeout','--signal=TERM','--kill-after=10','1200','python3','-u','-c','import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))']}],'init_containers':[],'resource_spec':{'name':'n6lv.n.i40.2c8g'}},'scheduling':{'priority':'NORMAL','quota_type':'RESERVED'},'rolling_update_strategy':{'max_surge':'0%','max_unavailable':'100%'},'termination_grace_period_seconds':15}
receipt={'name':NAME,'key':KEY,'gpu_count':0,'cpu':2,'memory_gib':8,'deadline_seconds':1200,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'scope':'Read-only inventory of three RLinf directories, plus temporary 4KiB write probe; no deletion.'}
(HERE/'inventory_submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
fd,path=tempfile.mkstemp(prefix='b1k-storage-',suffix='.yaml');os.fchmod(fd,0o600)
try:
 with os.fdopen(fd,'w') as f:yaml.safe_dump(spec,f)
 result=subprocess.run([CLI,'cci','apps','create',NAME,'--workspace-name=share-space','--config='+path],capture_output=True,text=True,timeout=90)
 receipt['returncode']=result.returncode
 if result.returncode:receipt['error']=re.sub(r'https?://\S+','[URL omitted]',result.stderr)[:1000]
 actual=subprocess.run([CLI,'cci','apps','describe',NAME,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,timeout=40,check=True)
 d=json.loads(actual.stdout);receipt.update(uid=d['uid'],state=d['state'])
finally:os.unlink(path);client.close()
(HERE/'inventory_submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
