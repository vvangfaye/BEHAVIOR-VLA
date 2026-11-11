import sys,json,os
from pathlib import Path
import boto3
from botocore.config import Config
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
c=aoss_client();cred=c._request_signer._credentials.get_frozen_credentials();i=boto3.client('s3',endpoint_url='https://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn',region_name='cn-sh-01b',aws_access_key_id=cred.access_key,aws_secret_access_key=cred.secret_key,config=Config(signature_version='s3v4',s3={'addressing_style':'path'},proxies={}))
p=json.loads((HERE/'storage_plan.json').read_text());checks=[]
for uid,position in [('checkpoint01r4',21),('checkpoint02r2r4',20),('checkpoint03r2r4',0)]:
 u=next(x for x in p['units'] if x['id']==uid);f=u['files'][position];key=f.get('existing_object_key',u['prefix']+'/files/'+f['path']);checks.append({'id':uid,'url':i.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=600),'range':f['size']>8*1024**2})
s='''import http.client,urllib.parse,ssl,json,base64\nrows=[]\nfor item in CHECKS:\n try:\n  u=urllib.parse.urlsplit(item['url']);c=http.client.HTTPSConnection(u.hostname,timeout=15,context=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt'));headers={'Cache-Control':'no-cache'}\n  if item['range']:headers['Range']='bytes=0-8388607'\n  c.request('GET',u.path+'?'+u.query,headers=headers);r=c.getresponse();row={'id':item['id'],'status':r.status,'headers':{k:r.getheader(k) for k in ['Content-Range','Content-Length','Via','X-Cache','Content-Encoding']}}\n  row['read_bytes']=len(r.read());c.close();rows.append(row)\n except Exception as e:rows.append({'id':item['id'],'error':type(e).__name__,'message':str(e) if 'http' not in str(e) else 'http error'})\nprint('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps({'checks':rows}).encode()).decode()+'B1KRUNTIMEEND',flush=True)\n'''.replace('CHECKS',repr(checks))
path=Path('/tmp/afs-network-probe.py');fd=os.open(path,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as out:out.write(s)
c.close();i.close()
