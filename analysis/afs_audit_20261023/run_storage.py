"""Send a reviewed storage plan to the existing worker without persisting signed URLs."""
import argparse,base64,hashlib,json,os,re,select,shlex,subprocess,sys,time,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--action',choices=['archive','release'],required=True);parser.add_argument('--units',nargs='+',required=True);parser.add_argument('--job');parser.add_argument('--uid');args=parser.parse_args()
 plan=json.loads((HERE/'storage_plan.json').read_text());config={k:v for k,v in plan.items() if k!='units'};config['action']=args.action;config['allow_mount_rebind']=True
 config['units']=[x for x in plan['units'] if x['id'] in args.units];assert len(config['units'])==len(args.units)
 c=aoss_client()
 def signed(method,key):return c.generate_presigned_url(method,Params={'Bucket':plan['bucket'],'Key':key},ExpiresIn=10800)
 for unit in config['units']:
  for f in unit['files']:
   f['put_url']=signed('put_object',unit['prefix']+'/files/'+f['path']);f['get_url']=signed('get_object',f.get('existing_object_key',unit['prefix']+'/files/'+f['path']))
  for label,name in [('manifest','MANIFEST.json'),('receipt','VERIFIED.json'),('complete','COMPLETE.json'),('release','RELEASED.json')]:
   unit[label+'_put']=signed('put_object',unit['prefix']+'/'+name);unit[label+'_get']=signed('get_object',unit['prefix']+'/'+name)
 c.close()
 script=(HERE/'storage_worker.py').read_text()+'''\nimport base64\ntry:\n result=run_all(CONFIG)\nexcept Exception as error:\n result={'error_type':type(error).__name__,'error':str(error) if 'https' not in str(error) else 'HTTP transport failed; URL omitted'}\nprint('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)\n'''.replace('run_all(CONFIG)','run_all('+repr(config)+')')
 packed=base64.b64encode(zlib.compress(script.encode(),6)).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')+'\nexit\n'
 assert len(command)<120000
 job=({'name':args.job,'uid':args.uid} if args.job and args.uid else plan['worker']);p=subprocess.Popen(['sco','acp','jobs','exec',job['name'],'--workspace-name=share-space','--worker-name=pt-'+job['uid']+'-worker-0'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 pending=b'';sent=False;start=time.monotonic();progress=[]
 poll_client=aoss_client() if args.action=='release' else None;last_poll=0;complete_seen=None;cloud_result=None
 try:
  while time.monotonic()-start<plan['max_seconds']+180:
   if poll_client and time.monotonic()-last_poll>10:
    last_poll=time.monotonic();released=[]
    for unit in config['units']:
     try:r=poll_client.get_object(Bucket=plan['bucket'],Key=unit['prefix']+'/RELEASED.json')
     except poll_client.exceptions.NoSuchKey:continue
     try:item=json.loads(r['Body'].read())
     finally:r['Body'].close()
     assert item['source']==unit['source'] and item['archive_id']==plan['archive_id'] and item['released_files']==len(unit['files'])
     released.append(item)
    if len(released)==len(config['units']):
     cloud_result={'action':'release','archive_id':plan['archive_id'],'units':released,'receipt_source':'AOSS verified release receipts'}
     if complete_seen is None:complete_seen=time.monotonic()
   if cloud_result and time.monotonic()-complete_seen>3:
    dest=HERE/(args.action+'_'+'_'.join(args.units)+'.json');dest.write_text(json.dumps(cloud_result,indent=2)+'\n');print(json.dumps(cloud_result),flush=True);return
   ready,_,_=select.select([p.stdout],[],[],1)
   if not ready:continue
   block=os.read(p.stdout.fileno(),65536)
   if not block:break
   pending+=block;pending=pending.translate(None,b'\x00\x01\x02\x03\x04')
   if not sent and re.search(rb'(?:# |\$ )',pending):p.stdin.write(command.encode());p.stdin.flush();sent=True;pending=b''
   match=re.search(rb'B1KRUNTIMEBEGIN([A-Za-z0-9+/=]+)B1KRUNTIMEEND',pending)
   if match:
    value=json.loads(base64.b64decode(match.group(1)));dest=HERE/(args.action+'_'+'_'.join(args.units)+'.json');dest.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
    if 'error_type' in value:raise RuntimeError('Remote storage operation failed; see sanitized receipt')
    return
   while b'\n' in pending:
    line,pending=pending.split(b'\n',1);pos=line.find(b'STORAGE_PROGRESS ')
    if pos>=0:
     try:value=json.loads(line[pos+len(b'STORAGE_PROGRESS '):])
     except Exception:continue
     progress.append(value);(HERE/'live_progress.json').write_text(json.dumps({'action':args.action,'latest':value,'events':progress},indent=2)+'\n');print(json.dumps(value),flush=True)
   if len(pending)>2*1024**2:raise RuntimeError('Output limit exceeded')
  raise RuntimeError('Worker transport ended without final receipt; source cleanup status requires inspection')
 finally:
  if poll_client:poll_client.close()
  p.terminate()
  try:p.wait(timeout=3)
  except subprocess.TimeoutExpired:p.kill();p.wait()
if __name__=='__main__':main()
