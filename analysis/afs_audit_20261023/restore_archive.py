"""Restore one archived unit to a NEW directory; never overwrite an existing tree."""
import argparse,ast,hashlib,json,os,re,shutil
from pathlib import Path
import boto3
from botocore.config import Config

def client():
 profile=Path(os.environ.get('SCO_CONFIG',str(Path.home()/'.config/sco')))/'profiles/default.toml'
 values={}
 if profile.is_file():
  values={k:ast.literal_eval(v) for k,v in re.findall(r'^(access_key_id|access_key_secret)\s*=\s*([\"\'].*[\"\'])\s*$',profile.read_text(),re.M)}
 return boto3.client('s3',endpoint_url='https://aoss.cn-sh-01b.sensecoreapi-oss.cn',region_name='cn-sh-01b',aws_access_key_id=values.get('access_key_id'),aws_secret_access_key=values.get('access_key_secret'),config=Config(signature_version='s3v4',s3={'addressing_style':'path'},proxies={}))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prefix',required=True);ap.add_argument('--destination',required=True);a=ap.parse_args()
 dest=Path(a.destination).resolve();assert not dest.exists(),'Destination must not exist'
 c=client();bucket='260010165';prefix=a.prefix.rstrip('/')
 def get(key):
  r=c.get_object(Bucket=bucket,Key=prefix+'/'+key)
  try:return r['Body'].read()
  finally:r['Body'].close()
 complete=json.loads(get('COMPLETE.json'));raw=get('MANIFEST.json');assert hashlib.sha256(raw).hexdigest()==complete['manifest_sha256'];manifest=json.loads(raw)
 assert hashlib.sha256(get('VERIFIED.json')).hexdigest()==complete['receipt_sha256']
 need=sum(f['size'] for f in manifest['files']);assert shutil.disk_usage(dest.parent).free>need+1024**3,'Insufficient filesystem space; also check personal AFS quota before restoring'
 dest.mkdir()
 for f in manifest['files']:
  relative=Path(f['path']);assert not relative.is_absolute() and '..' not in relative.parts
  path=dest/relative;path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0;r=c.get_object(Bucket=bucket,Key=f.get('existing_object_key',prefix+'/files/'+f['path']))
  try:
   with path.open('xb') as out:
    for chunk in r['Body'].iter_chunks(chunk_size=8*1024**2):out.write(chunk);h.update(chunk);count+=len(chunk)
  finally:r['Body'].close()
  assert count==f['size'] and h.hexdigest()==f['sha256'],'Restored file failed verification'
  path.chmod(f.get('mode',0o644))
 (dest/'RESTORE_VERIFIED.json').write_text(json.dumps({'prefix':prefix,'manifest_sha256':complete['manifest_sha256'],'files':len(manifest['files']),'bytes':need},indent=2)+'\n')
 c.close();print('Verified restoration:',dest)
if __name__=='__main__':main()
