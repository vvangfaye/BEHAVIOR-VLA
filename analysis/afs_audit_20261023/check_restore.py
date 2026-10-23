import hashlib,importlib.util,io,json,pathlib,tempfile
spec=importlib.util.spec_from_file_location('restore',pathlib.Path(__file__).with_name('restore_archive.py'));r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def encoded(x):return json.dumps(x).encode()
payload=b'known archived model bytes';manifest=encoded({'files':[{'path':'params/test','size':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'mode':0o640,'existing_object_key':'old-archive/model'}]});receipt=b'verified'
objects={'new/COMPLETE.json':encoded({'manifest_sha256':hashlib.sha256(manifest).hexdigest(),'receipt_sha256':hashlib.sha256(receipt).hexdigest()}),'new/MANIFEST.json':manifest,'new/VERIFIED.json':receipt,'old-archive/model':payload}
class Body(io.BytesIO):
 def iter_chunks(self,chunk_size):
  while True:
   v=self.read(chunk_size)
   if not v:return
   yield v
class Client:
 def get_object(self,**kwargs):return {'Body':Body(objects[kwargs['Key']])}
 def close(self):pass
r.client=lambda:Client()
with tempfile.TemporaryDirectory() as t:
 import sys
 dest=pathlib.Path(t)/'restored';sys.argv=['restore','--prefix','new','--destination',str(dest)];r.main();assert (dest/'params/test').read_bytes()==payload
print('PASS: restoration verifies metadata and follows referenced existing object keys')
