import hashlib,importlib.util,pathlib
spec=importlib.util.spec_from_file_location('worker',pathlib.Path(__file__).with_name('storage_worker.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
data=(b'valid archive bytes'*1000000)[:17*1024**2];calls=[]
class Response:
 def __init__(self,start,end,status):self.start=start;self.end=end;self.status=status
 def getheader(self,k,default=None):return {'Content-Range':f'bytes {self.start}-{self.end}/{len(data)}','Content-Length':str(self.end-self.start+1)}.get(k,default)
 def read(self,n=-1):return data[self.start:self.end+1]
class Connection:
 def __init__(self,*args,**kwargs):pass
 def request(self,method,path,headers):
  self.start,self.end=map(int,headers['Range'][6:].split('-'));calls.append((self.start,self.end));self.status=503 if len(calls)==1 else 206
 def getresponse(self):return Response(self.start,self.end,self.status)
 def close(self):pass
w.http.client.HTTPSConnection=Connection
result=w.ranged_get('https://aoss.cn-sh-01b.sensecoreapi-oss.cn/test',{'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
assert result['size']==len(data) and len(calls)>3
try:w.ranged_get('https://aoss.cn-sh-01b.sensecoreapi-oss.cn/test',{'size':len(data),'sha256':'0'*64})
except RuntimeError:pass
else:raise AssertionError('Incorrect whole-file hash accepted')
print('PASS: transient range failure retried; ordered full-file SHA enforced')
