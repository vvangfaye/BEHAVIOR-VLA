import hashlib,io,json,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as tar:
    for path in sorted((HERE/'wheels').glob('*.whl')):
        data=path.read_bytes();item=tarfile.TarInfo(path.name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
data=stream.getvalue();digest=hashlib.sha256(data).hexdigest()
key='behavior1k/serf-reproduction-20261025/dependencies/python-'+digest+'.tar.gz'
c=aoss_client()
try:
    c.put_object(Bucket='260010165',Key=key,Body=data,Metadata={'sha256':digest})
    assert c.get_object(Bucket='260010165',Key=key)['Body'].read()==data
    item={'key':key,'sha256':digest,'bytes':len(data)}
    (HERE/'python_dependencies.json').write_text(json.dumps(item,indent=2))
    print(json.dumps(item))
finally:c.close()
