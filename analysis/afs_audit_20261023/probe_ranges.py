import sys,json,time,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
s=importlib.util.spec_from_file_location('worker',HERE/'storage_worker.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
p=json.loads((HERE/'storage_plan.json').read_text());u=p['units'][0];c=aoss_client();url=c.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':u['prefix']+'/files/assets/subtask_mixture.json'},ExpiresIn=600);expected=json.loads((HERE/'local_readback_probe.json').read_text());t=time.monotonic();result=w.request(url,'GET',expected={'size':expected['bytes'],'sha256':expected['sha256']});result['seconds']=time.monotonic()-t;(HERE/'range_readback_probe.json').write_text(json.dumps(result,indent=2));print(result);c.close()
