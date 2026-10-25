import argparse,base64,json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--iteration',type=int,default=1);iteration=parser.parse_args().iteration
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
item=json.loads((HERE/'python_dependencies.json').read_text())
c=aoss_client()
try:
    config={'wheel_get':c.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':item['key']},ExpiresIn=7200),
            'wheel_sha256':item['sha256'],'iteration':iteration,
            'put':c.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':f'behavior1k/serf-reproduction-20261025/dev01/iteration{iteration:02d}.json'},ExpiresIn=7200)}
finally:c.close()
source=(HERE/'repair_imports_worker.py').read_text()+'\nrun('+repr(config)+')\n'
encoded=base64.b64encode(source.encode()).decode()
script='''import base64,json,subprocess
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/dev01')
previous=json.loads((p/'result.json').read_text())
assert set(previous['probes'])=={'sim','policy'}
script=p/'REPAIR_SCRIPT'
assert not script.exists()
script.write_bytes(base64.b64decode(ENCODED))
log=(p/'REPAIR_LOG').open('w')
process=subprocess.Popen(['python3','-u',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'pid':process.pid,'log':str(p/'REPAIR_LOG')}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KPROBE_END')
'''.replace('ENCODED',repr(encoded)).replace('REPAIR_SCRIPT',f'repair_{iteration:02d}.py').replace('REPAIR_LOG',f'repair_{iteration:02d}.log')
(HERE/'_repair_submit_private.py').write_text(script)
(HERE/'_repair_submit_private.py').chmod(0o600)
print('Prepared private repair submission; no credentials printed')
