"""Free a completed task's large tensors after verified archival, when needed."""
import json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
PROJECT=ROOT.parents[3]
sys.path.insert(0,str(PROJECT/'analysis/serf_multitask_20261029'))
from cluster_helpers import submit,write,describe
from internal_storage import internal_aoss_client

def start_completed_archive(slug):
    task=ROOT.parent/slug
    assert (task/'outcome.json').is_file()
    assert json.loads((task/'outcome.json').read_text())['total_optimizer_steps']==17
    for p in (task/'jobs').glob('*/submission.json'):
        r=json.loads(p.read_text());d=describe(r['job'])
        assert d['uid']==r['uid'] and d['state'] in {'SUCCEEDED','FAILED','STOPPED','SUSPENDED','TERMINATED','CANCELED'}
    plan=json.loads((task/'plan.json').read_text());base=pathlib.Path(plan['output'])
    paths=[str(base/'training/round_01'/f'episode_{i:02d}'/'rollout.pt') for i in range(1,33)]
    paths += [str(base/'checkpoints/round_01'/n) for n in ['trainable.pt','optimizer.pt']]
    client=internal_aoss_client();prefix=f'behavior1k/storage-reclaim-20261030/completed-{slug}-a1/'
    def url(method,key):return client.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=86400)
    try:
        files=[]
        for i,path in enumerate(paths):
            key=prefix+f'files/{i:04d}/'+pathlib.Path(path).name
            item={'path':path,'key':key,'identity_at_start':True,'put':url('put_object',key),'get':url('get_object',key)}
            for typ in ['manifest','complete']:
                k=prefix+f'files/{i:04d}/{typ}.json'
                item[typ+'_put']=url('put_object',k);item[typ+'_get']=url('get_object',k)
            files.append(item)
        worker=(ROOT/'archive_worker_streaming.py').read_text()
        script=ROOT/f'archive_completed_{slug}_worker.py';script.write_text(worker)
        config={'output':f'/mnt/afs/260010165/behavior_1k/operational_cleanup/20261030_completed_{slug}',
                'result_key':prefix+'status.json','status_put':url('put_object',prefix+'status.json'),
                'helpers_source':(ROOT/'archive_helpers_streaming.py').read_text(),
                'files':files,'allowed_roots':[str(base)],'deadline_seconds':7000}
        return submit(f'b1k-rl-archive-{slug}-1030-a1',script,config,ROOT/f'archive_completed_{slug}_submission.json',7200,False)
    finally:client.close()
