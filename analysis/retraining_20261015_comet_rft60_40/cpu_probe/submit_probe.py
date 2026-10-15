import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'bundle'))
from nebula.submit_oss import OSSClient
p=Path(sys.argv[1]);cfg=json.loads((p/'bundle/probe_job.json').read_text());client=OSSClient();key=client.key(cfg['output']);client.unused(key)
intent=p/'intent.json'
with intent.open('x') as f:json.dump({'run_id':cfg['run_id'],'observed_at':time.time(),'purpose':'real_two_source_CPU_batches_no_model_or_optimizer','gpus':0,'training':False},f)
client.put_new(key+'.claim.json',intent)
secrets=[os.environ[k] for k in ('AI_HUB_TOKEN','OSS_ACCESS_ID','OSS_ACCESS_KEY')]
args=['ai-hub-cli','train','mdl','--token='+secrets[0],'--namespace='+os.environ['AI_HUB_NAMESPACE'],'--name=behavior','--queue=demand_general','--entry=four_source_cpu_probe.py','--worker_count=1','--file.cluster_file=probe_cluster.json','--oss_access_id='+secrets[1],'--oss_access_key='+secrets[2],'--oss_bucket=traffic-model','--oss_endpoint='+os.environ['OSS_ENDPOINT'],'--env=OSS_ACCESS_KEY_ID='+secrets[1]+',OSS_ACCESS_KEY_SECRET='+secrets[2],'--job_name='+cfg['run_id'],'--algo_name=pytorch2120','--gpu_card_type=l20z','--oss_appendable=true','--max_failover_times=0','--debug_mode']
r={'run_id':cfg['run_id'],'output':cfg['output'],'state':'unknown','gpus':0,'training':False}
try:
    result=subprocess.run(args,cwd=p/'bundle',capture_output=True,text=True,timeout=120)
    text=result.stdout+result.stderr
    for secret in secrets:text=text.replace(secret,'[REDACTED]')
    (p/'submission.log').write_text(text)
    r.update(state='submitted' if result.returncode==0 else 'failed_or_unknown',exit_code=result.returncode,identifiers=sorted(set(re.findall(r'(?i)\b(?:iid|instanceId|instance_id|task_id)\W+([A-Za-z0-9_-]+)',text))))
finally:
    (p/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
