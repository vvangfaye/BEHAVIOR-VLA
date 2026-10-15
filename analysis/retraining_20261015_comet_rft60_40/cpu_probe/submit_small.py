"""One bounded CPU-only AIHub creation after proven upload-before-create failure."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent


def main():
    failed=json.loads((HERE/'receipt.json').read_text())
    log=(HERE/'submission.log').read_text()
    assert failed['state']=='failed_or_unknown' and not failed['identifiers']
    assert '上传OSS失败' in log and '开始上传OSS' in log
    assert '创建任务成功' not in log and '任务提交成功' not in log
    output=json.loads((HERE/'bundle/probe_job.json').read_text())['output']
    receipt_path=HERE/'small_submit_receipt.json'
    intent=dict(run_id='b1k-two-source-cpu-20261015-1624',created_at=time.time(),gpus=0,training=False,
                method='small_bootstrap_after_upload_before_create_failure',maximum_creations=1,automatic_resubmit=False)
    with (HERE/'small_submit_intent.json').open('x') as stream:json.dump(intent,stream,indent=2)
    secret=[os.environ[k] for k in ['AI_HUB_TOKEN','OSS_ACCESS_ID','OSS_ACCESS_KEY']]
    args=['/Users/faye/.local/bin/ai-hub-cli','train','mdl','--token='+secret[0],
        '--namespace='+os.environ['AI_HUB_NAMESPACE'],'--name=behavior','--queue=demand_general',
        '--entry=bootstrap.py','--worker_count=1','--file.cluster_file=probe_cluster.json',
        '--oss_access_id='+secret[1],'--oss_access_key='+secret[2],'--oss_bucket=traffic-model',
        '--oss_endpoint='+os.environ['OSS_ENDPOINT'],
        '--env=OSS_ACCESS_KEY_ID='+secret[1]+',OSS_ACCESS_KEY_SECRET='+secret[2],
        '--job_name='+intent['run_id'],'--algo_name=pytorch2120','--gpu_card_type=l20z',
        '--oss_appendable=true','--max_failover_times=0','--debug_mode']
    result=dict(intent,state='unknown',output=output)
    try:
        cp=subprocess.run(args,cwd=HERE/'small_submit',capture_output=True,text=True,timeout=120)
        text=cp.stdout+cp.stderr
        for value in secret:text=text.replace(value,'[REDACTED]')
        (HERE/'small_submission.log').write_text(text)
        result.update(state='submitted' if cp.returncode==0 else 'failed_or_unknown',exit_code=cp.returncode,
            identifiers=sorted(set(re.findall(r'(?i)\b(?:iid|instanceId|instance_id|task_id)\W+([A-Za-z0-9_-]+)',text))))
    except Exception as e:result['error_type']=type(e).__name__
    finally:receipt_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
