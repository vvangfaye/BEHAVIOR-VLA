"""Copy trusted CPU-only committed-file measurements to the capacity gate."""
import json
import os
import subprocess
import time
from control import HERE, CAMPAIGN, aoss_client, save, CLI, ENV, describe


def main():
    receipt=json.loads((HERE/'owned_storage_submission.json').read_text())
    plan=json.loads((CAMPAIGN/'water/plan.json').read_text())
    client=aoss_client();deadline=time.time()+24*3600
    try:
        while time.time()<deadline:
            if (CAMPAIGN/'water/outcome.json').exists():
                job=describe(receipt['name']);assert job['uid']==receipt['uid']
                if job['state'] not in {'SUCCEEDED','FAILED','SUSPENDED','STOPPED'}:
                    subprocess.run([CLI,'acp','jobs','stop',receipt['name'],'--workspace-name=share-space'],
                                   env=ENV,check=True,capture_output=True,timeout=45)
                return
            try:
                response=client.get_object(Bucket='260010165',Key=receipt['result_key'])
                assert response['ContentLength']<1024**2
                value=json.loads(response['Body'].read())
                assert value['experiment']==plan['experiment'] and value['output']==plan['output']
                assert value['method']=='committed_regular_files_no_external_hardlinks'
                assert 0<=value['committed_bytes']<=plan['runtime_storage']['max_persistent_bytes']
                assert 0<=time.time()-value['observed_at']<180
                save(CAMPAIGN/'water/owned_storage_latest.json',value)
            except Exception as error:
                save(HERE/'owned_storage_watch_error.json',{'at':time.time(),'type':type(error).__name__})
            time.sleep(30)
    finally:
        client.close()


if __name__=='__main__':
    for key in list(os.environ):
        if key.lower() in {'http_proxy','https_proxy','all_proxy'}:os.environ.pop(key)
    main()
