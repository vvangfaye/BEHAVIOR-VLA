"""Release eval fan-out only after the corrected case executes real controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from deployment import HERE, ADAPT, ENV, aoss_client, save


def main():
    deadline=time.time()+1800
    marker=HERE/'seed_fix_canary_verified.json'
    plan=json.loads((HERE/'jobs/e-original-301s1-a2/worker_plan.json').read_text())
    receipt=json.loads((HERE/'jobs/e-original-301s1-a2/submission.json').read_text())
    assert plan['seed']==plan['model_seed']==0 and plan['evaluation']['cases']==[[301,1]]
    client=aoss_client()
    environment={**ENV,'PATH':'/home/faye/.local/bin:'+ENV.get('PATH','')}
    try:
        while time.time()<deadline and not marker.exists():
            try:
                status=json.loads(client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'status.json')['Body'].read())
                if status['state']=='failed':
                    save(HERE/'canary_watch_failure.json',status);return
                if status.get('stage') not in ('baseline','evaluation','evaluation_complete'):
                    time.sleep(20);continue
                output=HERE/'canary_live_probe.json'
                result=subprocess.run([str(Path(sys.executable).absolute()),
                    str(ADAPT/'forensics_20261023/exec_probe.py'),
                    '--job','b1k-water-owned-1101-a1','--uid','022086aa475a4d81b129974ff8285e0a',
                    '--script',str(HERE/'canary_probe.py'),'--output',str(output),'--timeout','45'],
                    env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=55)
                if result.returncode==0:
                    data=json.loads(output.read_text());files=data['files']
                    identity=files.get('identities/cmp_baseline_r0_a2.json',{})
                    spec=files.get('baseline/i301_s1/spec.json',{})
                    progress=files.get('baseline/i301_s1/progress.json',{})
                    if progress.get('steps',0)>0:
                        assert identity['loaded_policy_hash']==plan['source_identity']['cold_trainable_hash']
                        assert identity['frozen_hash']==plan['source_identity']['frozen_hash']
                        assert spec['seed']==1 and spec['instance']==301 and spec['task']==plan['task']
                        save(marker,{'at':time.time(),'job':receipt['job'],'uid':receipt['uid'],
                            'complete_episode':False,'real_controls_verified':True,'steps':progress['steps'],
                            'model_seed':0,'evaluation_seed':1,'loaded_policy_hash':identity['loaded_policy_hash']})
                        return
            except Exception as error:
                save(HERE/'canary_watch_error.json',{'at':time.time(),'type':type(error).__name__})
            time.sleep(30)
    finally:client.close()


if __name__=='__main__':main()
