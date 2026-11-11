"""One account-level disk check before a new task expands to parallel workers."""
import json
from pathlib import Path
import subprocess
import os
import time


def admit_storage(plan, baseline_probe=False):
    guard=plan['storage_admission']
    assert guard['account_root']=='/mnt/afs/260010165'
    if baseline_probe:
        root=Path(plan['output']);root.mkdir(parents=True,exist_ok=True)
        probe=root/f'.baseline_write_probe_{os.getpid()}'
        with probe.open('xb') as stream:
            stream.write(b'baseline-only-capacity-check');stream.flush();os.fsync(stream.fileno())
        probe.unlink()
        evidence={'passed':True,'scope':'baseline_only','full_training_clearance':False,
                  'shared_baseline_limit_bytes':8*1024**3,'observed_at':time.time()}
        (root/'baseline_storage_admission.json').write_text(json.dumps(evidence,indent=2)+'\n')
        return evidence
    proof=Path(plan['output'])/'storage_inventory.json'
    if proof.is_file():
        evidence=json.loads(proof.read_text())
        assert evidence['complete'] and evidence['experiment']==plan['experiment']
        assert evidence['account_root']==guard['account_root']
        assert 0<=time.time()-evidence['observed_at']<=6*3600,'Storage inventory is stale'
        assert evidence['new_task_reserve_bytes']>=plan['runtime_storage']['max_persistent_bytes']
        assert evidence['passed'],'Insufficient conservative account headroom'
        if 'required_remaining_bytes' in evidence:
            from capacity_admission import acceptable
            assert acceptable(evidence,plan),'Invalid shared capacity reservation'
        root=Path(plan['output']);probe=root/f'.write_probe_{os.getpid()}'
        with probe.open('xb') as stream:
            stream.write(b'capacity-check');stream.flush();os.fsync(stream.fileno())
        probe.unlink()
        (root/'storage_admission.json').write_text(json.dumps(evidence,indent=2)+'\n')
        return evidence
    if plan.get('experiment','').startswith('rlinf_multitask_'):
        raise RuntimeError('Waiting for CPU-only storage inventory; never scan the whole account on a GPU worker')
    result=subprocess.run(['nice','-n','15','du','-sx','--block-size=1',guard['account_root']],
                          capture_output=True,text=True,timeout=guard['scan_timeout_seconds'],check=True)
    used=int(result.stdout.split()[0])
    reserve=plan['runtime_storage']['max_persistent_bytes']
    evidence={'account_allocated_bytes':used,'new_task_reserve_bytes':reserve,
              'policy_quota_bytes':guard['quota_bytes'],
              'passed':used+reserve<=guard['quota_bytes']}
    root=Path(plan['output']);root.mkdir(parents=True,exist_ok=True)
    (root/'storage_admission.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if not evidence['passed']:
        raise RuntimeError('Account storage headroom is smaller than the frozen task reserve')
    return evidence
