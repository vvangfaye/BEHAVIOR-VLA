"""One account-level disk check before a new task expands to parallel workers."""
import json
from pathlib import Path
import subprocess


def admit_storage(plan):
    guard=plan['storage_admission']
    assert guard['account_root']=='/mnt/afs/260010165'
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
