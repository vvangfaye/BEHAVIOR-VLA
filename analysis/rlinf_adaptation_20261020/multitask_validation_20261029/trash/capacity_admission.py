"""One sequential training reservation; the other task may finish its baseline."""
import time
GIB=1024**3

def evidence(plan,quota,eligible,other_baseline_active):
    row=quota['rows'][0]
    reserve=plan['runtime_storage']['max_persistent_bytes']
    other=8*GIB if other_baseline_active else 0
    # Separate from this campaign's reserved outputs: tolerate concurrent writers.
    margin=20*GIB
    required=reserve+other+margin
    return {'complete':True,'passed':bool(eligible and row['quota_bytes']-row['used_bytes']>=required),
            'experiment':plan['experiment'],'account_root':'/mnt/afs/260010165',
            'account_allocated_bytes':row['used_bytes'],'quota_lower_bound_bytes':row['quota_bytes'],
            'new_task_reserve_bytes':reserve,'other_remaining_commitments_bytes':other,
            'headroom_margin_bytes':margin,'required_remaining_bytes':required,
            'observed_at':quota['observed_at'],'quota_source':quota['method'],
            'training_slot_eligible':eligible,'training_serialized':True,
            'baseline_parallelism_preserved':True,'no_directory_scan':True}

def acceptable(proof,plan,now=None):
    now=time.time() if now is None else now
    return bool(proof.get('complete') and proof.get('passed')
        and proof.get('experiment')==plan['experiment']
        and proof.get('account_root')=='/mnt/afs/260010165'
        and 0<=now-proof.get('observed_at',0)<=6*3600
        and proof.get('new_task_reserve_bytes',0)>=plan['runtime_storage']['max_persistent_bytes']
        and proof.get('account_allocated_bytes',0)+proof.get('required_remaining_bytes',0)<=proof.get('quota_lower_bound_bytes',0)
        and proof.get('required_remaining_bytes',0)>=proof.get('new_task_reserve_bytes',0)+proof.get('other_remaining_commitments_bytes',0)+proof.get('headroom_margin_bytes',0))
