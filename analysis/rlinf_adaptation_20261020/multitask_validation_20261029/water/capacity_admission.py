"""Reserve the unspent portion of a task's fixed persistent-storage budget."""
import json
from pathlib import Path
import time
GIB=1024**3


def _valid_owned(proof,plan,now):
    if not isinstance(proof,dict):return False
    amount=proof.get('committed_bytes')
    return (proof.get('experiment')==plan['experiment']
        and proof.get('output')==plan['output']
        and proof.get('method')=='committed_regular_files_no_external_hardlinks'
        and isinstance(amount,int) and not isinstance(amount,bool)
        and 0<=amount<=plan['runtime_storage']['max_persistent_bytes']
        and 0<=now-proof.get('observed_at',0)<=180)


def _owned(plan,now):
    folder=Path(__file__).resolve().parent
    if not (folder/'plan.json').exists():
        folder=folder/('water' if plan['task']=='bringing_water' else 'trash')
    try:proof=json.loads((folder/'owned_storage_latest.json').read_text())
    except (OSError,ValueError):return None
    return proof if _valid_owned(proof,plan,now) else None


def evidence(plan,quota,eligible,other_baseline_active):
    row=quota['rows'][0];now=time.time()
    reserve=plan['runtime_storage']['max_persistent_bytes']
    ownership=_owned(plan,now)
    if ownership and ownership['observed_at']>quota['observed_at']:ownership=None
    spent=ownership['committed_bytes'] if ownership else 0
    other=8*GIB if other_baseline_active else 0
    margin=20*GIB;remaining=reserve-spent;required=remaining+other+margin
    return {'complete':True,'passed':bool(eligible and row['quota_bytes']-row['used_bytes']>=required),
            'experiment':plan['experiment'],'account_root':'/mnt/afs/260010165',
            'account_allocated_bytes':row['used_bytes'],'quota_lower_bound_bytes':row['quota_bytes'],
            'new_task_reserve_bytes':reserve,'committed_task_bytes':spent,
            'remaining_task_reserve_bytes':remaining,'owned_storage':ownership,
            'other_remaining_commitments_bytes':other,'headroom_margin_bytes':margin,
            'required_remaining_bytes':required,'observed_at':quota['observed_at'],
            'quota_source':quota['method'],'training_slot_eligible':eligible,
            'training_serialized':True,'baseline_parallelism_preserved':True,'no_directory_scan':True}


def acceptable(proof,plan,now=None):
    now=time.time() if now is None else now
    reserve=proof.get('new_task_reserve_bytes',0);spent=proof.get('committed_task_bytes',0)
    if not isinstance(spent,int) or isinstance(spent,bool) or not 0<=spent<=reserve:return False
    if spent:
        owned=proof.get('owned_storage')
        # The committed files are immutable. Validate their measurement against
        # the quota snapshot; a queued worker may consume the same valid proof.
        if not _valid_owned(owned,plan,proof.get('observed_at',0)) or owned['committed_bytes']!=spent:return False
    remaining=reserve-spent
    return bool(proof.get('complete') and proof.get('passed')
        and proof.get('experiment')==plan['experiment']
        and proof.get('account_root')=='/mnt/afs/260010165'
        and 0<=now-proof.get('observed_at',0)<=6*3600
        and reserve>=plan['runtime_storage']['max_persistent_bytes']
        and proof.get('account_allocated_bytes',0)+proof.get('required_remaining_bytes',0)<=proof.get('quota_lower_bound_bytes',0)
        and proof.get('required_remaining_bytes',0)>=remaining+proof.get('other_remaining_commitments_bytes',0)+proof.get('headroom_margin_bytes',0)
        and proof.get('headroom_margin_bytes',0)>=20*GIB)
