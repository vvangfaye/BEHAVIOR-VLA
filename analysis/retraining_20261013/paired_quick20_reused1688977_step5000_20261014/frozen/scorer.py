"""Training-seen regression against immutable historical native seed0 results.

Historical seed provenance is source+first-process evidence, not a server ACK.
The candidate must additionally prove actual explicit seed0 application/ACK.
"""
import math
from statistics import mean


def validate(bundle,pairs,role):
    if bundle.get('kind')!='artifact_accepted_historical_reuse_regression':
        raise ValueError('Explicit historical-reuse evidence required')
    if bundle.get('policy_seed')!=0 or bundle.get('environment_seed')!=0:
        raise ValueError('Must match historical native seed0')
    expected={tuple(p) for p in pairs};rows={}
    if len(expected)!=20 or len({p[0] for p in expected})!=20:
        raise ValueError('Exactly20 distinct task-instance pairs required')
    for row in bundle['records']:
        key=(row['task'],row['instance_id'])
        if key not in expected or key in rows or row.get('artifact_accepted') is not True:
            raise ValueError('Unaccepted or duplicate/out-of-scope result')
        if row.get('policy_seed')!=0 or row.get('environment_seed')!=0 or row.get('steps',0)<=0:
            raise ValueError('Native seed/step identity mismatch')
        if role=='candidate':
            if row.get('policy_seed_acknowledged') is not True:
                raise ValueError('Candidate seed ACK missing')
        elif role=='baseline':
            if row.get('policy_seed_acknowledged') is not False or row.get('policy_seed_source_verified') is not True:
                raise ValueError('Historical source evidence cannot be presented as a seed ACK')
        else:raise ValueError('Unknown role')
        q=row['q_score']
        if type(row['model_success']) is not bool or not math.isfinite(q) or not 0<=q<=1 or row['model_success']!=(q==1):
            raise ValueError('Invalid native success/Q')
        if not (row.get('terminated') or row.get('truncated')):
            raise ValueError('Native episode not complete')
        rows[key]=row
    if set(rows)!=expected:raise ValueError('Incomplete paired results')
    return rows


def compare(baseline,candidate,pairs,**unused):
    if baseline['protocol_sha256']!=candidate['protocol_sha256']:
        raise ValueError('Paired native settings differ')
    old=validate(baseline,pairs,'baseline');new=validate(candidate,pairs,'candidate')
    if baseline['checkpoint_manifest_sha256']==candidate['checkpoint_manifest_sha256']:
        raise ValueError('Same checkpoint cannot stand in for trained candidate')
    improved=[];regressed=[];details=[]
    for task,instance in pairs:
        a,b=old[task,instance],new[task,instance]
        if not a['model_success'] and b['model_success']:improved.append([task,instance])
        if a['model_success'] and not b['model_success']:regressed.append([task,instance])
        details.append(dict(task=task,instance_id=instance,baseline_success=a['model_success'],candidate_success=b['model_success'],baseline_q=a['q_score'],candidate_q=b['q_score']))
    return dict(kind='historical_baseline_reused_training_seen_regression',pairs=20,tasks=20,
        baseline_checkpoint=1688977,candidate_completed_updates=5000,
        baseline_successes=sum(r['model_success'] for r in old.values()),candidate_successes=sum(r['model_success'] for r in new.values()),
        baseline_success_rate=mean(r['model_success'] for r in old.values()),candidate_success_rate=mean(r['model_success'] for r in new.values()),
        baseline_mean_q=mean(r['q_score'] for r in old.values()),candidate_mean_q=mean(r['q_score'] for r in new.values()),
        improved_pairs=improved,regressed_pairs=regressed,per_instance=details,
        policy_seed=0,environment_seed=0,historical_seed_acknowledged=False,
        historical_seed_evidence='frozen key0 initialization plus first-instance fresh-process evidence',
        training_seen=True,heldout_generalization_claim=False,training_continues_to20000=True)
