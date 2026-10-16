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


def compare(baseline,candidate,pairs,candidate_step=10000,**unused):
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
        baseline_checkpoint=1688977,candidate_completed_updates=candidate_step,
        baseline_successes=sum(r['model_success'] for r in old.values()),candidate_successes=sum(r['model_success'] for r in new.values()),
        baseline_success_rate=mean(r['model_success'] for r in old.values()),candidate_success_rate=mean(r['model_success'] for r in new.values()),
        baseline_mean_q=mean(r['q_score'] for r in old.values()),candidate_mean_q=mean(r['q_score'] for r in new.values()),
        improved_pairs=improved,regressed_pairs=regressed,per_instance=details,
        policy_seed=0,environment_seed=0,historical_seed_acknowledged=False,
        historical_seed_evidence='frozen key0 initialization plus first-instance fresh-process evidence',
        training_seen=True,heldout_generalization_claim=False,training_continues_to20000=True)


def compare_candidates(previous,candidate,pairs,previous_step=1000,candidate_step=10000):
    if previous['protocol_sha256'] != candidate['protocol_sha256']:
        raise ValueError('1000 and10000 native settings differ')
    old=validate(previous,pairs,'candidate');new=validate(candidate,pairs,'candidate')
    if previous['checkpoint_manifest_sha256'] == candidate['checkpoint_manifest_sha256']:
        raise ValueError('1000 and10000 checkpoint identity must differ')
    details=[];improved=[];regressed=[]
    for task,instance in pairs:
        a,b=old[task,instance],new[task,instance]
        if not a['model_success'] and b['model_success']:improved.append([task,instance])
        if a['model_success'] and not b['model_success']:regressed.append([task,instance])
        details.append(dict(task=task,instance_id=instance,baseline_success=a['model_success'],
                            candidate_success=b['model_success'],baseline_q=a['q_score'],candidate_q=b['q_score']))
    return dict(baseline_checkpoint=previous_step,candidate_completed_updates=candidate_step,pairs=20,
                baseline_successes=sum(r['model_success'] for r in old.values()),
                candidate_successes=sum(r['model_success'] for r in new.values()),
                baseline_success_rate=mean(r['model_success'] for r in old.values()),
                candidate_success_rate=mean(r['model_success'] for r in new.values()),
                baseline_mean_q=mean(r['q_score'] for r in old.values()),
                candidate_mean_q=mean(r['q_score'] for r in new.values()),
                improved_pairs=improved,regressed_pairs=regressed,per_instance=details)


def three_way(baseline,previous,candidate,pairs):
    identities={b['checkpoint_manifest_sha256'] for b in (baseline,previous,candidate)}
    if len(identities)!=3:raise ValueError('Three distinct checkpoint manifests required')
    versus_baseline=compare(baseline,candidate,pairs,candidate_step=10000)
    versus_previous=compare_candidates(previous,candidate,pairs)
    previous_vs_baseline=compare(baseline,previous,pairs,candidate_step=1000)
    old=validate(baseline,pairs,'baseline');mid=validate(previous,pairs,'candidate');new=validate(candidate,pairs,'candidate')
    summaries=[]
    for step,rows in ((1688977,old),(1000,mid),(10000,new)):
        summaries.append(dict(checkpoint=step,successes=sum(r['model_success'] for r in rows.values()),
                              success_rate=mean(r['model_success'] for r in rows.values()),
                              mean_q=mean(r['q_score'] for r in rows.values())))
    comparisons={'10000_vs1688977':versus_baseline,'10000_vs1000':versus_previous,
                 '1000_vs1688977':previous_vs_baseline}
    for value in comparisons.values():
        rows=value['per_instance']
        value['mean_q_delta']=value['candidate_mean_q']-value['baseline_mean_q']
        value['q_improved_pairs']=[[r['task'],r['instance_id']] for r in rows if r['candidate_q']>r['baseline_q']+1e-9]
        value['q_regressed_pairs']=[[r['task'],r['instance_id']] for r in rows if r['candidate_q']<r['baseline_q']-1e-9]
    details=[]
    for task,instance in pairs:
        details.append(dict(task=task,instance_id=instance,checkpoints={str(step):dict(success=rows[task,instance]['model_success'],
                  q=rows[task,instance]['q_score']) for step,rows in ((1688977,old),(1000,mid),(10000,new))}))
    return dict(kind='three_checkpoint_training_seen_regression',pairs=20,tasks=20,
                summaries=summaries,comparisons=comparisons,per_instance=details,
                policy_seed=0,environment_seed=0,historical_baseline_acknowledged=False,
                trained_checkpoints_actual_seed_acknowledged=True,training_seen=True,
                heldout_generalization_claim=False,training_continues_to20000=True)


def four_way(baseline,step1000,step10000,candidate,pairs):
    bundles=[(1688977,baseline),(1000,step1000),(10000,step10000),(20000,candidate)]
    if len({b['checkpoint_manifest_sha256'] for _,b in bundles})!=4:
        raise ValueError('Four distinct checkpoint manifests required')
    protocol=baseline['protocol_sha256']
    rows={}
    for step,bundle in bundles:
        if bundle['protocol_sha256']!=protocol:
            raise ValueError('Four native evaluation protocols differ')
        if 'completed_updates' in bundle and bundle['completed_updates']!=step:
            raise ValueError('Candidate checkpoint progress mislabeled')
        rows[step]=validate(bundle,pairs,'baseline' if step==1688977 else 'candidate')
    summaries=[dict(checkpoint=step,successes=sum(r['model_success'] for r in rows[step].values()),
                    success_rate=mean(r['model_success'] for r in rows[step].values()),
                    mean_q=mean(r['q_score'] for r in rows[step].values())) for step,_ in bundles]
    comparisons={}
    for i,(old_step,old) in enumerate(bundles):
        for step,new in bundles[i+1:]:
            result=(compare(old,new,pairs,candidate_step=step) if old_step==1688977
                    else compare_candidates(old,new,pairs,previous_step=old_step,candidate_step=step))
            result['training_continues_to20000']=False
            result['training_target_completed']=True
            result['mean_q_delta']=result['candidate_mean_q']-result['baseline_mean_q']
            detail=result['per_instance']
            result['q_improved_pairs']=[[r['task'],r['instance_id']] for r in detail if r['candidate_q']>r['baseline_q']+1e-9]
            result['q_regressed_pairs']=[[r['task'],r['instance_id']] for r in detail if r['candidate_q']<r['baseline_q']-1e-9]
            comparisons[f'{step}_vs{old_step}']=result
    detail=[dict(task=task,instance_id=instance,checkpoints={str(step):dict(
                success=rows[step][task,instance]['model_success'],q=rows[step][task,instance]['q_score'])
                for step,_ in bundles}) for task,instance in pairs]
    return dict(kind='four_checkpoint_training_seen_regression',pairs=20,tasks=20,
                summaries=summaries,comparisons=comparisons,per_instance=detail,
                policy_seed=0,environment_seed=0,historical_baseline_acknowledged=False,
                trained_checkpoints_actual_seed_acknowledged=True,training_seen=True,
                heldout_generalization_claim=False,training_continues_to20000=False,
                training_target_completed=True)


def five_way(baseline,old1000,old10000,old20000,candidate,pairs):
    bundles=[('1688977',1688977,baseline),('previous_1000',1000,old1000),('previous_10000',10000,old10000),('previous_20000',20000,old20000),('rft60_40_1000',1000,candidate)]
    if len({b['checkpoint_manifest_sha256'] for _,_,b in bundles})!=5:raise ValueError('Five distinct model identities required')
    protocol=baseline['protocol_sha256'];rows={}
    for name,step,bundle in bundles:
        if bundle['protocol_sha256']!=protocol:raise ValueError('Native protocols differ')
        if 'completed_updates' in bundle and bundle['completed_updates']!=step:raise ValueError('Mislabeled checkpoint step')
        rows[name]=validate(bundle,pairs,'baseline' if name=='1688977' else 'candidate')
    summaries=[dict(model=name,checkpoint_step=step,successes=sum(r['model_success'] for r in rows[name].values()),success_rate=mean(r['model_success'] for r in rows[name].values()),mean_q=mean(r['q_score'] for r in rows[name].values())) for name,step,_ in bundles]
    comparisons={}
    for name,step,old in bundles[:-1]:
        diff=compare(old,candidate,pairs,candidate_step=1000) if name=='1688977' else compare_candidates(old,candidate,pairs,previous_step=step,candidate_step=1000)
        diff.update(baseline_model=name,candidate_model='rft60_40_1000',training_continues_to20000=False,mean_q_delta=diff['candidate_mean_q']-diff['baseline_mean_q'])
        detail=diff['per_instance'];diff['q_improved_pairs']=[[r['task'],r['instance_id']] for r in detail if r['candidate_q']>r['baseline_q']+1e-9];diff['q_regressed_pairs']=[[r['task'],r['instance_id']] for r in detail if r['candidate_q']<r['baseline_q']-1e-9]
        comparisons['rft60_40_1000_vs_'+name]=diff
    details=[dict(task=task,instance_id=instance,models={name:dict(success=rows[name][task,instance]['model_success'],q=rows[name][task,instance]['q_score']) for name,_,_ in bundles}) for task,instance in pairs]
    return dict(kind='five_model_training_seen_regression',pairs=20,summaries=summaries,comparisons=comparisons,per_instance=details,policy_seed=0,environment_seed=0,training_seen=True,heldout_generalization_claim=False,historical_baseline_acknowledged=False,trained_checkpoints_actual_seed_acknowledged=True,new_candidate_rollouts=20,new_historical_rollouts=0,training_modified=False,training_continues_to20000=False)
