#!/usr/bin/env python3
"""Freeze and compare independently reserved samples from strict full-panel runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from subtask_improvement import digest, load_plan as load_collection_plan
from subtask_sample_panel import load_panel
from summarize_subtask_closed_loop import summarize, _result_path

KIND = 'subtask_paired_retest'
SEED_PROTOCOL = 'per_sample_first_request_jax_v1'


def policy_seed(seed, fingerprint, sample_id):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Retest seed must be a uint32 integer')
    return int(digest([SEED_PROTOCOL,seed,fingerprint,sample_id])[:8],16)


def evaluator_digest():
    root=Path(__file__).resolve().parents[1]
    files=['scripts/eval_b1k_subtask_samples.py','scripts/subtask_completion.py',
           'scripts/collect_b1k_subtasks.py','scripts/eval_b1k.sh',
           'openpi/src/openpi/shared/collection_seeding.py','openpi/scripts/b1k/serve_b1k.py',
           'openpi/src/openpi/serving/websocket_b1k_server.py',
           'BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/network_utils.py']
    return digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})


def freeze_retest(panel, collection_plan, control_tasks, *, seed=1, max_steps=2048, per_group=10):
    if type(per_group) is not int or per_group < 1 or type(max_steps) is not int or not 1 <= max_steps <= 4096:
        raise ValueError('Retest limits must be positive and bounded')
    policy_seed(seed,panel['fingerprint'],'validate')
    if not collection_plan.get('checkpoint_identity'):raise ValueError('Retest requires the verified collection checkpoint identity')
    held_out={tuple(key) for key in collection_plan['held_out_instances']}
    if not held_out:raise ValueError('Reserve independent instance groups before collecting data')
    targets={(g['task'],g['subtask']) for g in collection_plan['selected_groups']}
    controls=sorted(set(control_tasks))
    if not targets or not controls or set(controls)&{t for t,_ in targets}:
        raise ValueError('Retest requires target groups and distinct control tasks')
    candidates=[s for s in panel['samples'] if (s['task'],s['task_instance_id']) in held_out]
    selections=[];counts=[]
    for role,groups in [('target',sorted(targets)),('control',[(t,None) for t in controls])]:
        for task,prompt in groups:
            rows=[s for s in candidates if s['task']==task and (prompt is None or s['subtask']==prompt)]
            rows.sort(key=lambda s:digest([seed,s['sample_id']]))
            first,rest,seen=[],[],set()
            for row in rows:
                if row['task_instance_id'] in seen:rest.append(row)
                else:first.append(row);seen.add(row['task_instance_id'])
            rows=(first+rest)[:per_group]
            if not rows:raise ValueError(f'No reserved samples for {role} group: {task}/{prompt}')
            selections.extend(dict(sample_id=s['sample_id'],role=role,
                policy_seed=policy_seed(seed,panel['fingerprint'],s['sample_id'])) for s in rows)
            counts.append(dict(role=role,task=task,subtask=prompt,samples=len(rows),
                distinct_instances=len({s['task_instance_id'] for s in rows})))
    payload=dict(schema_version=1,kind=KIND,panel_fingerprint=panel['fingerprint'],
        collection_plan_fingerprint=collection_plan['fingerprint'],
        base_checkpoint_identity=collection_plan['checkpoint_identity'],
        held_out_instances=sorted(map(list,held_out)),target_groups=sorted(map(list,targets)),
        control_tasks=controls,samples=selections,groups=counts,seed=seed,seed_protocol=SEED_PROTOCOL,
        max_steps=max_steps,conditioning='task_and_subtask',official_benchmark=False,
        held_out_scope='this_fine_tuning_experiment_only')
    return {**payload,'fingerprint':digest(payload)}


def validate_plan(plan,panel,collection_plan):
    payload={k:v for k,v in plan.items() if k!='fingerprint'}
    if plan.get('fingerprint')!=digest(payload) or plan.get('schema_version')!=1 or plan.get('kind')!=KIND:
        raise ValueError('Modified or unsupported retest plan')
    if plan.get('panel_fingerprint')!=panel['fingerprint'] or plan.get('collection_plan_fingerprint')!=collection_plan['fingerprint']:
        raise ValueError('Retest does not match frozen source plans')
    if plan.get('held_out_instances')!=collection_plan['held_out_instances'] or plan.get('base_checkpoint_identity')!=collection_plan['checkpoint_identity']:
        raise ValueError('Reserved groups or base checkpoint changed')
    if plan.get('conditioning')!='task_and_subtask' or plan.get('seed_protocol')!=SEED_PROTOCOL or type(plan.get('max_steps')) is not int or not 1<=plan['max_steps']<=4096:
        raise ValueError('Invalid retest inference contract')
    targets={tuple(g) for g in plan['target_groups']}
    if targets!={(g['task'],g['subtask']) for g in collection_plan['selected_groups']}:
        raise ValueError('Retest targets differ from collected target groups')
    controls=set(plan['control_tasks']);held_out={tuple(g) for g in plan['held_out_instances']}
    if not controls or controls&{t for t,_ in targets}:raise ValueError('Control tasks overlap target tasks')
    samples={s['sample_id']:s for s in panel['samples']};seen=set();roles=set()
    for row in plan['samples']:
        sid=row['sample_id']
        if sid not in samples or sid in seen:raise ValueError('Unknown or duplicate retest sample')
        seen.add(sid);s=samples[sid];role=row['role'];roles.add(role)
        if (s['task'],s['task_instance_id']) not in held_out:raise ValueError('Retest sample is not reserved from fine-tuning')
        if (role=='target' and (s['task'],s['subtask']) not in targets) or (role=='control' and s['task'] not in controls) or role not in ('target','control'):
            raise ValueError('Retest role differs from target/control group')
        if row['policy_seed']!=policy_seed(plan['seed'],panel['fingerprint'],sid):raise ValueError('Retest sample seed changed')
    if roles!={'target','control'}:raise ValueError('Retest must include target and control evidence')
    observed_targets={(samples[r['sample_id']]['task'],samples[r['sample_id']]['subtask']) for r in plan['samples'] if r['role']=='target'}
    observed_controls={samples[r['sample_id']]['task'] for r in plan['samples'] if r['role']=='control'}
    if observed_targets!=targets or observed_controls!=controls:raise ValueError('Retest selections omit requested target or control groups')
    return plan


def read_verified_run(root,panel,plan,receipt):
    root=Path(root)
    for name in ['_SUCCESS','manifest.json','submission_contract.json']:
        if not (root/name).is_file():raise ValueError(f'Incomplete strict retest run: {name}')
    manifest=json.loads((root/'manifest.json').read_text())
    marker=json.loads((root/'_SUCCESS').read_text())
    contract=json.loads((root/'submission_contract.json').read_text())
    if manifest.get('fingerprint')!=digest({k:v for k,v in manifest.items() if k!='fingerprint'}):
        raise ValueError('Retest run manifest changed')
    if receipt.get('evaluation_kind')!='training_subtask_closed_loop' or receipt.get('exit_code')!=0 or receipt.get('panel_fingerprint')!=panel['fingerprint']:
        raise ValueError('Retest submission identity is invalid')
    if manifest.get('run_id')!=receipt.get('run_id') or contract.get('run_id')!=receipt.get('run_id'):
        raise ValueError('Retest run IDs disagree')
    if manifest.get('sample_max_steps')!=plan['max_steps'] or str(manifest.get('child_environment',{}).get('PI05_PAIRED_RETEST_SEED'))!=str(plan['seed']):
        raise ValueError('Retest limits or per-sample seed contract differ')
    settings=manifest['child_environment']
    if settings.get('SUBTASK_PREDICTION')!='1' or settings.get('PROVIDED_SUBTASK_AS_PROMPT')!='0' or settings.get('MAX_TOKEN_LEN')!='512':
        raise ValueError('Retest must preserve joint task/subtask inference')
    if contract.get('checkpoint_manifest_sha256')!=receipt.get('checkpoint_manifest_sha256') or contract.get('checkpoint_step')!=receipt.get('checkpoint_step') or contract.get('conditioning')!='task_and_subtask':
        raise ValueError('Retest checkpoint or training conditioning is not verified')
    if str(contract.get('paired_retest_seed'))!=str(plan['seed']):raise ValueError('Retest source contract seed differs')
    if not isinstance(contract.get('evaluator_code_sha256'),str) or len(contract['evaluator_code_sha256'])!=64:
        raise ValueError('Retest lacks evaluator code identity')
    if marker.get('expected_segment_rollouts')!=len(panel['samples']) or marker.get('expected_episode_jobs')!=len(panel['episodes']):
        raise ValueError('Retest completion marker differs from scheduled panel')
    results=root/'results'
    summary=summarize(panel,results,checkpoint=manifest['checkpoint'],require_videos=True,run_manifest=manifest)
    if not summary['complete']:raise ValueError('Strict retest result validation failed; infrastructure failures cannot count as policy failures')
    if marker.get('run_id')!=receipt['run_id'] or marker.get('panel_fingerprint')!=panel['fingerprint'] or marker.get('summary_sha256')!=digest(summary):
        raise ValueError('Retest completion marker does not bind the validated run summary')
    rows={}
    for selection in plan['samples']:
        sample=next(s for s in panel['samples'] if s['sample_id']==selection['sample_id'])
        row=json.loads(_result_path(results,sample).read_text())
        if row.get('policy_seed')!=selection['policy_seed'] or row.get('seed_protocol')!=SEED_PROTOCOL:
            raise ValueError('Actual rollout lacks matched per-sample policy seed evidence')
        if row.get('restoration',{}).get('full_snapshot_format')!='named_simulator_state_v1':
            raise ValueError('Retest lacks exact named-state restoration evidence')
        rows[sample['sample_id']]=row
    return rows,contract,summary


def paired_statistics(rows,*,seed=1,replicates=2000):
    """Resample whole task-instance groups, retaining correlated sample outcomes."""
    if type(replicates) is not int or replicates<200:raise ValueError('At least 200 bootstrap replicates are required')
    if not rows:return dict(pairs=0,base_success_rate=None,finetuned_success_rate=None,delta=None,
                            instance_groups=0,delta_ci95=None,conservative_delta_ci95=None,evidence_sufficient=False)
    base=sum(r['base_success'] for r in rows);fine=sum(r['finetuned_success'] for r in rows)
    clusters={}
    for row in rows:
        values=clusters.setdefault((row['task'],row['task_instance_id']),[0,0])
        values[0]+=int(row['finetuned_success'])-int(row['base_success']);values[1]+=1
    a=np.asarray(list(clusters.values()),dtype=float);rng=np.random.default_rng(seed)
    draws=a[rng.integers(0,len(a),size=(replicates,len(a)))].sum(axis=1)
    interval=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist() if len(a)>1 else None
    # Bootstrap collapses at all-win/all-tie boundaries. Also bound the paired
    # mean using independent instance clusters with ranges [-n_i, n_i].
    delta=(fine-base)/len(rows)
    margin=math.sqrt(2*math.log(40)*float(np.square(a[:,1]).sum())/len(rows)**2)
    conservative=[max(-1.,delta-margin),min(1.,delta+margin)]
    enough=len(a)>=8
    return dict(pairs=len(rows),base_successes=base,finetuned_successes=fine,
        base_success_rate=base/len(rows),finetuned_success_rate=fine/len(rows),delta=(fine-base)/len(rows),
        gains=sum(not r['base_success'] and r['finetuned_success'] for r in rows),
        losses=sum(r['base_success'] and not r['finetuned_success'] for r in rows),
        instance_groups=len(a),delta_ci95=interval,conservative_delta_ci95=conservative,evidence_sufficient=enough,
        supported_improvement=bool(enough and conservative[0]>0),
        conservative_interval_method='Hoeffding_independent_bounded_instance_clusters',
        bootstrap_unit='task_and_task_instance',bootstrap_replicates=replicates)


def compare_runs(panel,plan,collection_plan,base_root,fine_root,base_receipt,fine_receipt,provenance,training_receipt):
    validate_plan(plan,panel,collection_plan)
    base,base_contract,_=read_verified_run(base_root,panel,plan,base_receipt)
    fine,fine_contract,_=read_verified_run(fine_root,panel,plan,fine_receipt)
    identity=plan['base_checkpoint_identity']
    if any(base_receipt.get(k)!=identity.get(k) for k in ('checkpoint_step','checkpoint_manifest_sha256','checkpoint_source')):
        raise ValueError('Base model differs from the collection baseline')
    if fine_receipt['checkpoint_manifest_sha256']==base_receipt['checkpoint_manifest_sha256']:
        raise ValueError('Retest checkpoints must have distinct verified manifests')
    for key in ('evaluator_code_sha256','sample_max_steps','conditioning','runtime_identity'):
        if base_contract.get(key)!=fine_contract.get(key):raise ValueError(f'Retest runtime contracts differ: {key}')
    runtime=base_contract.get('runtime_identity')
    if not isinstance(runtime,dict) or set(runtime)!={'policy_environment','evaluation_environment','assets_manifest','norm_stats','runtime_requirements'} or any(not isinstance(v,str) or len(v)!=64 for v in runtime.values()):
        raise ValueError('Retest runtime dependency identities are missing')
    reserved={tuple(g) for g in plan['held_out_instances']}
    source_groups={(r['task_name'],r['task_instance_id']) for r in provenance}
    if not provenance or source_groups&reserved:raise ValueError('Exported training data leaks reserved instances')
    expected=dict(complete=True,conditioning='task_and_subtask',collection_plan_fingerprint=collection_plan['fingerprint'],
        export_provenance_sha256=digest(provenance),init_checkpoint_manifest_sha256=base_receipt['checkpoint_manifest_sha256'],
        output_checkpoint_manifest_sha256=fine_receipt['checkpoint_manifest_sha256'])
    if any(training_receipt.get(k)!=v for k,v in expected.items()) or not training_receipt.get('training_run_id'):
        raise ValueError('Completed training provenance does not connect exported data and the two models')
    samples={s['sample_id']:s for s in panel['samples']};pairs=[];excluded=[]
    for selection in plan['samples']:
        sid=selection['sample_id'];a,b=base[sid],fine[sid];sample=samples[sid]
        for key in ('sha256','selected_frame','selected_state_size','topology_transition_steps'):
            if a['restoration'].get(key)!=b['restoration'].get(key):raise ValueError('Paired start-state evidence differs')
        if a['metric_eligible']!=b['metric_eligible'] or a['setup_exclusion_reason']!=b['setup_exclusion_reason']:
            raise ValueError('Paired setup eligibility differs; investigate state restoration')
        row=dict(sample_id=sid,task=sample['task'],task_instance_id=sample['task_instance_id'],
                 subtask=sample['subtask'],role=selection['role'],policy_seed=selection['policy_seed'])
        if not a['metric_eligible']:
            excluded.append({**row,'reason':a['setup_exclusion_reason']});continue
        pairs.append({**row,'base_success':a['success'],'finetuned_success':b['success'],
                      'base_steps':a['steps'],'finetuned_steps':b['steps']})
    per_group=[]
    for task,prompt in sorted({(samples[r['sample_id']]['task'],samples[r['sample_id']]['subtask']) for r in plan['samples']}):
        selected=[r for r in pairs if r['task']==task and r['subtask']==prompt]
        planned=[r for r in plan['samples'] if samples[r['sample_id']]['task']==task and samples[r['sample_id']]['subtask']==prompt]
        per_group.append(dict(task=task,subtask=prompt,role=planned[0]['role'],scheduled_pairs=len(planned),
            setup_excluded=sum(r['task']==task and r['subtask']==prompt for r in excluded),**paired_statistics(selected,seed=plan['seed'])))
    target=paired_statistics([r for r in pairs if r['role']=='target'],seed=plan['seed'])
    control=paired_statistics([r for r in pairs if r['role']=='control'],seed=plan['seed'])
    return dict(schema_version=1,kind=KIND,complete=True,official_benchmark=False,
        plan_fingerprint=plan['fingerprint'],held_out_scope=plan['held_out_scope'],
        training_run_id=training_receipt['training_run_id'],target=target,control=control,
        control_regression_observed=bool(control['delta'] is not None and control['delta']<0),
        per_group=per_group,matched_pairs=pairs,setup_excluded=excluded,
        policy_improvement_confirmed=bool(target.get('supported_improvement') and control['evidence_sufficient'] and control['conservative_delta_ci95'] and control['conservative_delta_ci95'][0]>=0),
        limitations=['Public training states are held out only from this fine-tuning experiment.',
                     'Bootstrap intervals are exploratory; conservative bounds assume independent reserved task-instance clusters.',
                     'Intervals describe uncertainty across comparable instances, not official benchmark generalization.',
                     'Fixed annotated subtasks measure execution, not high-level subtask selection or official competition score.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    freeze=commands.add_parser('freeze');compare=commands.add_parser('compare')
    for parser in (freeze,compare):
        parser.add_argument('--panel',required=True,type=Path);parser.add_argument('--collection-plan',required=True,type=Path)
        parser.add_argument('--output',required=True,type=Path)
    freeze.add_argument('--control-tasks',required=True,nargs='+');freeze.add_argument('--seed',type=int,default=1)
    freeze.add_argument('--max-steps',type=int,default=2048);freeze.add_argument('--samples-per-group',type=int,default=10)
    for name in ('plan','base-root','fine-root','base-receipt','fine-receipt','provenance','training-receipt'):
        compare.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();panel=load_panel(a.panel);collection,_=load_collection_plan(a.collection_plan,panel)
    def read(path):return json.loads(path.read_text())
    if a.command=='freeze':result=freeze_retest(panel,collection,a.control_tasks,seed=a.seed,max_steps=a.max_steps,per_group=a.samples_per_group)
    else:result=compare_runs(panel,read(a.plan),collection,a.base_root,a.fine_root,read(a.base_receipt),read(a.fine_receipt),read(a.provenance),read(a.training_receipt))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'kind':result['kind'],'output':str(a.output),'complete':result.get('complete',False)}))


if __name__=='__main__':main()
