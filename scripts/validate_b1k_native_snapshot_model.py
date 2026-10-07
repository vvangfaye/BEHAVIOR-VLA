#!/usr/bin/env python3
"""Actual model inference from verified reconstructed native starts, never official scoring."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import shutil
import diagnose_b1k_grasp_restore as diag
import eval_b1k_subtask_samples as runner
import validate_b1k_grasp_reconstruction as reconstruction

KIND='subtask_native_snapshot_model_validation'
INPUT_DIRECTORY='native_snapshot_input'
MANIFEST_KIND='verified_reconstructed_native_snapshots_v1'
PROTOCOL='known_native_reconstructed_start_v1'
MAX_INPUT_BYTES=128*1024**2


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value):
    if isinstance(value,dict):return all(_finite(v) for v in value.values())
    if isinstance(value,list):return all(_finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def validate_plan(plan):
    if (len(plan['sample_ids'])>2 or type(plan.get('max_steps')) is not int
            or not 1<=plan['max_steps']<=2048 or plan.get('paired_seed')!=1
            or type(plan.get('paired_seed')) is not int or plan.get('state_protocol')!=PROTOCOL
            or plan.get('snapshot_input_directory')!=INPUT_DIRECTORY
            or plan.get('checkpoint_step')!=1375000
            or plan.get('metric_eligible') is not False or plan.get('official_benchmark') is not False
            or plan.get('exact_legacy_grasp_restoration_verified') is not False):
        raise ValueError('Native model validation requires at most two reconstructed starts, seed1, checkpoint1375000 and explicit non-scoring fidelity labels')
    hashes=plan.get('native_snapshot_sha256_by_sample',{})
    if set(hashes)!=set(plan['sample_ids']):raise ValueError('Native plan does not bind every selected snapshot')
    for field in ('snapshot_manifest_sha256','checkpoint_manifest_sha256'):
        value=plan.get(field)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Native model validation lacks frozen input identity')
    if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in hashes.values()):
        raise ValueError('Native snapshot hashes are invalid')


def _safe_path(root,name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p)!=name:
        raise ValueError('Unsafe native snapshot input path')
    target=Path(root).joinpath(*p.parts)
    if any(x.is_symlink() for x in [target,*target.parents] if x==Path(root) or Path(root) in x.parents):
        raise ValueError('Native input must not use symlinks')
    if target.suffix not in ('.json','.mp4','.pt') and name!='_SUCCESS':
        raise ValueError('Unexpected native snapshot input type')
    return target


def validate_input_artifacts(root,panel,plan):
    validate_plan(plan);root=Path(root);manifest_path=root/'manifest.json'
    if _sha(manifest_path)!=plan['snapshot_manifest_sha256']:
        raise ValueError('Native input manifest changed')
    manifest=json.loads(manifest_path.read_text())
    if (manifest.get('kind')!=MANIFEST_KIND or manifest.get('state_protocol')!=PROTOCOL
            or manifest.get('panel_fingerprint')!=panel['fingerprint']
            or manifest.get('sample_ids')!=plan['sample_ids']
            or manifest.get('source_iid')!=plan.get('source_iid')
            or manifest.get('source_run_id')!=plan.get('source_run_id')
            or manifest.get('exact_legacy_grasp_restoration_verified') is not False):
        raise ValueError('Native input source contract differs')
    files=manifest.get('files');names=[x.get('path') for x in files] if isinstance(files,list) else []
    if not names or len(names)!=len(set(names)) or sum(x['bytes'] for x in files)>MAX_INPUT_BYTES:
        raise ValueError('Native input inventory is missing, duplicated or exceeds its budget')
    for item in files:
        p=_safe_path(root,item['path'])
        if type(item.get('bytes')) is not int or item['bytes']<=0 or not p.is_file() or p.stat().st_size!=item['bytes'] or _sha(p)!=item['sha256']:
            raise ValueError('Native snapshot input bytes changed')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=manifest_path}
    if actual!=set(names):raise ValueError('Native input has extra or missing files')
    source_plan=json.loads((root/'validation_plan.json').read_text())
    if source_plan.get('kind')!=reconstruction.KIND or source_plan.get('max_steps')!=32 or source_plan.get('sample_ids')!=plan['sample_ids']:
        raise ValueError('Native source experiment does not match selected starts')
    if json.loads((root/'parent_panel.json').read_text())!=panel:
        raise ValueError('Native source panel changed')
    contract=json.loads((root/'validation_contract.json').read_text())
    if contract.get('run_id')!=manifest['source_run_id'] or contract.get('checkpoint_step')!=plan['checkpoint_step']:
        raise ValueError('Native source checkpoint or run differs')
    if json.loads((root/'_SUCCESS').read_text())!={'kind':reconstruction.KIND,'complete':True}:
        raise ValueError('Native source was not successfully completed')
    samples=[next(s for s in panel['samples'] if s['sample_id']==sid) for sid in plan['sample_ids']]
    summary=reconstruction.validate_results(root,panel,source_plan,samples)
    if summary!=json.loads((root/'validation_summary.json').read_text()) or summary['source_inferred_retention_passed']!=len(samples):
        raise ValueError('Native source reconstruction did not pass every retention case')
    results={}
    for path in root.glob('pi05/*/grasp_reconstruction/*.json'):
        d=json.loads(path.read_text())
        if plan['native_snapshot_sha256_by_sample'][d['sample_id']]!=d['known_native_snapshot']['sha256']:
            raise ValueError('Native source snapshot differs from the model plan')
        results[d['sample_id']]=(path,d)
    return manifest,results


def pack_input_artifacts(root,destination,panel,plan):
    manifest,_=validate_input_artifacts(root,panel,plan);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for item in manifest['files']:
        target=destination/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(root)/item['path'],target)
    shutil.copyfile(Path(root)/'manifest.json',destination/'manifest.json')
    validate_input_artifacts(destination,panel,plan)


def _to_device(value,device):
    import torch
    if isinstance(value,torch.Tensor):return value.to(device=device)
    if isinstance(value,dict):return {k:_to_device(v,device) for k,v in value.items()}
    if isinstance(value,list):return [_to_device(v,device) for v in value]
    if isinstance(value,tuple):return tuple(_to_device(v,device) for v in value)
    return value


def validate_restored_observation(proof,current,joints,links,native_robot_sha):
    if not _finite(current) or not _finite(joints):raise ValueError('Nonfinite native restart observation')
    if proof['native_robot_source_sha256']!=native_robot_sha:
        raise ValueError('Native robot implementation differs from the verified snapshot source')
    if joints!=proof['initial_robot_joint_positions']:
        raise ValueError('Native restart changed initial full robot joints')
    errors=reconstruction.pose_errors(proof['known_native_snapshot']['before'],current)
    if max(errors.values())>1e-5:raise ValueError('Native restart changed initial object or EEF poses')
    for arm,proposal in proof['proposals'].items():
        row=current['arms'][arm]
        if (row['held_object']!=proposal['object'] or row['constraint_joint_type']!='FixedJoint'
                or row['release_counter'] is not None
                or links.get(arm)!=proof['physical_contact_probe']['contacts'][arm]['link']):
            raise ValueError('Native restart did not restore its recorded grasp constraint')
    return {'state_protocol':PROTOCOL,'known_native_restart_verified':True,
            'full_robot_joints_match':True,'pose_errors':errors,'initial_observation':current,
            'native_robot_source_sha256':native_robot_sha,'restored_grasp_links':links,
            'exact_legacy_grasp_restoration_verified':False}


class ActionAudit:
    def __init__(self):self.rows=[]
    def record(self,frame,action,terminated,truncated):
        from subtask_trajectory import array
        state=array(frame['observation.state']).reshape(-1);action=array(action).reshape(-1)
        if state.shape!=(61,) or action.shape!=(23,) or not all(math.isfinite(float(v)) for v in [*state,*action]):
            raise ValueError('Actual model actions require finite 61-wide preceding state and 23-wide controls')
        self.rows.append({'step':len(self.rows),'preceding_observation_state':state.tolist(),
                          'executed_action':action.tolist(),'terminated':bool(terminated),'truncated':bool(truncated)})


def run(base,evaluator,args,panel,plan,samples,cache,input_root):
    import omnigibson as og
    import torch
    manifest,proofs=validate_input_artifacts(input_root,panel,plan)
    out=Path(args.output_dir)/'native_snapshot_validation'
    for sample in samples:
        path,proof=proofs[sample['sample_id']]
        if cache['sha256']!=proof['raw_sha256']:raise ValueError('Native snapshot raw source differs')
        native_path=path.parent/'native_snapshots'/(path.stem+'.pt')
        state=torch.load(native_path,map_location='cpu',weights_only=True)
        state=_to_device(state,evaluator.robot.get_joint_positions().device)
        materialized={'state':state,'state_format':runner.FULL_SNAPSHOT_FORMAT,
                      'transition_steps':[s for s in sorted(cache['transition_by_step']) if s<sample['start']],
                      'restoration':{**copy.deepcopy(cache['sample_evidence'][sample['sample_id']]),
                         'full_snapshot':True,'full_snapshot_format':runner.FULL_SNAPSHOT_FORMAT,
                         'native_snapshot_sha256':_sha(native_path),'state_protocol':PROTOCOL,
                         'original_legacy_grasp_verified':False,'source_iid':manifest['source_iid']}}
        def verify_restart(_evaluator):
            names=list(proof['source_observation']['arms'][next(iter(proof['proposals']))]['objects'])
            objects=[_evaluator.env.scene.object_registry('name',n) for n in names]
            if any(o is None for o in objects):raise ValueError('Native snapshot movable objects are missing')
            observation=diag.observe(_evaluator,objects,frame=sample['start'])
            links={a:_evaluator.robot._ag_obj_constraint_params[a]['target_link_name'] for a in proof['proposals']}
            return {'native_snapshot_restart':validate_restored_observation(proof,observation,
                    _evaluator.robot.get_joint_positions().tolist(),links,
                    runner.file_sha256(Path(og.__file__).parent/'robots/robot.py'))}
        base.seed_everything(plan['paired_seed']);audit=ActionAudit()
        rollout=runner.run_sample(base,evaluator,args,sample,cache,materialized,out,panel['fingerprint'],
                                  trajectory_writer=audit,restoration_validator=verify_restart)
        result={'kind':KIND,'sample_id':sample['sample_id'],'panel_fingerprint':panel['fingerprint'],
                'state_protocol':PROTOCOL,'source_iid':manifest['source_iid'],'source_run_id':manifest['source_run_id'],
                'snapshot_manifest_sha256':plan['snapshot_manifest_sha256'],
                'native_snapshot_sha256':_sha(native_path),'paired_seed':plan['paired_seed'],
                'official_benchmark':False,'metric_eligible':False,'exact_legacy_grasp_restoration_verified':False,
                'model_rollout':rollout,'actual_model_control_steps':len(audit.rows),
                'observed_policy_subtask_reports':len(rollout['observed_subtasks']),
                'action_audit':audit.rows,'video_sha256':None}
        if rollout['metric_eligible']:
            result['video_sha256']=runner.file_sha256(out/'videos'/(runner._artifact_stem(sample)+'.mp4'))
        runner.atomic_write_json(out/'json'/(runner._artifact_stem(sample)+'.json'),result)
        print(f'NATIVE_SNAPSHOT_MODEL_SAMPLE sample={sample["sample_id"]} steps={rollout["steps"]} success={rollout["success"]} setup_eligible={rollout["metric_eligible"]}',flush=True)


def validate_results(root,panel,plan,samples):
    validate_plan(plan);expected={s['sample_id']:s for s in samples};seen=set();eligible=successful=steps=0
    for path in Path(root).glob('pi05/*/native_snapshot_validation/json/*.json'):
        d=json.loads(path.read_text());sid=d.get('sample_id')
        if sid not in expected or sid in seen:raise ValueError('Unknown or duplicate native model validation result')
        if (d.get('kind')!=KIND or d.get('state_protocol')!=PROTOCOL or d.get('metric_eligible') is not False
                or d.get('official_benchmark') is not False or d.get('exact_legacy_grasp_restoration_verified') is not False
                or d.get('panel_fingerprint')!=panel['fingerprint'] or d.get('paired_seed')!=plan['paired_seed']
                or d.get('source_iid')!=plan.get('source_iid') or d.get('source_run_id')!=plan.get('source_run_id')
                or d.get('snapshot_manifest_sha256')!=plan['snapshot_manifest_sha256']):
            raise ValueError('Native model validation provenance or fidelity labels differ')
        if not _finite(d):raise ValueError('Nonfinite native model validation evidence')
        sample=expected[sid];r=d['model_rollout']
        if d.get('native_snapshot_sha256')!=plan['native_snapshot_sha256_by_sample'][sid] or r.get('restoration',{}).get('native_snapshot_sha256')!=d['native_snapshot_sha256']:
            raise ValueError('Native model snapshot identity differs')
        runner.EvaluationQueue._validate_result(r,sample['task'],sample['task_instance_id'],sample['rollout_id'])
        for field,source in [('sample_id','sample_id'),('episode_index','episode_index'),('raw_episode_id','raw_episode_id'),('subtask','subtask'),('segment_start','start')]:
            if r.get(field)!=sample[source]:raise ValueError('Native model result changed sample identity')
        restart=r.get('restoration',{}).get('native_snapshot_restart',{})
        if (restart.get('known_native_restart_verified') is not True or restart.get('full_robot_joints_match') is not True
                or restart.get('state_protocol')!=PROTOCOL or restart.get('exact_legacy_grasp_restoration_verified') is not False
                or max(restart.get('pose_errors',{'missing':1}).values())>1e-5):
            raise ValueError('Native model inference lacks actual initial-state verification')
        if type(r.get('metric_eligible')) is not bool:raise ValueError('Native setup eligibility must be explicit')
        n=r['steps'];rows=d.get('action_audit')
        if type(n) is not int or not 0<=n<=plan['max_steps'] or not isinstance(rows,list) or len(rows)!=n or d.get('actual_model_control_steps')!=n:
            raise ValueError('Native model controls are incomplete')
        if [a.get('step') for a in rows]!=list(range(n)):
            raise ValueError('Native model action order differs')
        for a in rows:
            if len(a.get('preceding_observation_state',[]))!=61 or len(a.get('executed_action',[]))!=23 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in [*a['preceding_observation_state'],*a['executed_action']]):
                raise ValueError('Native model action audit has invalid dimensions or numbers')
        if r['metric_eligible']:
            reports=r.get('observed_subtasks',[]);sources=r.get('observed_subtask_sources',[])
            if n==0 or reports!=[sample['subtask']]*n or sources!=['provided']*n or d.get('observed_policy_subtask_reports')!=n:
                raise ValueError('Native model lacks fixed-subtask policy response evidence')
            video=path.parent.parent/'videos'/(path.stem+'.mp4')
            if not video.is_file() or video.stat().st_size==0 or runner.file_sha256(video)!=d.get('video_sha256'):
                raise ValueError('Native model video is missing or changed')
            eligible+=1;successful+=int(r['success'])
        elif n!=0 or r.get('result_state')!='setup_excluded' or not r.get('setup_exclusion_reason'):
            raise ValueError('Native model setup exclusion lacks evidence')
        steps+=n;seen.add(sid)
    if seen!=set(expected) or eligible==0:raise ValueError('Native validation requires complete results and actual model actions')
    return {'kind':KIND,'complete':True,'samples':len(seen),'eligible_diagnostic_rollouts':eligible,
            'successful_diagnostic_rollouts':successful,'setup_excluded':len(seen)-eligible,
            'actual_model_control_steps':steps,'state_protocol':PROTOCOL,'known_native_restarts_verified':len(seen),
            'metric_eligible':False,'official_benchmark':False,'exact_legacy_grasp_restoration_verified':False}
