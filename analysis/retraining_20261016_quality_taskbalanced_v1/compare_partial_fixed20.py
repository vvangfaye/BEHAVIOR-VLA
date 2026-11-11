"""Report only finished, accepted matched episodes; missing results are never failures."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_candidate import validate, summary

ROOT=Path(__file__).resolve().parent


def accepted_records(plan, reports, bundle_sha):
    rows=[];failed=[]
    for index,pair in enumerate(plan['pairs']):
        id=f'candidate-{index:02d}'
        report=reports.get(id)
        if not report:continue
        if (report.get('id')!=id or report.get('protocol_sha256')!=plan['protocol_sha256']
                or report.get('worker_bundle_sha256')!=bundle_sha):
            raise ValueError('Report identity differs: '+id)
        if report.get('state')!='accepted':
            failed.append(id)
            continue
        if report.get('checkpoint_manifest_sha256')!=plan['expected_checkpoint_manifest_sha256']:
            raise ValueError('Checkpoint identity differs: '+id)
        record=report['record']
        if [record['task'],record['instance_id']]!=pair:
            raise ValueError('Report was assigned to another fixed episode: '+id)
        rows.append(record)
    return rows,failed


def calculate(plan, reports, bundle_sha, models):
    records,failed=accepted_records(plan,reports,bundle_sha)
    keys=[(r['task'],r['instance_id']) for r in records]
    candidate=validate(dict(protocol_sha256=plan['protocol_sha256'],records=records),keys,plan['protocol_sha256'])
    total=len(plan['pairs']);missing=total-len(records)
    out=dict(kind='partial_fixed20_matched_observation',completed_updates=plan['candidate_step'],
             accepted_episodes=len(records),planned_episodes=total,unobserved_or_unaccepted_episodes=missing,
             infrastructure_failed_reports=failed,complete=len(records)==total,
             candidate=summary(candidate,keys) if keys else None,comparisons={},
             missing_scored_as_failure=False,model_improvement_proven=False,
             warning='Completed-subset comparisons can change as remaining episodes finish; do not treat as final20 accuracy.')
    if not keys:return out
    successes=sum(candidate[k]['model_success'] for k in keys)
    out['full_panel_success_rate_bounds']=[successes/total,(successes+missing)/total]
    for name,old in models.items():
        common=[k for k in keys if k in old]
        if not common:continue
        out['comparisons'][name]=dict(matched_completed_episodes=len(common),
            candidate_same_scope=summary(candidate,common),reference_same_scope=summary(old,common),
            mean_q_delta=sum(candidate[k]['q_score']-old[k]['q_score'] for k in common)/len(common),
            success_gains=[list(k) for k in common if candidate[k]['model_success'] and not old[k]['model_success']],
            success_losses=[list(k) for k in common if old[k]['model_success'] and not candidate[k]['model_success']])
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=ROOT/'paired_fixed20_step1000_5x4')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();campaign=args.campaign.resolve()
    if campaign.parent!=ROOT:raise ValueError('Only the current quality-trial campaign roots')
    plan=json.loads((campaign/'plan.json').read_text())
    reports={p.stem:json.loads(p.read_text()) for p in (campaign/'reports').glob('candidate-*.json')}
    sha=hashlib.sha256((campaign/'frozen_sha256.json').read_bytes()).hexdigest()
    manifest=json.loads((ROOT/'evaluation_history/manifest.json').read_text())
    if manifest['protocol_sha256']!=plan['protocol_sha256']:raise ValueError('Historical protocol differs')
    models={}
    for name,source in manifest['models'].items():
        p=ROOT/'evaluation_history'/source['file'];raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('Historical data changed')
        models[name]=validate(json.loads(raw),source['pairs'],plan['protocol_sha256'],baseline=name=='1688977')
    result=calculate(plan,reports,sha,models)
    if args.output:
        if args.output.resolve().parent!=campaign:raise ValueError('Write report only inside selected campaign')
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
