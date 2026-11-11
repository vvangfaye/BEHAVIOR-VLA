"""Ensure a failed, missing, or changed offline result cannot enable serving."""
import json
from pathlib import Path
import tempfile

import vla_runtime
from serve_posterior_support import verify_gate,digest


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='posterior-gate-') as tmp:
        root=Path(tmp);folder=root/'formal_posterior';folder.mkdir()
        (root/'plan.json').write_text('{}')
        (root/'analysis_plan.json').write_text('{}')
        names=('test.npz','test.json','result.json','completion.json')
        for name in names:(folder/name).write_text(name)
        report=dict(ready=True,screen_passed=True,plan_sha256=digest(root/'plan.json'),
                    analysis_plan_sha256=digest(root/'analysis_plan.json'),
                    source_hashes={'posterior':{n:digest(folder/n) for n in names}})
        path=root/'reanalysis.json'
        def write():path.write_text(json.dumps(report))
        def rejected(name):
            try:verify_gate(root,'posterior')
            except (ValueError,FileNotFoundError,KeyError):checks.append(name)
            else:raise AssertionError(name+' was accepted')
        rejected('missing_report')
        report['screen_passed']=False;write();rejected('failed_screen')
        report['screen_passed']=True;write()
        rejected('missing_training_contribution')
        (root/'training_contribution_gate_plan.json').write_text('{}')
        contribution=dict(ready=True,primary_gate_passed=True,posthoc_gate_passed=True,
                          training_contribution_gate_passed=True,
                          source_primary_report_sha256=digest(path),
                          gate_plan_sha256=digest(root/'training_contribution_gate_plan.json'))
        contribution_path=root/'training_contribution_result.json'
        def write_contribution():contribution_path.write_text(json.dumps(contribution))
        for flag in ('ready','primary_gate_passed','posthoc_gate_passed','training_contribution_gate_passed'):
            contribution[flag]=False;write_contribution();rejected('failed_contribution_'+flag)
            contribution[flag]=True
        write_contribution()
        weights,plan_hash=verify_gate(root,'posterior')
        assert weights==folder/'checkpoint.pt' and plan_hash==report['plan_sha256']
        checks.append('matching_screen_accepted')
        (folder/'result.json').write_text('changed');rejected('changed_result')
        (folder/'result.json').write_text('result.json')
        (root/'plan.json').write_text('changed');rejected('changed_plan')
        (root/'plan.json').write_text('{}')
        (root/'analysis_plan.json').write_text('changed');rejected('changed_analysis_plan')
        (root/'analysis_plan.json').write_text('{}')
        contribution['source_primary_report_sha256']='changed'
        write_contribution();rejected('changed_primary_contribution_source')
        contribution['source_primary_report_sha256']=digest(path);write_contribution()
        (root/'training_contribution_gate_plan.json').write_text('changed')
        rejected('changed_contribution_plan')
        (root/'training_contribution_gate_plan.json').write_text('{}')
        del report['source_hashes']['posterior']['test.npz'];write();rejected('partial_source_manifest')
    result=dict(passed=True,checks=checks,
                scope='Gate verification with temporary synthetic files; not a trained policy or simulator run.')
    Path(__file__).with_name('posterior_native_gate_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
