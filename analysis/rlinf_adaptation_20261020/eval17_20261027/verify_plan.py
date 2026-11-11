"""Verify complete disjoint coverage and early paired results before submission."""
import ast
import json
from pathlib import Path
import tempfile

from eval_utils import assigned_cases, paired_report, schedule

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE/'plan.json').read_text())
assert plan['no_training'] and plan['checkpoint']['completed_updates']==17
assert plan['evaluation']['cases']==[[i,s] for i in range(301,311) for s in range(3)]
jobs = [schedule(plan,r) for r in (0,1)]
flat = [x for group in jobs for x in group]
assert len(flat)==54 and len(set(flat))==54
assert all(len(group)==27 for group in jobs)
assert sum(x[0]=='final' for x in flat)==30
assert sum(x[0]=='baseline' for x in flat)==24
assert {tuple(x) for x in assigned_cases(plan,0)}.isdisjoint({tuple(x) for x in assigned_cases(plan,1)})
assert {tuple(x[1:]) for group in jobs for x in group[:3]}=={tuple(x) for x in plan['reuse']['baseline_cases']}
for name in ('driver.py','bootstrap.py','deploy.py'):
    ast.parse((HERE/name).read_text())
tree=ast.parse((HERE/'driver.py').read_text())
calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
assert not calls.intersection({'build_optimizer','update_policy','save_checkpoint','collect','warmup_critic'})
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    initial={'physical_state':{'obj':[[0,0,0],[0,0,0,1]]},'proprio':[0,1],
             'physics_dt':1/120,'rendering_dt':1/30,'observation_dt':1/30}
    for phase,success in [('baseline',True),('final',False)]:
        folder=root/phase/'i301_s0';folder.mkdir(parents=True)
        (folder/'initial.json').write_text(json.dumps(initial))
        (folder/'result.json').write_text(json.dumps({'instance':301,'seed':0,'score':int(success),'success':success}))
    report=paired_report(root,[[301,0],[302,0]])
    assert report['completed_pairs']==1 and not report['comparison_complete']
    assert report['comparison_valid'] and report['success_to_failure']==1
    initial['physical_state']['obj'][0][0]=1.
    (root/'final/i301_s0/initial.json').write_text(json.dumps(initial))
    assert not paired_report(root,[[301,0]])['comparison_valid']
result={'passed':True,'new_episodes':54,'shard_episodes':[27,27],
        'trained_episodes':30,'baseline_new_episodes':24,'baseline_reused':6,
        'no_training_calls':True,'pairing_mismatch_rejected':True,'partial_results_not_reported_as_complete':True}
(HERE/'test_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
