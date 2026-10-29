"""Reuse the reviewed submission implementation with task-local identities."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
s=(ROOT.parent/'modem_parallel_20261029/deploy.py').read_text()
def replace(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a))
    s=s.replace(a,b)
replace('ROOT=HERE.parents[2]\nBASE=HERE.parent','ROOT=HERE.parents[3]\nBASE=HERE.parents[1]')
replace("    migration=json.loads((HERE/'migration_ready.json').read_text())\n    assert migration['old_worker_stopped'] and migration['first_case_complete']",
        "    assert plan['metadata_verified']\n    migration={'previous_generation_gpu_seconds_upper_bound':0}")
start=s.index("    dependency_status=json.loads(Path(plan['dependency_status_file']).read_text())")
end=s.index("    receipts=[",start)
s=s[:start]+s[end:]
replace("job=f'b1k-mdmp-1029-{label}r{rank}a{attempt}'","job=f\"{plan['job_prefix']}-{label}r{rank}a{attempt}\"")
replace("prefix=f'behavior1k/rlinf-modem-20261028/parallel8_20261029/{task}/'","prefix=plan['aoss_prefix']+task+'/'")
replace("'driver.py','goals.py','eval_utils.py','resource_watch.py','plan.json']", "'driver.py','goals.py','eval_utils.py','resource_watch.py','storage_guard.py','plan.json']")
(ROOT/'deploy_template.py').write_text(s)
print('Generated task-local submitter; no cloud operations')
