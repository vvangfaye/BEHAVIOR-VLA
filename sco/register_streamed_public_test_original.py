"""Register a complete streamed original after frozen-scope and byte verification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from submit_public_test_shard import build_payload

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evidence', type=Path, required=True)
p.add_argument('--index', type=int, required=True)
p.add_argument('--original', type=Path, required=True)
p.add_argument('--transfer-proof', type=Path, required=True)
p.add_argument('--assignment-proof', type=Path, required=True)
a = p.parse_args()
r = a.evidence.resolve()
label = 'a'+str(a.index).zfill(3)
submission = json.loads((r/'production_submissions'/(label+'.json')).read_text())
expected = build_payload(json.loads((r/'preparation_ready.json').read_text()),json.loads((r/'production_plan.json').read_text()),json.loads((r/'pilot_accepted.json').read_text()),a.index,submission['name'],hashlib.sha256((r/'pilot_driver_source.py').read_bytes()).hexdigest())
assert submission['generation']==0 and all(submission[k]==v for k,v in expected.items())
assignment = json.loads(a.assignment_proof.read_text())
assert assignment['assignment_root']==submission['assignment_root'] and assignment['observer_job']['uid']==submission['uid']
assert assignment['status']['source_manifest_sha256']==submission['source_manifest_sha256'] and assignment['status']['checkpoint_step']==1688977
proof = json.loads(a.transfer_proof.read_text())
assert proof['state'] in ('original_bytes_verified','verified_existing_original')
file = a.original.resolve()
assert file.parent==r/'production_artifacts'/label
raw = file.read_bytes()
assert len(raw)==proof['bytes'] and hashlib.sha256(raw).hexdigest()==proof['sha256']
item = json.loads(raw)
assert item['task']==submission['task'] and item['instance_id'] in submission['instance_ids'] and item['rollout_id']==0
assert file.name=='%s_%s_0.json'%(item['task'],item['instance_id'])
assert proof['remote_path']==str(Path(submission['assignment_root'])/'results/pi05'/item['task']/'json'/file.name)
assert any(row['instance_id']==item['instance_id'] and row['state']=='succeeded' for row in assignment['queue'])
assert type(item['steps']) is int and item['steps']>0 and type(item['success']) is bool
score = item['q_score']['final']
assert type(score) in (int,float) and math.isfinite(score) and 0<=score<=1
assert item['conditioning']['mode']=='whole_task' and item['conditioning']['oracle'] is False and item['conditioning']['privileged'] is False
assert any(e['source']=='generated' for e in item['subtask_events']) and all(e['source'] in ('generated','cached') for e in item['subtask_events'])
assert math.isfinite(item['time']['normalized_time']) and item['time']['normalized_time']>0
assert all(math.isfinite(item['normalized_agent_distance'][s]) for s in ('base','left','right'))
row = dict(instance_id=item['instance_id'],bytes=len(raw),sha256=proof['sha256'],steps=item['steps'],success=item['success'],q_score=score)
report_path = r/('production_partial_results_'+label+'_current.json')
report = json.loads(report_path.read_text()) if report_path.exists() else dict(state='original_bytes_verified_full_shard_audit_pending',job_uid=submission['uid'],source_manifest_sha256=submission['source_manifest_sha256'],results=[])
assert report['job_uid']==submission['uid'] and report['source_manifest_sha256']==submission['source_manifest_sha256']
existing = [x for x in report['results'] if x['instance_id']==item['instance_id']]
assert not existing or existing==[row]
if not existing:report['results'].append(row)
report['results'].sort(key=lambda x:x['instance_id'])
report.update(observed_at=proof.get('completed_at',assignment['observed_at']),streamed_transfer_proof=str(a.transfer_proof.resolve().relative_to(r)),assignment_proof=str(a.assignment_proof.resolve().relative_to(r)))
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(index=a.index,originals_verified=len(report['results']),new_original=not existing,original_bytes=len(raw),sha256=proof['sha256'],subtask_events=len(item['subtask_events']),steps=item['steps'],success=item['success'],q_score=score,formally_accepted=False)))
