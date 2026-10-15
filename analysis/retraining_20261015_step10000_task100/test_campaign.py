import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'frozen'))
import controller
import candidate_checks
import submit_helpers

def read(path):return json.loads(path.read_text())

def test_scope_covers_every_task_once_without_repeating_twenty():
    plan=read(ROOT/'plan.json');reused={(x['report']['record']['task'],301) for x in plan['reuse_candidates']}
    pairs=set(map(tuple,plan['pairs']))
    assert len(pairs)==100 and len({p[0] for p in pairs})==100 and {p[1] for p in pairs}=={301}
    assert len(reused)==20 and len(pairs-reused)==80
    assert len(read(ROOT/'baseline_accepted.json')['records'])==100

def test_protocol_semantics_match_previously_accepted_closed_loop():
    old=read(ROOT.parent/'retraining_20261013/paired_quick20_reused1688977_step10000_20261014/protocol.json')
    new=read(ROOT/'protocol.json')
    assert {k:v for k,v in old.items() if k!='pairs'}=={k:v for k,v in new.items() if k!='pairs'}
    assert not new['external_agent'] and new['native_task_step_limits'] and new['write_video']

def test_native_server_and_evaluator_commands_preserved():
    old=ROOT.parent/'retraining_20261013/paired_quick20_reused1688977_step10000_20261014/frozen/worker.py'
    def commands(path):
        tree=ast.parse(path.read_text());out={}
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in ('server_cmd','eval_cmd'):
                        out[target.id]=ast.dump(node.value)
        return out
    assert commands(old)==commands(ROOT/'frozen/worker.py')

def test_unknown_or_starting_requests_consume_pending_slots():
    def row(state,gpu=1):return dict(intent={'gpu':gpu},observation={'state':state})
    rows={str(i):row(s) for i,s in enumerate(['RUNNING','STARTING','UNKNOWN','SUCCEEDED','QUEUEING'])}
    rows['cpu']=row('RUNNING',0)
    assert controller.counts(rows)==(3,4,1)

def test_existing_intent_never_calls_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(submit_helpers,'ROOT',tmp_path);(tmp_path/'intents').mkdir()
    (tmp_path/'intents/candidate-000.json').write_text('{}')
    with pytest.raises(ValueError,match='One creation per exact intent'):
        submit_helpers.submit(None,{},'candidate-000','evaluate',role='candidate')

def test_partial_results_do_not_count_missing_tasks_as_failures(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    baseline=read(ROOT/'baseline_accepted.json');(tmp_path/'baseline_accepted.json').write_text(json.dumps(baseline))
    reused=[dict(record=x['report']['record']) for x in read(ROOT/'plan.json')['reuse_candidates']]
    rows={'preflight':dict(intent={'gpu':0},report=dict(state='accepted',reused_candidate_results=reused))}
    result=controller.summarize(read(ROOT/'plan.json'),rows)
    assert result['completed_pairs']==20 and not result['complete']
    assert result['completed_pair_metrics']['candidate_success_rate']==0.05
    assert not (tmp_path/'comparison.json').exists()

class Response(io.BytesIO):pass
class Opener:
    def __init__(self,body):self.body=body;self.requests=[]
    def open(self,request,timeout):
        if isinstance(request,str):return Response(self.body)
        self.requests.append(request)
        raise urllib.error.HTTPError('unused',412,'exists',{},None)

def test_existing_artifact_must_match_exact_bytes(tmp_path,monkeypatch):
    path=tmp_path/'source';path.write_bytes(b'original video')
    opener=Opener(b'original video');monkeypatch.setattr(candidate_checks.urllib.request,'build_opener',lambda *a:opener)
    receipt=candidate_checks.publish_file(path,dict(key='exact',put='https://unused/put',get='https://unused/get'))
    assert receipt['full_destination_readback'] and receipt['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert opener.requests[0].get_header('If-none-match')=='*'
    opener.body=b'changed video'
    with pytest.raises(AssertionError):candidate_checks.publish_file(path,dict(key='exact',put='https://unused/put',get='https://unused/get'))

def test_frozen_overlay_hashes_cover_actual_submitted_files():
    frozen=read(ROOT/'frozen_sha256.json')
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'frozen').glob('*.py')}==frozen
