import copy
import json
from pathlib import Path

import pytest
import full_episode_failure_recovery as recovery
from full_episode_collection import digest, save, submission_generation


@pytest.fixture
def chain(tmp_path, monkeypatch):
    root=tmp_path/'root';history=root/'submission_history/0001';history.mkdir(parents=True)
    campaign=tmp_path/'campaign';campaign.mkdir()
    original=dict(roots=[str(root)],deadline_epoch=10000)
    save(campaign/'authorization.json',original)
    old=dict(index=1,task='task',instance_ids=[301,302],plan_fingerprint='f'*64,
             name='behavior-full-ep-ffffffff-a0001-g0',uid='old',generation=0)
    save(history/'g0.json',old)
    proof=dict(uid='old',state='FAILED',verified=True,old_writer_stopped=True,completion_present=False,
        missing_instance_ids=[302],preserved_episodes=[dict(instance_id=301,receipt_key='old-key',receipt_sha256='a'*64)])
    save(history/'proof.json',proof)
    failures=[dict(previous_uid='old',root=str(root),index=1,eligible=True,category='multipart_put_URLError')]
    failures.extend(dict(previous_uid='other'+str(i),eligible=False) for i in range(77))
    save(campaign/'failure_recovery_authorization.json',dict(kind='user_failure78_recovery',
        original_authorization_sha256=digest((campaign/'authorization.json').read_bytes()),
        roots=[str(root)],deadline_epoch=10000,max_additional_attempts_per_shard=1,max_pending_tasks=5,failures=failures))
    ext=dict(kind='user_failure78_recovery',campaign=str(campaign),generation=1,previous_uid='old',
        history='submission_history/0001/g0.json',history_sha256=digest((history/'g0.json').read_bytes()),
        evidence='submission_history/0001/proof.json',evidence_sha256=digest((history/'proof.json').read_bytes()),
        preserved_episodes=proof['preserved_episodes'],missing_instance_ids=[302])
    new=dict(old,name=old['name'][:-1]+'1',generation=1,uid='new',failure_recovery=ext)
    monkeypatch.setattr(recovery.time,'time',lambda:100)
    return root,campaign,old,new


def test_recovery_preserves_old_generation_and_only_missing_instances(chain):
    root,c,old,new=chain
    assert recovery.validate_preparation(root,old,new['failure_recovery'])['missing_instance_ids']==[302]
    assert submission_generation(root,new)==1
    assert json.loads((root/'submission_history/0001/g0.json').read_text())==old


@pytest.mark.parametrize('mutation', ['changed_missing','changed_history','wrong_generation','wrong_scope','stop','expired'])
def test_recovery_fails_closed(chain,mutation,monkeypatch):
    root,c,old,new=chain;ext=copy.deepcopy(new['failure_recovery'])
    if mutation=='changed_missing': ext['missing_instance_ids']=[301,302]
    if mutation=='changed_history': ext['history_sha256']='0'*64
    if mutation=='wrong_generation': ext['generation']=2
    if mutation=='wrong_scope':old['uid']='unlisted'
    if mutation=='stop':(root/'STOP').touch()
    if mutation=='expired':monkeypatch.setattr(recovery.time,'time',lambda:10001)
    with pytest.raises((AssertionError,StopIteration)):
        recovery.validate_preparation(root,old,ext)


def test_recovery_results_remain_readable_after_deadline_without_new_admission(chain,monkeypatch):
    root,c,old,new=chain
    monkeypatch.setattr(recovery.time,'time',lambda:10001)
    assert submission_generation(root,new)==1
    with pytest.raises(AssertionError):recovery.validate_preparation(root,old,new['failure_recovery'])


def final1000_authorization(chain):
    root, campaign, old, new = chain
    original = json.loads((campaign / 'authorization.json').read_text())
    original.update(kind='native_full_episode_independent_final1000', total_episodes=1000)
    save(campaign / 'authorization.json', original)
    auth = json.loads((campaign / 'failure_recovery_authorization.json').read_text())
    auth.update(kind='user_final1000_failure5_recovery', target_episodes=1000,
                preserve_all_existing_episodes=True,
                original_authorization_sha256=digest((campaign / 'authorization.json').read_bytes()))
    auth['failures'] = [dict(previous_uid='old', root=str(root), index=1, eligible=True,
                             category='abrupt_mid_episode_worker_exit')]
    auth['failures'].extend(dict(previous_uid='uid'+str(i), root=str(root), index=i, eligible=False)
                            for i in [24, 36, 78, 84])
    save(campaign / 'failure_recovery_authorization.json', auth)
    new['failure_recovery']['kind'] = auth['kind']
    return root, campaign, old, new


def test_final1000_finite_repair_preserves_data_and_skips_completed_retries(chain):
    root, campaign, old, new = final1000_authorization(chain)
    assert recovery.validate_preparation(root, old, new['failure_recovery'])['missing_instance_ids'] == [302]
    assert submission_generation(root, new) == 1
    (root / 'submissions').mkdir()
    save(root / 'submissions/0001.json', new)
    assert recovery.choose_recovery(campaign, 'unused', None) is None


@pytest.mark.parametrize('mutation', ['extra_failure', 'changed_index', 'extra_attempt', 'changed_original'])
def test_final1000_repair_cannot_expand_scope_or_budget(chain, mutation):
    root, campaign, old, new = final1000_authorization(chain)
    p = campaign / 'failure_recovery_authorization.json'
    auth = json.loads(p.read_text())
    if mutation == 'extra_failure': auth['failures'].append(dict(previous_uid='extra', index=90))
    if mutation == 'changed_index': auth['failures'][1]['index'] = 25
    if mutation == 'extra_attempt': auth['max_additional_attempts_per_shard'] = 2
    if mutation == 'changed_original': auth['original_authorization_sha256'] = '0'*64
    save(p, auth)
    with pytest.raises(AssertionError): recovery.load_authorization(campaign)


def test_published_coverage_never_recounts_preserved_or_partial_archives():
    plan = dict(run_id='test-run', fingerprint='f'*64)
    prefix = 'behavior1k/full-episode-collection/test-run/ffffffffffff/episodes/task/'
    old = prefix + '301/attempt-1.json'
    fresh = prefix + '302/attempt-1-g1.json'
    record = dict(task='task', instance_ids=[301,302], generation=1,
        failure_recovery=dict(preserved_episodes=[dict(instance_id=301,receipt_key=old)],
                              missing_instance_ids=[302]))
    objects = {old:100, old[:-5]+'.tar':200, fresh:100,
               prefix+'302/attempt-1.tar':200}
    covered, missing = recovery.published_coverage(plan,[record],objects)
    assert [v['instance_id'] for v in covered] == [301]
    assert [v['instance_id'] for v in missing] == [302]
    objects[fresh[:-5]+'.tar'] = 200
    covered, missing = recovery.published_coverage(plan,[record],objects)
    assert len(covered) == 2 and not missing
    with pytest.raises(AssertionError):recovery.published_coverage(plan,[record,record],objects)
