import copy
import json

import numpy as np
import pytest

from codex_jacobian_validation import assess, verify, CONTRACT


def probe_rows():
    return [dict(joint=i, sign=sign, actual_joint_motion_rad=.004,
        actual_translation_m=[.001*sign,0,0], predicted_translation_m=[.00099*sign,0,0],
        old_row_predicted_translation_m=[0,.001*sign,0], base_translation_m=1e-7,
        base_rotation_rad=1e-6) for i in range(7) for sign in (-1,1)]


def test_physical_motion_compares_candidates_and_accepts_accurate_one():
    result = assess(probe_rows())
    assert result['passed'] is True and result['translation_relative_error'] == pytest.approx(.01)
    assert result['old_row_translation_relative_error'] == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize('mutation', ['reverse','zero','missing','duplicate','base','nan'])
def test_rejects_wrong_prediction_or_invalid_observation(mutation):
    rows = probe_rows()
    if mutation == 'reverse':
        for r in rows: r['predicted_translation_m'][0] *= -1
    elif mutation == 'zero':
        for r in rows: r['actual_translation_m'] = [0,0,0]
    elif mutation == 'missing': rows.pop()
    elif mutation == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    elif mutation == 'base': rows[0]['base_translation_m'] = .01
    else: rows[0]['predicted_translation_m'][0] = float('nan')
    with pytest.raises(ValueError): assess(rows)


def test_verifier_binds_plan_and_excludes_training_data(tmp_path):
    plan = {'fingerprint':'a'*64}
    rows = probe_rows()
    result = dict(contract=CONTRACT, plan_fingerprint=plan['fingerprint'],
        passed=True, diagnostic_only=True, training_eligible=False,
        model_calls=0, probe_controls=42, rows=rows, assessment=assess(rows))
    path = tmp_path/'probe.json'
    path.write_text(json.dumps(result))
    assert verify(path,plan)['contract'] == CONTRACT
    for key, value in [('training_eligible',True),('model_calls',1),('probe_controls',41),('plan_fingerprint','b'*64)]:
        altered = dict(result, **{key:value});path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):verify(path,plan)


@pytest.mark.parametrize('offset', [0, 6])
def test_probe_restores_state_and_never_calls_model(monkeypatch, tmp_path, offset):
    from types import SimpleNamespace
    import codex_collection_actions as actions
    import codex_subtask_qualification as qualification
    import collect_b1k_codex_subtask as collector
    from codex_jacobian_validation import run

    robot = SimpleNamespace(q=np.zeros(28), articulation_root_path='/robot',
        eef_link_names={'left':'left_eef'}, control_limits={'position':(-np.ones(28),np.ones(28))})
    robot.get_joint_positions = lambda: robot.q.copy()
    robot.get_position_orientation = lambda: (np.zeros(3),np.array([0.,0.,0.,1.]))
    selected_row = 3 if offset == 6 else 2
    jac = np.zeros((1,4,6,28+offset));jac[0,selected_row,:3,10+offset:17+offset] = np.array([
        [1,.2,.1,.5,.1,.4,.2],[.1,1,.3,.1,.5,.2,.4],[.2,.3,1,.1,.2,.5,.1]])
    backend = SimpleNamespace(views=SimpleNamespace(
        get_all_relative_jacobians=lambda path:jac,
        get_member_view_indices=lambda path,members:np.array([0]),
        get_link_index=lambda path,link:3),
        pose=lambda r,arm:(jac[0,selected_row,:3,10+offset:17+offset]@r.q[10:17],np.array([0,0,0,1])))
    class Correction:
        def __init__(self, r):
            self.backend=backend
            self.action_indices={'arm_left':np.arange(7,14)}
            self.joint_indices={'arm_left':np.arange(10,17)}
        def begin(self, value): assert value['gripper']=='open'
        def next_action(self):return np.zeros(23)
    class Policy:
        def __init__(self, values,label): self.values=values;self.index=0
    original = object()
    evaluator = SimpleNamespace(robot=robot,policy=original,steps=0)
    def step():
        assert isinstance(evaluator.policy,Policy)
        robot.q[10:17]=evaluator.policy.values[evaluator.policy.index][7:14]
        evaluator.policy.index+=1;evaluator.steps+=1
        return False,False
    evaluator.step=step
    restored=[]
    def restore(*args):robot.q[:]=0;restored.append(True)
    monkeypatch.setattr(collector,'_restore',restore)
    monkeypatch.setattr(actions,'BoundedCartesianCorrection',Correction)
    monkeypatch.setattr(qualification,'OriginalActions',Policy)
    plan={'fingerprint':'a'*64,'sample':{'sample_id':'test','subtask':'pick up'},'source_plan':{}}
    result=run(evaluator,plan,{},object(),tmp_path/'validation.json')
    assert result['passed'] and result['probe_controls']==42
    assert evaluator.steps==42 and len(restored)==15
    assert evaluator.policy is original
    assert not robot.q.any()
