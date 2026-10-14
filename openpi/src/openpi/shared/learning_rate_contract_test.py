import dataclasses
import json
import numpy as np
import optax
import jax.numpy as jnp
import pytest
from openpi.shared import learning_rate_contract as lr
from openpi.training.optimizer import CosineDecaySchedule


def fixture(tmp_path):
    schedule=CosineDecaySchedule(warmup_steps=200,peak_lr=2.5e-6,decay_steps=20000,decay_lr=5e-7)
    expected=lr.make_contract(schedule,peak_limit=2.5e-6)
    path=tmp_path/'lr.json';path.write_text(json.dumps(expected))
    return schedule,expected,path


def test_actual_optax_schedule_obeys_frozen_curve_and_peak(tmp_path):
    schedule,expected,path=fixture(tmp_path)
    assert lr.check_schedule(schedule,path)==expected
    for step in [0,1,200,1000,5000,10000,20000,40000]:
        assert float(schedule.create()(step))==pytest.approx(lr.expected_rate(expected,step),rel=2e-5)
    assert float(schedule.create()(40000))==pytest.approx(5e-7)


@pytest.mark.parametrize('changes',[dict(decay_steps=40000),dict(peak_lr=1.3755e-5),dict(warmup_steps=1000)])
def test_budget_extension_or_lr_jump_rejected(tmp_path,changes):
    schedule,_,path=fixture(tmp_path)
    with pytest.raises(ValueError):lr.check_schedule(dataclasses.replace(schedule,**changes),path)


def test_resume_preserves_real_optimizer_schedule_clock_and_contract(tmp_path):
    schedule,expected,path=fixture(tmp_path);tx=optax.chain(optax.clip_by_global_norm(1),optax.sgd(schedule.create()))
    state=tx.init(jnp.array(1.));grad=jnp.array(1.)
    for _ in range(7):_,state=tx.update(grad,state)
    lr.check_optimizer_clock(state,7)
    saved=state;update,next_state=tx.update(grad,saved)
    assert float(-update)==pytest.approx(lr.expected_rate(expected,7),rel=2e-5)
    lr.check_optimizer_clock(next_state,8)
    with pytest.raises(ValueError,match='counter'):lr.check_optimizer_clock(saved,0)
    lr.save(tmp_path,expected);lr.check_resume(tmp_path,expected)
    with pytest.raises(ValueError,match='changed'):lr.check_resume(tmp_path,lr.make_contract(dataclasses.replace(schedule,decay_steps=40000)))
    with pytest.raises(ValueError,match='without'):lr.check_resume(tmp_path,None)
