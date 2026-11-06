"""Only pre-action proposal/context invariants; no simulator calls."""
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from proposal_outcome_probe import ProposalOutcome,ProposalTransform,preaction_context,prepare_inputs
from test_world_model import synthetic


def test_current_actual_length_and_done_do_not_enter_current_input(tmp_path):
    d=synthetic(tmp_path/'data.npz')
    d['executed_steps']=d['action_mask'].sum(1)
    context=preaction_context(d)
    t=ProposalTransform(d,context,8)
    before=prepare_inputs(d,context,t)
    # Only row 10's actual outcome length changes. It can affect future time,
    # but may not change row 10's own pre-action input.
    changed={key:value.copy() for key,value in d.items()}
    changed['executed_steps'][10]=3
    changed['action_mask'][10]=np.arange(16)<3
    changed['done'][10]=True
    after=prepare_inputs(changed,preaction_context(changed),t)
    for key in before:
        torch.testing.assert_close(before[key][:11],after[key][:11],atol=0,rtol=0)
    assert not torch.equal(before['time_features'][11],after['time_features'][11])


def test_request_mask_shared_but_actions_ablatable(tmp_path):
    d=synthetic(tmp_path/'data.npz')
    context=preaction_context(d,max_steps=310)
    assert context['remaining_steps'][19]==6
    assert context['request_mask'][19].sum()==6
    t=ProposalTransform(d,context,8);b=prepare_inputs(d,context,t)
    torch.manual_seed(7);full=ProposalOutcome(3,8,32,True)
    torch.manual_seed(7);no_action=ProposalOutcome(3,8,32,False)
    for name,p in full.state_dict().items():
        torch.testing.assert_close(p,no_action.state_dict()[name])
    altered={**b,'actions':b['actions']+2}
    for key,p in no_action(b).items():
        torch.testing.assert_close(p,no_action(altered)[key])
    assert not torch.equal(full(b)['reward'],full(altered)['reward'])
