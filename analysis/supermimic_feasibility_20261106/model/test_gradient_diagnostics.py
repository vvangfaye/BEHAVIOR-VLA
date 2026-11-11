"""Synthetic gradient agreement tests; no evidence about physical correctness."""
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from gradient_diagnostics import analyze
from train_world_model import Transform, WorldModel, load_data
from test_world_model import synthetic


def checkpoint_fixture(tmp_path):
    path=tmp_path/'data.npz'
    synthetic(path)
    torch.set_num_threads(2)
    d=load_data(path)
    t=Transform(d,8)
    torch.manual_seed(21)
    model=WorldModel(3,8,hidden=32)
    positive=model.state_dict()
    negative=copy.deepcopy(positive)
    negative['net.4.weight'][-1]*=-1
    negative['net.4.bias'][-1]*=-1
    config={'goals':3,'visual_dim':8,'use_action':True,'use_visual':True,'hidden':32}
    package={'schema':'b1k_small_world_model_probe_v1','name':'full','config':config,
        'transform':t.save(),'model_states':[positive,copy.deepcopy(positive),negative],
        'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    full=tmp_path/'full.pt'
    torch.save(package,full)
    package={**package,'name':'no_action','config':{**config,'use_action':False}}
    control=tmp_path/'no_action.pt'
    torch.save(package,control)
    return path,full,control


def test_known_agreeing_and_opposite_model_gradients(tmp_path):
    data,full,control=checkpoint_fixture(tmp_path)
    result=analyze(data,full,no_action=control,batch_size=24,threads=2)
    pairs=result['agreement']['pairs']
    assert pairs[0]['members']==[0,1]
    assert pairs[0]['raw']['cosine']['mean']==pytest.approx(1,abs=1e-6)
    assert pairs[1]['raw']['cosine']['mean']==pytest.approx(-1,abs=1e-6)
    assert pairs[0]['raw']['sign_agreement_active_coordinate_fraction']['mean']==1
    assert result['perturbation']['by_member'][0]['predicted_positive_step_gain']['mean']>0
    assert result['perturbation']['by_member'][2]['predicted_positive_step_gain']['mean']<0
    assert result['perturbation']['observed_normalized_rms']['mean']==pytest.approx(.02,abs=1e-7)
    assert result['no_action_control']['max_abs_action_gradient']==0
    assert result['no_action_control']['max_abs_prediction_change']==0
    d=load_data(data)
    assert (d['split'][result['test_indices']]==2).all()
    assert d['_full_chunk'][result['test_indices']].all()


def test_hash_mismatch_rejected(tmp_path):
    data,full,_=checkpoint_fixture(tmp_path)
    package=torch.load(full,weights_only=True)
    package['data_sha256']='wrong'
    torch.save(package,full)
    with pytest.raises(ValueError,match='SHA256'):
        analyze(data,full,batch_size=4,threads=2)
