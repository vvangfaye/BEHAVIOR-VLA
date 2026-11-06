#!/usr/bin/env python3
"""Diagnose agreement of fitted models, not correctness of environment gradients."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bridge.score import FrozenWorldModelScore, world_model_class
from train_world_model import load_data


def stats(values):
    value=np.asarray(values,dtype=float).reshape(-1)
    value=value[np.isfinite(value)]
    if not len(value):
        return {"n":0,"mean":None,"median":None,"p10":None,"p90":None,"min":None,"max":None}
    return {"n":len(value),"mean":float(value.mean()),"median":float(np.median(value)),
        "p10":float(np.quantile(value,.1)),"p90":float(np.quantile(value,.9)),
        "min":float(value.min()),"max":float(value.max())}


class NoActionScore(nn.Module):
    """Explicit control; a non-action checkpoint is never forged into the bridge."""
    def __init__(self,checkpoint):
        super().__init__()
        if checkpoint["name"]!="no_action" or checkpoint["config"]["use_action"]:
            raise ValueError("Expected a no_action checkpoint")
        for key,value in checkpoint["transform"].items():
            self.register_buffer(key,value.detach().clone().float())
        self.models=nn.ModuleList()
        for parameters in checkpoint["model_states"]:
            model=world_model_class()(**checkpoint["config"])
            model.load_state_dict(parameters,strict=True)
            model.requires_grad_(False).eval()
            self.models.append(model)

    def forward(self,obs,goals,visual,actions,*,goal_valid,visual_valid):
        filled=torch.where(visual_valid[:,None],visual,self.visual_mean)
        projected=(filled-self.visual_mean)@self.components/self.visual_scale
        inputs={"state":(obs-self.state_mean)/self.state_scale,"goals":goals,"goal_valid":goal_valid,
            "visual":projected,"visual_valid":visual_valid[:,None].float(),
            "actions":(actions.flatten(1)-self.action_mean)/self.action_scale,
            "action_mask":torch.ones((len(obs),16),dtype=obs.dtype)}
        return torch.stack([m(inputs)["reward"][:,0] for m in self.models]).mean(0)*self.reward_scale[0]+self.reward_mean[0]


def agreement(gradients,scale,tolerance=1e-12):
    g=gradients.detach().cpu().numpy().reshape(len(gradients),len(gradients[0]),-1)
    s=scale.detach().cpu().numpy().reshape(-1)
    result={"norms_by_member":[],"pairs":[],"zero_tolerance":tolerance}
    for i,raw in enumerate(g):
        result["norms_by_member"].append({"member":i,"raw_action_gradient_l2":stats(np.linalg.norm(raw,axis=1)),
            "normalized_action_gradient_l2":stats(np.linalg.norm(raw*s,axis=1))})
    for i,j in itertools.combinations(range(len(g)),2):
        pair={"members":[i,j]}
        for name,factor in [("raw",1.),("normalized",s)]:
            a,b=g[i]*factor,g[j]*factor
            norm_a,norm_b=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1)
            denominator=norm_a*norm_b
            valid=(norm_a>tolerance)&(norm_b>tolerance)
            cosine=np.sum(a[valid]*b[valid],axis=1)/denominator[valid]
            active=(np.abs(a)>tolerance)|(np.abs(b)>tolerance)
            agrees=(np.sign(a)==np.sign(b))&active
            sign_rate=np.divide(agrees.sum(1),active.sum(1),out=np.full(len(a),np.nan),where=active.sum(1)>0)
            pair[name]={"cosine":stats(cosine),"cosine_rows_excluded_zero_norm":int((~valid).sum()),
                "sign_agreement_active_coordinate_fraction":stats(sign_rate),
                "active_coordinate_fraction":float(active.mean())}
        result["pairs"].append(pair)
    return result


def analyze(data,checkpoint,*,no_action=None,batch_size=256,perturb_rms=.02,threads=8,seed=7):
    if not 1<=batch_size<=256 or perturb_rms<=0:
        raise ValueError("batch_size must be 1..256 and perturb_rms positive")
    torch.set_num_threads(threads)
    d=load_data(data)
    digest=hashlib.sha256(data.read_bytes()).hexdigest()
    trained=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if trained.get("data_sha256") not in (None,digest):
        raise ValueError("Checkpoint and data SHA256 do not match")
    eligible=np.flatnonzero((d["split"]==2)&d["_full_chunk"])
    rng=np.random.default_rng(seed)
    ids=np.sort(rng.choice(eligible,size=min(batch_size,len(eligible)),replace=False))
    if not len(ids):
        raise ValueError("No full test chunks")
    obs=torch.as_tensor(d["obs_state"][ids],dtype=torch.float32)
    goals=torch.as_tensor(d["goals"][ids],dtype=torch.float32)
    visual=torch.as_tensor(d["visual_features"][ids],dtype=torch.float32)
    kwargs={"goal_valid":torch.as_tensor(d["goal_valid"][ids],dtype=torch.float32),
        "visual_valid":torch.as_tensor(d["visual_valid"][ids],dtype=torch.bool)}
    actions=torch.as_tensor(d["execution_actions"][ids],dtype=torch.float32)
    scale=trained["transform"]["action_scale"].float().reshape(1,16,23)
    bridges=[]
    gradients=[]
    scores=[]
    for weights in trained["model_states"]:
        member={**trained,"model_states":[weights]}
        bridge=FrozenWorldModelScore(member).cpu().eval()
        raw=actions.clone().requires_grad_()
        prediction=bridge(obs,goals,visual,raw,**kwargs)
        if not prediction.requires_grad:
            raise ValueError("Full model score is disconnected from raw actions")
        gradient=torch.autograd.grad(prediction.sum(),raw)[0]
        if not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite fitted-model action gradient")
        assert all(p.grad is None and not p.requires_grad for p in bridge.parameters())
        gradients.append(gradient.detach());scores.append(prediction.detach());bridges.append(bridge)
    stacked=torch.stack(gradients)
    normalized_mean_gradient=(stacked*scale).mean(0)
    norm=normalized_mean_gradient.square().flatten(1).mean(1).sqrt()
    direction=torch.where((norm>1e-12)[:,None,None],
        normalized_mean_gradient/norm.clamp_min(1e-12)[:,None,None],torch.zeros_like(normalized_mean_gradient))
    normalized_delta=perturb_rms*direction
    physical_delta=normalized_delta*scale
    with torch.no_grad():
        original=torch.stack(scores)
        positive=torch.stack([b(obs,goals,visual,actions+physical_delta,**kwargs) for b in bridges])
        negative=torch.stack([b(obs,goals,visual,actions-physical_delta,**kwargs) for b in bridges])
    gain=positive-original
    result={"scope":"Fitted-model internal gradient agreement only; no simulator, intervention or actor update.",
        "score_route":"FrozenWorldModelScore with one checkpoint member per bridge; all normalization/PCA/action-score operations remain torch until after autograd.",
        "gradient_coordinates":"Derivative of predicted raw reward with respect to raw execution_actions; normalized-coordinate comparisons multiply by training action_scale.",
        "device":"cpu","threads":threads,"data_sha256":digest,
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "members":len(bridges),"rows":len(ids),"test_indices":ids.tolist(),
        "eligible_test_full_chunks":len(eligible),"agreement":agreement(stacked,scale),
        "perturbation":{"rms_in_training_normalized_action_coordinates":perturb_rms,
            "observed_normalized_rms":stats(normalized_delta.square().flatten(1).mean(1).sqrt().numpy()),
            "raw_action_rms":stats(physical_delta.square().flatten(1).mean(1).sqrt().numpy()),
            "zero_mean_gradient_rows":int((norm<=1e-12).sum()),
            "direction":"ensemble mean gradient of predicted raw reward, transformed to normalized action coordinates",
            "by_member":[{"member":i,"predicted_positive_step_gain":stats(gain[i].numpy()),
                "positive_gain_fraction":float((gain[i]>0).float().mean()),
                "predicted_negative_step_gain":stats((negative[i]-original[i]).numpy())} for i in range(len(bridges))],
            "ensemble_mean_predicted_gain":stats(gain.mean(0).numpy()),
            "cross_member_gain_variance":stats(gain.var(0,unbiased=False).numpy()),
            "original_cross_member_reward_variance":stats(original.var(0,unbiased=False).numpy()),
            "perturbed_cross_member_reward_variance":stats(positive.var(0,unbiased=False).numpy())},
        "limitations":["Agreement does not establish a correct environment derivative: all models can share bias.",
            "The differentiated head predicts one-step raw reward, not Q or long-horizon return.",
            "Perturbed commands are evaluated only inside fitted models and are not executed or claimed safe/feasible.",
            "RMS perturbation uses train action scaling; mixed physical command units make raw norms hard to interpret.",
            "No real policy or success metric is changed by this diagnostic."]}
    if no_action is not None:
        control=torch.load(no_action,map_location="cpu",weights_only=True)
        if control.get("data_sha256") not in (None,digest):
            raise ValueError("Control and data SHA256 do not match")
        model=NoActionScore(control).eval()
        raw=actions.clone().requires_grad_()
        output=model(obs,goals,visual,raw,**kwargs)
        grad=torch.autograd.grad(output.sum(),raw,allow_unused=True)[0] if output.requires_grad else None
        with torch.no_grad():
            changed=model(obs,goals,visual,actions+physical_delta,**kwargs)
        result["no_action_control"]={"members":len(model.models),
            "action_gradient":"None/disconnected" if grad is None else "zero" if not grad.count_nonzero() else "NONZERO",
            "max_abs_action_gradient":0. if grad is None else float(grad.abs().max()),
            "max_abs_prediction_change":float((changed-output.detach()).abs().max()),
            "note":"Handled explicitly without passing a non-action checkpoint to the action-required bridge."}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--no-action",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--perturb-rms",type=float,default=.02)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--seed",type=int,default=7)
    args=parser.parse_args()
    output=analyze(args.data,args.checkpoint,no_action=args.no_action,batch_size=args.batch_size,
        perturb_rms=args.perturb_rms,threads=args.threads,seed=args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"members":output["members"],"rows":output["rows"],
        "gradient_pairs":output["agreement"]["pairs"],
        "ensemble_mean_predicted_gain":output["perturbation"]["ensemble_mean_predicted_gain"],
        "no_action_control":output.get("no_action_control")},ensure_ascii=False))


if __name__=="__main__":
    main()
