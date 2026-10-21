"""Development check: full-data weak-moment solve on fixed learned features."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from multimodal_score import events, generate, marginal_score


ROOT = Path(__file__).resolve().parent


def evaluate(prediction, truth, advantages, gradient):
    estimate = advantages.T@prediction/len(prediction)
    error = float(np.linalg.norm(estimate-gradient)/np.linalg.norm(gradient))
    cosine = (estimate*gradient).sum(1)/(np.linalg.norm(estimate,axis=1)*np.linalg.norm(gradient,axis=1))
    mse = float(np.square(prediction-truth).mean()/np.square(truth).mean())
    return dict(relative_gradient_error=error, event_gradient_cosines=cosine.tolist(),
                relative_score_mse=mse, passed=bool(error<.25 and np.all(cosine>.9) and mse<.25))


def main():
    plan_path = ROOT/'head_solve_plan.json'
    plan = json.loads(plan_path.read_text())
    original = json.loads((ROOT/'multimodal_plan.json').read_text())
    output = ROOT/'head_solve_result.json'
    if output.exists():
        raise RuntimeError('Existing result; no automatic rerun')
    torch.set_num_threads(4)
    started = time.monotonic(); rows = []
    for dimension in plan['dimensions']:
        for seed in plan['seeds']:
            torch.manual_seed(seed)
            rng = torch.Generator().manual_seed(seed+1000+dimension)
            xtrain, tangent = generate(original['training_samples'], dimension, rng)
            xtest, _ = generate(original['test_samples'], dimension, rng)
            width = original['hidden_width']
            model = nn.Sequential(nn.Linear(dimension,width),nn.SiLU(),nn.Linear(width,width),
                                  nn.SiLU(),nn.Linear(width,3))
            opt = torch.optim.AdamW(model.parameters(),lr=original['learning_rate'],
                                   weight_decay=original['weight_decay'])
            for update in range(original['updates']):
                if time.monotonic()-started > plan['max_cpu_seconds']:
                    raise TimeoutError('Fixed CPU time cap reached')
                index = torch.randint(len(xtrain),(original['batch_size'],),generator=rng)
                x = xtrain[index].clone().requires_grad_(True)
                v = (2*torch.randint(2,(len(index),3),generator=rng)-1).float()
                projected = (model(x)*v).sum(1)
                derivative = torch.autograd.grad(projected.sum(),x,create_graph=True)[0][:,0]
                loss = (.5*projected.square()-derivative*(tangent[index]*v).sum(1)).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.inference_mode():
                sgd_prediction = torch.cat([model(x) for x in xtest.split(10000)]).numpy().astype(np.float64)
            features = model[:-1].double().eval()
            gram = torch.zeros((width+1,width+1),dtype=torch.float64)
            rhs = torch.zeros((width+1,3),dtype=torch.float64)
            eps = plan['feature_derivative_half_step']
            with torch.inference_mode():
                for begin in range(0,len(xtrain),2000):
                    x = xtrain[begin:begin+2000].double()
                    h = tangent[begin:begin+2000].double()
                    phi = torch.cat([features(x),torch.ones((len(x),1),dtype=torch.float64)],1)
                    plus,minus = x.clone(),x.clone();plus[:,0]+=eps;minus[:,0]-=eps
                    derivative = torch.cat([(features(plus)-features(minus))/(2*eps),
                                            torch.zeros((len(x),1),dtype=torch.float64)],1)
                    gram += phi.T@phi/len(xtrain)
                    rhs += derivative.T@h/len(xtrain)
                ridge = plan['ridge_trace_multiplier']*float(gram.trace())/(width+1)
                penalty = torch.eye(width+1,dtype=torch.float64)*ridge;penalty[-1,-1]=0
                weights = torch.linalg.solve(gram+penalty,rhs)
                solved_prediction = []
                for x in xtest.double().split(10000):
                    phi = torch.cat([features(x),torch.ones((len(x),1),dtype=torch.float64)],1)
                    solved_prediction.append(phi@weights)
                solved_prediction = torch.cat(solved_prediction).numpy()
            test = xtest.numpy().astype(np.float64)
            truth = marginal_score(test);advantage,gradient=events(test)
            row = dict(dimension=dimension,seed=seed,ridge=ridge,
                       sgd=evaluate(sgd_prediction,truth,advantage,gradient),
                       solved=evaluate(solved_prediction,truth,advantage,gradient),
                       moment_residual=float(torch.linalg.norm(gram@weights-rhs)),
                       regularized_condition_number=float(torch.linalg.cond(gram+penalty)))
            rows.append(row);print(json.dumps(row),flush=True)
    report = dict(complete=True,observed_at=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  original_plan_sha256=hashlib.sha256((ROOT/'multimodal_plan.json').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started,results=rows,
                  development_gate_passed=all(r['solved']['passed'] for r in rows),
                  scope='Development analysis on the earlier synthetic design; no native VLA or task outcome claim.',
                  new_gpu_jobs=0,new_simulator_episodes=0,goal_complete=False)
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
