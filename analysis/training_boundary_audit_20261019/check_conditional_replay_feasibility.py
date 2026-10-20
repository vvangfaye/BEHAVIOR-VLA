"""CPU-only Gaussian counterchecks; not a VLA performance experiment."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20261020)
out = {"scope": "Analytic cost model and whitened Gaussian posterior; no neural model, GPU, or environment used."}

# Standardized target is N(0,I). Each original state is already stationary.
# A frozen independence proposal is N(shift*1, scale^2 I).
def replay_case(dim, scale, shift, n=6000, draws=16):
    x = rng.normal(size=(n, dim))
    first = x[:, 0].copy()
    total = first.copy()
    accepted = 0
    def log_weight(z):
        return -.5 * np.sum(z*z, axis=1) + .5 * np.sum(((z-shift)/scale)**2, axis=1)
    wx = log_weight(x)
    for _ in range(draws-1):
        y = shift + scale*rng.normal(size=(n, dim))
        wy = log_weight(y)
        take = np.log(rng.random(n)) < np.minimum(0, wy-wx)
        x[take] = y[take]
        wx[take] = wy[take]
        accepted += int(take.sum())
        total += x[:, 0]
    average = total/draws
    ratio = float(np.var(average)/np.var(first))
    return {"dim": dim, "proposal_sd_ratio": scale, "proposal_mean_shift_per_coordinate_in_target_sd": shift,
            "chains": n, "draws_including_original": draws,
            "acceptance": accepted/(n*(draws-1)), "conditional_variance_ratio": ratio,
            "effective_iid_draws_for_first_coordinate": 1/ratio,
            "mean_of_average": float(average.mean()), "sd_of_average": float(average.std())}

out["gaussian_posterior_replay"] = [replay_case(d,s,m) for d in [16,128,368]
                                      for s,m in [(1.,0.),(1.1,0.),(1.,.2),(1.2,.2)]]

# rho = fraction of gradient variance removable by exact conditional averaging.
# e_over_f = environment+other fixed cost / one joint-gradient evaluation.
# Assumes independent posterior draws and no posterior-training/proposal overhead.
out["optimistic_cost_frontier"] = []
for rho in [.1,.5,.9]:
    for e_over_f in [1,10,100]:
        candidates=[]
        for m in range(1,129):
            variance = 1-rho+rho/m
            cost = (e_over_f+m)/(e_over_f+1)
            candidates.append((1/(variance*cost),m,variance,cost))
        speed,m,v,c=max(candidates)
        out["optimistic_cost_frontier"].append({"removable_variance_fraction":rho,"fixed_cost_over_one_gradient":e_over_f,
            "best_integer_draws":m,"variance_ratio":v,"cost_ratio":c,"inverse_variance_per_cost_gain":speed})

out["guardrails"] = [
    "Gaussian posterior whitening is given, not learned; no claim about a VLA proposal's accuracy.",
    "Finite Monte Carlo errors affect acceptance and variance ratios; no confidence-interval claim.",
    "Inverse-variance efficiency does not establish task success or convergence speed.",
    "Cost frontier omits proposal fitting and correlated draws, making it optimistic.",
    "An invariant kernel started at the recorded latent needs a proposal fixed independently of that latent conditional on context/action; fitting on that same latent can invalidate this argument."
]
dest=Path(__file__).with_name("conditional_replay_feasibility_20261020.json")
dest.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
