#!/usr/bin/env python3
"""Multi-iteration toy training at a matched interaction budget (same rollouts per iteration for every method).

Methods (all start from the same flow-matched "SFT" sampler of espo_toy.py):
  espo   : ESPO step with return-predictive features (Q hidden + output), unwhitened, damping lam, theta += eta*delta
  ppo    : Flow-SDE PPO: stochastic K-step sampler with Gaussian transitions, chain log-prob, clip 0.2, 4 epochs
  awfm   : advantage-weighted conditional flow matching (Adam steps on the surrogate)
  qpath  : pathwise ascent on a fitted return predictor through the sampler (Adam steps)
  exact  : oracle ascent on the exact objective gradient (reference only; uses the true reward)
Advantages for every method: per-state mean baseline over that iteration's noisy rollouts.
J is evaluated on the method's own deployment sampler (ODE for all but ppo, SDE for ppo) with fixed eval noise.
"""
import argparse
import json
import math

import torch
from torch.func import functional_call

import espo_toy as T

torch.set_default_dtype(torch.float64)
SDE_SIGMA = 0.35  # per-step transition std of the PPO sampler (x_{k+1} = x_k + v/K + sigma/sqrt(K) * xi)


def sde_sample(model, params, h, eps0, xis, K):
    """Returns final actions and per-step (mean, x_next) for the chain log-prob."""
    x = eps0
    means, nexts = [], []
    s = SDE_SIGMA / math.sqrt(K)
    for k in range(K):
        t = torch.full_like(x[:, :1], k / K)
        mu = x + functional_call(model, params, (x, t, h)) / K
        x_next = mu + s * xis[k]
        means.append(mu)
        nexts.append(x_next)
        x = x_next
    return x, means, nexts


def chain_logprob(model, params, h, eps0, nexts, K):
    s = SDE_SIGMA / math.sqrt(K)
    x = eps0
    lp = 0.0
    for k in range(K):
        t = torch.full_like(x[:, :1], k / K)
        mu = x + functional_call(model, params, (x, t, h)) / K
        lp = lp + (-0.5 * ((nexts[k] - mu) / s) ** 2).sum(-1)
        x = nexts[k]
    return lp


def J_eval(model, params, H, K, eps_eval, sde_noise=None):
    hs = H.repeat_interleave(eps_eval.shape[0] // H.shape[0], dim=0)
    with torch.no_grad():
        if sde_noise is None:
            u = T.sample(model, params, hs, eps_eval, K)
        else:
            u, _, _ = sde_sample(model, params, hs, eps_eval, sde_noise, K)
    return T.reward(hs, u).mean().item()


def run(method, lr, seed, args):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = T.Velocity()
    H = torch.randn(args.states, 2, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    offset = torch.tensor([0.6, -0.4]) if T.DU == 2 else 0.5 * torch.randn(T.DU, generator=torch.Generator().manual_seed(7)) / (T.DU / 2) ** 0.5
    for _ in range(1500):  # same SFT start as espo_toy.py (biased, too-wide demo distribution)
        hb = H[torch.randint(0, args.states, (256,), generator=gen)]
        demo = T.target(hb) + offset + args.demo_std * torch.randn(256, T.DU, generator=gen)
        x0 = torch.randn(256, T.DU, generator=gen)
        t = torch.rand(256, 1, generator=gen)
        loss = ((model((1 - t) * x0 + t * demo, t, hb) - (demo - x0)) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in model.parameters():
        p.requires_grad_(False)
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    K, S, n = args.K, args.states, args.rollouts
    eps_eval = torch.randn(S * 2048, T.DU, generator=gen)
    sde_eval = [torch.randn(S * 2048, T.DU, generator=gen) for _ in range(K)] if method == "ppo" else None
    curve = [J_eval(model, params, H, K, eps_eval, sde_eval)]
    adam_state = {}
    for it in range(args.iters):
        hs = H[:, None].expand(-1, n, -1).reshape(-1, 2)
        eps0 = torch.randn(hs.shape[0], T.DU, generator=gen)
        if method == "ppo":
            xis = [torch.randn(hs.shape[0], T.DU, generator=gen) for _ in range(K)]
            with torch.no_grad():
                U, _, nexts = sde_sample(model, params, hs, eps0, xis, K)
                old_lp = chain_logprob(model, params, hs, eps0, nexts, K)
        elif method == "espo" and getattr(args, "espo_rollout_sde", 0):
            # ablation: behavior rollouts from the SDE sampler while ESPO targets the ODE sampler (ESPO v1 on LIBERO)
            xis = [torch.randn(hs.shape[0], T.DU, generator=gen) for _ in range(K)]
            with torch.no_grad():
                U, _, _ = sde_sample(model, params, hs, eps0, xis, K)
        else:
            with torch.no_grad():
                U = T.sample(model, params, hs, eps0, K)
        R = (T.reward(hs, U) + args.reward_noise * torch.randn(hs.shape[0], generator=gen)).view(S, n)
        A = (R - R.mean(1, keepdim=True))
        if method == "espo":
            q = T.fit_q(hs, U, R.reshape(-1), steps=800, seed=seed * 1000 + it)
            d = T.espo_direction(model, params, H, {"U": U.view(S, n, T.DU), "A": A},
                                 T.make_phi(getattr(args, "espo_kind", "qfeat"), q), K, args.model_samples,
                                 eta=1.0, lam=args.lam, cg_iters=40, gen=gen, whiten=bool(getattr(args, "espo_whiten", 0)))
            if getattr(args, "verbose", False):
                print(f"  it {it} J {curve[-1]:.3f} |delta| {d.norm():.4f} |theta| {T.flat(params).norm():.3f} "
                      f"batch-success {R.mean():.3f} signal-states {(A.abs().sum(1) > 0).sum().item()}", flush=True)
            params = T.unflat(T.flat(params) + lr * d, params)
        elif method == "espo_reg":
            # Regression form (RLinf-friendly, no JVP): targets Y_s = Phi_s(theta0) + eta * b_s from real returns,
            # then a few Adam steps on sum_s ||Phi_s(theta) - Y_s||^2 through the native sampler (fixed noise).
            q = T.fit_q(hs, U, R.reshape(-1), steps=800, seed=seed * 1000 + it)
            phi = T.make_phi("qfeat", q)
            Mreg = args.model_samples
            e2 = torch.randn(S * Mreg, T.DU, generator=gen)
            h2 = H.repeat_interleave(Mreg, dim=0)
            with torch.no_grad():
                Phi0 = phi(h2, T.sample(model, params, h2, e2, K)).view(S, Mreg, -1).mean(1)
                Fr = phi(hs, U).view(S, n, -1)
                b = (A[..., None] * (Fr - Phi0[:, None])).mean(1)
                Y = Phi0 + args.reg_eta * b
            p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
            o = torch.optim.Adam(list(p.values()), lr=lr)
            for e in range(args.epochs):
                o.zero_grad()
                Phi = phi(h2, T.sample(model, p, h2, e2, K)).view(S, Mreg, -1).mean(1)
                loss = ((Phi - Y) ** 2).sum(-1).mean()
                loss.backward()
                o.step()
            params = {k: v.detach() for k, v in p.items()}
        elif method in ("awfm", "qpath", "exact", "ppo"):
            p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
            o = torch.optim.Adam(list(p.values()), lr=lr)
            if adam_state:
                o.load_state_dict(adam_state)
            if method == "qpath":
                q = T.fit_q(hs, U, R.reshape(-1), steps=800, seed=seed * 1000 + it)
            An = (A / (A.std() + 1e-8)).reshape(-1)
            for e in range(args.epochs):
                o.zero_grad()
                if method == "awfm":
                    x0 = torch.randn(U.shape, generator=gen)
                    t = torch.rand(U.shape[0], 1, generator=gen)
                    xt = (1 - t) * x0 + t * U
                    v = functional_call(model, p, (xt, t, hs))
                    loss = (An * ((v - (U - x0)) ** 2).sum(-1)).mean()
                elif method == "qpath":
                    e2 = torch.randn(S * args.model_samples, T.DU, generator=gen)
                    h2 = H.repeat_interleave(args.model_samples, dim=0)
                    loss = -q(h2, T.sample(model, p, h2, e2, K)).mean()
                elif method == "exact":
                    e2 = torch.randn(S * args.model_samples, T.DU, generator=gen)
                    h2 = H.repeat_interleave(args.model_samples, dim=0)
                    loss = -T.reward(h2, T.sample(model, p, h2, e2, K)).mean()
                else:  # ppo
                    lp = chain_logprob(model, p, hs, eps0, nexts, K)
                    ratio = torch.exp(lp - old_lp)
                    loss = -torch.min(ratio * An, ratio.clamp(0.8, 1.2) * An).mean()
                loss.backward()
                o.step()
            adam_state = o.state_dict()
            params = {k: v.detach() for k, v in p.items()}
        curve.append(J_eval(model, params, H, K, eps_eval, sde_eval))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="espo,ppo,awfm,qpath,exact")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=4, help="optimizer steps per iteration for Adam-based methods")
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--model-samples", type=int, default=256)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--reward-noise", type=float, default=0.3)
    ap.add_argument("--tune-seeds", default="0,1")
    ap.add_argument("--test-seeds", default="200,201,202,203,204")
    ap.add_argument("--json", default="train_results.json")
    ap.add_argument("--du", type=int, default=2)
    ap.add_argument("--task", default="smooth", choices=["smooth", "binary"])
    ap.add_argument("--success-radius", type=float, default=0.5)
    ap.add_argument("--demo-std", type=float, default=0.5)
    ap.add_argument("--ppo-grid", default="3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--reg-eta", type=float, default=3.0, help="target scale for espo_reg")
    ap.add_argument("--espo-kind", default="qfeat", choices=["qfeat", "moments", "mean", "qval"])
    ap.add_argument("--espo-whiten", type=int, default=0)
    ap.add_argument("--espo-rollout-sde", type=int, default=0)
    ap.add_argument("--fixed-lr", type=float, default=None, help="skip tuning and use this step size")
    args = ap.parse_args()
    T.DU, T.TASK, T.SUCCESS_RADIUS = args.du, args.task, args.success_radius
    grids = {"espo": [0.3, 1.0, 3.0, 10.0], "ppo": [3e-4, 1e-3, 3e-3, 1e-2], "awfm": [3e-4, 1e-3, 3e-3, 1e-2],
             "qpath": [3e-4, 1e-3, 3e-3, 1e-2], "exact": [3e-4, 1e-3, 3e-3, 1e-2],
             "espo_reg": [3e-4, 1e-3, 3e-3, 1e-2]}
    grids["ppo"] = [float(x) for x in args.ppo_grid.split(",")]
    tune = [int(s) for s in args.tune_seeds.split(",")]
    test = [int(s) for s in args.test_seeds.split(",")]
    out = {"args": vars(args), "tuning": {}, "test": {}}
    for m in args.methods.split(","):
        scores = {}
        for lr in ([args.fixed_lr] if args.fixed_lr is not None else grids[m]):
            if args.fixed_lr is not None:
                scores[lr] = float("nan")
                continue
            curves = [run(m, lr, s, args) for s in tune]
            # tuning criterion: mean improvement averaged over the whole run (rewards fast AND stable gains)
            scores[lr] = sum(sum(c) / len(c) - c[0] for c in curves) / len(curves)
        best = args.fixed_lr if args.fixed_lr is not None else max(scores, key=scores.get)
        out["tuning"][m] = {"scores": scores, "best_lr": best}
        curves = [run(m, best, s, args) for s in test]
        out["test"][m] = curves
        n = len(curves)
        mean = [sum(c[i] - c[0] for c in curves) / n for i in range(len(curves[0]))]
        se = [math.sqrt(sum((c[i] - c[0] - mean[i]) ** 2 for c in curves) / max(n - 1, 1) / n) for i in range(len(curves[0]))]
        print(f"{m:6s} lr={best:<7g} dJ@10 {mean[10]:+.3f}±{se[10]:.3f}  dJ@20 {mean[20]:+.3f}±{se[20]:.3f}  "
              f"dJ@{args.iters} {mean[-1]:+.3f}±{se[-1]:.3f}   (J0 {sum(c[0] for c in curves) / n:.3f})", flush=True)
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
