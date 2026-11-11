#!/usr/bin/env python3
"""Sequential long-horizon toy: chunked point-mass reaching with success termination (LIBERO-like credit assignment).

Episode: goal g (one of G fixed goals), start p0 ~ N(0, 0.3^2 I). At each of up to T decisions the policy outputs an
executed chunk u in R^{2c} (c sub-steps); position update p += 0.25 * sum_k tanh(u_k). Success (reward 1, episode
terminates) when ||p - g|| < r after a decision; otherwise truncation after T decisions (reward 0).
State h = [p, g, t/T] (5-d). Policy = native K-step flow ODE sampler conditioned on h (same family as espo_toy.py).
Each iteration: n_ep episodes; per-decision advantages by GAE (gamma 0.99, lambda 0.95) from a value MLP refit on
returns each iteration (shared by all methods). Methods: espo (CG, Q features), espo_reg (regression form, as in RLinf
v1), ppo (Flow-SDE chain log-prob, clip 0.2), awfm, qpath. Metric: success rate under the deployment sampler.
"""
import argparse
import json
import math

import torch
from torch import nn
from torch.func import functional_call

torch.set_default_dtype(torch.float64)
DH, C, T_MAX, K = 5, 4, 8, 5
DU = 2 * C
SDE_SIGMA = 0.35


class Velocity(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DU + 1 + DH, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, DU))

    def forward(self, x, t, h):
        return self.net(torch.cat([x, t, h], -1))


def ode_sample(model, params, h, eps):
    x = eps
    for k in range(K):
        x = x + functional_call(model, params, (x, torch.full_like(x[:, :1], k / K), h)) / K
    return x


def sde_sample(model, params, h, eps, xis):
    x, nexts = eps, []
    s = SDE_SIGMA / math.sqrt(K)
    for k in range(K):
        mu = x + functional_call(model, params, (x, torch.full_like(x[:, :1], k / K), h)) / K
        x = mu + s * xis[k]
        nexts.append(x)
    return x, nexts


def chain_logprob(model, params, h, eps, nexts):
    s = SDE_SIGMA / math.sqrt(K)
    x, lp = eps, 0.0
    for k in range(K):
        mu = x + functional_call(model, params, (x, torch.full_like(x[:, :1], k / K), h)) / K
        lp = lp + (-0.5 * ((nexts[k] - mu) / s) ** 2).sum(-1)
        x = nexts[k]
    return lp


def step_pos(p, u):
    return p + 0.25 * torch.tanh(u.view(-1, C, 2)).sum(1)


class MLP(nn.Module):
    def __init__(self, din, width=64, dout=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, dout))

    def forward(self, x):
        return self.net(x)


class QNet(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(DH + DU, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.head = nn.Linear(width, 1)

    def features(self, h, u):
        z = self.body(torch.cat([h, u], -1))
        return torch.cat([z, self.head(z)], -1)

    def forward(self, h, u):
        return self.head(self.body(torch.cat([h, u], -1))).squeeze(-1)


def fit(net, X, y, steps, lr=3e-3, seed=0, fn=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mu, sd = y.mean(), y.std() + 1e-8
    for _ in range(steps):
        opt.zero_grad()
        pred = fn(net, X) if fn else net(X).squeeze(-1)
        loss = ((pred - (y - mu) / sd) ** 2).mean()
        loss.backward()
        opt.step()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, mu, sd


def rollout(model, params, goals, n_ep, radius, gen, sde=False):
    """Returns per-decision tensors (h, u, eps, nexts, ep_id, t) and episode outcomes."""
    G = goals.shape[0]
    gi = torch.arange(n_ep) % G
    g = goals[gi]
    p = 0.3 * torch.randn(n_ep, 2, generator=gen)
    alive = torch.ones(n_ep, dtype=torch.bool)
    rec = {k: [] for k in ["h", "u", "eps", "ep", "t", "r", "done"]}
    nexts_all = []
    success = torch.zeros(n_ep)
    for t in range(T_MAX):
        idx = alive.nonzero().squeeze(-1)
        if idx.numel() == 0:
            break
        h = torch.cat([p[idx], g[idx], torch.full((idx.numel(), 1), t / T_MAX)], -1)
        eps = torch.randn(idx.numel(), DU, generator=gen)
        with torch.no_grad():
            if sde:
                xis = [torch.randn(idx.numel(), DU, generator=gen) for _ in range(K)]
                u, nexts = sde_sample(model, params, h, eps, xis)
                nexts_all.append(torch.stack(nexts, 1))
            else:
                u = ode_sample(model, params, h, eps)
        pn = step_pos(p[idx], u)
        succ = ((pn - g[idx]).norm(dim=-1) < radius)
        p = p.clone(); p[idx] = pn
        done = succ | (t == T_MAX - 1)
        rec["h"].append(h); rec["u"].append(u); rec["eps"].append(eps); rec["ep"].append(idx)
        rec["t"].append(torch.full((idx.numel(),), t)); rec["r"].append(succ.double()); rec["done"].append(done)
        success[idx[succ]] = 1.0
        alive[idx[succ]] = False
    out = {k: torch.cat(v) for k, v in rec.items()}
    if sde:
        out["nexts"] = torch.cat(nexts_all)
    return out, success


def gae(data, vnet, vmu, vsd, n_ep, gamma=0.99, lam=0.95):
    """Per-decision GAE advantages and returns, episode by episode (success terminates, truncation bootstraps 0)."""
    with torch.no_grad():
        V = vnet(data["h"]).squeeze(-1) * vsd + vmu
    adv = torch.zeros_like(V)
    ret = torch.zeros_like(V)
    for e in range(n_ep):
        m = (data["ep"] == e).nonzero().squeeze(-1)
        if m.numel() == 0:
            continue
        m = m[torch.argsort(data["t"][m])]
        a_next, v_next = 0.0, 0.0
        for j in reversed(range(m.numel())):
            i = m[j]
            last = j == m.numel() - 1
            delta = data["r"][i] + (0.0 if last else gamma * v_next) - V[i]
            a_next = delta + (0.0 if last else gamma * lam * a_next)
            adv[i] = a_next
            ret[i] = a_next + V[i]
            v_next = V[i]
    return adv, ret


def espo_cg(model, params, data, A, q, M, lam, gen, cg_iters=30):
    """Per-sample ESPO target b_i = A_i (phi(h_i,u_i) - E phi(h_i,.)); matrix-free damped LS via JVP/VJP + CG."""
    from torch.func import jvp, vjp
    h, u = data["h"], data["u"]
    N = h.shape[0]
    eps = torch.randn(N * M, DU, generator=gen)
    hs = h.repeat_interleave(M, 0)

    def Phi(p):
        return q.features(hs, ode_sample(model, p, hs, eps)).view(N, M, -1).mean(1)

    with torch.no_grad():
        b = A[:, None] * (q.features(h, u) - Phi(params))
    flat = lambda d: torch.cat([v.reshape(-1) for v in d.values()])

    def unflat(v):
        out, i = {}, 0
        for k, t in params.items():
            out[k] = v[i:i + t.numel()].view_as(t); i += t.numel()
        return out
    _, vjp_fn = vjp(Phi, params)
    Tt = lambda w: flat(vjp_fn(w)[0])
    Tv = lambda v: jvp(Phi, (params,), (unflat(v),))[1]
    G = lambda v: Tt(Tv(v)) / N + lam * v
    rhs = Tt(b) / N
    x = torch.zeros_like(rhs)
    if not torch.isfinite(rhs).all() or rhs.norm() < 1e-12:
        return unflat(x)
    r = rhs.clone(); pdir = r.clone(); rs = r @ r
    for _ in range(cg_iters):
        Gp = G(pdir); a = rs / (pdir @ Gp)
        x = x + a * pdir; r = r - a * Gp
        rs_new = r @ r
        if rs_new.sqrt() < 1e-8 * rhs.norm():
            break
        pdir = r + (rs_new / rs) * pdir; rs = rs_new
    return unflat(x)


def run(method, lr, seed, args):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    goals = torch.randn(args.goals, 2, generator=gen) * 1.2
    model = Velocity()
    # "SFT": imitate a noisy, biased straight-line expert so there is room to improve
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(2000):
        gi = torch.randint(0, args.goals, (256,), generator=gen)
        p = 0.3 * torch.randn(256, 2, generator=gen) + 0.5 * torch.randn(256, 2, generator=gen)
        tt = torch.randint(0, T_MAX, (256, 1), generator=gen).double()
        h = torch.cat([p, goals[gi], tt / T_MAX], -1)
        step = (goals[gi] - p) / C / 0.25 * 0.6 + 0.25 * torch.randn(256, 2, generator=gen) + torch.tensor([0.3, -0.2])
        demo = torch.atanh(step.clamp(-0.95, 0.95)).repeat(1, C) + args.demo_std * torch.randn(256, DU, generator=gen)
        x0 = torch.randn(256, DU, generator=gen); tau = torch.rand(256, 1, generator=gen)
        loss = ((model((1 - tau) * x0 + tau * demo, tau, h) - (demo - x0)) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    sde = method == "ppo"
    egen = torch.Generator().manual_seed(10_000 + seed)
    def evaluate(prm):
        g2 = torch.Generator().manual_seed(10_000 + seed)
        return rollout(model, prm, goals, 1024, args.radius, g2, sde=sde)[1].mean().item()
    curve = [evaluate(params)]
    adam_state = {}
    for it in range(args.iters):
        data, succ = rollout(model, params, goals, args.episodes, args.radius, gen, sde=sde)
        # value function on Monte-Carlo returns-to-go (gamma 0.99), then GAE
        rtg = torch.zeros(data["h"].shape[0])
        for e in range(args.episodes):
            m = (data["ep"] == e).nonzero().squeeze(-1)
            m = m[torch.argsort(data["t"][m])]
            g_ = 0.0
            for j in reversed(range(m.numel())):
                g_ = data["r"][m[j]] + 0.99 * g_
                rtg[m[j]] = g_
        vnet, vmu, vsd = fit(MLP(DH), data["h"], rtg, 300, seed=seed * 1000 + it)
        A, ret = gae(data, vnet, vmu, vsd, args.episodes)
        A = (A - A.mean()) / (A.std() + 1e-8)
        if method in ("espo", "espo_reg", "qpath"):
            q, _, _ = fit(QNet(), (data["h"], data["u"]), ret, 400, seed=seed * 1000 + it,
                          fn=lambda net, X: net(X[0], X[1]))
        if method == "espo":
            d = espo_cg(model, params, data, A, q, args.model_samples, args.lam, gen)
            params = {k: params[k] + lr * d[k] for k in params}
        else:
            p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
            o = torch.optim.Adam(list(p.values()), lr=lr)
            if adam_state:
                o.load_state_dict(adam_state)
            h, u = data["h"], data["u"]
            if method == "espo_reg":
                M = args.model_samples
                e2 = torch.randn(h.shape[0] * M, DU, generator=gen); h2 = h.repeat_interleave(M, 0)
                with torch.no_grad():
                    Phi0 = q.features(h2, ode_sample(model, params, h2, e2)).view(h.shape[0], M, -1).mean(1)
                    Y = Phi0 + args.reg_eta * A[:, None] * (q.features(h, u) - Phi0)
            if method == "ppo":
                with torch.no_grad():
                    nexts = [data["nexts"][:, k] for k in range(K)]
                    old_lp = chain_logprob(model, params, h, data["eps"], nexts)
            for _ in range(args.epochs):
                o.zero_grad()
                if method == "awfm":
                    x0 = torch.randn(u.shape, generator=gen); tau = torch.rand(u.shape[0], 1, generator=gen)
                    v = functional_call(model, p, ((1 - tau) * x0 + tau * u, tau, h))
                    loss = (A * ((v - (u - x0)) ** 2).sum(-1)).mean()
                elif method == "qpath":
                    e2 = torch.randn(h.shape[0], DU, generator=gen)
                    loss = -q(h, ode_sample(model, p, h, e2)).mean()
                elif method == "espo_reg":
                    Phi = q.features(h2, ode_sample(model, p, h2, e2)).view(h.shape[0], M, -1).mean(1)
                    loss = ((Phi - Y) ** 2).sum(-1).mean()
                else:  # ppo
                    lp = chain_logprob(model, p, h, data["eps"], nexts)
                    ratio = torch.exp(lp - old_lp)
                    loss = -torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A).mean()
                loss.backward(); o.step()
            adam_state = o.state_dict()
            params = {k: v.detach() for k, v in p.items()}
        curve.append(evaluate(params))
        if args.verbose:
            print(f"  {method} it {it} success {curve[-1]:.3f} batch {succ.mean():.3f}", flush=True)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="espo,espo_reg,ppo,awfm,qpath")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--goals", type=int, default=8)
    ap.add_argument("--radius", type=float, default=0.35)
    ap.add_argument("--demo-std", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--model-samples", type=int, default=16)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--reg-eta", type=float, default=3.0)
    ap.add_argument("--tune-seeds", default="0,1")
    ap.add_argument("--test-seeds", default="300,301,302,303,304")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default="seq_results.json")
    args = ap.parse_args()
    grids = {"espo": [0.3, 1.0, 3.0, 10.0], "espo_reg": [3e-4, 1e-3, 3e-3, 1e-2], "ppo": [1e-3, 3e-3, 1e-2, 3e-2],
             "awfm": [3e-4, 1e-3, 3e-3, 1e-2], "qpath": [3e-4, 1e-3, 3e-3, 1e-2]}
    tune = [int(s) for s in args.tune_seeds.split(",")]
    test = [int(s) for s in args.test_seeds.split(",")]
    out = {"args": vars(args), "tuning": {}, "test": {}}
    for m in args.methods.split(","):
        scores = {lr: sum(sum(c) / len(c) - c[0] for c in [run(m, lr, s, args) for s in tune]) / len(tune) for lr in grids[m]}
        best = max(scores, key=scores.get)
        curves = [run(m, best, s, args) for s in test]
        out["tuning"][m] = {"scores": scores, "best_lr": best}
        out["test"][m] = curves
        n = len(curves)
        mean = lambda i: sum(c[i] for c in curves) / n
        se = lambda i: math.sqrt(sum((c[i] - mean(i)) ** 2 for c in curves) / max(n - 1, 1) / n)
        print(f"{m:8s} lr={best:<7g} success@0 {mean(0):.3f}  @5 {mean(5):.3f}±{se(5):.3f}  @10 {mean(10):.3f}±{se(10):.3f}  "
              f"@{args.iters} {mean(-1):.3f}±{se(-1):.3f}", flush=True)
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
