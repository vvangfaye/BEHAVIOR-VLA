#!/usr/bin/env python3
"""Toy check of the Execution-Space Policy Optimization (ESPO) update against the exact policy gradient.

Policy: a small native finite-step flow sampler u = F_theta(h, eps) (K Euler steps of an MLP velocity field,
x_0 = eps ~ N(0, I)), the same structure as the pi0.5 action expert. Reward r(h, u) = -||u - u*(h)||^2 (+ noise),
so the exact objective gradient grad J = E[grad_theta r(h, F_theta(h, eps))] is available by reparameterization.

ESPO step (as specified in the CVPR registration / RL survey thread):
  per state s: desired feature shift  b_s = eta * mean_i A_i * (phi(h_s, u_i) - E_pi[phi(h_s, .)])  from real rollouts
               sampler response       T_s = d/dtheta E_eps[phi(h_s, F_theta(h_s, eps))]  from extra model samples
  delta* = argmin_delta  sum_s ||T_s delta - b_s||^2_{C_s^-1} + lam ||delta||^2   (CG with JVP/VJP, matrix-free)
Feature choices: 'mean' phi=u, 'moments' phi=[u, u^2], 'qfeat' hidden features of a return predictor fitted to rollouts.
Reports cos(delta, grad J) and the best achievable J improvement along delta, versus baselines that use the same rollouts:
  'awfm'   advantage-weighted flow-matching gradient (FPO/AWR-style surrogate),
  'qpath'  pathwise gradient through the sampler of the fitted return predictor (Diffusion-QL style).
"""
import argparse
import json
import math

import torch
from torch import nn
from torch.func import functional_call, jvp, vjp

torch.set_default_dtype(torch.float64)

# Toy configuration (set by the training script): action dim DU, reward TASK in {"smooth", "binary"}.
DU = 2
TASK = "smooth"
SUCCESS_RADIUS = 0.5
_G = torch.Generator().manual_seed(12345)
_PROJ = {}


class Velocity(nn.Module):
    def __init__(self, dx=None, dh=2, width=64):
        dx = DU if dx is None else dx
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dx + 1 + dh, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, dx))

    def forward(self, x, t, h):
        return self.net(torch.cat([x, t, h], dim=-1))


def sample(model, params, h, eps, K):
    """Native K-step Euler sampler, differentiable in params. h: [N, dh], eps: [N, dx]."""
    x = eps
    for k in range(K):
        t = torch.full_like(x[:, :1], k / K)
        x = x + functional_call(model, params, (x, t, h)) / K
    return x


def target(h):
    base = torch.stack([torch.sin(1.5 * h[:, 0]) + 0.5 * h[:, 1], 0.8 * h[:, 0] - torch.cos(h[:, 1])], dim=-1)
    if DU == 2:
        return base
    if DU not in _PROJ:  # fixed random nonlinear lift of the 2-d target to DU dims (same for all seeds)
        _PROJ[DU] = (torch.randn(2, DU, generator=_G) / 2 ** 0.5, torch.randn(DU, generator=_G))
    W, c = _PROJ[DU]
    return torch.tanh(base @ W + 0.3 * c)


def reward(h, u):
    d2 = ((u - target(h)) ** 2).sum(-1)
    if TASK == "binary":
        return (d2 < SUCCESS_RADIUS ** 2).to(u.dtype)
    return -d2


def J(model, params, H, K, eps):
    hs = H.repeat_interleave(eps.shape[0] // H.shape[0], dim=0)
    return reward(hs, sample(model, params, hs, eps, K)).mean()


def flat(d):
    return torch.cat([v.reshape(-1) for v in d.values()])


def unflat(vec, like):
    out, i = {}, 0
    for k, v in like.items():
        n = v.numel()
        out[k] = vec[i:i + n].view_as(v)
        i += n
    return out


class QNet(nn.Module):
    def __init__(self, dh=2, du=None, width=32):
        super().__init__()
        du = DU if du is None else du
        self.body = nn.Sequential(nn.Linear(dh + du, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.head = nn.Linear(width, 1)

    def features(self, h, u):
        z = self.body(torch.cat([h, u], dim=-1))
        return torch.cat([z, self.head(z)], dim=-1)

    def forward(self, h, u):
        return self.head(self.body(torch.cat([h, u], dim=-1))).squeeze(-1)


def fit_q(Hr, Ur, Rr, steps=1500, seed=0):
    torch.manual_seed(seed)
    q = QNet()
    opt = torch.optim.Adam(q.parameters(), lr=3e-3)
    mu, sd = Rr.mean(), Rr.std() + 1e-8
    for _ in range(steps):
        opt.zero_grad()
        loss = ((q(Hr, Ur) - (Rr - mu) / sd) ** 2).mean()
        loss.backward()
        opt.step()
    for p in q.parameters():
        p.requires_grad_(False)
    return q


def make_phi(kind, q):
    if kind == "mean":
        return lambda h, u: u
    if kind == "moments":
        return lambda h, u: torch.cat([u, u * u], dim=-1)
    if kind == "qfeat":
        return lambda h, u: q.features(h, u)
    if kind == "qval":
        return lambda h, u: q(h, u)[..., None]
    raise ValueError(kind)


def espo_direction(model, params, H, rollouts, phi, K, M, eta, lam, cg_iters, gen, whiten=True):
    """Matrix-free ESPO solve. rollouts: dict with per-state tensors U [S, n, du], A [S, n]."""
    S = H.shape[0]
    eps = torch.randn(S * M, DU, generator=gen)
    hs = H.repeat_interleave(M, dim=0)

    def feat_means(p):  # Phi_s(theta) for all states: [S, dphi]
        u = sample(model, p, hs, eps, K)
        f = phi(hs, u)
        return f.view(S, M, -1).mean(1), f.view(S, M, -1)

    with torch.no_grad():
        Phi0, F0 = feat_means(params)
        C = torch.einsum("smi,smj->sij", F0 - Phi0[:, None], F0 - Phi0[:, None]) / M
        C = C + 1e-3 * torch.eye(C.shape[-1]) * C.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None].clamp_min(1e-6)
        Cinv = torch.linalg.inv(C) if whiten else torch.eye(C.shape[-1]).expand_as(C).clone()
        Fr = phi(H[:, None].expand(-1, rollouts["U"].shape[1], -1).reshape(-1, 2), rollouts["U"].reshape(-1, DU))
        Fr = Fr.view(S, rollouts["U"].shape[1], -1)
        b = eta * (rollouts["A"][..., None] * (Fr - Phi0[:, None])).mean(1)  # [S, dphi]

    def T(v):  # JVP: [S, dphi]
        return jvp(lambda p: feat_means(p)[0], (params,), (unflat(v, params),))[1]

    _, vjp_fn = vjp(lambda p: feat_means(p)[0], params)

    def Tt(w):  # VJP
        return flat(vjp_fn(w)[0])

    def G(v):
        return Tt(torch.einsum("sij,sj->si", Cinv, T(v))) + lam * v

    rhs = Tt(torch.einsum("sij,sj->si", Cinv, b))
    x = torch.zeros_like(rhs)
    if not torch.isfinite(rhs).all() or rhs.norm() < 1e-12:
        return x  # no learning signal (e.g. every rollout at every state had the same outcome): no update
    r = rhs.clone()
    p = r.clone()
    rs = r @ r
    for _ in range(cg_iters):
        Gp = G(p)
        a = rs / (p @ Gp)
        x = x + a * p
        r = r - a * Gp
        rs_new = r @ r
        if rs_new.sqrt() < 1e-10 * rhs.norm():
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def awfm_direction(model, params, H, rollouts, K, gen):
    """Gradient of the advantage-weighted conditional flow-matching surrogate (FPO / AWR style)."""
    S, n = rollouts["A"].shape
    hs = H[:, None].expand(-1, n, -1).reshape(-1, 2)
    u = rollouts["U"].reshape(-1, DU)
    A = rollouts["A"].reshape(-1)
    x0 = torch.randn(u.shape, generator=gen)
    t = torch.rand(u.shape[0], 1, generator=gen)
    xt = (1 - t) * x0 + t * u
    p = {k: v.detach().requires_grad_(True) for k, v in params.items()}
    v = functional_call(model, p, (xt, t, hs))
    loss = (A * ((v - (u - x0)) ** 2).sum(-1)).mean()   # minimizing pulls toward A>0 actions, away from A<0
    g = torch.autograd.grad(loss, list(p.values()))
    return -torch.cat([x.reshape(-1) for x in g])


def qpath_direction(model, params, H, q, K, M, gen):
    S = H.shape[0]
    eps = torch.randn(S * M, DU, generator=gen)
    hs = H.repeat_interleave(M, dim=0)
    p = {k: v.detach().requires_grad_(True) for k, v in params.items()}
    val = q(hs, sample(model, p, hs, eps, K)).mean()
    g = torch.autograd.grad(val, list(p.values()))
    return torch.cat([x.reshape(-1) for x in g])


def best_step(model, params, d, H, K, eps_eval, J0):
    d = d / (d.norm() + 1e-12)
    best = (0.0, 0.0)
    base = flat(params)
    for alpha in [10 ** e for e in torch.linspace(-3, 0.5, 15).tolist()]:
        Jn = J(model, unflat(base + alpha * d, params), H, K, eps_eval).item()
        if Jn - J0 > best[0]:
            best = (Jn - J0, alpha)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--rollouts", type=int, default=16, help="real rollouts per state")
    ap.add_argument("--model-samples", type=int, default=64, help="extra model samples per state (M)")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--reward-noise", type=float, default=0.3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--lams", default="1e-4", help="comma list of damping values for the ESPO solve")
    ap.add_argument("--kinds", default="mean,moments,qfeat,qval")
    ap.add_argument("--whiten", default="1", help="comma list of 1/0")
    args = ap.parse_args()
    rows = []
    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        model = Velocity()
        params = {k: v.detach() for k, v in model.named_parameters()}
        H = torch.randn(args.states, 2, generator=gen)
        # "SFT" start: flow-match onto a biased, too-wide demo distribution so there is room to improve.
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(1500):
            hb = H[torch.randint(0, args.states, (256,), generator=gen)]
            demo = target(hb) + torch.tensor([0.6, -0.4]) + 0.5 * torch.randn(256, 2, generator=gen)
            x0 = torch.randn(256, 2, generator=gen)
            t = torch.rand(256, 1, generator=gen)
            loss = ((model((1 - t) * x0 + t * demo, t, hb) - (demo - x0)) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        params = {k: v.detach().clone() for k, v in model.named_parameters()}
        for prm in model.parameters():
            prm.requires_grad_(False)
        eps_eval = torch.randn(args.states * 4096, 2, generator=gen)
        pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
        Jtrue = J(model, pg, H, args.K, eps_eval)
        g_true = torch.cat([x.reshape(-1) for x in torch.autograd.grad(Jtrue, list(pg.values()))])
        J0 = Jtrue.item()
        # real rollouts from the current policy, observed noisy returns, per-state mean baseline
        n = args.rollouts
        hs = H[:, None].expand(-1, n, -1).reshape(-1, 2)
        with torch.no_grad():
            U = sample(model, params, hs, torch.randn(hs.shape[0], 2, generator=gen), args.K)
        R = reward(hs, U) + args.reward_noise * torch.randn(hs.shape[0], generator=gen)
        R = R.view(args.states, n)
        A = R - R.mean(1, keepdim=True)
        roll = {"U": U.view(args.states, n, 2), "A": A}
        q = fit_q(hs, U, R.reshape(-1), seed=seed)
        dirs = {}
        for kind in args.kinds.split(","):
            for lam in [float(x) for x in args.lams.split(",")]:
                for wh in [w == "1" for w in args.whiten.split(",")]:
                    dirs[f"espo_{kind}_l{lam:g}_w{int(wh)}"] = espo_direction(
                        model, params, H, roll, make_phi(kind, q), args.K, args.model_samples, eta=1.0, lam=lam,
                        cg_iters=50, gen=gen, whiten=wh)
        dirs["awfm"] = awfm_direction(model, params, H, roll, args.K, gen)
        dirs["qpath"] = qpath_direction(model, params, H, q, args.K, args.model_samples, gen)
        row = {"seed": seed, "J0": J0}
        for name, d in dirs.items():
            cos = (d @ g_true / (d.norm() * g_true.norm() + 1e-12)).item()
            gain, alpha = best_step(model, params, d, H, args.K, eps_eval, J0)
            row[name] = {"cos": cos, "best_gain": gain, "alpha": alpha}
        gain_true, _ = best_step(model, params, g_true, H, args.K, eps_eval, J0)
        row["true_grad_gain"] = gain_true
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in row.items()}))
    names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    print("\nmean over seeds (cos with exact gradient | best J gain along direction, exact-gradient gain "
          f"{sum(r['true_grad_gain'] for r in rows) / len(rows):.4f})")
    for nm in names:
        c = sum(r[nm]["cos"] for r in rows) / len(rows)
        g = sum(r[nm]["best_gain"] for r in rows) / len(rows)
        print(f"  {nm:24s} cos {c:+.3f}   gain {g:.4f}")
    if args.json:
        json.dump({"args": vars(args), "rows": rows}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
