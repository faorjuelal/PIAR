"""Physics-informed generative Full-Partial Reconstruction Mapping (FPRM).

FPRM (Wu et al., Nat. Commun. 2026) maps the delay reconstruction of a complete series
(M_F) onto the reconstruction of an incomplete series (M_P) and uses the map to impute
the missing values. Here the Gaussian-process regressor of the original framework is
replaced by a conditional normalizing flow (neural spline flow, zuko) that returns the
full conditional distribution p(y_t | c_t) of the missing values given a context c_t:

    c_t = [ delay window of the complete channel,
            observed neighbours of the incomplete channel (zero-filled) + their mask,
            (optional) forcing phase (cos Omega t, sin Omega t) ]

Baselines share exactly the same context: GPR (per output, original FPRM uses only the
delay window), a conditional VAE and a conditional DDPM. The physics-informed variant adds
pointwise residuals of the holonomic constraint and of the equation of motion evaluated on
reparameterised flow samples.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import zuko
from scipy.signal import savgol_filter
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from .chaos import trajectory
from .stochastic import baoab_series
from .embedding import ami, first_minimum, fnn_fraction

torch.set_num_threads(1)
F32 = torch.float32


# ============================================================================ data
def forced_channels(p, n_samples=40000, spp=128, per_period=32, n_transient=400, seed=0, th0=0.0, om0=0.0):
    """Channels of the driven PIAR sampled per_period times per forcing period."""
    tr = trajectory(p, th0, om0, n_transient + n_samples // per_period, spp=spp, save_per_period=per_period)
    tr = tr[n_transient * per_period:]
    t, th, om = tr[:, 0], tr[:, 1], tr[:, 2]
    return {"t": t, "theta": th, "s": np.sin(th), "c": np.cos(th), "omega": om,
            "dt": t[1] - t[0], "phase": p.Omega * t}


def langevin_channels(lam, beta, gamma, T, n_samples=40000, dt_sample=0.2, dt=0.01, seed=0):
    every = int(round(dt_sample / dt))
    th, om = baoab_series(math.pi, 0.0, dt, (n_samples + 2000) * every, every, lam, beta, gamma, T, seed)
    th, om = th[2000:], om[2000:]
    t = np.arange(len(th)) * dt_sample
    return {"t": t, "theta": th, "s": np.sin(th), "c": np.cos(th), "omega": om, "dt": dt_sample,
            "phase": None}


def choose_embedding(x, max_lag=60, E_max=8):
    I = ami(x, max_lag=max_lag)
    tau = first_minimum(I)
    f = fnn_fraction(x, tau, E_max=E_max)
    E = int(np.argmax(f < 0.01) + 1) if np.any(f < 0.01) else int(np.argmin(f) + 1)
    return tau, E, I, f


def build_context(x_full, y_part, mask, tau, m, K, phase=None, idx=None):
    """Context rows for the time indices idx (default: all admissible)."""
    N = len(x_full)
    halo = max(m * tau, K)
    if idx is None:
        idx = np.arange(halo, N - halo)
    lags = np.arange(-m, m + 1) * tau
    XF = np.stack([x_full[idx + l] for l in lags], axis=1)
    nb = [k for k in range(-K, K + 1) if k != 0]
    d = y_part.shape[1]
    YP, MK = [], []
    for k in nb:
        mk = mask[idx + k]                       # (n, d) availability
        YP.append(np.where(mk, y_part[idx + k], 0.0))
        MK.append(mk.astype(float))
    ctx = [XF, np.concatenate(YP, axis=1), np.concatenate(MK, axis=1)]
    if phase is not None:
        ctx.append(np.stack([np.cos(phase[idx]), np.sin(phase[idx])], axis=1))
    return np.concatenate(ctx, axis=1), idx, XF.shape[1]


# ============================================================================ physics residuals
class Physics:
    """Pointwise residuals. mode 'C1': s known, targets (c, omega); 'C2'/'C3': c known, target s."""

    def __init__(self, mode, s=None, c=None, dt=None, lam=None, beta=None, gamma=None, A=None,
                 Omega=None, t=None, win=7, poly=4):
        self.mode = mode
        self.lam, self.beta, self.gamma, self.A, self.Omega = lam, beta, gamma, A, Omega
        self.t = t
        if mode == "C1":
            self.x = s
        else:
            self.x = c
        self.xd = savgol_filter(self.x, win, poly, deriv=1, delta=dt)
        self.xdd = savgol_filter(self.x, win, poly, deriv=2, delta=dt)

    def residuals(self, Y, idx, ymean, ystd):
        """Y: (..., n, d) samples in standardized units; returns list of residual tensors."""
        Yp = Y * ystd + ymean
        x = torch.as_tensor(self.x[idx], dtype=F32)
        xd = torch.as_tensor(self.xd[idx], dtype=F32)
        xdd = torch.as_tensor(self.xdd[idx], dtype=F32)
        out = []
        if self.mode == "C1":
            s, sd, sdd = x, xd, xdd
            c, om = Yp[..., 0], Yp[..., 1]
            out.append(c ** 2 + s ** 2 - 1.0)                            # holonomic
            out.append(c * om - sd)                                      # kinematic: s' = c omega
            if self.A is not None:
                f = torch.as_tensor(self.A * np.cos(self.Omega * self.t[idx]), dtype=F32)
                rhs = self.beta * s - self.lam * s * c - self.gamma * om + f
                out.append(sdd + s * om ** 2 - c * rhs)                  # EOM (times cos theta)
        else:
            c, cd, cdd = x, xd, xdd
            s = Yp[..., 0]
            out.append(s ** 2 + c ** 2 - 1.0)
            if self.mode == "C3" and self.A is not None:
                f = torch.as_tensor(self.A * np.cos(self.Omega * self.t[idx]), dtype=F32)
                R = (-s ** 2 * cdd - c * cd ** 2 - self.gamma * cd * s ** 2
                     + self.lam * s ** 4 * c - self.beta * s ** 4 - f * s ** 3)
                out.append(R)                                            # EOM (times sin^3 theta)
        return out


# ============================================================================ models
class Standardizer:
    def __init__(self, X):
        self.m = X.mean(axis=0)
        self.s = X.std(axis=0) + 1e-8

    def __call__(self, X):
        return (X - self.m) / self.s

    def inv(self, Z):
        return Z * self.s + self.m


def _batches(n, bs, rng):
    perm = rng.permutation(n)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


def _split(n, rng, frac=0.1, n_min=200):
    """Hold out a validation subset of the labelled rows (early stopping)."""
    perm = rng.permutation(n)
    nv = max(n_min, int(frac * n))
    return perm[nv:], perm[:nv]


class _Checkpoint:
    """Keeps the parameters with the lowest validation loss among eligible evaluations."""

    def __init__(self, modules):
        self.modules, self.best, self.state, self.best_step = modules, float("inf"), None, -1

    def update(self, val, step):
        if val < self.best:
            self.best, self.best_step = val, step
            self.state = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in self.modules]

    def restore(self):
        if self.state is not None:
            for m, s in zip(self.modules, self.state):
                m.load_state_dict(s)


class NSFImputer:
    name = "nsf"

    def __init__(self, dim, ctx_dim, hidden=(128, 128), transforms=3, bins=8, steps=2500, lr=1e-3,
                 bs=256, physics=None, w_phys=0.0, hinge=0.0, seed=0):
        torch.manual_seed(seed)
        self.flow = zuko.flows.NSF(features=dim, context=ctx_dim, transforms=transforms,
                                   hidden_features=hidden, bins=bins)
        self.steps, self.lr, self.bs, self.seed = steps, lr, bs, seed
        self.physics, self.w_phys, self.hinge = physics, w_phys, hinge
        self.dim = dim
        self.eval_every = 50

    def _phys_loss(self, dist, idx_b, ym, ys):
        ys_ = dist.rsample((8,))                                          # (8, B, d) reparameterised
        rs = self.physics.residuals(ys_, idx_b, ym, ys)
        tot = 0.0
        for r, sc in zip(rs, self.r_scale):
            z2 = (r / sc) ** 2
            # hinge > 0: discrepancy band -- residuals within sqrt(hinge) noise units are not penalised
            tot = tot + (torch.relu(z2 - self.hinge).mean() if self.hinge > 0 else z2.mean())
        return tot

    def fit(self, X, Y, idx, X_col=None, idx_col=None):
        """X_col/idx_col: collocation rows (all time indices, labelled or not) for the residuals."""
        self.sx, self.sy = Standardizer(X), Standardizer(Y)
        Xt = torch.as_tensor(self.sx(X), dtype=F32)
        Yt = torch.as_tensor(self.sy(Y), dtype=F32)
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.Adam(self.flow.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps, eta_min=self.lr * 0.05)
        ym, ys = torch.as_tensor(self.sy.m, dtype=F32), torch.as_tensor(self.sy.s, dtype=F32)
        if self.physics is not None and self.w_phys > 0:
            with torch.no_grad():
                r0 = self.physics.residuals(Yt, idx, ym, ys)
            self.r_scale = [float(torch.sqrt((r ** 2).mean()) + 1e-6) for r in r0]
        Xc = torch.as_tensor(self.sx(X_col), dtype=F32) if X_col is not None else None
        itr, iva = _split(len(Xt), rng)
        ck = _Checkpoint([self.flow])
        step = 0
        t0 = time.time()
        while step < self.steps:
            for bb in _batches(len(itr), self.bs, rng):
                b = itr[bb]
                dist = self.flow(Xt[b])
                loss = -dist.log_prob(Yt[b]).mean()
                ramp = min(1.0, max(0.0, (step - 0.1 * self.steps) / (0.1 * self.steps)))
                if self.physics is not None and self.w_phys > 0 and ramp > 0:
                    if Xc is None:
                        lp = self._phys_loss(dist, idx[b], ym, ys)
                    else:
                        bc = rng.integers(0, len(Xc), self.bs)
                        lp = self._phys_loss(self.flow(Xc[bc]), idx_col[bc], ym, ys)
                    loss = loss + ramp * self.w_phys * lp
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 5.0)
                opt.step()
                sched.step()
                step += 1
                if step >= 0.2 * self.steps and (step % self.eval_every == 0 or step == self.steps):
                    with torch.no_grad():
                        ck.update(float(-self.flow(Xt[iva]).log_prob(Yt[iva]).mean()), step)
                if step >= self.steps:
                    break
        ck.restore()
        self.best_step, self.val_nll = ck.best_step, ck.best
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def sample(self, X, n=64):
        Xt = torch.as_tensor(self.sx(X), dtype=F32)
        S = self.flow(Xt).sample((n,)).numpy()                           # (n, N, d)
        return S * self.sy.s + self.sy.m


class CVAEImputer:
    name = "cvae"

    def __init__(self, dim, ctx_dim, latent=4, hidden=128, steps=2500, lr=1e-3, bs=256, seed=0):
        torch.manual_seed(seed)
        self.enc = nn.Sequential(nn.Linear(dim + ctx_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 2 * latent))
        self.dec = nn.Sequential(nn.Linear(latent + ctx_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 2 * dim))
        self.latent, self.steps, self.lr, self.bs, self.seed, self.dim = latent, steps, lr, bs, seed, dim

    def fit(self, X, Y, idx=None):
        self.sx, self.sy = Standardizer(X), Standardizer(Y)
        Xt = torch.as_tensor(self.sx(X), dtype=F32)
        Yt = torch.as_tensor(self.sy(Y), dtype=F32)
        params = list(self.enc.parameters()) + list(self.dec.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps, eta_min=self.lr * 0.05)
        rng = np.random.default_rng(self.seed)
        itr, iva = _split(len(Xt), rng)
        ck = _Checkpoint([self.enc, self.dec])
        step = 0
        t0 = time.time()
        while step < self.steps:
            for bb in _batches(len(itr), self.bs, rng):
                b = itr[bb]
                h = self.enc(torch.cat([Yt[b], Xt[b]], 1))
                mu, logv = h[:, :self.latent], h[:, self.latent:].clamp(-8, 8)
                z = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
                o = self.dec(torch.cat([z, Xt[b]], 1))
                m, ls = o[:, :self.dim], o[:, self.dim:].clamp(-7, 3)
                nll = (0.5 * ((Yt[b] - m) / torch.exp(ls)) ** 2 + ls).sum(1)
                kl = 0.5 * (torch.exp(logv) + mu ** 2 - 1 - logv).sum(1)
                beta = min(1.0, step / (0.3 * self.steps))               # KL warm-up
                loss = (nll + beta * kl).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                if step >= 0.3 * self.steps and (step % 50 == 0 or step == self.steps):
                    ck.update(self._neg_elbo(Xt[iva], Yt[iva]), step)
                if step >= self.steps:
                    break
        ck.restore()
        self.best_step = ck.best_step
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def _neg_elbo(self, X, Y):
        g = torch.Generator().manual_seed(0)
        h = self.enc(torch.cat([Y, X], 1))
        mu, logv = h[:, :self.latent], h[:, self.latent:].clamp(-8, 8)
        z = mu + torch.exp(0.5 * logv) * torch.randn(mu.shape, generator=g)
        o = self.dec(torch.cat([z, X], 1))
        m, ls = o[:, :self.dim], o[:, self.dim:].clamp(-7, 3)
        nll = (0.5 * ((Y - m) / torch.exp(ls)) ** 2 + ls).sum(1)
        kl = 0.5 * (torch.exp(logv) + mu ** 2 - 1 - logv).sum(1)
        return float((nll + kl).mean())

    @torch.no_grad()
    def sample(self, X, n=64):
        Xt = torch.as_tensor(self.sx(X), dtype=F32)
        N = len(Xt)
        z = torch.randn(n, N, self.latent)
        o = self.dec(torch.cat([z, Xt.expand(n, N, -1)], -1))
        m, ls = o[..., :self.dim], o[..., self.dim:].clamp(-7, 3)
        S = (m + torch.exp(ls) * torch.randn_like(m)).numpy()
        return S * self.sy.s + self.sy.m


class DDPMImputer:
    name = "diffusion"

    def __init__(self, dim, ctx_dim, hidden=192, steps=3000, lr=1e-3, bs=256, T=100, seed=0, schedule="cosine"):
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(dim + ctx_dim + 16, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.T = T
        if schedule == "cosine":                  # Nichol & Dhariwal (2021): abar_T ~ 0, so sampling from N(0, I) is consistent
            sgrid = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
            f = torch.cos((sgrid + 0.008) / 1.008 * math.pi / 2) ** 2
            ab = f / f[0]
            b = torch.clamp(1 - ab[1:] / ab[:-1], max=0.999).float()
        else:                                      # linear schedule of Ho et al. (2020) scaled to T steps
            b = torch.linspace(1e-4, 0.02, T)
        self.betas = b
        self.alphas = 1 - b
        self.abar = torch.cumprod(self.alphas, 0)
        self.steps, self.lr, self.bs, self.seed, self.dim = steps, lr, bs, seed, dim

    def _temb(self, t):
        f = torch.exp(torch.linspace(0, math.log(1000.0), 8))
        a = t[:, None].float() / self.T * f[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def fit(self, X, Y, idx=None):
        self.sx, self.sy = Standardizer(X), Standardizer(Y)
        Xt = torch.as_tensor(self.sx(X), dtype=F32)
        Yt = torch.as_tensor(self.sy(Y), dtype=F32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps, eta_min=self.lr * 0.05)
        rng = np.random.default_rng(self.seed)
        itr, iva = _split(len(Xt), rng)
        ck = _Checkpoint([self.net])
        g = torch.Generator().manual_seed(0)
        tv = torch.randint(0, self.T, (len(iva), 4), generator=g)
        ev = torch.randn((len(iva), 4, self.dim), generator=g)
        step = 0
        t0 = time.time()
        while step < self.steps:
            for bb in _batches(len(itr), self.bs, rng):
                b = itr[bb]
                y0 = Yt[b]
                t = torch.randint(0, self.T, (len(b),))
                eps = torch.randn_like(y0)
                ab = self.abar[t][:, None]
                yt = torch.sqrt(ab) * y0 + torch.sqrt(1 - ab) * eps
                pred = self.net(torch.cat([yt, Xt[b], self._temb(t)], 1))
                loss = ((pred - eps) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                if step >= 0.3 * self.steps and (step % 50 == 0 or step == self.steps):
                    with torch.no_grad():
                        vl = 0.0
                        for j in range(4):
                            ab = self.abar[tv[:, j]][:, None]
                            yt = torch.sqrt(ab) * Yt[iva] + torch.sqrt(1 - ab) * ev[:, j]
                            pr = self.net(torch.cat([yt, Xt[iva], self._temb(tv[:, j])], 1))
                            vl += float(((pr - ev[:, j]) ** 2).mean()) / 4
                    ck.update(vl, step)
                if step >= self.steps:
                    break
        ck.restore()
        self.best_step = ck.best_step
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def sample(self, X, n=64):
        Xt = torch.as_tensor(self.sx(X), dtype=F32)
        N = len(Xt)
        C = Xt.repeat(n, 1)
        y = torch.randn(n * N, self.dim)
        for t in reversed(range(self.T)):
            tt = torch.full((n * N,), t)
            eps = self.net(torch.cat([y, C, self._temb(tt)], 1))
            a, ab, b = self.alphas[t], self.abar[t], self.betas[t]
            y = (y - b / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
            if t > 0:
                y = y + torch.sqrt(b) * torch.randn_like(y)
        S = y.reshape(n, N, self.dim).numpy()
        return S * self.sy.s + self.sy.m


class GPRImputer:
    """Independent GP per output dimension (original FPRM); Gaussian predictive."""
    name = "gpr"

    def __init__(self, dim, n_max=1500, seed=0, cols=None):
        self.dim, self.n_max, self.seed, self.cols = dim, n_max, seed, cols

    def fit(self, X, Y, idx=None):
        rng = np.random.default_rng(self.seed)
        if self.cols is not None:
            X = X[:, self.cols]
        self.sx, self.sy = Standardizer(X), Standardizer(Y)
        sel = rng.choice(len(X), min(self.n_max, len(X)), replace=False)
        Xs, Ys = self.sx(X[sel]), self.sy(Y[sel])
        self.gps = []
        t0 = time.time()
        for k in range(self.dim):
            kern = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.ones(Xs.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e1))
            gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=0, random_state=self.seed)
            gp.fit(Xs, Ys[:, k])
            self.gps.append(gp)
        self.train_seconds = time.time() - t0
        return self

    def predict(self, X):
        if self.cols is not None:
            X = X[:, self.cols]
        Xs = self.sx(X)
        mus, sds = [], []
        for gp in self.gps:
            m, s = gp.predict(Xs, return_std=True)
            mus.append(m)
            sds.append(s)
        M = np.stack(mus, 1) * self.sy.s + self.sy.m
        S = np.stack(sds, 1) * self.sy.s
        return M, S

    def sample(self, X, n=64, seed=0):
        M, S = self.predict(X)
        rng = np.random.default_rng(seed)
        return M[None] + S[None] * rng.standard_normal((n,) + M.shape)


# ============================================================================ metrics
def crps_samples(S, y):
    """Sample CRPS per point and dim. S: (n, N, d), y: (N, d)."""
    n = S.shape[0]
    term1 = np.mean(np.abs(S - y[None]), axis=0)
    Ssort = np.sort(S, axis=0)
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None, None]
    term2 = np.sum(w * Ssort, axis=0) / (n * n)          # = 0.5 E|X - X'|
    return term1 - term2


def evaluate(S, y, gauss=None):
    """Point (median) and probabilistic metrics; returns dict averaged over dims."""
    med = np.median(S, axis=0)
    out = {}
    rho, nrmse = [], []
    for k in range(y.shape[1]):
        rho.append(np.corrcoef(med[:, k], y[:, k])[0, 1])
        nrmse.append(np.sqrt(np.mean((med[:, k] - y[:, k]) ** 2)) / np.std(y[:, k]))
    out["rho"] = float(np.mean(rho))
    out["nrmse"] = float(np.mean(nrmse))
    if gauss is not None:
        M, Sd = gauss
        z = (y - M) / Sd
        crps = Sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))
        out["crps"] = float(np.mean(crps / np.std(y, axis=0)))
        lo, hi = M - 1.645 * Sd, M + 1.645 * Sd
    else:
        out["crps"] = float(np.mean(crps_samples(S, y) / np.std(y, axis=0)))
        lo, hi = np.percentile(S, 5, axis=0), np.percentile(S, 95, axis=0)
    out["coverage90"] = float(np.mean((y >= lo) & (y <= hi)))
    # sign-sensitive diagnostics on the first target (branch of the angle)
    out["p_correct_sign"] = float(np.mean(np.mean(np.sign(S[..., 0]) == np.sign(y[None, :, 0]), axis=0)))
    out["nrmse_abs"] = float(np.sqrt(np.mean((np.median(np.abs(S[..., 0]), 0) - np.abs(y[:, 0])) ** 2)) / np.std(y[:, 0]))
    return out
