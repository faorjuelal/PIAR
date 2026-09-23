"""E3 - Physics-informed generative FPRM with measurable observables (Secs. V and VI).

Configurations (observables normalised by calibration: s = F_s/(kL) = sin(theta), c = y_c/L = cos(theta)):
  C0  control   theta complete -> theta-dot incomplete (Savitzky-Golay derivative suffices)
  C1  forced    s complete -> (c, omega) incomplete, joint 2-D target
  C2  Langevin  c complete -> s incomplete (autonomous: mirror symmetry theta -> -theta)
  C3  forced    c complete -> s incomplete, with / without forcing phase in the context
Usage: python e3_fprm.py C1|C2|C3|C0   (writes results/e3_<cfg>.json and cache files)
"""
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.signal import savgol_filter

from piar.physics import PIAR
from piar import fprm as F

torch.set_num_threads(1)
CFG = sys.argv[1] if len(sys.argv) > 1 else "C1"
RATES = (0.3, 0.6, 0.9)
SEEDS = (0, 1)
SIGMA_N = 0.01
N_TEST = 1500
NS = 64
STEPS = 2000
STEPS_BASELINE = 6000                              # cVAE and DDPM (their validation loss still improved at 2000)
GP_NMAX = 1200
ONLY = [m for m in os.environ.get("PIAR_METHODS", "").split(",") if m]   # rerun a subset and merge into the JSON
N_SERIES = 40000
TAG = ""
if os.environ.get("PIAR_SMOKE") == "1":            # quick end-to-end check of the pipeline
    RATES, SEEDS, N_TEST, NS, STEPS, GP_NMAX, N_SERIES, TAG = (0.9,), (0,), 200, 16, 40, 150, 8000, "smoke_"
t00 = time.time()

p_forced = PIAR.dimensionless(-0.8, gamma=0.25, A=0.8, Omega=2 / 3, forcing="torque")


def noisy(x, rng):
    return x + SIGMA_N * rng.standard_normal(x.shape)


def sliced_w1(P, Q, n_proj=64, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_proj):
        v = rng.standard_normal(P.shape[1])
        v /= np.linalg.norm(v)
        a, b = np.sort(P @ v), np.sort(Q @ v)
        n = min(len(a), len(b))
        qa = np.quantile(a, np.linspace(0, 1, n))
        qb = np.quantile(b, np.linspace(0, 1, n))
        d.append(np.mean(np.abs(qa - qb)))
    return float(np.mean(d))


def run_methods(methods, X, Y, idx, ytrue, tr, te, dim, physics=None, cols_xf=None, center_col=None, seed=0,
                extra_eval=None, te_extra=None):
    out = {}
    samples_keep = {}
    for mname in methods:
        t0 = time.time()
        m = None
        if mname == "static":
            m = F.GPRImputer(dim, n_max=GP_NMAX, seed=seed, cols=[center_col]).fit(X[tr], Y[tr])
            M, Sd = m.predict(X[te])
            S = M[None] + Sd[None] * np.random.default_rng(seed).standard_normal((NS,) + M.shape)
            r = F.evaluate(S, ytrue[te], gauss=(M, Sd))
        elif mname in ("gpr", "gpr_ctx"):
            cols = cols_xf if mname == "gpr" else None
            m = F.GPRImputer(dim, n_max=GP_NMAX, seed=seed, cols=cols).fit(X[tr], Y[tr])
            M, Sd = m.predict(X[te])
            S = M[None] + Sd[None] * np.random.default_rng(seed).standard_normal((NS,) + M.shape)
            r = F.evaluate(S, ytrue[te], gauss=(M, Sd))
        elif mname == "cvae":
            m = F.CVAEImputer(dim, X.shape[1], steps=STEPS_BASELINE if not TAG else STEPS, seed=seed).fit(X[tr], Y[tr])
            S = m.sample(X[te], NS)
            r = F.evaluate(S, ytrue[te])
        elif mname == "diffusion":
            m = F.DDPMImputer(dim, X.shape[1], steps=STEPS_BASELINE if not TAG else STEPS, seed=seed,
                              schedule="cosine").fit(X[tr], Y[tr])
            S = m.sample(X[te], NS)
            r = F.evaluate(S, ytrue[te])
        elif mname == "nsf":
            m = F.NSFImputer(dim, X.shape[1], steps=STEPS, seed=seed).fit(X[tr], Y[tr], idx[tr])
            S = m.sample(X[te], NS)
            r = F.evaluate(S, ytrue[te])
        elif mname == "nsf_phys":
            m = F.NSFImputer(dim, X.shape[1], steps=STEPS, seed=seed, physics=physics, w_phys=1.0,
                             hinge=4.0).fit(X[tr], Y[tr], idx[tr], X_col=X, idx_col=idx)
            S = m.sample(X[te], NS)
            r = F.evaluate(S, ytrue[te])
        else:
            raise ValueError(mname)
        r["seconds"] = time.time() - t0
        r["best_step"] = int(getattr(m, "best_step", -1)) if m is not None else -1
        if extra_eval is not None and te_extra is not None and len(te_extra):
            if mname in ("static", "gpr", "gpr_ctx"):
                Me, Se = m.predict(X[te_extra])
                Sx = Me[None] + Se[None] * np.random.default_rng(seed + 1).standard_normal((4,) + Me.shape)
            else:
                Sx = m.sample(X[te_extra], 4)
            r.update(extra_eval(Sx, te_extra))
        out[mname] = r
        samples_keep[mname] = S
        print(f"   {mname:10s} rho={r['rho']:.3f} nrmse={r['nrmse']:.3f} crps={r['crps']:.4f} "
              f"cov90={r['coverage90']:.2f} psign={r['p_correct_sign']:.3f} ({r['seconds']:.0f}s, "
              f"total {time.time()-t00:.0f}s)", flush=True)
    return out, samples_keep


results = {"cfg": CFG, "sigma_n": SIGMA_N, "rates": RATES, "seeds": SEEDS, "runs": {}}

if CFG in ("C1", "C3"):
    ch = F.forced_channels(p_forced, n_samples=N_SERIES)
elif CFG == "C2":
    ch = F.langevin_channels(lam=0.5, beta=1.0, gamma=0.1, T=0.6, n_samples=N_SERIES, seed=7)
elif CFG == "C0":
    ch = F.forced_channels(p_forced, n_samples=N_SERIES)

rng_noise = np.random.default_rng(123)
if CFG == "C1":
    x_full_true = ch["s"]
    y_true = np.stack([ch["c"], ch["omega"]], 1)
elif CFG in ("C2", "C3"):
    x_full_true = ch["c"]
    y_true = ch["s"][:, None]
else:
    x_full_true = ch["theta"]
    y_true = ch["omega"][:, None]
x_full = noisy(x_full_true, rng_noise)
y_obs = noisy(y_true, rng_noise)

tau, E, I, fnn = F.choose_embedding(x_full)
m = max(1, int(math.ceil((E - 1) / 2)))
results["embedding"] = {"tau": int(tau), "E": int(E), "m": m, "fnn": fnn.tolist(), "ami": I[:40].tolist()}
print(f"{CFG}: tau={tau} E={E} (window 2m+1={2*m+1})", flush=True)
K = 3

physics = None
if CFG == "C1":
    physics = F.Physics("C1", s=x_full, dt=ch["dt"], lam=p_forced.lam, beta=p_forced.beta,
                        gamma=p_forced.gamma, A=p_forced.A, Omega=p_forced.Omega, t=ch["t"])
elif CFG == "C2":
    physics = F.Physics("C2", c=x_full, dt=ch["dt"])
elif CFG == "C3":
    physics = F.Physics("C3", c=x_full, dt=ch["dt"], lam=p_forced.lam, beta=p_forced.beta,
                        gamma=p_forced.gamma, A=p_forced.A, Omega=p_forced.Omega, t=ch["t"])

cache = {}
for r in RATES:
    for seed in SEEDS:
        rng = np.random.default_rng(1000 * seed + int(100 * r))
        mask = rng.random(y_true.shape) > r            # True = observed
        if CFG == "C1":
            mask[:, 1] = mask[:, 0]                    # c and omega lost together (same sensor record)
        variants = [("nophase", None)]
        if CFG == "C1":
            variants = [("phase", ch["phase"])]
        if CFG == "C3":
            variants = [("phase", ch["phase"]), ("nophase", None)]
        for vname, phase in variants:
            X, idx, n_xf = F.build_context(x_full, y_obs, mask, tau, m, K, phase=phase)
            obs_rows = mask[idx, 0]
            tr = np.where(obs_rows)[0]
            te_all = np.where(~obs_rows)[0]
            te = np.sort(rng.choice(te_all, min(N_TEST, len(te_all)), replace=False))
            Y = y_obs[idx]
            yt = y_true[idx]
            key = f"r{r}_s{seed}_{vname}"
            print(f"[{key}] train={len(tr)} test={len(te)} ctx_dim={X.shape[1]}", flush=True)
            if CFG == "C0":
                # Savitzky-Golay derivative of the complete (noisy) angle
                sg = savgol_filter(x_full, 11, 4, deriv=1, delta=ch["dt"])[idx][te]
                Srep = np.repeat(sg[None, :, None], 2, axis=0)
                rs = F.evaluate(Srep, yt[te])
                print("   savgol   ", {k: round(v, 4) for k, v in rs.items()}, flush=True)
                res_m, S_keep = run_methods(["nsf"], X, Y, idx, yt, tr, te, 1, seed=seed)
                res_m["savgol"] = rs
            elif CFG == "C1":
                def extra(S, te_):
                    # sliced W1 on the stroboscopic section in (s, c, omega): one posterior draw per row
                    s_ = x_full_true[idx[te_]]
                    P = np.stack([s_, S[0, :, 0], S[0, :, 1]], 1)
                    Q = np.stack([s_, yt[te_, 0], yt[te_, 1]], 1)
                    return {"sw1_section": sliced_w1(P, Q), "n_strobe": int(len(te_))}
                te_strobe = te_all[(idx[te_all] % 32) == 0]
                meths = ["static", "gpr", "gpr_ctx", "cvae", "diffusion", "nsf", "nsf_phys"]
                meths = [mm for mm in meths if not ONLY or mm in ONLY]
                res_m, S_keep = run_methods(meths, X, Y, idx, yt, tr, te, 2, physics=physics,
                                            cols_xf=list(range(n_xf)), center_col=m, seed=seed, extra_eval=extra,
                                            te_extra=te_strobe)
            elif CFG == "C2":
                meths = ["static", "gpr", "gpr_ctx", "cvae", "diffusion", "nsf", "nsf_phys"]
                meths = [mm for mm in meths if not ONLY or mm in ONLY]
                res_m, S_keep = run_methods(meths, X, Y, idx, yt, tr, te, 1, physics=physics,
                                            cols_xf=list(range(n_xf)), center_col=m, seed=seed)
            else:  # C3
                meths = ["gpr", "nsf", "nsf_phys"]
                res_m, S_keep = run_methods(meths, X, Y, idx, yt, tr, te, 1, physics=physics,
                                            cols_xf=list(range(n_xf)) + ([X.shape[1] - 2, X.shape[1] - 1] if phase is not None else []),
                                            center_col=m, seed=seed)
            if ONLY:
                prev = json.load(open(f"results/{TAG}e3_{CFG}.json"))
                prev["runs"].setdefault(key, {}).update(res_m)
                prev.setdefault("reruns", {})[",".join(ONLY)] = "cVAE/DDPM: 6000 steps, cosine DDPM schedule"
                results = prev
            else:
                results["runs"][key] = res_m
            if seed == 0 and not ONLY:
                d_ = y_obs.shape[1]
                nb_obs = X[te][:, n_xf + 2 * K * d_: n_xf + 4 * K * d_].sum(1)
                cache[key] = {"te": te, "idx_te": idx[te], "ytrue": yt[te], "x_te": x_full_true[idx[te]],
                              "nb_obs": nb_obs, "samples": {k: v[:, :600] for k, v in S_keep.items()}}
            json.dump(results, open(f"results/{TAG}e3_{CFG}.json", "w"), indent=1)
if not ONLY:
    np.save(f"results/{TAG}e3_{CFG}_cache.npy", cache, allow_pickle=True)
np.save(f"results/{TAG}e3_{CFG}_series.npy", {"t": ch["t"][:3000], "x_full": x_full[:3000], "y_true": y_true[:3000],
                                         "y_obs": y_obs[:3000]}, allow_pickle=True)
print("done", f"{time.time()-t00:.0f}s")
