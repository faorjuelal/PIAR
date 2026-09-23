"""E4 - Visual reconstruction of the chaotic PIAR at 90% missing data (task C1, first mask).

The benchmark of e3_fprm.py reports scores on 1500 random missing entries. Here the same
data, noise, mask, context, and training settings are used to impute *every* missing entry
inside a display window and every missing point of the stroboscopic section, so that the
recovered dynamics can be seen. Two engines are compared: the physics-regularized
conditional neural spline flow (NSF + physics) and the delay-coordinate Gaussian process of
the original FPRM (GPR on the delay window).

Outputs: results/e4_reconstruction.npz, results/e4_reconstruction.json
"""
import json
import math
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from piar.physics import PIAR
from piar import fprm as F
from piar.chaos import trajectory

torch.set_num_threads(1)
t00 = time.time()
R, SEED, N_SERIES, SIGMA_N, K, NS = 0.9, 0, 40000, 0.01, 3, 256
p = PIAR.dimensionless(-0.8, gamma=0.25, A=0.8, Omega=2 / 3, forcing="torque")

# ---------------------------------------------------------------- data exactly as in e3_fprm.py (C1)
ch = F.forced_channels(p, n_samples=N_SERIES)
rng_noise = np.random.default_rng(123)
x_true = ch["s"]
y_true = np.stack([ch["c"], ch["omega"]], 1)
x_full = x_true + SIGMA_N * rng_noise.standard_normal(x_true.shape)
y_obs = y_true + SIGMA_N * rng_noise.standard_normal(y_true.shape)
tau, E, _, _ = F.choose_embedding(x_full)
m = max(1, int(math.ceil((E - 1) / 2)))
physics = F.Physics("C1", s=x_full, dt=ch["dt"], lam=p.lam, beta=p.beta, gamma=p.gamma, A=p.A,
                    Omega=p.Omega, t=ch["t"])
rng = np.random.default_rng(1000 * SEED + int(100 * R))
mask = rng.random(y_true.shape) > R
mask[:, 1] = mask[:, 0]
X, idx, n_xf = F.build_context(x_full, y_obs, mask, tau, m, K, phase=ch["phase"])
obs_rows = mask[idx, 0]
tr = np.where(obs_rows)[0]
te_all = np.where(~obs_rows)[0]
te = np.sort(rng.choice(te_all, min(1500, len(te_all)), replace=False))
Y, yt = y_obs[idx], y_true[idx]
print(f"tau={tau} E={E} m={m} train={len(tr)} missing={len(te_all)}", flush=True)

# ---------------------------------------------------------------- engines
nsf = F.NSFImputer(2, X.shape[1], steps=2000, seed=SEED, physics=physics, w_phys=1.0,
                   hinge=4.0).fit(X[tr], Y[tr], idx[tr], X_col=X, idx_col=idx)
print(f"NSF+physics trained {time.time()-t00:.0f}s", flush=True)
gpr = F.GPRImputer(2, n_max=1200, seed=SEED, cols=list(range(n_xf))).fit(X[tr], Y[tr])
print(f"GPR trained {time.time()-t00:.0f}s", flush=True)

# consistency with the benchmark (same test entries, 64 samples)
S_n = nsf.sample(X[te], 64)
r_n = F.evaluate(S_n, yt[te])
Mg, Sg = gpr.predict(X[te])
S_g = Mg[None] + Sg[None] * np.random.default_rng(SEED).standard_normal((64,) + Mg.shape)
r_g = F.evaluate(S_g, yt[te], gauss=(Mg, Sg))
print("check NSF+phys", {k: round(v, 4) for k, v in r_n.items()}, flush=True)
print("check GPR     ", {k: round(v, 4) for k, v in r_g.items()}, flush=True)

# ---------------------------------------------------------------- display window: every missing entry
PER = 32                                   # samples per forcing period
w0 = 20 * PER                               # start of the window (row index)
w1 = w0 + 8 * PER                           # eight forcing periods
rows = np.arange(w0, w1)
miss = rows[~obs_rows[rows]]
Sw_n = nsf.sample(X[miss], NS)                           # (NS, n, 2)
Mw, Sdw = gpr.predict(X[miss])
q = lambda S: np.percentile(S, [5, 50, 95], axis=0)       # (3, n, 2)
Qn = q(Sw_n)

# ---------------------------------------------------------------- stroboscopic section: every missing point
strobe = np.where((idx % PER) == PER - 1)[0]              # rows at t = n T_d (sample j <-> t=(j+1)dt)
strobe_miss = strobe[~obs_rows[strobe]]
Ssec_n = nsf.sample(X[strobe_miss], 1)[0]
Msec, Sdsec = gpr.predict(X[strobe_miss])
Ssec_g = Msec + Sdsec * np.random.default_rng(SEED + 1).standard_normal(Msec.shape)

# ---------------------------------------------------------------- dense truth for the window
n_tr = 400
dense = trajectory(p, 0.0, 0.0, n_tr + N_SERIES // PER, spp=128, save_per_period=128)[n_tr * 128:]
d_lo, d_hi = 4 * idx[w0] + 3 - 8, 4 * idx[w1 - 1] + 3 + 8
td, thd, omd = dense[d_lo:d_hi, 0], dense[d_lo:d_hi, 1], dense[d_lo:d_hi, 2]
assert np.allclose(dense[4 * idx[w0] + 3, 1], ch["theta"][idx[w0]])

np.savez_compressed(
    "results/e4_reconstruction.npz",
    t=ch["t"][idx[rows]], rows=rows, obs=obs_rows[rows], y_true=yt[rows], y_obs=Y[rows],
    miss_t=ch["t"][idx[miss]], Qn=Qn, gpr_mean=Mw, gpr_sd=Sdw,
    td=td, thd=thd, omd=omd,
    sec_true_s=x_true[idx[strobe]], sec_true=yt[strobe],
    sec_miss_s=x_true[idx[strobe_miss]], sec_nsf=Ssec_n, sec_gpr=Ssec_g, sec_miss_true=yt[strobe_miss],
)
json.dump({"tau": int(tau), "E": int(E), "m": m, "n_train": int(len(tr)), "n_missing": int(len(te_all)),
           "window_rows": [int(w0), int(w1)], "n_missing_window": int(len(miss)),
           "n_section": int(len(strobe)), "n_section_missing": int(len(strobe_miss)),
           "check_nsf_phys": r_n, "check_gpr": r_g, "seconds": time.time() - t00},
          open("results/e4_reconstruction.json", "w"), indent=2)
print("done", f"{time.time()-t00:.0f}s")
