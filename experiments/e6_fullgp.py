"""E6 - The original FPRM with all labels (task C1, 90% missing data).

The benchmark of e3_fprm.py fits the Gaussian process of the original FPRM to a random subset of
1200 labelled rows (exact GPR scales as the cube of the training-set size). Here the same model is
fitted to all labelled rows (about 4000 at 90% missing data) to quantify how much of the gap to the
generative engines is due to the subset. Same data, noise, masks, test entries and metrics as e3.
Output: results/e6_fullgp.json
"""
import json
import math
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from piar.physics import PIAR
from piar import fprm as F

R, NS, SIGMA_N, K = 0.9, 64, 0.01, 3
p = PIAR.dimensionless(-0.8, gamma=0.25, A=0.8, Omega=2 / 3, forcing="torque")
ch = F.forced_channels(p, n_samples=40000)
rng_noise = np.random.default_rng(123)
x_true = ch["s"]
y_true = np.stack([ch["c"], ch["omega"]], 1)
x_full = x_true + SIGMA_N * rng_noise.standard_normal(x_true.shape)
y_obs = y_true + SIGMA_N * rng_noise.standard_normal(y_true.shape)
tau, E, _, _ = F.choose_embedding(x_full)
m = max(1, int(math.ceil((E - 1) / 2)))
out = {"tau": int(tau), "E": int(E), "runs": {}}
for seed in (0, 1):
    t0 = time.time()
    rng = np.random.default_rng(1000 * seed + int(100 * R))
    mask = rng.random(y_true.shape) > R
    mask[:, 1] = mask[:, 0]
    X, idx, n_xf = F.build_context(x_full, y_obs, mask, tau, m, K, phase=ch["phase"])
    obs_rows = mask[idx, 0]
    tr = np.where(obs_rows)[0]
    te_all = np.where(~obs_rows)[0]
    te = np.sort(rng.choice(te_all, min(1500, len(te_all)), replace=False))
    Y, yt = y_obs[idx], y_true[idx]
    gp = F.GPRImputer(2, n_max=len(tr), seed=seed, cols=list(range(n_xf))).fit(X[tr], Y[tr])
    M, Sd = gp.predict(X[te])
    S = M[None] + Sd[None] * np.random.default_rng(seed).standard_normal((NS,) + M.shape)
    r = F.evaluate(S, yt[te], gauss=(M, Sd))
    r["n_train"] = int(len(tr))
    r["seconds"] = time.time() - t0
    out["runs"][f"r{R}_s{seed}"] = r
    print(seed, {k: round(v, 4) for k, v in r.items()}, flush=True)
out["mean"] = {k: float(np.mean([out["runs"][s][k] for s in out["runs"]])) for k in ("rho", "nrmse", "crps", "coverage90", "p_correct_sign")}
print("mean", out["mean"])
json.dump(out, open("results/e6_fullgp.json", "w"), indent=1)
