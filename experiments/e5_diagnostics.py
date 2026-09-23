"""E5 - Diagnostics quoted in Sec. VI: (i) why the original FPRM saturates in task C1 (library size of
the delay-window mapping); (ii) conditioning of the C3 equation-of-motion residual.

A nearest-neighbour cross map on the same symmetric delay window as the original FPRM
(tau = 5, m = 2) predicts (c, theta_dot) at the benchmark test entries using as library either
all observed entries or a random subset of 1200 entries (the training-set size of the GPR).
Output: results/e5_diagnostics.json
"""
import json

import numpy as np
from scipy.spatial import cKDTree

from piar.physics import PIAR
from piar import fprm as F

p = PIAR.dimensionless(-0.8, gamma=0.25, A=0.8, Omega=2 / 3, forcing="torque")
ch = F.forced_channels(p, n_samples=40000)
rng_noise = np.random.default_rng(123)
x = ch["s"] + 0.01 * rng_noise.standard_normal(40000)
y_true = np.stack([ch["c"], ch["omega"]], 1)
y_obs = y_true + 0.01 * rng_noise.standard_normal(y_true.shape)
tau, m = 5, 2
idx = np.arange(m * tau, len(x) - m * tau)
V = np.stack([x[idx + l] for l in np.arange(-m, m + 1) * tau], 1)
out = {}
for r in (0.3, 0.6, 0.9):
    rng = np.random.default_rng(int(100 * r))
    mask = rng.random(y_true.shape) > r
    obs = mask[idx, 0]
    lib, te = np.where(obs)[0], np.where(~obs)[0]
    te = np.sort(rng.choice(te, 1500, replace=False))
    for name, sub in (("all_observed", lib), ("gpr_size_1200", np.sort(np.random.default_rng(0).choice(lib, 1200, replace=False)))):
        _, j = cKDTree(V[sub]).query(V[te], k=1)
        pred, tru = y_obs[idx[sub[j]]], y_true[idx[te]]
        rho = [float(np.corrcoef(pred[:, k], tru[:, k])[0, 1]) for k in range(2)]
        out[f"r{r}_{name}"] = {"library": int(len(sub)), "rho_c": rho[0], "rho_omega": rho[1], "rho_mean": float(np.mean(rho))}
        print(f"r={r} {name:14s} library={len(sub):5d} rho(c)={rho[0]:.3f} rho(omega)={rho[1]:.3f} mean={np.mean(rho):.3f}")
# (ii) conditioning of Eq. (res_c3): the sign-discriminating term A cos(Omega t) s^3 versus the noise
# s^2 * delta(c'') injected by the Savitzky-Golay second derivative (window 7, degree 4)
from scipy.signal import savgol_coeffs
gain = float(np.sqrt((savgol_coeffs(7, 4, deriv=2, delta=ch["dt"]) ** 2).sum()))
dcdd = 0.01 * gain
thr = dcdd / p.A
frac = float(np.mean(np.abs(ch["s"] * np.cos(ch["phase"])) < thr))
out["c3_residual_conditioning"] = {"sg_second_derivative_noise_gain": gain, "delta_cdd": dcdd,
                                   "threshold_abs_s_cos": thr, "fraction_below_threshold": frac}
print("C3 residual:", out["c3_residual_conditioning"])
json.dump(out, open("results/e5_diagnostics.json", "w"), indent=1)
