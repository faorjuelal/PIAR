"""Fig. reconstruction: the chaotic PIAR recovered from 10% of the collar/velocity record (C1).

Reads results/e4_reconstruction.npz. (a),(b) every missing entry of an eight-period window:
posterior median and central 90% interval of the physics-regularized flow versus the Gaussian
predictive mean of the delay-coordinate GPR (original FPRM). (c),(d) stroboscopic Poincare
section rebuilt from the imputed collar height and angular velocity at every missing section point.
"""
import json

import numpy as np
import matplotlib.pyplot as plt

from piar.physics import PIAR
from piar.smooth import hermite_upsample
from piar.style import set_style, savefig, CAT, COL2, INK, MUTED, panel_label

set_style()
Z = np.load("results/e4_reconstruction.npz")
p = PIAR.dimensionless(-0.8, gamma=0.25, A=0.8, Omega=2 / 3, forcing="torque")
Td = 2 * np.pi / p.Omega
t = Z["t"]
t0 = t[0]
obs = Z["obs"]
tf, thf, omf = hermite_upsample(Z["td"], Z["thd"], Z["omd"], lambda a, b, c: p.accel(a, b, c), factor=6)


def sliced_w1(P, Q, n_proj=128, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_proj):
        v = rng.standard_normal(P.shape[1])
        v /= np.linalg.norm(v)
        a, b = np.sort(P @ v), np.sort(Q @ v)
        n = min(len(a), len(b))
        out.append(np.mean(np.abs(np.quantile(a, np.linspace(0, 1, n)) - np.quantile(b, np.linspace(0, 1, n)))))
    return float(np.mean(out))


fig = plt.figure(figsize=(COL2, 4.2))
gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.35, 1.0], hspace=0.34, wspace=0.5, top=0.86)
Qn = Z["Qn"]                                         # (3, n, 2): 5th, 50th, 95th percentiles
mt = (Z["miss_t"] - t0) / Td
labels = (r"collar height $c=\cos\theta$", r"angular velocity $\dot\theta\sqrt{L/g}$")
truth_dense = (np.cos(thf), omf)
for k in range(2):
    ax = fig.add_subplot(gs[k, :2])
    ax.plot((tf - t0) / Td, truth_dense[k], color=MUTED, lw=0.9, zorder=1, label="truth")
    ax.plot((t[obs] - t0) / Td, Z["y_obs"][obs, k], "o", ms=3.0, color=INK, mew=0, zorder=4, label="observed (10\\%)")
    ax.errorbar(mt, Qn[1, :, k], yerr=[Qn[1, :, k] - Qn[0, :, k], Qn[2, :, k] - Qn[1, :, k]], fmt="o", ms=2.2,
                color=CAT[0], ecolor=CAT[0], elinewidth=0.5, capsize=0, mew=0, zorder=3,
                label="NSF + physics: median and 90\\% interval")
    ax.plot(mt, Z["gpr_mean"][:, k], "s", ms=1.9, color=CAT[1], mew=0, zorder=2, label="original FPRM (GPR)")
    ax.set_xlim(0, (t[-1] - t0) / Td)
    ax.set_ylabel(labels[k], fontsize=8.5)
    if k == 0:
        ax.tick_params(labelbottom=False)
        ax.set_ylim(-1.25, 1.25)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=8,
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0, frameon=False)
    else:
        ax.set_xlabel(r"time $t/T_d$ (forcing periods)")
    panel_label(ax, "ab"[k], x=-0.1, y=1.05)

# Poincare sections -----------------------------------------------------------------------
th_true = np.arctan2(Z["sec_true_s"], Z["sec_true"][:, 0])
om_true = Z["sec_true"][:, 1]
res = {}
for k, (key, col, name) in enumerate((("sec_nsf", CAT[0], "NSF + physics"), ("sec_gpr", CAT[1], "original FPRM"))):
    ax = fig.add_subplot(gs[k, 2])
    ax.plot(th_true, om_true, ".", ms=1.6, color=MUTED, mew=0, zorder=1)
    S = Z[key]
    th_i = np.arctan2(Z["sec_miss_s"], np.clip(S[:, 0], -1.5, 1.5))
    ax.plot(th_i, S[:, 1], ".", ms=1.8, color=col, mew=0, zorder=2)
    P = np.column_stack([Z["sec_miss_s"], S[:, 0], S[:, 1]])
    Q = np.column_stack([Z["sec_miss_s"], Z["sec_miss_true"][:, 0], Z["sec_miss_true"][:, 1]])
    res[name] = sliced_w1(P, Q)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.1, 2.4)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_ylabel(r"$\dot\theta(nT_d)\sqrt{L/g}$", fontsize=8.5)
    ax.set_title(rf"{name}: $W_1={res[name]:.3f}$", fontsize=8.5, pad=3)
    if k == 0:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r"$\theta(nT_d)$ (rad)")
    panel_label(ax, "cd"[k], x=-0.5, y=1.0)
savefig(fig, "fig_reconstruction")

# numbers for the text: interval coverage and errors inside the window
yt_m = Z["y_true"][~obs]
cov = np.mean((yt_m >= Qn[0]) & (yt_m <= Qn[2]), axis=0)
rmse_n = np.sqrt(np.mean((Qn[1] - yt_m) ** 2, axis=0))
rmse_g = np.sqrt(np.mean((Z["gpr_mean"] - yt_m) ** 2, axis=0))
out = {"window_coverage90_nsf": cov.tolist(), "window_rmse_nsf": rmse_n.tolist(), "window_rmse_gpr": rmse_g.tolist(),
       "section_sliced_w1": {("original FPRM (GPR)" if k_ == "original FPRM" else k_): v_ for k_, v_ in res.items()}, "n_section_missing": int(len(Z["sec_miss_s"])), "n_window_missing": int(mt.size)}
json.dump(out, open("results/fig_reconstruction.json", "w"), indent=1)
print(out)
