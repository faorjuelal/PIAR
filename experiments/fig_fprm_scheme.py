"""Fig. fprm_scheme: the full--partial reconstruction mapping illustrated on the chaotic PIAR.

(a) complete channel s = sin(theta) and incomplete channel c = cos(theta) (60% missing, task C1),
(b) attractor in the state space, (c) delay reconstruction M_F from s, (d) delay reconstruction M_P
from c, of which only the fully observed delay pairs are available.
"""
import math

import numpy as np
import matplotlib.pyplot as plt

from piar.physics import PIAR
from piar import fprm as F
from piar.chaos import trajectory
from piar.smooth import hermite_upsample, break_on_wrap
from piar.style import set_style, savefig, CAT, COL2, INK, MUTED, panel_label

set_style()
p = PIAR.dimensionless(-0.8, gamma=0.25, A=0.8, Omega=2 / 3, forcing="torque")
N, PER, NTR, R = 40000, 32, 400, 0.6
ch = F.forced_channels(p, n_samples=N)
rng_noise = np.random.default_rng(123)
s_obs = ch["s"] + 0.01 * rng_noise.standard_normal(N)
y_true = np.stack([ch["c"], ch["omega"]], 1)
y_obs = y_true + 0.01 * rng_noise.standard_normal(y_true.shape)
rng = np.random.default_rng(int(100 * R))
mask = rng.random(y_true.shape) > R
mask[:, 1] = mask[:, 0]
tau = 5                                             # selected by mutual information (Sec. V)

dense = trajectory(p, 0.0, 0.0, NTR + N // PER, spp=128, save_per_period=128)[NTR * 128:]
acc = lambda t, x, v: p.accel(t, x, v)
Td = 2 * np.pi / p.Omega

fig = plt.figure(figsize=(COL2, 4.55))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.3], hspace=0.66, wspace=0.62, right=0.89)

# (a) time series ------------------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
j0, j1 = 64, 64 + 6 * PER
d0, d1 = 4 * j0 + 3 - 4, 4 * (j1 - 1) + 3 + 4
tf, thf, _ = hermite_upsample(dense[d0:d1, 0], dense[d0:d1, 1], dense[d0:d1, 2], acc, factor=6)
t0 = ch["t"][j0]
x_ = (tf - t0) / Td
ax.plot(x_, np.sin(thf), color=CAT[0], lw=1.0, label=r"$s=F_s/(kL)=\sin\theta$ (complete)")
ax.plot(x_, np.cos(thf), color=MUTED, lw=0.8, label=r"$c=y_c/L=\cos\theta$ (true)")
tj = (ch["t"][j0:j1] - t0) / Td
ob = mask[j0:j1, 0]
ax.plot(tj, s_obs[j0:j1], "o", ms=1.8, color=CAT[0], mew=0)
ax.plot(tj[ob], y_obs[j0:j1, 0][ob], "o", ms=3.0, color=CAT[1], mec="white", mew=0.3, label=r"$c$ observed")
ax.plot(tj[~ob], ch["c"][j0:j1][~ob], "o", ms=3.0, mfc="none", mec=CAT[1], mew=0.6, label=r"$c$ missing")
ax.set_xlim(tj[0], tj[-1])
ax.set_ylim(-1.18, 1.18)
ax.set_xlabel(r"time $t/T_d$ (forcing periods)")
ax.set_ylabel(r"observable")
ax.legend(loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.3), handlelength=1.4, columnspacing=0.9)
panel_label(ax, "a", x=-0.055, y=1.02)

# (b)-(d) cross-mapping test: is the hidden channel a function on the delay reconstruction? ----
from scipy.spatial import cKDTree
from matplotlib.colors import LogNorm
from piar.style import SEQ_W


def delay(x, tau, E):
    n = len(x) - (E - 1) * tau
    return np.stack([x[i * tau: i * tau + n] for i in range(E)], 1), n


def crossmap(x, hidden, tau, E, phase=None, theiler=PER):
    """Hidden value at t versus hidden value at the nearest neighbour of t in the delay space of x."""
    V, n = delay(x, tau, E)
    if phase is not None:
        ph = phase[(E - 1) * tau: (E - 1) * tau + n]
        V = np.column_stack([V, np.cos(ph), np.sin(ph)])
    h = hidden[(E - 1) * tau: (E - 1) * tau + n]
    tree = cKDTree(V)
    d, j = tree.query(V, k=12)
    ii = np.arange(n)[:, None]
    ok = np.abs(j - ii) > theiler                      # exclude temporal neighbours
    first = np.argmax(ok, axis=1)
    nn = j[np.arange(n), first]
    return h, h[nn]


c_noisy = y_obs[:, 0]
s_noisy = s_obs
panels = [
    ("b", s_noisy, ch["c"], 5, 5, None, r"C1: $s$ complete $\to$ $c$", r"$c(t)$", r"$c(t^\ast)$"),
    ("c", c_noisy, ch["s"], 6, 5, None, r"C3: $c$ complete $\to$ $s$", r"$s(t)$", r"$s(t^\ast)$"),
    ("d", c_noisy, ch["s"], 6, 5, ch["phase"], r"C3 with forcing phase", r"$s(t)$", r"$s(t^\ast)$"),
]
rho = {}
for k, (letter, x, hid, tau_, E_, ph, title, xl, yl) in enumerate(panels):
    axk = fig.add_subplot(gs[1, k])
    h, hn = crossmap(x, hid, tau_, E_, phase=ph)
    rho[letter] = float(np.corrcoef(h, hn)[0, 1])
    H, xe, ye = np.histogram2d(h, hn, bins=90, range=[[-1.05, 1.05], [-1.05, 1.05]])
    Hm = np.ma.masked_where(H == 0, H)
    im = axk.pcolormesh(xe, ye, Hm.T, cmap=SEQ_W, norm=LogNorm(vmin=1, vmax=H.max()), rasterized=True)
    axk.plot([-1, 1], [-1, 1], color=MUTED, lw=0.5, ls=(0, (3, 2)))
    axk.set_aspect("equal")
    axk.set_xlim(-1.05, 1.05)
    axk.set_ylim(-1.05, 1.05)
    axk.set_xticks([-1, 0, 1])
    axk.set_yticks([-1, 0, 1])
    axk.set_xlabel(xl)
    axk.set_ylabel(yl)
    axk.set_title(title, fontsize=8.5)
    axk.text(0.5, 0.97, rf"$\rho={rho[letter]:.2f}$", transform=axk.transAxes, fontsize=8.5, va="top", ha="center")
    panel_label(axk, letter, x=-0.28, y=1.08)
cb = fig.colorbar(im, ax=fig.axes[1:], fraction=0.015, pad=0.015)
cb.set_label("pairs", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
print("cross-map rho:", rho)
import json
json.dump({"crossmap_rho": rho, "pairs_observed_60pct": None}, open("results/fig_fprm_scheme.json", "w"), indent=1)
savefig(fig, "fig_fprm_scheme")
