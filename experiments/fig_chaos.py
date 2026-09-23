"""Figures of Sec. IV from results/e2_chaos.npz: fig_route.pdf and fig_attractor.pdf."""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

from piar.physics import PIAR
from piar.smooth import hermite_upsample, break_on_wrap
from piar.style import set_style, savefig, CAT, COL2, INK, MUTED, SEQ_W, DIV, panel_label

set_style()
Z = np.load("results/e2_chaos.npz")
R = json.load(open("results/e2_chaos.json"))
GAMMA, OMEGA = 0.25, 2.0 / 3.0
A_REF = 0.80
A_C = R["melnikov"]["A_c_Omega_2_3"]
A_INF = R["feigenbaum"]["A_inf"]
An = np.array(R["feigenbaum"]["A_n"])
deltas = R["feigenbaum"]["delta_n"]

# ============================================================== route to chaos
fig = plt.figure(figsize=(COL2, 3.75))
gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], width_ratios=[1.6, 1.6, 1.05], hspace=0.12, wspace=0.36)
ax = fig.add_subplot(gs[0, :2])
A = Z["A_grid"]
th = Z["bif_theta"]
ax.scatter(np.repeat(A, th.shape[1]), th.ravel(), s=0.05, color=INK, marker=".", linewidths=0, rasterized=True)
ax.set_ylim(-np.pi, np.pi)
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax.set_ylabel(r"$\theta(nT_d)$ (rad)")
ax.set_xlim(A[0], A[-1])
ax.tick_params(labelbottom=False)
for a_, lab, ls in ((A_C, r"$A_c$", (0, (3, 2))), (A_INF, r"$A_\infty$", "-"), (A_REF, r"$A_\mathrm{ref}$", "-")):
    ax.axvline(a_, color=CAT[1] if lab == r"$A_\mathrm{ref}$" else MUTED, lw=0.6, ls=ls)
    ax.text(a_, np.pi * 1.03, lab, fontsize=8.5, ha="center", va="bottom", color=INK)
panel_label(ax, "a", x=-0.09, y=1.07)
ax2 = fig.add_subplot(gs[1, :2], sharex=ax)
ax2.axhline(0, color=MUTED, lw=0.5)
ax2.plot(A, Z["lyap1_A"], color=CAT[0], lw=0.6)
for a_, ls in ((A_C, (0, (3, 2))), (A_INF, "-")):
    ax2.axvline(a_, color=MUTED, lw=0.6, ls=ls)
ax2.axvline(A_REF, color=CAT[1], lw=0.6)
ax2.set_ylabel(r"$\lambda_1\sqrt{L/g}$")
ax2.set_xlabel(r"forcing amplitude $A/(g/L)$")
panel_label(ax2, "b", x=-0.08, y=0.92)
ax3 = fig.add_subplot(gs[:, 2])
Az, thz = Z["zoom_A"], Z["zoom_theta"]
ax3.scatter(np.repeat(Az, thz.shape[1]), thz.ravel(), s=0.12, color=INK, marker=".", linewidths=0, rasterized=True)
for a_ in An[:4]:
    ax3.axvline(a_, color=CAT[1], lw=0.5)
lo, hi = np.percentile(thz, [0.5, 99.5])
ax3.set_ylim(lo - 0.05, hi + 0.05)
ax3.set_xlim(Az[0], Az[-1])
ax3.set_xticks([0.66, 0.67])
ax3.set_xlabel(r"$A/(g/L)$")
ax3.set_ylabel(r"$\theta(nT_d)$ (rad)")
txt = "\n".join([rf"$\delta_{i+1}={d:.3f}$" for i, d in enumerate(deltas[:4])])
ax3.text(0.05, 0.04, txt, transform=ax3.transAxes, fontsize=8, va="bottom", ha="left",
         bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
panel_label(ax3, "c", y=1.01)
savefig(fig, "fig_route")

# ============================================================== strange attractor
p = PIAR.dimensionless(-0.8, gamma=GAMMA, A=A_REF, Omega=OMEGA, forcing="torque")
fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.45), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0], wspace=0.78))
ax = axs[0]
tr = Z["traj"][: 128 * 40]
acc = lambda t, x, v: p.accel(t, x, v)
tf, thf, omf = hermite_upsample(tr[:, 0], tr[:, 1], tr[:, 2], acc, factor=6)
thp, omp = break_on_wrap(thf, omf)
ax.plot(thp, omp, color=CAT[0], lw=0.35, alpha=0.9, rasterized=True)
q = PIAR.dimensionless(-0.8)
sg = np.linspace(-np.pi, np.pi, 2001)
up, dn = q.separatrix(sg)
ax.plot(sg, up, color=INK, lw=0.8, ls=(0, (3, 2)))
ax.plot(sg, dn, color=INK, lw=0.8, ls=(0, (3, 2)))
ax.plot(0, 0, "x", color=INK, ms=4, mew=0.9)
ax.set_xlim(-np.pi, np.pi)
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax.set_xlabel(r"$\theta$ (rad)")
ax.set_ylabel(r"$\dot\theta\sqrt{L/g}$")
panel_label(ax, "a", y=1.04)
ax = axs[1]
sec = Z["section"]
thw = (sec[:, 0] + np.pi) % (2 * np.pi) - np.pi
H, xe, ye = np.histogram2d(thw, sec[:, 1], bins=[240, 170], range=[[-np.pi, np.pi], [sec[:, 1].min(), sec[:, 1].max()]])
Hm = np.ma.masked_where(H == 0, H)
im = ax.pcolormesh(xe, ye, Hm.T, cmap=SEQ_W, norm=LogNorm(vmin=1, vmax=H.max()), rasterized=True)
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax.set_xlabel(r"$\theta(nT_d)$ (rad)")
ax.set_ylabel(r"$\dot\theta(nT_d)\sqrt{L/g}$")
cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.06, aspect=22)
cb.set_label("counts", fontsize=8.5, labelpad=1)
cb.ax.tick_params(labelsize=8)
panel_label(ax, "b", y=1.04)
ax = axs[2]
chart, Ac, Om = Z["chart"], Z["chart_A"], Z["chart_Omega"]
im = ax.pcolormesh(Ac, Om, chart, cmap=DIV, norm=TwoSlopeNorm(vcenter=0.0, vmin=chart.min(), vmax=chart.max()),
                   shading="nearest", rasterized=True)
import matplotlib.patheffects as pe
ax.plot(Z["melnikov_Ac"], Om, color="white", lw=1.1, path_effects=[pe.Stroke(linewidth=2.0, foreground=INK), pe.Normal()])
ax.plot([A_REF], [OMEGA], marker="*", color="white", mec=INK, mew=0.5, ms=7, ls="none")
ax.text(1.06, 1.24, r"$A_c(\Omega)$", fontsize=8.5, color="white", ha="right", va="center")
ax.set_xlim(Ac[0], Ac[-1])
ax.set_xlabel(r"$A/(g/L)$")
ax.set_ylabel(r"$\Omega\sqrt{L/g}$")
cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.06, aspect=22)
cb.set_label(r"$\lambda_1\sqrt{L/g}$", fontsize=8.5, labelpad=1)
cb.ax.tick_params(labelsize=8)
panel_label(ax, "c", y=1.04)
savefig(fig, "fig_attractor")
print("ok")
