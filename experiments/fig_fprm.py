"""Figure and table for the generative FPRM benchmark (Sec. VI).

Reads results/e3_C{0,1,2,3}.json and results/e3_C2_cache.npy.
Writes figures/fig_fprm.pdf and manuscript/tables/tab_fprm.tex.
Usage: python fig_fprm.py [smoke]
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from piar.style import set_style, savefig, C, CAT, COL2, INK, MUTED, SEQ_W, panel_label

TAG = "smoke_" if len(sys.argv) > 1 and sys.argv[1] == "smoke" else ""
set_style()
R = {c: json.load(open(f"results/{TAG}e3_{c}.json")) for c in ("C1", "C2", "C3", "C0")
     if os.path.exists(f"results/{TAG}e3_{c}.json")}

STY = {  # method: (label, color, linestyle, marker)
    "static": ("static GPR", CAT[5], ":", "v"),
    "gpr": ("original FPRM (GPR)", CAT[1], "--", "s"),
    "gpr_ctx": ("GPR (full context)", CAT[1], "-", "s"),
    "cvae": ("cVAE", CAT[4], "-", "D"),
    "diffusion": ("DDPM", CAT[3], "-", "^"),
    "nsf": ("NSF", CAT[2], "-", "o"),
    "nsf_phys": ("NSF + physics", CAT[0], "-", "o"),
}


def collect(cfg, variant, metric):
    """mean and half-range over seeds for each method and missing fraction."""
    runs = R[cfg]["runs"]
    rates = R[cfg]["rates"]
    out = {}
    for r in rates:
        for key, res in runs.items():
            rr, s, v = key.split("_")
            if float(rr[1:]) != r or v != variant:
                continue
            for mth, d in res.items():
                if metric in d:
                    out.setdefault(mth, {}).setdefault(r, []).append(d[metric])
    table = {}
    for mth, dd in out.items():
        rs = sorted(dd)
        vals = [np.array(dd[r]) for r in rs]
        table[mth] = (np.array(rs), np.array([v.mean() for v in vals]), np.array([np.ptp(v) / 2 for v in vals]))
    return table


def panel_metric(ax, cfg, variant, metric, methods, ylabel, logy=True):
    tab = collect(cfg, variant, metric)
    for mth in methods:
        if mth not in tab:
            continue
        lab, col, ls, mk = STY[mth]
        r, m, h = tab[mth]
        ax.errorbar(100 * r, m, yerr=h, color=col, ls=ls, marker=mk, ms=3.2, lw=1.0, capsize=1.5,
                    mec="white", mew=0.4, label=lab)
    ax.set_xlabel("missing fraction (\\%)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks([30, 60, 90])


fig, axs = plt.subplots(2, 3, figsize=(COL2, 4.75))

# (a), (b): symmetry-induced bimodality, C2 at the largest missing fraction, no observed neighbour
cache = np.load(f"results/{TAG}e3_C2_cache.npy", allow_pickle=True).item()
rmax = max(R["C2"]["rates"])
key = f"r{rmax}_s0_nophase"
cc = cache[key]
sel = cc["nb_obs"][: cc["samples"]["nsf"].shape[1]] == 0
cgrid = np.linspace(-1, 1, 400)
for ax, mth, title, letter in ((axs[0, 0], "nsf", "NSF", "a"),
                               (axs[0, 1], "gpr_ctx", "GPR (full context)", "b")):
    S = cc["samples"][mth]                                  # (n_samples, n_rows, 1)
    n = S.shape[1]
    cx = np.broadcast_to(cc["x_te"][:n][None, :], S.shape[:2])[:, sel].ravel()
    sy = S[:, sel, 0].ravel()
    H, xe, ye = np.histogram2d(cx, sy, bins=[70, 70], range=[[-1.02, 0.25], [-1.25, 1.25]])
    im = ax.pcolormesh(xe, ye, H.T, cmap=SEQ_W, norm=LogNorm(vmin=1, vmax=max(H.max(), 2)), rasterized=True)
    ax.plot(cgrid, np.sqrt(1 - cgrid ** 2), color=INK, lw=0.6)
    ax.plot(cgrid, -np.sqrt(1 - cgrid ** 2), color=INK, lw=0.6)
    ax.set_xlim(-1.02, 0.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlabel(r"collar height $c=y_c/L$")
    ax.set_ylabel(r"spring force $s=F_s/(kL)$")
    ax.text(0.97, 0.52, title, transform=ax.transAxes, ha="right", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    panel_label(ax, letter, y=1.03)

meths = ["static", "gpr", "gpr_ctx", "cvae", "diffusion", "nsf", "nsf_phys"]
ax = axs[0, 2]
panel_metric(ax, "C2", "nophase", "crps", meths, "CRPS / std (C2)")
panel_label(ax, "c", y=1.03)

ax = axs[1, 0]
if "C1" in R:
    panel_metric(ax, "C1", "phase", "crps", meths, "CRPS / std (C1)")
panel_label(ax, "d", y=1.03)

ax = axs[1, 1]
if "C1" in R:
    panel_metric(ax, "C1", "phase", "sw1_section", meths, r"sliced $W_1$, Poincar\'e section (C1)")
panel_label(ax, "e", y=1.03)

ax = axs[1, 2]
if "C3" in R:
    for variant in ("phase", "nophase"):
        tab = collect("C3", variant, "crps")
        for mth in ("gpr", "nsf", "nsf_phys"):
            if mth not in tab:
                continue
            lab, col, ls_m, mk = STY[mth]
            r, m, h = tab[mth]
            if variant == "phase":        # filled markers, the line style of the shared legend
                ax.errorbar(100 * r, m, yerr=h, color=col, ls=ls_m, marker=mk, ms=3.6, lw=1.0, capsize=1.5,
                            mec="white", mew=0.4)
            else:                         # open markers and dotted lines: without the forcing phase
                ax.errorbar(100 * r, m, yerr=h, color=col, ls=":", marker=mk, ms=3.6, lw=1.0, capsize=1.5,
                            mfc="white", mec=col, mew=0.8)
    ax.set_yscale("log")
    ax.set_xticks([30, 60, 90])
    ax.set_xlabel("missing fraction (\\%)")
    ax.set_ylabel("CRPS / std (C3)")
    ax.set_ylim(top=3.0)
    ax.legend(handles=[Line2D([], [], color=INK, ls="-", marker="o", ms=3.6, mec="white", mew=0.4,
                              label="with forcing phase"),
                       Line2D([], [], color=INK, ls=":", marker="o", ms=3.6, mfc="white", mec=INK, mew=0.8,
                              label="without phase")],
              fontsize=8, loc="upper left", handlelength=2.2, borderaxespad=0.2)
panel_label(ax, "f", y=1.03)

handles = [Line2D([], [], color=STY[m][1], ls=STY[m][2], marker=STY[m][3], ms=3.2, mec="white", mew=0.4,
                  label=STY[m][0]) for m in meths]
fig.tight_layout(h_pad=1.1, w_pad=1.3, rect=(0, 0, 1, 0.9))
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.0),
           handlelength=2.2, columnspacing=1.2, frameon=False)
savefig(fig, TAG + "fig_fprm")

# ------------------------------------------------------------------ LaTeX table
os.makedirs("manuscript/tables", exist_ok=True)


def fmt(v, nd=3):
    return f"${v:.{nd}f}$" if v < 0 else f"{v:.{nd}f}"


rows = []
for cfg, variant, cap in (("C1", "phase", r"C1: $s\to(c,\dot\theta)$, forced"),
                          ("C2", "nophase", r"C2: $c\to s$, Langevin"),
                          ("C3", "phase", r"C3: $c\to s$, forced, with phase"),
                          ("C3", "nophase", r"C3: $c\to s$, forced, no phase")):
    if cfg not in R:
        continue
    rmax = max(R[cfg]["rates"])
    stats = {mt: collect(cfg, variant, mt) for mt in ("rho", "nrmse", "crps", "coverage90", "p_correct_sign",
                                                       "sw1_section")}
    rows.append(r"\multicolumn{7}{l}{\textit{" + cap + r"}}\\")
    for mth in meths:
        if mth not in stats["crps"]:
            continue

        def g(mt):
            if mth not in stats[mt]:
                return "--"
            r, m, h = stats[mt][mth]
            i = int(np.argmin(np.abs(r - rmax)))
            return fmt(m[i], 3 if mt != "coverage90" else 2)
        name = STY[mth][0] + (" + phase" if (cfg == "C3" and variant == "phase" and mth == "gpr") else "")
        rows.append(f"{name} & {g('rho')} & {g('nrmse')} & {g('crps')} & {g('coverage90')} & "
                    f"{g('p_correct_sign')} & {g('sw1_section')}\\\\")
head = ["\\begin{tabular}{lcccccc}", "Imputer & $\\rho$ & NRMSE & CRPS & cov. & sign & $W_1$\\\\", "\\hline"]
with open(f"manuscript/tables/{TAG}tab_fprm.tex", "w") as fh:
    fh.write("\n".join(head + rows + ["\\end{tabular}"]) + "\n")
print("\n".join(rows))
