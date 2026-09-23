"""E1 - Model and the five canonical regimes of the conservative PIAR (Secs. II and III).

The five parameter sets span the three stability regimes of the upright position
(kL = Mg, kL > Mg, kL < Mg) and two initial speeds. Every case starts at the upright
position, theta0 = 0, with a small angular velocity.

Trajectories are integrated with DOP853 (rtol = 1e-12) and sampled through the
seventh-order dense-output interpolant at 25 001 instants over 100 s, so that every
curve is drawn from a continuous, smooth representation of the solution.

Outputs
  figures/fig_model.pdf       schematic, potentials of the five cases, pitchfork, critical scales
  figures/fig_cases.pdf       5 x 4 atlas: U(theta), K(theta_dot), phase diagram, theta(t)
  figures/fig_portraits.pdf   phase portraits on the cylinder (log-density + separatrix + orbit)
  figures/cases/case<i>_en.*, caso<i>_es.*   one 2 x 2 figure per case (English / Spanish)
  results/e1_cases.json
"""
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle, Arc
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

from piar.physics import PIAR
from piar.stochastic import baoab_hist2d
from piar.style import set_style, savefig, CAT, COL2, INK, MUTED, SEQ_W, panel_label

set_style()
os.makedirs("figures/cases", exist_ok=True)

T_WINDOW = 100.0          # s, observation window of every case
N_DENSE = 25001           # dense-output samples (dt = 4 ms)

CASES = [  # id, k [N/m], L [m], M [kg], g [m/s^2], theta0 [rad], omega0 [rad/s]
    (1, 2.0, 2.0, 2.0, 2.0, 0.0, 0.1),
    (2, 2.0, 3.0, 4.0, 1.0, 0.0, 0.1),
    (3, 1.0, 4.0, 5.0, 10.0, 0.0, 0.1),
    (4, 0.1, 4.0, 2.0, 0.1, 0.0, 0.4),
    (5, 0.1, 3.0, 5.0, 0.1, 0.0, 0.4),
]
COLS = {i: CAT[i - 1] for i in range(1, 6)}
REGIME = {1: r"$kL=Mg$", 2: r"$kL>Mg$", 3: r"$kL<Mg$", 4: r"$kL>Mg$", 5: r"$kL<Mg$"}


# ----------------------------------------------------------------------------- dynamics
def simulate(p, th0, om0):
    f = lambda t, y: [y[1], p.beta * math.sin(y[0]) - p.lam * math.sin(y[0]) * math.cos(y[0])]
    sol = solve_ivp(f, (0.0, T_WINDOW), [th0, om0], method="DOP853", rtol=1e-12, atol=1e-13,
                    dense_output=True)
    t = np.linspace(0.0, T_WINDOW, N_DENSE)
    th, om = sol.sol(t)
    return t, th, om, sol


def period_exact(p, E):
    """Exact period (s) of the orbit with mechanical energy E (J) through theta = 0."""
    I = p.inertia
    w = lambda th: math.sqrt(max(2.0 * (E - float(p.U(th))) / I, 0.0))
    if E > p.E_sep:                                       # rotation: time to advance 2 pi
        pts = [0.0, math.pi] + ([p.theta_star, 2 * math.pi - p.theta_star] if p.mu > 0 else [])
        val, _ = quad(lambda th: 1.0 / w(th), 0.0, 2 * math.pi, points=sorted(pts), epsabs=1e-13,
                      epsrel=1e-12, limit=800)
        return val, "rotation", float("nan")
    # libration about the upright centre (mu > 0 and E below the tilted saddles)
    th_m = brentq(lambda th: float(p.U(th)) - E, 1e-12, p.theta_star, xtol=1e-15, rtol=1e-15)
    g = lambda psi: th_m * math.cos(psi) / max(w(th_m * math.sin(psi)), 1e-300)
    val, _ = quad(g, 0.0, 0.5 * math.pi, epsabs=1e-13, epsrel=1e-12, limit=800)
    return 4.0 * val, "libration", th_m


def numerical_period(t, th, kind):
    if kind == "rotation":                              # successive crossings of theta = 2 pi n
        n = np.floor(th / (2 * np.pi))
        idx = np.where(np.diff(n) != 0)[0]
        tc = [t[i] + (2 * np.pi * max(n[i], n[i + 1]) - th[i]) * (t[i + 1] - t[i]) / (th[i + 1] - th[i])
              for i in idx]
    else:                                               # upward zero crossings
        idx = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
        tc = [t[i] - th[i] * (t[i + 1] - t[i]) / (th[i + 1] - th[i]) for i in idx]
    return float(np.mean(np.diff(tc))) if len(tc) > 1 else float("nan")


# ----------------------------------------------------------------------------- run cases
results = {"window_s": T_WINDOW, "n_dense": N_DENSE, "cases": []}
runs = {}
sol_cache = {}
for cid, k, L, M, g, th0, om0 in CASES:
    p = PIAR(k=k, L=L, M=M, g=g)
    t, th, om, sol = simulate(p, th0, om0)
    E = p.energy(th, om)
    E0 = float(p.energy(th0, om0))
    T_ex, kind, th_m = period_exact(p, E0)
    T_num = numerical_period(t, th, kind)
    rec = {
        "case": cid, "k": k, "L": L, "M": M, "g": g, "theta0": th0, "omega0": om0,
        "kL": k * L, "Mg": M * g, "mu": p.mu, "lam": p.lam, "beta": p.beta,
        "t_c": math.sqrt(L / g), "MgL": p.MgL, "E0": E0, "E0_over_MgL": E0 / p.MgL,
        "Esep_over_MgL": p.E_sep / p.MgL, "motion": kind,
        "amplitude_rad": th_m, "period_exact_s": T_ex, "period_numeric_s": T_num,
        "turns_in_window": float((th[-1] - th[0]) / (2 * np.pi)) if kind == "rotation" else 0.0,
        "omega_min": float(om.min()), "omega_max": float(om.max()),
        "K_max": float(0.5 * p.inertia * om.max() ** 2) if kind == "rotation" else float(0.5 * p.inertia * np.max(om ** 2)),
        "max_rel_energy_error": float(np.max(np.abs(E - E0)) / p.MgL),
        "nfev": int(sol.nfev),
        "theta_star": p.theta_star if p.mu > 0 else None,
        "barrier_over_MgL": (p.barrier / p.MgL) if p.mu > 0 else None,
    }
    results["cases"].append(rec)
    runs[cid] = (p, t, th, om, E0)
    sol_cache[cid] = sol
    print(f"case {cid}: mu={p.mu:+.3f} E0/MgL={E0/p.MgL:.4f} Esep/MgL={p.E_sep/p.MgL:.4f} {kind:9s} "
          f"T_exact={T_ex:.4f}s T_num={T_num:.4f}s turns={rec['turns_in_window']:.2f} "
          f"amp={th_m:.4f} |dE|/MgL={rec['max_rel_energy_error']:.1e} nfev={sol.nfev}")

# independent check with the symplectic Velocity Verlet scheme (compensated summation), dt = T/2000
from piar import integrators as INT
for rec in results["cases"]:
    p, t, th, om, E0 = runs[rec["case"]]
    dt = rec["period_exact_s"] / 2000.0
    nst = int(np.ceil(T_WINDOW / dt))
    v = INT.verlet(p, rec["theta0"], rec["omega0"], dt, nst)
    th_v = v["theta"] + 2 * np.pi * v["winding"]
    ok = v["t"] <= T_WINDOW
    ref = sol_cache[rec["case"]].sol(v["t"][ok])[0]
    rec["verlet_dt_s"] = dt
    rec["verlet_max_rel_energy_error"] = float(np.max(np.abs(p.energy(v["theta"], v["omega"]) - E0)) / p.MgL)
    rec["verlet_max_angle_difference_rad"] = float(np.max(np.abs(th_v[ok] - ref)))
    print(f"case {rec['case']}: Verlet dt={dt:.4f}s |dE|/MgL={rec['verlet_max_rel_energy_error']:.1e} "
          f"max|dtheta|={rec['verlet_max_angle_difference_rad']:.1e} rad")
# critical slowing down in case 1: fraction of time within 45 degrees of the upright position
p1, t1, th1, om1, _ = runs[1]
thw1 = (th1 + np.pi) % (2 * np.pi) - np.pi
results["cases"][0]["fraction_time_within_45deg_of_upright"] = float(np.mean(np.abs(thw1) < np.pi / 4))
print("case 1 fraction within 45 deg:", results["cases"][0]["fraction_time_within_45deg_of_upright"])

json.dump(results, open("results/e1_cases.json", "w"), indent=2)


# ----------------------------------------------------------------------------- helpers
def pi_ticks(ax, lo, hi, axis="x", max_ticks=6):
    """Ticks at multiples of pi (or pi/2) for angle axes spanning [lo, hi]."""
    span = hi - lo
    step = np.pi if span > 2.5 * np.pi else np.pi / 2 if span > 0.8 * np.pi else None
    if step is None:
        return
    while span / step > max_ticks:
        step *= 2
    ticks = np.arange(math.ceil(lo / step) * step, hi + 1e-9, step)
    labels = []
    for tk in ticks:
        n = int(round(tk / (np.pi / 2)))
        if n == 0:
            labels.append(r"$0$")
        elif n % 2 == 0:
            m = n // 2
            labels.append(r"$\pi$" if m == 1 else r"$-\pi$" if m == -1 else rf"${m}\pi$")
        else:
            labels.append(rf"${'-' if n < 0 else ''}\pi/2$" if abs(n) == 1 else rf"${n}\pi/2$")
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks)
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(labels)


def draw_case(axs, cid, lang="en", titles=True):
    """Four diagnostic panels: U(theta), K(theta_dot), phase diagram, theta(t)."""
    p, t, th, om, E0 = runs[cid]
    col = COLS[cid]
    lw = 1.1
    T = {"en": ("Potential energy", "Kinetic energy", "Phase diagram", "Position"),
         "es": ("Energ\\'ia potencial", "Energ\\'ia cin\\'etica", "Diagrama de fase", "Posici\\'on")}[lang]
    rotation = results["cases"][cid - 1]["motion"] == "rotation"

    # [1] potential energy along the trajectory, U(theta)
    ax = axs[0]
    lo, hi = th.min(), th.max()
    if rotation:
        xlo, xhi = lo, hi
    else:   # libration: show the whole upright well, bounded by the tilted saddles
        xm = 1.25 * p.theta_star
        xlo, xhi = -xm, xm
    grid = np.linspace(xlo, xhi, 4000)
    ax.plot(grid, p.U(grid), color=MUTED, lw=0.7, zorder=1)
    ax.plot(th, p.U(th), color=col, lw=lw, zorder=2)
    ax.axhline(E0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$U$ (J)")
    ytop = max(E0, p.U(grid).max())
    ybot = p.U(grid).min()
    ax.set_ylim(ybot - 0.06 * (ytop - ybot), ytop + 0.14 * (ytop - ybot))
    ax.text(xhi - 0.01 * (xhi - xlo), E0, r"$E$", ha="right", va="bottom", fontsize=8.5, color=INK)

    # [2] kinetic energy versus angular velocity, K = I omega^2 / 2
    ax = axs[1]
    wlo, whi = om.min(), om.max()
    pad = 0.08 * (whi - wlo)
    wg = np.linspace(wlo - pad, whi + pad, 800)
    ax.plot(wg, 0.5 * p.inertia * wg ** 2, color=MUTED, lw=0.7, zorder=1)
    ax.plot(om, 0.5 * p.inertia * om ** 2, color=col, lw=lw, zorder=2)
    ax.set_xlim(wg[0], wg[-1])
    ax.set_xlabel(r"$\dot\theta$ (rad/s)")
    ax.set_ylabel(r"$K$ (J)")

    # [3] phase diagram with the separatrix
    ax = axs[2]
    ax.plot(th, om, color=col, lw=lw, zorder=3)
    sg = np.linspace(xlo, xhi, 6000)
    up, dn = p.separatrix(sg)
    ax.plot(sg, up, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.plot(sg, dn, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=2)
    if rotation:
        ylo, yhi = 0.0, 1.08 * om.max()
    else:
        yy = np.nanmax(np.abs(up))
        ylo, yhi = -1.12 * yy, 1.12 * yy
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(xlo, xhi)
    # equilibria inside the window
    for th_e, kind in p.equilibria():
        for n in range(int(math.floor(xlo / (2 * np.pi))) - 1, int(math.ceil(xhi / (2 * np.pi))) + 2):
            for te in ((th_e, -th_e) if kind == "saddle" and th_e > 0 else (th_e,)):
                x = te + 2 * np.pi * n
                if xlo <= x <= xhi and ylo <= 0 <= yhi:
                    if kind == "centre":
                        ax.plot(x, 0, "o", ms=3.0, color=INK, mec="white", mew=0.4, zorder=4, clip_on=False)
                    else:
                        ax.plot(x, 0, "x", ms=3.4, color=INK, mew=0.8, zorder=4, clip_on=False)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")

    # [4] position versus time
    ax = axs[3]
    ax.plot(t, th, color=col, lw=lw)
    ax.set_xlim(0, T_WINDOW)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$\theta$ (rad)")
    if titles:
        for a, tt in zip(axs, T):
            a.set_title(tt, fontsize=9)
    for a in (axs[0], axs[2]):
        if not rotation:
            pi_ticks(a, xlo, xhi) if (xhi - xlo) > 0.8 * np.pi else None
    return p


# ============================================================================ Fig. model
fig = plt.figure(figsize=(COL2, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
th_s = 0.6
x, y = math.sin(th_s), math.cos(th_s)
xg = -1.25                                                  # vertical guide of the collar, d = 1.25 L
for dx in (-0.035, 0.035):
    ax.plot([xg + dx, xg + dx], [-1.2, 1.3], color=MUTED, lw=0.8)
circ = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.sin(circ), np.cos(circ), color=MUTED, lw=0.5, ls=(0, (2, 2)))      # path of the mass
ax.plot([0, 0], [0, 1.22], color=MUTED, lw=0.5, ls=(0, (2, 2)))
ax.plot([0, x], [0, y], color=INK, lw=1.8, solid_capstyle="round", zorder=3)
x0, x1 = xg + 0.06, x - 0.07
xs = np.linspace(x0 + 0.05, x1 - 0.05, 500)
ys = y + 0.045 * np.sin(2 * np.pi * 12 * (xs - xs[0]) / (xs[-1] - xs[0]))
ax.plot([xg + 0.035, x0 + 0.05], [y, y], color=CAT[0], lw=0.9)
ax.plot(xs, ys, color=CAT[0], lw=0.9)
ax.plot([x1 - 0.05, x], [y, y], color=CAT[0], lw=0.9)
ax.add_patch(Rectangle((xg - 0.07, y - 0.08), 0.14, 0.16, facecolor="#d9d9d9", edgecolor=INK, lw=0.6, zorder=4))
ax.add_patch(Circle((x, y), 0.085, facecolor=CAT[1], edgecolor=INK, lw=0.6, zorder=5))
ax.add_patch(Circle((0, 0), 0.035, facecolor=INK, zorder=5))
ax.add_patch(Arc((0, 0), 0.7, 0.7, theta1=90 - math.degrees(th_s), theta2=90, color=INK, lw=0.6))
ax.text(0.1, 0.45, r"$\theta$", fontsize=10)
ax.text(x + 0.1, y + 0.05, r"$M$", fontsize=10)
ax.text(x / 2 + 0.14, y / 2 - 0.12, r"$L$", fontsize=10)
ax.text((xg + x) / 2 - 0.05, y + 0.1, r"$k$", fontsize=10, color=CAT[0])
ax.annotate("", xy=(xg - 0.16, y), xytext=(xg - 0.16, 0.0),
            arrowprops=dict(arrowstyle="<->", lw=0.5, color=INK, shrinkA=0, shrinkB=0))
ax.text(xg - 0.22, y / 2, r"$y_c$", fontsize=9.5, ha="right", va="center")
ax.plot([xg + 0.035, 0], [0, 0], color=MUTED, lw=0.5, ls=(0, (2, 2)))
ax.annotate("", xy=(xg + 0.04, -0.14), xytext=(0.0, -0.14),
            arrowprops=dict(arrowstyle="<->", lw=0.5, color=INK, shrinkA=0, shrinkB=0))
ax.text(xg / 2, -0.2, r"$d$", fontsize=9.5, ha="center", va="top")
ax.annotate("", xy=(0.82, 0.05), xytext=(0.82, -0.3), arrowprops=dict(arrowstyle="<|-", lw=0.6, color=INK))
ax.text(0.87, -0.15, r"$g$", fontsize=9.5)
ax.set_xlim(-1.75, 1.2)
ax.set_ylim(-1.18, 1.32)
ax.set_aspect("equal")
ax.axis("off")
panel_label(ax, "a", x=0.0, y=0.98)

ax = fig.add_subplot(gs[0, 1])
thg = np.linspace(-np.pi, np.pi, 1201)
for cid, *_ in CASES:
    p = runs[cid][0]
    q = PIAR.dimensionless(p.mu)
    ax.plot(thg, q.U(thg), color=COLS[cid], lw=0.9, label=rf"{cid}: $\mu={p.mu:+.2f}$".replace("+0.00", "0"))
ax.set_xlim(-np.pi, np.pi)
pi_ticks(ax, -np.pi, np.pi)
ax.set_xlabel(r"$\theta$ (rad)")
ax.set_ylabel(r"$U(\theta)/MgL$")
ax.legend(loc="lower center", fontsize=8, handlelength=1.4, ncol=1, bbox_to_anchor=(0.5, -0.01), labelspacing=0.3)
panel_label(ax, "b", x=-0.12)

ax = fig.add_subplot(gs[1, 0])
mus = np.linspace(-1, 1.3, 800)
ax.plot(mus[mus > 0], 0 * mus[mus > 0], color=INK, lw=1.2)
ax.plot(mus[mus <= 0], 0 * mus[mus <= 0], color=INK, lw=0.8, ls=(0, (2, 1.5)))
mp = mus[mus > 0]
ts = np.arccos(1 / (1 + mp))
ax.plot(mp, ts, color=MUTED, lw=0.9, ls=(0, (2, 1.5)))
ax.plot(mp, -ts, color=MUTED, lw=0.9, ls=(0, (2, 1.5)))
ax.plot(mus, np.pi + 0 * mus, color=INK, lw=1.2)
ax.plot(mus, -np.pi + 0 * mus, color=INK, lw=1.2)
for cid, *_ in CASES:
    mu = runs[cid][0].mu
    ax.axvline(mu, color=COLS[cid], lw=1.0, ymin=0.08, ymax=0.92, zorder=0)
    ax.text(mu, 3.62, str(cid), color=COLS[cid], fontsize=9, ha="center", va="bottom")
ax.text(0.70, 0.18, r"$\theta=0$", fontsize=8.5)
ax.text(0.62, 1.20, r"$\pm\theta^*$", fontsize=8.5, color=MUTED)
ax.text(-0.97, np.pi - 0.62, r"$\theta=\pm\pi$", fontsize=8.5)
ax.set_ylim(-3.9, 3.9)
ax.set_xlim(-1.0, 1.3)
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax.set_xlabel(r"$\mu=kL/(Mg)-1$")
ax.set_ylabel(r"equilibrium angle (rad)")
panel_label(ax, "c", x=-0.12, y=1.08)

ax = fig.add_subplot(gs[1, 1])
mm = np.logspace(-3, 0.3, 300)
ax.loglog(mm, np.sqrt(mm), color=CAT[0], lw=1.0, label=r"$\omega_0\sqrt{L/g}=\mu^{1/2}$")
ax.loglog(mm, np.arccos(1 / (1 + mm)), color=CAT[2], lw=1.0, label=r"$\theta^*\simeq(2\mu)^{1/2}$")
ax.loglog(mm, mm ** 2 / (2 * (1 + mm)), color=CAT[1], lw=1.0, label=r"$\Delta U/MgL\simeq\mu^2/2$")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"frequency, tilt, and barrier")
ax.legend(loc="lower right", fontsize=8, handlelength=1.4)
ax.set_ylim(1e-7, 30)
panel_label(ax, "d", x=-0.12, y=1.08)
savefig(fig, "fig_model")

# ============================================================================ Fig. cases (atlas)
fig = plt.figure(figsize=(COL2, 7.9))
gs = fig.add_gridspec(5, 4, left=0.085, right=0.995, top=0.925, bottom=0.055, hspace=1.12, wspace=0.62)
for r, (cid, k, L, M, g, th0, om0) in enumerate(CASES):
    axs_r = [fig.add_subplot(gs[r, c_]) for c_ in range(4)]
    p = draw_case(axs_r, cid, "en", titles=(r == 0))
    lab = (rf"\textbf{{Case {cid}}}\quad $k={k:g}$ N/m, $L={L:g}$ m, $M={M:g}$ kg, $g={g:g}$ m/s$^2$;"
           rf"\quad $\theta_0=0$, $\dot\theta_0={om0:g}$ rad/s;\quad {REGIME[cid]}, $\mu={p.mu:+.2f}$").replace("+0.00", "0")
    bb = axs_r[0].get_position()
    fig.text(0.005, bb.y1 + (0.036 if r == 0 else 0.017), lab, fontsize=8.5, ha="left", va="bottom", color=INK)
    for c_, letter in enumerate("abcd"):
        axs_r[c_].text(-0.03, 1.03, rf"({letter}{cid})", transform=axs_r[c_].transAxes, fontsize=8,
                       ha="right", va="bottom", color=INK)
savefig(fig, "fig_cases")

# ============================================================================ per-case figures
SUP = {"en": "Case {i}: $\\theta_0=0$, $\\dot\\theta_0={w:g}$ rad/s; $k={k:g}$ N/m, $L={L:g}$ m, $M={M:g}$ kg, $g={g:g}$ m/s$^2$ ($\\mu={mu:+.2f}$)",
       "es": "Caso {i}: $\\theta_0=0$, $\\omega_0={w:g}$ rad/s; $k={k:g}$ N/m, $L={L:g}$ m, $M={M:g}$ kg, $g={g:g}$ m/s$^2$ ($\\mu={mu:+.2f}$)"}
for cid, k, L, M, g, th0, om0 in CASES:
    for lang in ("en", "es"):
        fig, a = plt.subplots(2, 2, figsize=(5.6, 4.4))
        p = draw_case(a.ravel(), cid, lang, titles=True)
        fig.suptitle(SUP[lang].format(i=cid, w=om0, k=k, L=L, M=M, g=g, mu=p.mu).replace("+0.00", "0"), fontsize=8.5)
        fig.tight_layout(h_pad=1.2, w_pad=1.2)
        stem = f"cases/case{cid}_en" if lang == "en" else f"cases/caso{cid}_es"
        savefig(fig, stem)

# ============================================================================ Fig. portraits
T_BATH, OM_MAX = 0.5, 3.4
fig, axs = plt.subplots(1, 5, figsize=(COL2, 2.05))
for ax, (cid, *_rest) in zip(axs, CASES):
    p, t, th, om, E0 = runs[cid]
    q = PIAR.dimensionless(p.mu)
    H = baoab_hist2d(math.pi, 0.0, 0.02, 12_000_000, q.lam, q.beta, 0.02, T_BATH, cid, 150, 120, OM_MAX)
    xe = np.linspace(-np.pi, np.pi, H.shape[0] + 1)
    ye = np.linspace(-OM_MAX, OM_MAX, H.shape[1] + 1)
    Hm = np.ma.masked_where(H == 0, H)
    im = ax.pcolormesh(xe, ye, Hm.T, cmap=SEQ_W, norm=LogNorm(vmin=1, vmax=H.max()), shading="flat",
                       rasterized=True)
    sg = np.linspace(-np.pi, np.pi, 4001)
    up, dn = q.separatrix(sg)
    ax.plot(sg, up, color=INK, lw=0.7)
    ax.plot(sg, dn, color=INK, lw=0.7)
    # the orbit of the case, in natural units, wrapped on the cylinder without spurious jumps
    thw = (th + np.pi) % (2 * np.pi) - np.pi
    omn = om * math.sqrt(p.L / p.g)
    jumps = np.where(np.abs(np.diff(thw)) > np.pi)[0] + 1
    ax.plot(np.insert(thw, jumps, np.nan), np.insert(omn, jumps, np.nan), color=COLS[cid], lw=1.2)
    for th_e, kind in q.equilibria():
        for te in ((th_e, -th_e) if abs(th_e - math.pi) < 1e-9 else (th_e,)):
            if kind == "centre":
                ax.plot(te, 0, "o", ms=3.0, color=INK, mec="white", mew=0.4, zorder=5)
            else:
                ax.plot(te, 0, "x", ms=3.6, color=INK, mew=0.9, zorder=5)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-OM_MAX, OM_MAX)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_yticks([-2, 0, 2])
    ax.set_title(rf"({'abcde'[cid - 1]}) $\mu={p.mu:+.2f}$".replace("+0.00", "0"), fontsize=8.5, loc="left")
    ax.set_xlabel(r"$\theta$ (rad)")
axs[0].set_ylabel(r"$\dot\theta\,\sqrt{L/g}$")
for ax in axs[1:]:
    ax.tick_params(labelleft=False)
cb = fig.colorbar(im, ax=axs, fraction=0.018, pad=0.012)
cb.set_label(r"occupation", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
savefig(fig, "fig_portraits")
print("done")
