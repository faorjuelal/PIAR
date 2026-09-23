"""Interactive laboratory: one function that renders the six dashboard panels (Sec. VIII).

    fig = dashboard_figure(mu=-0.8, gamma=0.25, A=0.8, Omega=2/3, th0=0.1, om0=0.0)

The Jupyter notebook notebooks/PIAR_Dashboard.ipynb wraps this function with ipywidgets
sliders; the same function produces the static snapshot figures/fig_dashboard.pdf.
"""
from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import welch

from matplotlib.collections import LineCollection

from .physics import PIAR
from .chaos import trajectory
from .smooth import hermite_upsample, break_on_wrap
from .style import CAT, INK, MUTED, SEQ_W, COL2, panel_label

TWO_PI = 2.0 * math.pi


def _wrap(x):
    return (x + math.pi) % TWO_PI - math.pi


def simulate(mu, gamma, A, Omega, th0, om0, n_periods=600, spp=128, save=128):
    """Trajectory of the (possibly damped and driven) PIAR in natural units.

    Every RK4 step is stored (save = spp); curves are then drawn through the cubic Hermite
    interpolant of piar.smooth, which uses theta' = omega and omega' = acceleration and is
    therefore smooth at any zoom level.
    """
    p = PIAR.dimensionless(mu, gamma=gamma, A=A, Omega=Omega, forcing="torque")
    tr = trajectory(p, th0, om0, n_periods, spp=spp, save_per_period=save)
    return p, tr[:, 0], tr[:, 1], tr[:, 2]


def smooth_segment(p, t, th, om, start, stop, factor=8):
    """Hermite-refined copy of the samples [start, stop)."""
    sl = slice(max(0, start), stop)
    return hermite_upsample(t[sl], th[sl], om[sl], lambda a, b, c: p.accel(a, b, c), factor=factor)


def fading_line(ax, x, y, color, lw0=0.3, lw1=2.0, zorder=2):
    """Motion trail: a continuous line whose opacity and width grow towards the present."""
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    w = np.linspace(0.0, 1.0, len(segs))
    # opaque colours blended towards white (overlapping semi-transparent segments would add up)
    base = np.array(plt.matplotlib.colors.to_rgb(color))
    f = (0.06 + 0.94 * w ** 1.6)[:, None]
    rgba = np.column_stack([1.0 - f * (1.0 - base), np.ones(len(segs))])
    lc = LineCollection(segs, colors=rgba, linewidths=lw0 + (lw1 - lw0) * w, capstyle="butt", zorder=zorder)
    ax.add_collection(lc)


def dashboard_figure(mu=-0.8, gamma=0.25, A=0.8, Omega=2 / 3, th0=0.1, om0=0.0, n_periods=1500,
                     trail=0.75, figsize=(COL2, 4.4)):
    """trail: length of the motion trail in forcing periods."""
    p, t, th, om = simulate(mu, gamma, A, Omega, th0, om0, n_periods=n_periods)
    spp = 128
    thw = _wrap(th)
    u = lambda x: np.cos(x) + 0.5 * p.lam * np.sin(x) ** 2          # U / MgL
    fig, axs = plt.subplots(2, 3, figsize=figsize)

    # (a) schematic with motion trail
    ax = axs[0, 0]
    ax.set_aspect("equal")
    ax.plot([-1.3, 1.3], [-1.08, -1.08], color=INK, lw=0.8)
    n = len(th)
    _, th_tr, _ = smooth_segment(p, t, th, om, n - int(trail * spp) - 1, n, factor=8)
    fading_line(ax, np.sin(th_tr), np.cos(th_tr), CAT[0])             # smooth fading trail of the mass
    xs, ys = np.sin(th[-1]), np.cos(th[-1])
    xg = -1.15                                                        # vertical guide of the collar
    ax.plot([xg, xg], [-1.08, 1.25], color=MUTED, lw=1.0, zorder=0)
    ax.plot([0, xs], [0, ys], color=INK, lw=1.6)
    ax.plot([0], [0], "o", ms=3, color=INK)                          # pivot
    spring_x = np.linspace(xg, xs, 120)
    ax.plot(spring_x, ys + 0.04 * np.sin(np.linspace(0, 18 * math.pi, 120)), color=CAT[0], lw=0.8)
    ax.plot([xg], [ys], "s", ms=4, color="#dddddd", mec=INK, mew=0.6)
    ax.plot([xs], [ys], "o", ms=6, color=CAT[1], mec=INK, mew=0.5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.15, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    panel_label(ax, "a")

    # (b) energy landscape with the current energy
    ax = axs[0, 1]
    x = np.linspace(-math.pi, math.pi, 600)
    ax.plot(x, u(x), color=INK, lw=1.0)
    E = 0.5 * om ** 2 + u(th)
    ax.axhline(E[-1], color=CAT[0], lw=0.8, ls="--", label=r"current $E$")
    ax.axhline(p.E_sep / p.MgL, color=MUTED, lw=0.6, label=r"$E_{\mathrm{sep}}$")
    ax.plot(thw[-1], u(th[-1]), "o", color=CAT[1], ms=4, mec=INK, mew=0.4)
    ax.set_xticks([-math.pi, 0, math.pi])
    ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$U/MgL$")
    ax.legend(fontsize=8, loc="lower center", handlelength=1.6)
    panel_label(ax, "b")

    # (c) phase portrait: log occupation + separatrix + last trajectory segment + fixed points
    ax = axs[0, 2]
    omax = max(2.5, 1.1 * np.max(np.abs(om)))
    H, xe, ye = np.histogram2d(thw, om, bins=[72, 56], range=[[-math.pi, math.pi], [-omax, omax]])
    ax.pcolormesh(xe, ye, H.T, cmap=SEQ_W, norm=LogNorm(vmin=1, vmax=max(2, H.max())), rasterized=True)
    ps = np.sqrt(np.maximum(0.0, 2 * (p.E_sep / p.MgL - u(x))))
    ax.plot(x, ps, color=INK, lw=0.7)
    ax.plot(x, -ps, color=INK, lw=0.7)
    _, th_s, om_s = smooth_segment(p, t, th, om, n - 3 * spp, n, factor=8)
    tw, ow = break_on_wrap(th_s, om_s)
    ax.plot(tw, ow, color=CAT[0], lw=0.6, alpha=0.9)
    for th_e, kind in p.equilibria():
        st = kind == "centre"
        ax.plot(_wrap(th_e), 0, "o" if st else "X", ms=4, color=CAT[0] if st else CAT[1], mec="white", mew=0.4)
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-omax, omax)
    ax.set_xticks([-math.pi, 0, math.pi])
    ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot\theta\sqrt{L/g}$")
    panel_label(ax, "c")

    # (d) stroboscopic Poincare section
    ax = axs[1, 0]
    per = spp
    k0 = min(len(th) // 3, 100 * per)
    k0 += (per - 1 - k0 % per) % per                                  # sample at t = n T_d exactly
    ax.plot(thw[k0::per], om[k0::per], ".", ms=1.2, color=INK, rasterized=True)
    ax.set_xlim(-math.pi, math.pi)
    ax.set_xticks([-math.pi, 0, math.pi])
    ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_xlabel(r"$\theta(nT_d)$")
    ax.set_ylabel(r"$\dot\theta(nT_d)\sqrt{L/g}$")
    panel_label(ax, "d")

    # (e) measurable observables
    ax = axs[1, 1]
    tw_, thw_, _ = smooth_segment(p, t, th, om, n - 6 * per, n, factor=4)
    ax.plot(tw_ - tw_[0], np.sin(thw_), color=CAT[0], lw=0.8, label=r"$F_s/(kL)=\sin\theta$")
    ax.plot(tw_ - tw_[0], np.cos(thw_), color=CAT[1], lw=0.8, label=r"$y_c/L=\cos\theta$")
    ax.set_xlabel(r"$t\sqrt{g/L}$")
    ax.set_ylabel(r"$F_s/(kL)$, $y_c/L$")
    ax.set_ylim(-1.1, 1.1)
    panel_label(ax, "e")

    # (f) power spectrum of the spring force
    ax = axs[1, 2]
    xs_ = np.sin(th[len(th) // 3::4])                                 # 32 samples per forcing period
    fs = (per // 4) / (TWO_PI / Omega)
    f, P = welch(xs_ - xs_.mean(), fs=fs, nperseg=min(4096, len(xs_)), window="hann")
    ax.semilogy(f / (Omega / TWO_PI), P + 1e-14, color=CAT[0], lw=0.7)
    ax.set_xlim(0, 4.2)
    ax.set_xlabel(r"frequency / $f_d$")
    ax.set_ylabel(r"PSD of $\sin\theta$")
    panel_label(ax, "f")
    fig.tight_layout(h_pad=0.9, w_pad=0.8)
    return fig


def animate(mu=-0.8, gamma=0.25, A=0.8, Omega=2 / 3, th0=0.1, om0=0.0, seconds=20.0, fps=30,
            speed=1.0, trail=0.6, figsize=(4.8, 2.6)):
    """Smooth animation of the PIAR (schematic with a fading trail and the phase portrait).

    Frames are interpolated from the stored RK4 steps with the cubic Hermite interpolant, so the
    motion is continuous for any frame rate. In a notebook: ``HTML(animate().to_jshtml())``.
    ``speed`` is the number of natural time units shown per second of animation.
    """
    from matplotlib.animation import FuncAnimation

    Td = TWO_PI / Omega
    n_periods = int(math.ceil((seconds * speed + 2 * Td) / Td)) + 1
    p, t, th, om = simulate(mu, gamma, A, Omega, th0, om0, n_periods=n_periods)
    tf, thf, omf = hermite_upsample(t, th, om, lambda a, b, c: p.accel(a, b, c), factor=8)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.25)
    ax.set_ylim(-1.2, 1.3)
    ax.axis("off")
    xg = -1.15
    ax.plot([xg, xg], [-1.15, 1.25], color=MUTED, lw=1.0)
    trail_ln, = ax.plot([], [], color=CAT[0], lw=1.2, alpha=0.7)
    rod, = ax.plot([], [], color=INK, lw=2.0, solid_capstyle="round")
    spring, = ax.plot([], [], color=CAT[0], lw=0.9)
    collar, = ax.plot([], [], "s", ms=5, color="#dddddd", mec=INK, mew=0.6)
    bob, = ax.plot([], [], "o", ms=8, color=CAT[1], mec=INK, mew=0.6)
    x = np.linspace(-math.pi, math.pi, 400)
    u = np.cos(x) + 0.5 * p.lam * np.sin(x) ** 2
    ps = np.sqrt(np.maximum(0.0, 2 * (p.E_sep / p.MgL - u)))
    bx.plot(x, ps, color=MUTED, lw=0.7)
    bx.plot(x, -ps, color=MUTED, lw=0.7)
    omax = max(2.5, 1.1 * np.max(np.abs(omf)))
    bx.set_xlim(-math.pi, math.pi)
    bx.set_ylim(-omax, omax)
    bx.set_xticks([-math.pi, 0, math.pi])
    bx.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    bx.set_xlabel(r"$\theta$")
    bx.set_ylabel(r"$\dot\theta\sqrt{L/g}$")
    ph_ln, = bx.plot([], [], color=CAT[0], lw=0.8)
    ph_pt, = bx.plot([], [], "o", ms=4, color=CAT[1], mec=INK, mew=0.5)
    dt_f = np.median(np.diff(tf))
    n_trail = int(trail * Td / dt_f)
    frames = int(seconds * fps)

    def update(i):
        k = int(min(len(tf) - 1, (i / fps) * speed / dt_f))
        a = max(0, k - n_trail)
        xs, ys = np.sin(thf[k]), np.cos(thf[k])
        trail_ln.set_data(np.sin(thf[a:k + 1]), np.cos(thf[a:k + 1]))
        rod.set_data([0, xs], [0, ys])
        sx = np.linspace(xg, xs, 120)
        spring.set_data(sx, ys + 0.045 * np.sin(np.linspace(0, 18 * math.pi, 120)))
        collar.set_data([xg], [ys])
        bob.set_data([xs], [ys])
        b = max(0, k - 4 * n_trail)
        tw, ow = break_on_wrap(thf[b:k + 1], omf[b:k + 1])
        ph_ln.set_data(tw, ow)
        ph_pt.set_data([tw[-1]], [ow[-1]])
        return trail_ln, rod, spring, collar, bob, ph_ln, ph_pt

    fig.tight_layout()
    return FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)


CASES = {  # the five conservative regimes of Table I: k [N/m], L [m], M [kg], g [m/s^2], theta0 [rad], omega0 [rad/s]
    1: (2.0, 2.0, 2.0, 2.0, 0.0, 0.1),
    2: (2.0, 3.0, 4.0, 1.0, 0.0, 0.1),
    3: (1.0, 4.0, 5.0, 10.0, 0.0, 0.1),
    4: (0.1, 4.0, 2.0, 0.1, 0.0, 0.4),
    5: (0.1, 3.0, 5.0, 0.1, 0.0, 0.4),
}


def case_figure(k=2.0, L=2.0, M=2.0, g=2.0, th0=0.0, om0=0.1, t_end=100.0, n=25001, lang="es",
                color=None, figsize=(5.6, 4.4)):
    """Four diagnostics of a conservative case (potential energy vs angle, kinetic energy vs angular
    velocity, phase diagram with separatrix, angle vs time), drawn from the dense output of DOP853.

    ``case_figure(*CASES[2])`` reproduces one row of Fig. 2 of the manuscript.
    """
    from scipy.integrate import solve_ivp

    p = PIAR(k=k, L=L, M=M, g=g)
    f = lambda t, y: [y[1], p.beta * math.sin(y[0]) - p.lam * math.sin(y[0]) * math.cos(y[0])]
    sol = solve_ivp(f, (0.0, t_end), [th0, om0], method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True)
    t = np.linspace(0.0, t_end, n)
    th, om = sol.sol(t)
    E0 = float(p.energy(th0, om0))
    col = color or CAT[0]
    titles = {"es": ("Energía potencial", "Energía cinética", "Diagrama de fase", "Posición"),
              "en": ("Potential energy", "Kinetic energy", "Phase diagram", "Position")}[lang]
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    a = axs.ravel()
    libr = E0 < p.E_sep and p.mu > 0 and abs(th0) < p.theta_star
    lo, hi = (th.min(), th.max()) if not libr else (-1.25 * p.theta_star, 1.25 * p.theta_star)
    grid = np.linspace(lo, hi, 3000)
    a[0].plot(grid, p.U(grid), color=MUTED, lw=0.7)
    a[0].plot(th, p.U(th), color=col, lw=1.1)
    a[0].axhline(E0, color=INK, lw=0.6, ls=(0, (3, 2)))
    a[0].set_xlim(lo, hi)
    a[0].set_xlabel(r"$\theta$ (rad)")
    a[0].set_ylabel(r"$U$ (J)")
    wg = np.linspace(om.min(), om.max(), 400)
    a[1].plot(wg, 0.5 * p.inertia * wg ** 2, color=col, lw=1.1)
    a[1].set_xlabel(r"$\dot\theta$ (rad/s)")
    a[1].set_ylabel(r"$K$ (J)")
    a[2].plot(th, om, color=col, lw=1.1)
    up, dn = p.separatrix(grid)
    a[2].plot(grid, up, color=INK, lw=0.6, ls=(0, (3, 2)))
    a[2].plot(grid, dn, color=INK, lw=0.6, ls=(0, (3, 2)))
    a[2].set_xlim(lo, hi)
    if not libr:
        a[2].set_ylim(min(0.0, om.min()), 1.08 * max(om.max(), 0.0) if om.max() > 0 else 0.0)
    a[2].set_xlabel(r"$\theta$ (rad)")
    a[2].set_ylabel(r"$\dot\theta$ (rad/s)")
    a[3].plot(t, th, color=col, lw=1.1)
    a[3].set_xlim(0, t_end)
    a[3].set_xlabel(r"$t$ (s)")
    a[3].set_ylabel(r"$\theta$ (rad)")
    for ax, tt in zip(a, titles):
        ax.set_title(tt, fontsize=8)
    fig.suptitle(rf"$k={k:g}$, $L={L:g}$, $M={M:g}$, $g={g:g}$; $\mu={p.mu:+.2f}$", fontsize=8.5)
    fig.tight_layout()
    return fig
