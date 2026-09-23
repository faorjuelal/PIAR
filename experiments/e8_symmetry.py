"""E8 - Spatiotemporal symmetry of the reference strange attractor.

Checks that the attractor at A = 0.8 (mu = -0.8, gamma = 0.25, Omega = 2/3) is invariant under
(theta, t) -> (-theta, t + pi/Omega), Eq. (st_symmetry): the stroboscopic section at phase 0 is
compared with the mirror image of the section at phase pi (total-variation distance between 40x40
histograms), against the distance between the sections at phase 0 of two independent records of the
same length, which measures the sampling noise; the time averages of sin(theta) and of the angular
velocity must vanish. Three initial conditions, 2e5 forcing periods each; RK4 with 128 steps per
forcing period (as in Sec. IV).
Writes results/e8_symmetry.json.
"""
import json
import os

import numpy as np
from numba import njit

MU, GAMMA, A, OMEGA = -0.8, 0.25, 0.8, 2.0 / 3.0
LAM, BETA = 1.0 + MU, 1.0            # natural units: lambda = k/M -> 1 + mu, beta = g/L -> 1
N_TRANS, N_REC, SPP = 400, 200000, 128


@njit(cache=True)
def _rhs(t, th, om):
    return om, -GAMMA * om - LAM * np.sin(th) * np.cos(th) + BETA * np.sin(th) + A * np.cos(OMEGA * t)


@njit(cache=True)
def _run(th, om):
    h = 2.0 * np.pi / OMEGA / SPP
    t = 0.0
    s0 = np.empty((N_REC, 2))
    spi = np.empty((N_REC, 2))
    m_sin = 0.0
    m_om = 0.0
    cnt = 0
    for p in range(N_TRANS + N_REC):
        for k in range(SPP):
            if p >= N_TRANS:
                if k == 0:
                    s0[p - N_TRANS, 0] = th
                    s0[p - N_TRANS, 1] = om
                if k == SPP // 2:
                    spi[p - N_TRANS, 0] = th
                    spi[p - N_TRANS, 1] = om
                m_sin += np.sin(th)
                m_om += om
                cnt += 1
            k1a, k1b = _rhs(t, th, om)
            k2a, k2b = _rhs(t + h / 2, th + h / 2 * k1a, om + h / 2 * k1b)
            k3a, k3b = _rhs(t + h / 2, th + h / 2 * k2a, om + h / 2 * k2b)
            k4a, k4b = _rhs(t + h, th + h * k3a, om + h * k3b)
            th += h / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
            om += h / 6 * (k1b + 2 * k2b + 2 * k3b + k4b)
            t += h
    return s0, spi, m_sin / cnt, m_om / cnt


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _tv(a, b, bins):
    ha, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=bins)
    hb, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=bins)
    return float(0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum())


def main():
    bins = [np.linspace(-np.pi, np.pi, 41), np.linspace(-3.0, 3.0, 41)]
    out = {"params": {"mu": MU, "gamma": GAMMA, "A": A, "Omega": OMEGA, "periods": N_REC}, "runs": [],
           "baseline_pairs": []}
    sections = []
    for ic in [(0.0, 0.0), (1.0, -0.5), (-2.0, 1.0)]:
        s0, spi, m_sin, m_om = _run(*ic)
        a = np.c_[_wrap(s0[:, 0]), s0[:, 1]]
        b = np.c_[_wrap(-spi[:, 0]), -spi[:, 1]]           # mirror image of the section at phase pi
        sections.append(a)
        run = {"ic": ic, "mean_sin_theta": float(m_sin), "mean_omega": float(m_om),
               "tv_section0_vs_mirrored_pi": _tv(a, b, bins)}
        out["runs"].append(run)
        print(run, flush=True)
    for i in range(len(sections)):                         # sampling noise: independent records, same length
        for j in range(i + 1, len(sections)):
            pair = {"pair": [i, j], "tv_section0_independent": _tv(sections[i], sections[j], bins)}
            out["baseline_pairs"].append(pair)
            print(pair, flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/e8_symmetry.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
