"""Langevin dynamics of the PIAR with the BAOAB splitting (Leimkuhler & Matthews).

    thetaddot + gamma thetadot + lam sin cos - beta sin = sqrt(2 gamma T) xi(t),
    <xi(t) xi(t')> = delta(t - t').

The stationary density is proportional to exp(-(omega^2/2 + u(theta))/T), with u the
potential per unit inertia (energy unit M g L in dimensionless form). BAOAB is the
stochastic analogue of the kick-drift-kick splitting used for velocity Verlet.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

TWO_PI = 2.0 * math.pi


@njit(cache=True)
def _force(th, lam, beta):
    s = math.sin(th)
    return beta * s - lam * s * math.cos(th)


@njit(cache=True)
def baoab_series(th0, om0, dt, nsteps, every, lam, beta, gamma, T, seed):
    """Return theta (unwrapped) and omega sampled every `every` steps."""
    np.random.seed(seed)
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(max(T * (1.0 - c1 * c1), 0.0))
    ns = nsteps // every
    ths = np.empty(ns)
    oms = np.empty(ns)
    th = th0
    om = om0
    f = _force(th, lam, beta)
    k = 0
    for n in range(nsteps):
        om += 0.5 * dt * f
        th += 0.5 * dt * om
        om = c1 * om + c2 * np.random.standard_normal()
        th += 0.5 * dt * om
        f = _force(th, lam, beta)
        om += 0.5 * dt * f
        if (n + 1) % every == 0:
            ths[k] = th
            oms[k] = om
            k += 1
    return ths, oms


@njit(cache=True)
def baoab_hist2d(th0, om0, dt, nsteps, lam, beta, gamma, T, seed, nbx, nby, om_max):
    """Accumulate the (theta mod 2pi, omega) occupation histogram along one trajectory."""
    np.random.seed(seed)
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(max(T * (1.0 - c1 * c1), 0.0))
    H = np.zeros((nbx, nby))
    th = th0
    om = om0
    f = _force(th, lam, beta)
    for n in range(nsteps):
        om += 0.5 * dt * f
        th += 0.5 * dt * om
        om = c1 * om + c2 * np.random.standard_normal()
        th += 0.5 * dt * om
        f = _force(th, lam, beta)
        om += 0.5 * dt * f
        tw = th - TWO_PI * math.floor((th + math.pi) / TWO_PI)
        ix = int((tw + math.pi) / TWO_PI * nbx)
        iy = int((om + om_max) / (2 * om_max) * nby)
        if 0 <= ix < nbx and 0 <= iy < nby:
            H[ix, iy] += 1.0
    return H


@njit(cache=True)
def escape_count(th0, dt, nsteps, lam, beta, gamma, T, seed, theta_esc):
    """Number of escapes from the upright well |theta| < theta_esc (re-injected at theta0)."""
    np.random.seed(seed)
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(max(T * (1.0 - c1 * c1), 0.0))
    th = th0
    om = 0.0
    f = _force(th, lam, beta)
    n_esc = 0
    for n in range(nsteps):
        om += 0.5 * dt * f
        th += 0.5 * dt * om
        om = c1 * om + c2 * np.random.standard_normal()
        th += 0.5 * dt * om
        f = _force(th, lam, beta)
        om += 0.5 * dt * f
        if abs(th) > theta_esc:
            n_esc += 1
            th = th0
            om = 0.0
            f = _force(th, lam, beta)
    return n_esc
