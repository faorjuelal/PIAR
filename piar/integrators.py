"""Time integrators for the PIAR, compiled with numba.

* ``dop853``   adaptive explicit Runge-Kutta 8(5,3) (Dormand-Prince; Hairer's DOP853),
               same tableau and error norm as SciPy's implementation.
* ``verlet``   velocity Verlet (kick-drift-kick), symplectic, time-reversible,
               second order; optional Kahan compensated summation (Brouwer's law).
* ``rk4``      classical fixed-step RK4 (reference non-symplectic fixed-step scheme).

All work in (theta, omega) with the acceleration per unit inertia (``accel_nb``).
The angle is wrapped to (-pi, pi] at output only; winding is tracked separately.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.integrate._ivp import dop853_coefficients as _d

from .physics import accel_nb

_A = np.ascontiguousarray(_d.A[:12, :12])
_B = np.ascontiguousarray(_d.B)
_C = np.ascontiguousarray(_d.C[:12])
_E3 = np.ascontiguousarray(_d.E3)
_E5 = np.ascontiguousarray(_d.E5)
TWO_PI = 2.0 * math.pi


@njit(cache=True)
def _wrap(x):
    return x - TWO_PI * math.floor((x + math.pi) / TWO_PI)


@njit(cache=True)
def _dop853_core(th0, om0, t0, t1, rtol, atol, h0, lam, beta, gamma, A, Omega, ftype,
                 save_dt, max_saves, A_, B_, C_, E3_, E5_):
    K = np.zeros((13, 2))
    ts = np.empty(max_saves)
    ys = np.empty((max_saves, 2))
    wind = np.empty(max_saves)
    t = t0
    th = th0
    om = om0
    nwind = 0.0
    f0 = om
    f1 = accel_nb(t, th, om, lam, beta, gamma, A, Omega, ftype)
    nfev = 1
    h = h0
    next_save = t0 + save_dt
    nsave = 0
    naccept = 0
    nreject = 0
    rejected = False
    while t < t1:
        # land exactly on the next output time (and on t1) without disturbing the controller
        h_try = h
        truncated = False
        lim = min(next_save, t1) - t
        if h_try >= lim:
            h_try = lim
            truncated = True
        K[0, 0] = f0
        K[0, 1] = f1
        for s in range(1, 12):
            dy0 = 0.0
            dy1 = 0.0
            for j in range(s):
                dy0 += A_[s, j] * K[j, 0]
                dy1 += A_[s, j] * K[j, 1]
            yth = th + h_try * dy0
            yom = om + h_try * dy1
            K[s, 0] = yom
            K[s, 1] = accel_nb(t + C_[s] * h_try, yth, yom, lam, beta, gamma, A, Omega, ftype)
        nfev += 11
        b0 = 0.0
        b1 = 0.0
        for j in range(12):
            b0 += B_[j] * K[j, 0]
            b1 += B_[j] * K[j, 1]
        thn = th + h_try * b0
        omn = om + h_try * b1
        K[12, 0] = omn
        K[12, 1] = accel_nb(t + h_try, thn, omn, lam, beta, gamma, A, Omega, ftype)
        nfev += 1
        sc0 = atol + max(abs(th), abs(thn)) * rtol
        sc1 = atol + max(abs(om), abs(omn)) * rtol
        e50 = 0.0
        e51 = 0.0
        e30 = 0.0
        e31 = 0.0
        for j in range(13):
            e50 += E5_[j] * K[j, 0]
            e51 += E5_[j] * K[j, 1]
            e30 += E3_[j] * K[j, 0]
            e31 += E3_[j] * K[j, 1]
        e50 /= sc0
        e51 /= sc1
        e30 /= sc0
        e31 /= sc1
        n5 = e50 * e50 + e51 * e51
        n3 = e30 * e30 + e31 * e31
        if n5 == 0.0 and n3 == 0.0:
            err = 0.0
        else:
            err = abs(h_try) * n5 / math.sqrt((n5 + 0.01 * n3) * 2.0)
        if err < 1.0:
            t = t + h_try
            th = thn
            om = omn
            if th > math.pi or th <= -math.pi:
                w = math.floor((th + math.pi) / TWO_PI)
                th -= TWO_PI * w
                nwind += w
            f0 = K[12, 0]
            f1 = K[12, 1]
            naccept += 1
            if err == 0.0:
                fac = 10.0
            else:
                fac = min(10.0, 0.9 * err ** (-1.0 / 8.0))
            if rejected:
                fac = min(1.0, fac)
            rejected = False
            if not truncated:
                h = h_try * fac
            if t >= next_save - 1e-12 * max(1.0, abs(t)) and nsave < max_saves:
                ts[nsave] = t
                ys[nsave, 0] = th
                ys[nsave, 1] = om
                wind[nsave] = nwind
                nsave += 1
                next_save += save_dt
        else:
            h = h_try * max(0.2, 0.9 * err ** (-1.0 / 8.0))
            rejected = True
            nreject += 1
    return ts[:nsave], ys[:nsave], wind[:nsave], nfev, naccept, nreject, th, om, nwind


def dop853(p, th0, om0, t1, rtol=1e-10, atol=None, save_dt=None, h0=1e-2, t0=0.0,
           max_saves=2_000_000):
    """Adaptive DOP853. Returns dict with t, theta (wrapped), omega, winding, nfev."""
    if atol is None:
        atol = rtol
    if save_dt is None:
        save_dt = (t1 - t0) / 2000.0
    lam, beta, gamma, A, Omega, ftype = p.kernel_args()
    ts, ys, wind, nfev, nacc, nrej, th, om, nw = _dop853_core(
        float(th0), float(om0), float(t0), float(t1), float(rtol), float(atol), float(h0),
        lam, beta, gamma, A, Omega, ftype, float(save_dt), int(max_saves),
        _A, _B, _C, _E3, _E5)
    return {"t": ts, "theta": ys[:, 0], "omega": ys[:, 1], "winding": wind,
            "nfev": nfev, "naccept": nacc, "nreject": nrej,
            "final": (th, om, nw)}


@njit(cache=True)
def _kahan(s, c, x):
    y = x - c
    t = s + y
    c = (t - s) - y
    return t, c


@njit(cache=True)
def _verlet_core(th0, om0, dt, nsteps, save_every, lam, beta, compensated):
    ns = nsteps // save_every
    ts = np.empty(ns)
    ths = np.empty(ns)
    oms = np.empty(ns)
    wind = np.empty(ns)
    th = th0
    om = om0
    cth = 0.0
    com = 0.0
    nwind = 0.0
    a = accel_nb(0.0, th, om, lam, beta, 0.0, 0.0, 0.0, 0)
    k = 0
    for n in range(nsteps):
        if compensated:
            om, com = _kahan(om, com, 0.5 * dt * a)
            th, cth = _kahan(th, cth, dt * om)
        else:
            om += 0.5 * dt * a
            th += dt * om
        a = accel_nb(0.0, th, om, lam, beta, 0.0, 0.0, 0.0, 0)
        if compensated:
            om, com = _kahan(om, com, 0.5 * dt * a)
        else:
            om += 0.5 * dt * a
        if (n + 1) % save_every == 0:
            if th > math.pi or th <= -math.pi:
                w = math.floor((th + math.pi) / TWO_PI)
                th -= TWO_PI * w
                nwind += w
            ts[k] = (n + 1) * dt
            ths[k] = th
            oms[k] = om
            wind[k] = nwind
            k += 1
    return ts, ths, oms, wind


def verlet(p, th0, om0, dt, nsteps, save_every=1, compensated=True):
    """Velocity Verlet (kick-drift-kick) for the conservative PIAR."""
    if p.gamma != 0.0 or p.ftype != 0:
        raise ValueError("Verlet is used here for the conservative system only.")
    ts, ths, oms, wind = _verlet_core(float(th0), float(om0), float(dt), int(nsteps),
                                      int(save_every), p.lam, p.beta, bool(compensated))
    return {"t": ts, "theta": ths, "omega": oms, "winding": wind, "nfev": int(nsteps)}


@njit(cache=True)
def _rk4_core(th0, om0, t0, dt, nsteps, save_every, lam, beta, gamma, A, Omega, ftype):
    ns = nsteps // save_every
    ts = np.empty(ns)
    ths = np.empty(ns)
    oms = np.empty(ns)
    t = t0
    th = th0
    om = om0
    k = 0
    for n in range(nsteps):
        k1t = om
        k1o = accel_nb(t, th, om, lam, beta, gamma, A, Omega, ftype)
        k2t = om + 0.5 * dt * k1o
        k2o = accel_nb(t + 0.5 * dt, th + 0.5 * dt * k1t, k2t, lam, beta, gamma, A, Omega, ftype)
        k3t = om + 0.5 * dt * k2o
        k3o = accel_nb(t + 0.5 * dt, th + 0.5 * dt * k2t, k3t, lam, beta, gamma, A, Omega, ftype)
        k4t = om + dt * k3o
        k4o = accel_nb(t + dt, th + dt * k3t, k4t, lam, beta, gamma, A, Omega, ftype)
        th += dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om += dt / 6.0 * (k1o + 2 * k2o + 2 * k3o + k4o)
        t = t0 + (n + 1) * dt
        if (n + 1) % save_every == 0:
            ts[k] = t
            ths[k] = th
            oms[k] = om
            k += 1
    return ts, ths, oms


def rk4(p, th0, om0, dt, nsteps, save_every=1, t0=0.0):
    lam, beta, gamma, A, Omega, ftype = p.kernel_args()
    ts, ths, oms = _rk4_core(float(th0), float(om0), float(t0), float(dt), int(nsteps),
                             int(save_every), lam, beta, gamma, A, Omega, ftype)
    return {"t": ts, "theta": ths, "omega": oms, "nfev": 4 * int(nsteps)}


def energy_per_inertia(p, th, om):
    s = np.sin(th)
    return 0.5 * om ** 2 + p.beta * np.cos(th) + 0.5 * p.lam * s ** 2


def modified_energy_verlet(p, th, om, dt):
    """Truncated modified Hamiltonian of velocity Verlet (per unit inertia):
    H~ = H + dt^2 * ( -U'^2/24 + om^2 U''/12 ),  from the symmetric BCH formula."""
    s, c = np.sin(th), np.cos(th)
    Up = -p.beta * s + p.lam * s * c
    Upp = -p.beta * c + p.lam * np.cos(2 * th)
    return energy_per_inertia(p, th, om) + dt ** 2 * (-(Up ** 2) / 24.0 + om ** 2 * Upp / 12.0)


def exact_period(p, E):
    """Exact period of a libration about theta = pi (E < E_sep) or of a rotation (E > E_sep).
    E is the energy per unit inertia."""
    u = lambda th: p.beta * np.cos(th) + 0.5 * p.lam * np.sin(th) ** 2
    Esep = p.E_sep / p.inertia
    if E > Esep:
        val, _ = quad(lambda th: 1.0 / math.sqrt(2.0 * (E - u(th))), 0.0, TWO_PI,
                      epsabs=1e-14, epsrel=1e-13, limit=400)
        return val
    # libration about pi: phi = theta - pi, u(phi) even; turning point phi_m
    from scipy.optimize import brentq
    g = lambda ph: u(math.pi + ph) - E
    phm = brentq(g, 1e-12, math.pi - 1e-12, xtol=1e-15, rtol=1e-15)
    f = lambda psi: phm * math.cos(psi) / math.sqrt(max(2.0 * (E - u(math.pi + phm * math.sin(psi))), 1e-300))
    val, _ = quad(f, 0.0, 0.5 * math.pi, epsabs=1e-14, epsrel=1e-13, limit=400)
    return 4.0 * val
