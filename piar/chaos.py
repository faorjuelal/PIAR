"""Nonlinear-dynamics toolkit for the damped-driven PIAR (numba).

Extended phase space (theta, omega, phi = Omega t mod 2pi) = S^1 x R x S^1.
Divergence of the vector field is -gamma, so the stroboscopic (Poincare) map P has
det DP = exp(-2 pi gamma / Omega), and the Lyapunov spectrum of the 2D
non-autonomous flow obeys lambda_1 + lambda_3 = -gamma (lambda_2 = 0 is the time
direction of the extended flow).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.integrate import quad
from scipy.spatial import cKDTree

from .physics import accel_nb, jac21_nb

TWO_PI = 2.0 * math.pi


@njit(cache=True)
def _wrap(x):
    return x - TWO_PI * math.floor((x + math.pi) / TWO_PI)


@njit(cache=True)
def _rk4_step(t, th, om, dt, lam, beta, gamma, A, Omega, ftype):
    k1t = om
    k1o = accel_nb(t, th, om, lam, beta, gamma, A, Omega, ftype)
    k2t = om + 0.5 * dt * k1o
    k2o = accel_nb(t + 0.5 * dt, th + 0.5 * dt * k1t, k2t, lam, beta, gamma, A, Omega, ftype)
    k3t = om + 0.5 * dt * k2o
    k3o = accel_nb(t + 0.5 * dt, th + 0.5 * dt * k2t, k3t, lam, beta, gamma, A, Omega, ftype)
    k4t = om + dt * k3o
    k4o = accel_nb(t + dt, th + dt * k3t, k4t, lam, beta, gamma, A, Omega, ftype)
    return (th + dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t),
            om + dt / 6.0 * (k1o + 2 * k2o + 2 * k3o + k4o))


@njit(cache=True)
def _rk4_var_step(t, th, om, a, b, c, d, dt, lam, beta, gamma, A, Omega, ftype):
    """RK4 for state + 2x2 tangent matrix [[a, c], [b, d]] (columns = tangent vectors)."""
    # stage 1
    k1t = om
    k1o = accel_nb(t, th, om, lam, beta, gamma, A, Omega, ftype)
    j = jac21_nb(t, th, lam, beta, A, Omega, ftype)
    k1a = b
    k1b = j * a - gamma * b
    k1c = d
    k1d = j * c - gamma * d
    # stage 2
    t2 = t + 0.5 * dt
    th2 = th + 0.5 * dt * k1t
    om2 = om + 0.5 * dt * k1o
    a2 = a + 0.5 * dt * k1a
    b2 = b + 0.5 * dt * k1b
    c2 = c + 0.5 * dt * k1c
    d2 = d + 0.5 * dt * k1d
    k2t = om2
    k2o = accel_nb(t2, th2, om2, lam, beta, gamma, A, Omega, ftype)
    j = jac21_nb(t2, th2, lam, beta, A, Omega, ftype)
    k2a = b2
    k2b = j * a2 - gamma * b2
    k2c = d2
    k2d = j * c2 - gamma * d2
    # stage 3
    th3 = th + 0.5 * dt * k2t
    om3 = om + 0.5 * dt * k2o
    a3 = a + 0.5 * dt * k2a
    b3 = b + 0.5 * dt * k2b
    c3 = c + 0.5 * dt * k2c
    d3 = d + 0.5 * dt * k2d
    k3t = om3
    k3o = accel_nb(t2, th3, om3, lam, beta, gamma, A, Omega, ftype)
    j = jac21_nb(t2, th3, lam, beta, A, Omega, ftype)
    k3a = b3
    k3b = j * a3 - gamma * b3
    k3c = d3
    k3d = j * c3 - gamma * d3
    # stage 4
    t4 = t + dt
    th4 = th + dt * k3t
    om4 = om + dt * k3o
    a4 = a + dt * k3a
    b4 = b + dt * k3b
    c4 = c + dt * k3c
    d4 = d + dt * k3d
    k4t = om4
    k4o = accel_nb(t4, th4, om4, lam, beta, gamma, A, Omega, ftype)
    j = jac21_nb(t4, th4, lam, beta, A, Omega, ftype)
    k4a = b4
    k4b = j * a4 - gamma * b4
    k4c = d4
    k4d = j * c4 - gamma * d4
    s = dt / 6.0
    return (th + s * (k1t + 2 * k2t + 2 * k3t + k4t), om + s * (k1o + 2 * k2o + 2 * k3o + k4o),
            a + s * (k1a + 2 * k2a + 2 * k3a + k4a), b + s * (k1b + 2 * k2b + 2 * k3b + k4b),
            c + s * (k1c + 2 * k2c + 2 * k3c + k4c), d + s * (k1d + 2 * k2d + 2 * k3d + k4d))


@njit(cache=True)
def strobe_nb(th0, om0, n_transient, n_keep, spp, lam, beta, gamma, A, Omega, ftype):
    """Stroboscopic samples at t = n T_d (after n_transient periods)."""
    Td = TWO_PI / Omega
    dt = Td / spp
    th = th0
    om = om0
    out = np.empty((n_keep, 2))
    for n in range(n_transient + n_keep):
        t0 = n * Td
        for i in range(spp):
            th, om = _rk4_step(t0 + i * dt, th, om, dt, lam, beta, gamma, A, Omega, ftype)
        th = _wrap(th)
        if n >= n_transient:
            out[n - n_transient, 0] = th
            out[n - n_transient, 1] = om
    return out, th, om


def strobe(p, th0=0.0, om0=0.0, n_transient=300, n_keep=500, spp=128):
    lam, beta, gamma, A, Omega, ftype = p.kernel_args()
    out, th, om = strobe_nb(float(th0), float(om0), int(n_transient), int(n_keep), int(spp),
                            lam, beta, gamma, A, Omega, ftype)
    return out


@njit(cache=True)
def trajectory_nb(th0, om0, n_periods, spp, save_per_period, lam, beta, gamma, A, Omega, ftype):
    Td = TWO_PI / Omega
    dt = Td / spp
    every = spp // save_per_period
    out = np.empty((n_periods * save_per_period, 3))
    th = th0
    om = om0
    k = 0
    for n in range(n_periods):
        t0 = n * Td
        for i in range(spp):
            th, om = _rk4_step(t0 + i * dt, th, om, dt, lam, beta, gamma, A, Omega, ftype)
            if (i + 1) % every == 0:
                out[k, 0] = t0 + (i + 1) * dt
                out[k, 1] = th
                out[k, 2] = om
                k += 1
    return out


def trajectory(p, th0, om0, n_periods, spp=128, save_per_period=32):
    """Dense trajectory (t, theta_unwrapped, omega) of the driven system."""
    lam, beta, gamma, A, Omega, ftype = p.kernel_args()
    return trajectory_nb(float(th0), float(om0), int(n_periods), int(spp), int(save_per_period),
                         lam, beta, gamma, A, Omega, ftype)


@njit(cache=True)
def sweep_nb(A_values, th0, om0, n_transient, n_keep, spp, lam, beta, gamma, Omega, ftype):
    """Bifurcation sweep with continuation (the final state of one A seeds the next)."""
    nA = A_values.shape[0]
    th_out = np.empty((nA, n_keep))
    om_out = np.empty((nA, n_keep))
    th = th0
    om = om0
    for ia in range(nA):
        pts, th, om = strobe_nb(th, om, n_transient, n_keep, spp, lam, beta, gamma,
                                A_values[ia], Omega, ftype)
        th_out[ia] = pts[:, 0]
        om_out[ia] = pts[:, 1]
    return th_out, om_out


def bifurcation_sweep(p, A_values, th0=0.0, om0=0.0, n_transient=200, n_keep=120, spp=128):
    lam, beta, gamma, _, Omega, ftype = p.kernel_args()
    if ftype == 0:
        ftype = 1   # sweeping the forcing amplitude implies a forced system (torque default)
    return sweep_nb(np.asarray(A_values, dtype=np.float64), float(th0), float(om0),
                    int(n_transient), int(n_keep), int(spp), lam, beta, gamma, Omega, ftype)


@njit(cache=True)
def lyapunov_nb(th0, om0, n_transient, n_periods, spp, lam, beta, gamma, A, Omega, ftype):
    """Two Lyapunov exponents of the 2D non-autonomous flow (Benettin + Gram-Schmidt QR
    once per forcing period). Returns (l1, l2, running l1 per period)."""
    Td = TWO_PI / Omega
    dt = Td / spp
    th = th0
    om = om0
    for n in range(n_transient):
        t0 = n * Td
        for i in range(spp):
            th, om = _rk4_step(t0 + i * dt, th, om, dt, lam, beta, gamma, A, Omega, ftype)
    a, b, c, d = 1.0, 0.0, 0.0, 1.0
    s1 = 0.0
    s2 = 0.0
    run = np.empty(n_periods)
    for n in range(n_periods):
        t0 = (n_transient + n) * Td
        for i in range(spp):
            th, om, a, b, c, d = _rk4_var_step(t0 + i * dt, th, om, a, b, c, d, dt,
                                               lam, beta, gamma, A, Omega, ftype)
        th = _wrap(th)
        # Gram-Schmidt on columns v1 = (a, b), v2 = (c, d)
        r11 = math.sqrt(a * a + b * b)
        q1a = a / r11
        q1b = b / r11
        r12 = q1a * c + q1b * d
        c2 = c - r12 * q1a
        d2 = d - r12 * q1b
        r22 = math.sqrt(c2 * c2 + d2 * d2)
        s1 += math.log(r11)
        s2 += math.log(r22)
        a, b = q1a, q1b
        c, d = c2 / r22, d2 / r22
        run[n] = s1 / ((n + 1) * Td)
    T = n_periods * Td
    return s1 / T, s2 / T, run


def lyapunov(p, th0=0.0, om0=0.0, n_transient=300, n_periods=3000, spp=128):
    lam, beta, gamma, A, Omega, ftype = p.kernel_args()
    return lyapunov_nb(float(th0), float(om0), int(n_transient), int(n_periods), int(spp),
                       lam, beta, gamma, A, Omega, ftype)


@njit(cache=True, parallel=True)
def lyapunov_grid_nb(A_values, Omega_values, th0, om0, n_transient, n_periods, spp,
                     lam, beta, gamma, ftype):
    nA = A_values.shape[0]
    nO = Omega_values.shape[0]
    out = np.empty((nO, nA))
    for k in prange(nO * nA):
        io = k // nA
        ia = k % nA
        l1, l2, run = lyapunov_nb(th0, om0, n_transient, n_periods, spp, lam, beta, gamma,
                                  A_values[ia], Omega_values[io], ftype)
        out[io, ia] = l1
    return out


@njit(cache=True, parallel=True)
def lyapunov_sweep_nb(A_values, th0, om0, n_transient, n_periods, spp, lam, beta, gamma,
                      Omega, ftype):
    nA = A_values.shape[0]
    l1 = np.empty(nA)
    l2 = np.empty(nA)
    for ia in prange(nA):
        a, b, run = lyapunov_nb(th0, om0, n_transient, n_periods, spp, lam, beta, gamma,
                                A_values[ia], Omega, ftype)
        l1[ia] = a
        l2[ia] = b
    return l1, l2


def attractor_period(pts, tol=1e-6, max_period=64):
    """Smallest p such that the stroboscopic sequence is p-periodic (angles mod 2pi)."""
    th = pts[:, 0]
    om = pts[:, 1]
    n = len(th)
    for per in range(1, max_period + 1):
        if n <= 2 * per:
            break
        dth = np.abs(np.angle(np.exp(1j * (th[per:] - th[:-per]))))
        dom = np.abs(om[per:] - om[:-per])
        if np.max(dth) < tol and np.max(dom) < tol:
            return per
    return 0  # aperiodic (chaotic or quasi-periodic) within max_period


@njit(cache=True)
def monodromy_nb(th0, om0, n_per, spp, lam, beta, gamma, A, Omega, ftype):
    Td = TWO_PI / Omega
    dt = Td / spp
    th, om = th0, om0
    a, b, c, d = 1.0, 0.0, 0.0, 1.0
    for n in range(n_per):
        for i in range(spp):
            th, om, a, b, c, d = _rk4_var_step(n * Td + i * dt, th, om, a, b, c, d, dt,
                                               lam, beta, gamma, A, Omega, ftype)
    return a, b, c, d, th, om


def floquet_multipliers(p, th0, om0, period, spp=256):
    """Monodromy matrix of P^period at a point of a periodic orbit (t = 0 mod T_d)."""
    lam, beta, gamma, A, Omega, ftype = p.kernel_args()
    a, b, c, d, th, om = monodromy_nb(float(th0), float(om0), int(period), int(spp),
                                      lam, beta, gamma, A, Omega, ftype)
    Mx = np.array([[a, c], [b, d]])
    return np.linalg.eigvals(Mx), Mx


# ------------------------------------------------------------------------------------
# Melnikov threshold for the homoclinic loop of the upright saddle (mu < 0), torque forcing
# ------------------------------------------------------------------------------------
def melnikov_ratio(kappa, Omega, n=400_000):
    """(A/gamma)_c = I0 / |I1(Omega)| in dimensionless units (beta = 1, lam = kappa < 1).

    M(t0) = A I1 cos(Omega t0) - gamma I0 with I0 = int thetadot_h^2 dt and
    I1 = int thetadot_h(t) cos(Omega t) dt along the homoclinic orbit.
    Written with u = 2pi - theta to avoid cancellation near the saddle.
    """
    if kappa >= 1.0:
        raise ValueError("homoclinic loop through the upright saddle requires mu < 0")
    thd = lambda u: np.sqrt(np.maximum(0.0, 2 * (2 * np.sin(u / 2) ** 2 - 0.5 * kappa * np.sin(u) ** 2)))
    I0 = 2 * quad(lambda u: float(thd(u)), 0, math.pi, limit=500)[0]
    s = np.linspace(math.log(math.pi), math.log(1e-15), n)
    u = np.exp(s)
    w = u / thd(u)
    tau = np.concatenate([[0.0], np.cumsum(-0.5 * (w[1:] + w[:-1]) * np.diff(s))])
    I1 = 2 * np.trapezoid((np.cos(Omega * tau) * u)[::-1], s[::-1])
    return I0 / abs(I1), I0, I1


# ------------------------------------------------------------------------------------
# Correlation dimension (Grassberger-Procaccia) and PSD
# ------------------------------------------------------------------------------------
def correlation_sum(points, radii, theiler=0, max_pairs_points=20000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(points)
    if len(X) > max_pairs_points:
        X = X[rng.choice(len(X), max_pairs_points, replace=False)]
    tree = cKDTree(X)
    N = len(X)
    C = np.array([(tree.count_neighbors(tree, r) - N) / (N * (N - 1)) for r in radii])
    return C


def correlation_dimension(points, rmin, rmax, nr=16, seed=0):
    radii = np.logspace(np.log10(rmin), np.log10(rmax), nr)
    C = correlation_sum(points, radii, seed=seed)
    m = C > 0
    slope, intercept = np.polyfit(np.log(radii[m]), np.log(C[m]), 1)
    return slope, radii, C


# ------------------------------------------------------------------------------------
# Periodic orbits: Newton shooting on P^k, Floquet multipliers, period-doubling points
# ------------------------------------------------------------------------------------
def newton_periodic(p, x0, k, spp=256, tol=1e-12, maxit=30):
    """Fixed point of P^k (theta taken mod 2pi) by Newton shooting with the monodromy matrix."""
    lam, beta, gamma, A, Omega, ftype = p.kernel_args()
    x = np.array(x0, dtype=float)
    for it in range(maxit):
        a, b, c, d, th, om = monodromy_nb(x[0], x[1], int(k), int(spp), lam, beta, gamma, A, Omega, ftype)
        F = np.array([math.remainder(th - x[0], TWO_PI), om - x[1]])
        J = np.array([[a - 1.0, c], [b, d - 1.0]])
        dx = np.linalg.solve(J, -F)
        x = x + dx
        if np.max(np.abs(dx)) < tol:
            break
    a, b, c, d, th, om = monodromy_nb(x[0], x[1], int(k), int(spp), lam, beta, gamma, A, Omega, ftype)
    M = np.array([[a, c], [b, d]])
    res = max(abs(math.remainder(th - x[0], TWO_PI)), abs(om - x[1]))
    return x, np.linalg.eigvals(M), res


def min_real_multiplier(p, x, k, spp=256):
    x, mult, res = newton_periodic(p, x, k, spp=spp)
    # period doubling: a real multiplier crossing -1
    mr = np.real(mult[np.abs(np.imag(mult)) < 1e-9]) if np.any(np.abs(np.imag(mult)) < 1e-9) else np.array([0.0])
    return x, float(np.min(mr)), mult, res


def locate_period_doubling(p, x, k, A_lo, A_hi, spp=256, tol=1e-11):
    """Bisection in A for the crossing of a Floquet multiplier of P^k through -1."""
    x_lo = x
    for _ in range(80):
        A_mid = 0.5 * (A_lo + A_hi)
        pm = p.with_(A=A_mid)
        xm, mmin, mult, res = min_real_multiplier(pm, x_lo, k, spp)
        if res > 1e-8:
            A_hi = A_mid
            continue
        if mmin > -1.0:
            A_lo, x_lo = A_mid, xm
        else:
            A_hi = A_mid
        if A_hi - A_lo < tol:
            break
    return 0.5 * (A_lo + A_hi), x_lo


def minimal_period(p, x, kmax, spp=256, tol=1e-7):
    lam, beta, gamma, A, Omega, ftype = p.kernel_args()
    th, om = float(x[0]), float(x[1])
    for j in range(1, kmax + 1):
        a, b, c, d, th, om = monodromy_nb(th, om, 1, int(spp), lam, beta, gamma, A, Omega, ftype)
        # monodromy_nb integrates from t = 0, valid because P is autonomous in n (t mod T_d)
        if abs(math.remainder(th - x[0], TWO_PI)) < tol and abs(om - x[1]) < tol:
            return j
    return 0


def feigenbaum_cascade(p0, A_start, levels=5, spacing0=9e-3, delta_guess=4.67, spp=256):
    """Follow a period-doubling cascade: returns the bifurcation amplitudes A_n."""
    pts = strobe(p0.with_(A=A_start), 0.0, 0.0, n_transient=4000, n_keep=4, spp=spp)
    x = pts[-1]
    k = 1
    A = A_start
    out = []
    for level in range(levels):
        spacing = spacing0 / delta_guess ** level
        dA = spacing / 40.0
        x, mmin, mult, res = min_real_multiplier(p0.with_(A=A), x, k, spp)
        A_prev, x_prev = A, x
        while mmin > -1.0:
            A_prev, x_prev = A, x
            A_try = A + dA
            x_try, m_try, mult, res = min_real_multiplier(p0.with_(A=A_try), x, k, spp)
            if res > 1e-8 or minimal_period(p0.with_(A=A_try), x_try, k, spp) != k:
                dA *= 0.5
                if dA < 1e-12:
                    raise RuntimeError("continuation failed")
                continue
            A, x, mmin = A_try, x_try, m_try
        Ac, xc = locate_period_doubling(p0, x_prev, k, A_prev, A, spp=spp)
        out.append(Ac)
        # jump onto the period-2k orbit, a fraction of the expected window beyond A_c
        A = Ac + 0.3 * spacing / delta_guess
        pts = strobe(p0.with_(A=A), xc[0] + 1e-3, xc[1], n_transient=6000, n_keep=2 * k, spp=spp)
        x = pts[-1]
        k *= 2
        x, mmin, mult, res = min_real_multiplier(p0.with_(A=A), x, k, spp)
        if minimal_period(p0.with_(A=A), x, k, spp) != k:
            break
    return np.array(out)
