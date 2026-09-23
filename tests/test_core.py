"""Unit tests for the analytical invariants and the integrators."""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from piar.physics import PIAR
from piar import integrators as I


def test_closed_form_invariants():
    for mu in (0.05, 0.3, 1.0):
        p = PIAR.dimensionless(mu)
        # frequency about theta = 0 from U''(0)
        assert math.isclose(p.d2U(0.0) / p.inertia, p.omega0_sq, rel_tol=1e-12)
        # saddles and barrier
        ts = p.theta_star
        assert abs(p.dU(ts)) < 1e-12 and p.d2U(ts) < 0
        assert math.isclose(p.U(ts) - p.U(0.0), p.barrier, rel_tol=1e-12)
        assert math.isclose(p.U(ts), p.E_sep, rel_tol=1e-12)
        # saddle rate
        assert math.isclose(math.sqrt(-p.d2U(ts) / p.inertia), p.saddle_rate, rel_tol=1e-12)
    for mu in (-0.8, -0.3):
        p = PIAR.dimensionless(mu)
        assert p.d2U(0.0) < 0 and p.d2U(math.pi) > 0
        assert math.isclose(p.d2U(math.pi) / p.inertia, p.omega_pi_sq, rel_tol=1e-12)
        assert math.isclose(p.U(0.0), p.E_sep, rel_tol=1e-12)


def test_normal_form_and_quartic_maximum():
    for mu in (0.02, -0.02):
        p = PIAR.dimensionless(mu)
        th = 1e-2
        exact = p.accel(0.0, th, 0.0)
        nf = -mu * th + p.normal_form_c3 * th ** 3
        assert abs(exact - nf) < 1e-9
    p0 = PIAR.dimensionless(0.0)
    th = 1e-2
    assert math.isclose(p0.U(th) - p0.U(0.0), -th ** 4 / 8, rel_tol=1e-3)


def test_dop853_matches_scipy():
    p = PIAR.dimensionless(-0.75)
    out = I.dop853(p, 2.0, 0.0, 50.0, rtol=1e-11, atol=1e-12, save_dt=50.0)
    th, om, nw = out["final"]
    ref = solve_ivp(p.rhs, (0, 50), [2.0, 0.0], method="DOP853", rtol=1e-13, atol=1e-14)
    th_ref = ref.y[0, -1]
    assert abs((th + 2 * math.pi * nw) - th_ref) < 1e-8
    assert abs(om - ref.y[1, -1]) < 1e-8


def test_verlet_second_order_and_modified_energy():
    p = PIAR.dimensionless(-0.75)
    th0, om0 = 2.0, 0.0
    T = 20.0
    ref = solve_ivp(p.rhs, (0, T), [th0, om0], method="DOP853", rtol=1e-13, atol=1e-14)
    errs = []
    for dt in (0.02, 0.01, 0.005):
        o = I.verlet(p, th0, om0, dt, int(round(T / dt)), save_every=int(round(T / dt)))
        errs.append(abs(o["theta"][-1] + 2 * math.pi * o["winding"][-1] - ref.y[0, -1]))
    order = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(order - 2.0) < 0.15)
    # modified energy fluctuates as O(dt^4) while the energy fluctuates as O(dt^2)
    fl_H, fl_Ht = [], []
    for dt in (0.04, 0.02):
        o = I.verlet(p, th0, om0, dt, int(200 / dt), save_every=1)
        H = I.energy_per_inertia(p, o["theta"], o["omega"])
        Ht = I.modified_energy_verlet(p, o["theta"], o["omega"], dt)
        fl_H.append(np.ptp(H))
        fl_Ht.append(np.ptp(Ht))
    assert 3.5 < fl_H[0] / fl_H[1] < 4.5          # O(dt^2)
    assert 12.0 < fl_Ht[0] / fl_Ht[1] < 20.0      # O(dt^4)


def test_exact_period():
    p = PIAR.dimensionless(-0.75)
    for E in (-0.5, 0.5, 1.6):
        T = I.exact_period(p, E)
        # initial condition on the energy level at theta = pi
        om0 = math.sqrt(2 * (E - (p.beta * math.cos(math.pi))))
        sol = solve_ivp(p.rhs, (0, 10 * T), [math.pi, om0], method="DOP853", rtol=1e-12, atol=1e-13,
                        dense_output=True)
        # after 10 periods the state returns (mod 2pi)
        th10, om10 = sol.sol(10 * T)
        assert abs(math.remainder(th10 - math.pi, 2 * math.pi)) < 1e-7
        assert abs(om10 - om0) < 1e-7


def test_hermite_rendering_is_smooth_and_accurate():
    from piar.smooth import hermite_upsample, break_on_wrap
    t = np.linspace(0.0, 20.0, 101)                     # coarse grid, h = 0.2
    tf, xf, vf = hermite_upsample(t, np.sin(t), np.cos(t), lambda a, b, c: -b, factor=8)
    assert len(tf) == 8 * 100 + 1
    assert np.max(np.abs(xf - np.sin(tf))) < 1e-4       # O(h^4) interpolation error
    assert np.max(np.abs(vf - np.cos(tf))) < 1e-4
    th = np.linspace(0.0, 6 * np.pi, 500)               # three revolutions
    thw, _ = break_on_wrap(th, th)
    finite = thw[np.isfinite(thw)]
    assert np.all(np.abs(finite) <= np.pi + 1e-12)
    assert np.isnan(thw).sum() == 3                     # one break per crossing of the cut


def test_upright_libration_period_case2():
    # case 2 of Sec. III: k=2, L=3, M=4, g=1, theta0=0, omega0=0.1 rad/s -> libration in the upright well
    p = PIAR(k=2.0, L=3.0, M=4.0, g=1.0)
    E = float(p.energy(0.0, 0.1))
    assert E < p.E_sep
    sol = solve_ivp(lambda t, y: [y[1], p.accel(t, y[0], y[1])], (0, 40), [0.0, 0.1], method="DOP853",
                    rtol=1e-12, atol=1e-13, dense_output=True)
    tt = np.linspace(0, 40, 400001)
    th = sol.sol(tt)[0]
    up = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
    T = np.mean(np.diff(tt[up]))
    assert abs(T - 16.0323) < 2e-3
