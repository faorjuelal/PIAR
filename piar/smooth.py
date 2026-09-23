"""Smooth rendering of sampled trajectories.

A fixed-step integrator returns the state only at the grid points. Drawing straight
segments between them produces the polygonal, "cut" look of coarse animations. Because
both the angle and its time derivative are known at every grid point (theta' = omega,
omega' = acceleration from the equation of motion), each interval can be filled with the
cubic Hermite interpolant, which is C^1-continuous and third-order accurate, at no extra
cost in force evaluations beyond one acceleration per stored point.
"""
from __future__ import annotations

import numpy as np


def hermite_upsample(t, th, om, acc, factor=8):
    """Return (t, theta, omega) on a grid ``factor`` times finer.

    ``acc(t, theta, omega)`` is the acceleration of the equation of motion (vectorised).
    """
    t, th, om = map(np.asarray, (t, th, om))
    a = acc(t, th, om)
    h = np.diff(t)
    s = np.linspace(0.0, 1.0, factor, endpoint=False)[None, :]          # (1, f)
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    H = h[:, None]

    def interp(y, dy):
        return (h00 * y[:-1, None] + h10 * H * dy[:-1, None] + h01 * y[1:, None] + h11 * H * dy[1:, None]).ravel()

    tt = (t[:-1, None] + H * s).ravel()
    th_f = interp(th, om)
    om_f = interp(om, a)
    return np.append(tt, t[-1]), np.append(th_f, th[-1]), np.append(om_f, om[-1])


def break_on_wrap(theta, *ys):
    """Wrap theta to (-pi, pi] and insert NaN where the curve crosses the cut at +-pi,
    so that the orbit is drawn on the cylinder without spurious horizontal segments."""
    thw = (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi
    j = np.where(np.abs(np.diff(thw)) > np.pi)[0] + 1
    out = [np.insert(thw, j, np.nan)] + [np.insert(np.asarray(y, dtype=float), j, np.nan) for y in ys]
    return out
