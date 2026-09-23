"""Delay-coordinate embedding utilities (Takens): average mutual information (AMI)
for the delay tau and false nearest neighbours (Kennel et al., 1992) for the dimension."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def ami(x, max_lag=60, bins=48):
    """Average mutual information I(x_t; x_{t+lag}) in nats for lag = 0..max_lag."""
    x = np.asarray(x, dtype=float)
    edges = np.histogram_bin_edges(x, bins=bins)
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        a = x[: len(x) - lag]
        b = x[lag:]
        pab, _, _ = np.histogram2d(a, b, bins=[edges, edges])
        pab /= pab.sum()
        pa = pab.sum(axis=1, keepdims=True)
        pb = pab.sum(axis=0, keepdims=True)
        m = pab > 0
        out[lag] = np.sum(pab[m] * np.log(pab[m] / (pa @ pb)[m]))
    return out


def first_minimum(y):
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            return i
    return int(np.argmin(y))


def delay_embed(x, E, tau, symmetric=False):
    """Rows are delay vectors. symmetric=True centres the window on t (offline use)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if symmetric:
        m = (E - 1) // 2
        lags = np.arange(-m, m + 1) * tau
    else:
        lags = -np.arange(E) * tau
    lo = -lags.min() if lags.min() < 0 else 0
    hi = N - (lags.max() if lags.max() > 0 else 0)
    idx = np.arange(lo, hi)
    return np.stack([x[idx + l] for l in lags], axis=1), idx


def fnn_fraction(x, tau, E_max=8, rtol=15.0, atol=2.0, n_max=8000, seed=0):
    """Fraction of false nearest neighbours for E = 1..E_max (Kennel criterion)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    Ra = np.std(x)
    out = []
    for E in range(1, E_max + 1):
        Y, idx = delay_embed(x, E + 1, tau)
        Y = Y[:, ::-1]                       # oldest first
        if len(Y) > n_max:
            sel = rng.choice(len(Y), n_max, replace=False)
            Y = Y[sel]
        YE = Y[:, 1:]                        # E-dimensional vectors
        extra = Y[:, 0]                      # (E+1)-th coordinate
        tree = cKDTree(YE)
        d, j = tree.query(YE, k=2)
        d = d[:, 1]
        j = j[:, 1]
        valid = d > 0
        crit1 = np.abs(extra[valid] - extra[j[valid]]) / d[valid] > rtol
        d_new = np.sqrt(d[valid] ** 2 + (extra[valid] - extra[j[valid]]) ** 2)
        crit2 = d_new / Ra > atol
        out.append(float(np.mean(crit1 | crit2)))
    return np.array(out)
