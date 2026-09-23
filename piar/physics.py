r"""Analytical mechanics of the spring-coupled inverted pendulum (PIAR).

Geometry: massless rod of length L pivoted at the origin, point mass M at the tip,
horizontal spring of stiffness k kept horizontal by a collar sliding on a vertical
guide. theta is measured from the upward vertical.

    Lagrangian   L = 1/2 M L^2 thetadot^2 - U(theta)
    Potential    U = M g L cos(theta) + 1/2 k L^2 sin^2(theta)
    EOM          thetaddot + lam sin(theta) cos(theta) - beta sin(theta) = f(t)
    lam = k/M,   beta = g/L,   kappa = k L/(M g) = 1 + mu

Dimensionless units (default in the paper): M = L = g = 1, so beta = 1,
lam = kappa = 1 + mu, time unit sqrt(L/g), energy unit M g L, p = thetadot.
Damped-driven extension:
    thetaddot + gamma thetadot + lam sin cos - beta sin = A cos(Omega t)          (torque)
    thetaddot + gamma thetadot + lam sin cos - beta sin = A cos(Omega t) cos(theta) (base)
The base form follows from a horizontally shaken frame, A = X Omega^2 / L.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numba import njit

FORCING = {"none": 0, "torque": 1, "base": 2}


@dataclass(frozen=True)
class PIAR:
    k: float = 1.25
    L: float = 1.0
    M: float = 1.0
    g: float = 1.0
    gamma: float = 0.0
    A: float = 0.0
    Omega: float = 0.0
    forcing: str = "none"

    # ---------- factories ----------
    @classmethod
    def dimensionless(cls, mu: float, gamma: float = 0.0, A: float = 0.0,
                      Omega: float = 0.0, forcing: str | None = None) -> "PIAR":
        if forcing is None:
            forcing = "torque" if A != 0.0 else "none"
        return cls(k=1.0 + mu, L=1.0, M=1.0, g=1.0, gamma=gamma, A=A,
                   Omega=Omega, forcing=forcing)

    def with_(self, **kw) -> "PIAR":
        return replace(self, **kw)

    # ---------- derived parameters ----------
    @property
    def lam(self) -> float:
        return self.k / self.M

    @property
    def beta(self) -> float:
        return self.g / self.L

    @property
    def kappa(self) -> float:
        return self.k * self.L / (self.M * self.g)

    @property
    def mu(self) -> float:
        return self.kappa - 1.0

    @property
    def inertia(self) -> float:
        return self.M * self.L ** 2

    @property
    def MgL(self) -> float:
        return self.M * self.g * self.L

    @property
    def ftype(self) -> int:
        return FORCING[self.forcing]

    # ---------- closed-form invariants ----------
    @property
    def omega0_sq(self) -> float:
        """Small-oscillation frequency squared about theta = 0 (beta * mu)."""
        return self.beta * self.mu

    @property
    def omega_pi_sq(self) -> float:
        """Small-oscillation frequency squared about theta = pi (beta*(2+mu))."""
        return self.beta * (2.0 + self.mu)

    @property
    def theta_star(self) -> float:
        """Tilted saddles, cos(theta*) = 1/(1+mu); defined for mu > 0."""
        return math.acos(1.0 / self.kappa) if self.kappa > 1.0 else float("nan")

    @property
    def barrier(self) -> float:
        """U(theta*) - U(0) = MgL mu^2 / (2(1+mu)) for mu > 0."""
        return self.MgL * self.mu ** 2 / (2.0 * self.kappa) if self.mu > 0 else float("nan")

    @property
    def E_sep(self) -> float:
        if self.mu < 0:
            return self.MgL
        if self.mu > 0:
            return self.MgL * (self.kappa ** 2 + 1.0) / (2.0 * self.kappa)
        return self.MgL

    @property
    def saddle_rate(self) -> float:
        """Unstable eigenvalue of the separatrix saddle."""
        if self.mu < 0:
            return math.sqrt(-self.beta * self.mu)
        if self.mu > 0:
            return math.sqrt(self.beta * self.mu * (2 + self.mu) / (1 + self.mu))
        return 0.0

    @property
    def normal_form_c3(self) -> float:
        """theta'' = -mu theta + c3 theta^3 + ...  (dimensionless time), c3=(4 kappa-1)/6."""
        return (4.0 * self.kappa - 1.0) / 6.0

    def equilibria(self):
        out = [(0.0, "centre" if self.mu > 0 else ("saddle" if self.mu < 0 else "degenerate")),
               (math.pi, "centre")]
        if self.mu > 0:
            ts = self.theta_star
            out += [(ts, "saddle"), (-ts, "saddle")]
        return out

    # ---------- energies ----------
    def U(self, th):
        th = np.asarray(th)
        return self.MgL * np.cos(th) + 0.5 * self.k * self.L ** 2 * np.sin(th) ** 2

    def dU(self, th):
        th = np.asarray(th)
        return -self.MgL * np.sin(th) + self.k * self.L ** 2 * np.sin(th) * np.cos(th)

    def d2U(self, th):
        th = np.asarray(th)
        return -self.MgL * np.cos(th) + self.k * self.L ** 2 * np.cos(2 * th)

    def energy(self, th, om):
        return 0.5 * self.inertia * np.asarray(om) ** 2 + self.U(th)

    def hamiltonian(self, th, p):
        return np.asarray(p) ** 2 / (2 * self.inertia) + self.U(th)

    def separatrix(self, th):
        """Upper/lower branches thetadot(theta) of the separatrix level set."""
        om2 = 2.0 * (self.E_sep - self.U(th)) / self.inertia
        om = np.sqrt(np.where(om2 >= 0, om2, np.nan))
        return om, -om

    # ---------- dynamics ----------
    def accel(self, t, th, om):
        s, c = np.sin(th), np.cos(th)
        a = self.beta * s - self.lam * s * c - self.gamma * om
        if self.ftype == 1:
            a = a + self.A * np.cos(self.Omega * t)
        elif self.ftype == 2:
            a = a + self.A * np.cos(self.Omega * t) * c
        return a

    def rhs(self, t, y):
        return np.array([y[1], self.accel(t, y[0], y[1])])

    def kernel_args(self):
        return (self.lam, self.beta, self.gamma, self.A, self.Omega, self.ftype)


# --------------------------------------------------------------------------------------
# numba kernels (dimensionful lam, beta; per unit inertia)
# --------------------------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def accel_nb(t, th, om, lam, beta, gamma, A, Omega, ftype):
    s = math.sin(th)
    c = math.cos(th)
    a = beta * s - lam * s * c - gamma * om
    if ftype == 1:
        a += A * math.cos(Omega * t)
    elif ftype == 2:
        a += A * math.cos(Omega * t) * c
    return a


@njit(cache=True)
def jac21_nb(t, th, lam, beta, A, Omega, ftype):
    """d(accel)/d(theta)."""
    c = math.cos(th)
    j = beta * c - lam * math.cos(2.0 * th)
    if ftype == 2:
        j += -A * math.cos(Omega * t) * math.sin(th)
    return j


@njit(cache=True)
def energy_nb(th, om, lam, beta):
    """Energy per unit inertia: 1/2 om^2 + beta cos th + 1/2 lam sin^2 th."""
    s = math.sin(th)
    return 0.5 * om * om + beta * math.cos(th) + 0.5 * lam * s * s
