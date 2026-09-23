"""PIAR computational laboratory: spring-coupled inverted pendulum and the full-partial
reconstruction mapping (FPRM).

Modules
-------
physics      analytical mechanics, invariants, numba kernels
integrators  DOP853 (adaptive), velocity Verlet (symplectic), RK4, exact periods
chaos        Poincare maps, bifurcation sweeps, Lyapunov spectra, Floquet, Melnikov, D2
stochastic   BAOAB Langevin dynamics (occupation densities, stochastic records)
embedding    delay embedding, average mutual information, false nearest neighbours
fprm         generative FPRM: context builder, conditional NSF, cVAE, DDPM, GPR, physics residuals, metrics
smooth       smooth rendering of sampled trajectories (cubic Hermite interpolation, cylinder wrapping)
dashboard    interactive dashboard and smooth animation
style        publication style (Computer Modern, batlow/vik, validated categorical order)
"""
from .physics import PIAR  # noqa: F401

__version__ = "2.2.0"
