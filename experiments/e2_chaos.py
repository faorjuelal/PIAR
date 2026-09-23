"""E2 - Damped-driven PIAR: from the separatrix to the strange attractor (Sec. IV).

Reference parameters (natural units): mu = -0.8 (kappa = 0.2), gamma = 0.25, Omega = 2/3,
harmonic torque A cos(Omega t). The chaotic reference state A = 0.8 is the attractor that the
FPRM of Secs. V-VI reconstructs.

Outputs: results/e2_chaos.json (numbers) and results/e2_chaos.npz (arrays for fig_chaos.py)
"""
import json
import math
import time

import numpy as np

from piar.physics import PIAR
from piar import chaos as C

KAPPA, GAMMA, OMEGA = 0.2, 0.25, 2.0 / 3.0
p0 = PIAR.dimensionless(KAPPA - 1.0, gamma=GAMMA, A=0.8, Omega=OMEGA, forcing="torque")
res = {"params": {"mu": KAPPA - 1.0, "gamma": GAMMA, "Omega": OMEGA, "forcing": "torque"}}
arr = {}
t00 = time.time()

# ------------------------------------------------------------------ bifurcation diagram + hysteresis
A_grid = np.linspace(0.30, 1.40, 1101)
thu, omu = C.bifurcation_sweep(p0, A_grid, 0.0, 0.0, n_transient=300, n_keep=160, spp=128)
thd, omd = C.bifurcation_sweep(p0, A_grid[::-1], 0.0, 0.0, n_transient=300, n_keep=160, spp=128)
per_u = np.array([C.attractor_period(np.column_stack([thu[i], omu[i]]), tol=1e-5) for i in range(len(A_grid))])
per_d = np.array([C.attractor_period(np.column_stack([thd[i], omd[i]]), tol=1e-5)
                  for i in range(len(A_grid))])[::-1]
res["hysteresis_fraction"] = float(np.mean(per_u != per_d))
arr.update(A_grid=A_grid, bif_theta=thu, bif_omega=omu)
print(f"bifurcation {time.time()-t00:.0f}s", flush=True)

# ------------------------------------------------------------------ largest Lyapunov exponent along A
l1_A, l2_A = C.lyapunov_sweep_nb(A_grid, 0.0, 0.0, 300, 1500, 128, p0.lam, p0.beta, GAMMA, OMEGA, 1)
res["lyap_sum_rule_max_dev"] = float(np.max(np.abs(l1_A + l2_A + GAMMA)))
res["chaotic_fraction"] = float(np.mean(l1_A > 1e-3))
arr.update(lyap1_A=l1_A, lyap3_A=l2_A)
print(f"lyapunov sweep {time.time()-t00:.0f}s", flush=True)

# ------------------------------------------------------------------ Feigenbaum cascade (Floquet continuation)
An = C.feigenbaum_cascade(p0, 0.64, levels=6, spacing0=9e-3)
dn = np.diff(An)
deltas = dn[:-1] / dn[1:]
A_inf = An[-1] + dn[-1] / (4.669201609 - 1.0)
An_fine = C.feigenbaum_cascade(p0, 0.64, levels=2, spacing0=9e-3, spp=512)
res["feigenbaum"] = {"A_n": An.tolist(), "delta_n": deltas.tolist(), "A_inf": float(A_inf),
                     "A_n_spp512": An_fine.tolist()}
print("Feigenbaum:", An, deltas, f"{time.time()-t00:.0f}s", flush=True)

# ------------------------------------------------------------------ cascade magnification
p1_seed = C.strobe(p0.with_(A=0.40), 0.0, 0.0, n_transient=500, n_keep=4, spp=128)[-1]
Az = np.linspace(0.655, 0.6755, 700)
thz, _ = C.bifurcation_sweep(p0, Az, *p1_seed, n_transient=1500, n_keep=128, spp=128)
arr.update(zoom_A=Az, zoom_theta=thz)
print(f"cascade zoom {time.time()-t00:.0f}s", flush=True)

# ------------------------------------------------------------------ strange attractor at A = 0.8
sec = C.strobe(p0.with_(A=0.80), 0.0, 0.0, n_transient=500, n_keep=200000, spp=128)
arr["section"] = sec
l1s, l3s = [], []
for s in range(10):
    th0, om0 = sec[1000 * (s + 1)]
    l1, l3, _ = C.lyapunov(p0.with_(A=0.80), th0, om0, n_transient=50, n_periods=5000, spp=128)
    l1s.append(l1)
    l3s.append(l3)
l1s, l3s = np.array(l1s), np.array(l3s)
lam1, lam3 = l1s.mean(), l3s.mean()
se1 = l1s.std(ddof=1) / math.sqrt(len(l1s))
DKY = 2.0 + lam1 / abs(lam3)
radii = np.logspace(-4, np.log10(2e-3), 10)
Csum = C.correlation_sum(sec, radii, max_pairs_points=100000, seed=1)
D2 = float(np.polyfit(np.log(radii), np.log(Csum), 1)[0])
res["chaos_A0.80"] = {"lambda1": float(lam1), "lambda1_se": float(se1), "lambda3": float(lam3),
                      "sum": float(lam1 + lam3), "D_KY_flow": float(DKY), "D_KY_section": float(DKY - 1),
                      "D2_section": float(D2), "lyapunov_time": float(1.0 / lam1),
                      "lyapunov_time_forcing_periods": float(1.0 / lam1 / (2 * math.pi / OMEGA))}
print("Lyapunov:", res["chaos_A0.80"], f"{time.time()-t00:.0f}s", flush=True)

# smooth display trajectory on the attractor: every RK4 step is stored (128 per forcing period)
th_a, om_a = sec[-1]
tr = C.trajectory(p0.with_(A=0.80), th_a, om_a, n_periods=60, spp=128, save_per_period=128)
arr["traj"] = tr

# ------------------------------------------------------------------ Melnikov threshold + (A, Omega) chart
Om_grid = np.linspace(0.40, 1.60, 70)
A_chart = np.linspace(0.02, 1.60, 90)
chart = C.lyapunov_grid_nb(A_chart, Om_grid, 0.0, 0.0, 150, 400, 96, p0.lam, p0.beta, GAMMA, 1)
mel = np.array([C.melnikov_ratio(KAPPA, Om)[0] for Om in Om_grid])
r_ref, I0, I1 = C.melnikov_ratio(KAPPA, OMEGA)
res["melnikov"] = {"ratio_Omega_2_3": float(r_ref), "I0": float(I0), "I1": float(I1),
                   "A_c_Omega_2_3": float(GAMMA * r_ref)}
chaotic = chart > 1e-3
Amesh, Omesh = np.meshgrid(A_chart, Om_grid)
under = Amesh < np.interp(Omesh, Om_grid, GAMMA * mel)
res["melnikov"]["chaotic_fraction_below_curve"] = float(chaotic[under].mean())
res["melnikov"]["chaotic_fraction_above_curve"] = float(chaotic[~under].mean())
arr.update(chart=chart, chart_A=A_chart, chart_Omega=Om_grid, melnikov_Ac=GAMMA * mel)
print("Melnikov:", res["melnikov"], f"{time.time()-t00:.0f}s", flush=True)

json.dump(res, open("results/e2_chaos.json", "w"), indent=2)
np.savez_compressed("results/e2_chaos.npz", **arr)
print("done", f"{time.time()-t00:.0f}s")
