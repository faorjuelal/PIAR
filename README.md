# PIAR — computational laboratory for the spring-coupled inverted pendulum

Code, data, and article source for *From separatrix to strange attractor: Physics-informed normalizing
flows for recovering missing dynamics in a spring-coupled inverted pendulum*, by Fredy Alexander Orjuela
López (Universidad de los Andes), submitted to *Chaos* (AIP Publishing) in September 2026.
Repository: <https://github.com/faorjuelal/PIAR>.

The spring-coupled inverted pendulum (PIAR) is a rod held upright by a horizontal spring. A single
parameter, μ = kL/(Mg) − 1, controls its conservative dynamics; under damping and forcing it becomes
chaotic. The laboratory uses it as a benchmark for the full–partial reconstruction mapping (FPRM) of
Wu *et al.* (Nat. Commun. 2026, doi:10.1038/s41467-026-77922-1), which recovers missing sensor
records from the delay reconstruction of a complete one. The article covers:

- the model and its closed-form invariants (Sec. II);
- five canonical conservative regimes (Sec. III);
- the damped–driven route to a strange attractor (Sec. IV);
- the FPRM with a conditional neural-spline-flow engine and physics residuals (Sec. V);
- the reconstruction benchmark (Sec. VI).

## Try it in the browser

Open [`web/piar_simulation.html`](web/piar_simulation.html) in any browser. No installation is needed.
The page shows the five regimes of Table I and the chaotic state, with a smooth motion trail and four
live diagnostics: potential energy, kinetic energy, phase diagram, and position. The interface is in
English, with a Spanish option. The equation of motion is integrated with fourth-order Runge–Kutta at a
step of 0.002 (L/g)^(1/2), and the energy of the conservative cases stays constant to better than
10⁻¹⁰ MgL over 100 s.

## Repository layout

| Path | Content |
|---|---|
| `piar/` | Python package; its modules are listed below the table |
| `experiments/` | Experiments `e1_cases.py` … `e8_symmetry.py`, figure scripts `fig_*.py`, and `make_supplement.py` |
| `results/` | JSON files with every number reported in the article, cached arrays for the figure scripts, and `benchmark_runs.csv` with every metric of every benchmark run |
| `figures/` | Output folder of the figure scripts (PDF and 600-dpi PNG). `figures/cases/` holds one 2×2 figure per case, in English (`case<i>_en`) and Spanish (`caso<i>_es`) |
| `web/piar_simulation.html` | Browser simulation (single self-contained file) |
| `notebooks/PIAR_Dashboard.ipynb` | Interactive notebook with case figures, smooth animation, and a six-panel dashboard (ipywidgets) |
| `tests/` | Ten unit tests (`pytest -q`) |
| `manuscript/` | Article on the official AIP template (REVTeX 4.1 with `aip4-1.rtx`; template files included): `main.tex`, `sections/`, `tables/`, `refs.bib`, and `figures/` (the figures used in the article). Also `supplement.tex` with `supplement/` for the supplementary material, and `build.sh` |
| `tools/validate_palette.js` | Colour-vision-deficiency check of the categorical palette (`results/palette_check.txt`) |

Modules of `piar/`:

- `physics`: model, invariants, compiled right-hand sides.
- `integrators`: DOP853 port, compensated velocity Verlet, RK4, exact periods.
- `chaos`: Poincaré maps, continuation, Lyapunov spectra, Floquet multipliers, Feigenbaum cascade, Melnikov integrals, correlation dimension.
- `stochastic`: BAOAB Langevin dynamics.
- `embedding`: average mutual information, false nearest neighbours.
- `fprm`: conditioning vectors, conditional NSF, cVAE, DDPM, Gaussian processes, physics residuals, metrics.
- `smooth`: Hermite rendering, cylinder wrapping.
- `dashboard`: dashboard, case figures, animation.
- `style`: figure style.

## Installation and tests

```bash
pip install -r requirements.txt && pip install -e .
pytest -q                                   # 10 unit tests
```

The versions used for the article are pinned in `requirements.txt` (Python 3.11). The figure style uses
LaTeX for Computer Modern typography; `piar.style.set_style(usetex=False)` falls back to mathtext.

## Reproduction

```bash
python experiments/e1_cases.py              # Secs. II–III: fig_model, fig_cases, fig_portraits, case figures, Table I
python experiments/e2_chaos.py              # Sec. IV: bifurcations, Feigenbaum, Lyapunov, Melnikov (arrays in results/)
python experiments/fig_chaos.py             # fig_route, fig_attractor
python experiments/fig_fprm_scheme.py       # fig_fprm_scheme (model-free identifiability test)
for c in C0 C1 C2 C3; do python experiments/e3_fprm.py $c; done   # Sec. VI benchmark
python experiments/fig_fprm.py              # fig_fprm + manuscript/tables/tab_fprm.tex
python experiments/e4_reconstruction.py     # every missing entry of a window + Poincaré section (C1, 90%)
python experiments/fig_reconstruction.py    # fig_reconstruction
python experiments/e5_diagnostics.py        # library-size check and conditioning of the C3 residual
python experiments/e6_fullgp.py             # original FPRM with all labels (C1, 90%)
python experiments/e7_engine_cost.py        # cost of flow vs diffusion (Sec. VI F)
python experiments/e8_symmetry.py           # spatiotemporal symmetry of the reference attractor (Sec. IV D)
python experiments/fig_dashboard.py         # fig_dashboard
python experiments/make_supplement.py       # supplementary tables, Fig. S1, results/benchmark_runs.csv
cd manuscript && ./build.sh                 # main.pdf (reprint) and main_preprint.pdf (AIP review format)
pdflatex supplement && pdflatex supplement  # supplement.pdf (after main.pdf)
```

All random seeds are fixed in the scripts. `PIAR_SMOKE=1` runs a fast end-to-end check of `e3_fprm.py`;
its outputs are prefixed `smoke_` and are not used in the article.

Approximate run times on two CPU cores:

| Experiment | Run time |
|---|---|
| E1, E2 | about a minute each |
| E3 | about 3 h in total |
| E4 | about 4 min |
| E5 | a few seconds |
| E6 | about 16 min of Gaussian-process fitting |
| E7 | several minutes |
| E8 | about 10 s |

## Smooth rendering

Every curve is drawn from a continuous representation of the solution:

- **Conservative cases:** DOP853 dense output, 25 001 instants per 100 s.
- **Driven system:** cubic Hermite interpolation between RK4 steps (`piar.smooth.hermite_upsample`, which uses θ' = ω and ω' = acceleration).

Angles on the cylinder are broken at ±π (`piar.smooth.break_on_wrap`), so rotations are not crossed by spurious segments. The browser simulation stores every RK4 step in its trail.

## Provenance of the reported numbers

* `e3_C*.json`: full runs of `e3_fprm.py`; the cVAE and DDPM entries of C1 and C2 were recomputed with
  `PIAR_METHODS=cvae,diffusion` (6000 steps, cosine noise schedule) and merged into the same files.
* `e4_reconstruction.*`: retrains the C1 models of the first mask at 90% with identical settings (the
  benchmark scores are reproduced exactly) and imputes every missing entry of an eight-period window.
* The per-sample caches `results/e3_C{0,1,3}_cache.npy` are regenerated by `e3_fprm.py`; the cache of C2 is
  included because `fig_fprm.py` reads it.

## Citation and license

- **Citation:** metadata are given in `CITATION.cff` (read by GitHub's "Cite this repository").
- **License:** the code is released under the MIT License (`LICENSE`), and the numerical results and figures under CC BY 4.0.
