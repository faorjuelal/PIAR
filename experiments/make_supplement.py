"""Supplementary material: tables, embedding figure, and per-run data of the benchmark.

Reads results/e3_C{0,1,2,3}.json, e5_diagnostics.json, e6_fullgp.json, e7_engine_cost.json,
e8_symmetry.json. Writes
  manuscript/supplement/tabS_*.tex   tabular bodies included by manuscript/supplement.tex
  figures/figS_embedding.pdf/.png    average mutual information and false nearest neighbors
  results/benchmark_runs.csv         every metric of every run (task, variant, fraction, mask, imputer)
Usage: python experiments/make_supplement.py
"""
import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from piar.style import set_style, savefig, CAT, COL2, INK, MUTED, panel_label

OUT = "manuscript/supplement"
os.makedirs(OUT, exist_ok=True)
R = {c: json.load(open(f"results/e3_{c}.json")) for c in ("C0", "C1", "C2", "C3")}

NAMES = {"static": "static GPR", "gpr": "original FPRM (GPR)", "gpr_ctx": "GPR (full context)",
         "cvae": "cVAE", "diffusion": "DDPM", "nsf": "NSF", "nsf_phys": "NSF + physics",
         "savgol": "Savitzky--Golay derivative"}
PLAIN = {k: v.replace("--", "-") for k, v in NAMES.items()}
ORDER = ["savgol", "static", "gpr", "gpr_ctx", "cvae", "diffusion", "nsf", "nsf_phys"]
METRICS = [("rho", r"$\rho$", 4), ("nrmse", "NRMSE", 3), ("crps", "CRPS", 4),
           ("coverage90", "cov.", 3), ("p_correct_sign", "sign", 3), ("sw1_section", r"$W_1$", 4)]


def runs_of(cfg):
    """yield (variant, rate, mask, method, metrics) for every run of a task."""
    for key, res in R[cfg]["runs"].items():
        rr, s, v = key.split("_")
        for mth, d in res.items():
            yield v, float(rr[1:]), int(s[1:]), mth, d


# ------------------------------------------------------------------ per-run CSV
with open("results/benchmark_runs.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    cols = ["rho", "nrmse", "crps", "coverage90", "p_correct_sign", "nrmse_abs", "sw1_section", "best_step",
            "seconds"]
    w.writerow(["task", "variant", "missing_fraction", "mask", "imputer"] + cols)
    for cfg in ("C0", "C1", "C2", "C3"):
        for v, r, s, mth, d in sorted(runs_of(cfg), key=lambda t: (t[1], t[0], ORDER.index(t[3]), t[2])):
            w.writerow([cfg, v, r, s + 1, PLAIN[mth]] + [d.get(c, "") for c in cols])


# ------------------------------------------------------------------ benchmark tables
def cell(vals, nd):
    vals = np.asarray(vals, float)
    m, h = vals.mean(), np.ptp(vals) / 2
    ms = f"{m:.{nd}f}" if m >= 0 else f"$-${abs(m):.{nd}f}"
    return ms + r"\,$\pm$\," + f"{h:.{nd}f}"


def bench_table(cfg, variant, metrics, fname, extra_name=None):
    rows = []
    stats = {}
    for v, r, s, mth, d in runs_of(cfg):
        if v != variant:
            continue
        stats.setdefault((r, mth), []).append(d)
    rates = sorted({r for r, _ in stats})
    head = r"\begin{tabular}{l" + "c" * len(metrics) + "}"
    hdr = "Imputer & " + " & ".join(lab for _, lab, _ in metrics) + r"\\"
    body = [head, r"\toprule", hdr]
    for r in rates:
        body.append(r"\midrule")
        body.append(r"\multicolumn{" + str(len(metrics) + 1) + r"}{l}{\textit{" + f"{int(round(100 * r))}" +
                    r"\% missing}}\\")
        for mth in ORDER:
            if (r, mth) not in stats:
                continue
            ds = stats[(r, mth)]
            assert len(ds) == 2, (cfg, variant, r, mth)
            name = NAMES[mth]
            if extra_name and mth == "gpr":
                name += extra_name
            cells = []
            for key, _, nd in metrics:
                if mth == "savgol" and key == "coverage90":
                    cells.append("--")
                else:
                    cells.append(cell([d[key] for d in ds], nd))
            body.append(name + " & " + " & ".join(cells) + r"\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/{fname}", "w").write("\n".join(body) + "\n")


bench_table("C1", "phase", METRICS, "tabS_C1.tex")
bench_table("C2", "nophase", METRICS[:5], "tabS_C2.tex")
bench_table("C3", "phase", METRICS[:5], "tabS_C3_phase.tex", extra_name=" + phase")
bench_table("C3", "nophase", METRICS[:5], "tabS_C3_nophase.tex")
bench_table("C0", "nophase", METRICS[:5], "tabS_C0.tex")

# ------------------------------------------------------------------ embedding table and figure
CHAN = {"C0": r"$\theta$ (unwrapped), forced", "C1": r"$s$, forced", "C2": r"$c$, Langevin",
        "C3": r"$c$, forced"}
rows = [r"\begin{tabular}{llcccc}", r"\toprule",
        r"Task & Complete channel & $\tau$ (samples) & $d_E$ & $m$ & FNN fraction at $d_E$\\", r"\midrule"]
for cfg in ("C0", "C1", "C2", "C3"):
    e = R[cfg]["embedding"]
    rows.append(f"{cfg} & {CHAN[cfg]} & {e['tau']} & {e['E']} & {e['m']} & {100 * e['fnn'][e['E'] - 1]:.2f}\\%\\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tabS_embedding.tex", "w").write("\n".join(rows) + "\n")

set_style()
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.7))
for i, cfg in enumerate(("C0", "C1", "C2", "C3")):
    e = R[cfg]["embedding"]
    ami = np.array(e["ami"])
    axs[0].plot(np.arange(len(ami)), ami, color=CAT[i], lw=1.4, label=cfg)
    axs[0].plot(e["tau"], ami[e["tau"]], "o", ms=6, mfc=CAT[i], mec="white", mew=1.0, zorder=5)
    fnn = 100 * np.array(e["fnn"])
    d = np.arange(1, len(fnn) + 1)
    axs[1].semilogy(d, fnn, "-", color=CAT[i], lw=1.4, label=cfg)
    axs[1].semilogy(e["E"], fnn[e["E"] - 1], "o", ms=6, mfc=CAT[i], mec="white", mew=1.0, zorder=5)
axs[1].axhline(1.0, color=MUTED, lw=1.0, ls="--")
axs[1].text(1.0, 0.85, r"$1\%$", color=INK, fontsize=8, ha="left", va="top")
axs[0].set_xlabel(r"lag (samples)")
axs[0].set_ylabel(r"mutual information (nats)")
axs[0].set_yscale("log")
axs[0].set_ylim(0.05, 6.0)
axs[1].set_xlabel(r"embedding dimension $d$")
axs[1].set_ylabel(r"false nearest neighbors (\%)")
axs[1].set_xticks(range(1, 9))
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
           fontsize=8, frameon=False)
fig.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.9))
panel_label(axs[0], "a", x=-0.19, y=1.03)
panel_label(axs[1], "b", x=-0.19, y=1.03)
savefig(fig, "figS_embedding")

# ------------------------------------------------------------------ diagnostics
e5 = json.load(open("results/e5_diagnostics.json"))
rows = [r"\begin{tabular}{lcccc}", r"\toprule",
        r"Missing fraction & Library & $\rho(c)$ & $\rho(\dot\theta)$ & mean $\rho$\\", r"\midrule"]
for r in (0.3, 0.6, 0.9):
    for name, lab in (("all_observed", "all observed entries"), ("gpr_size_1200", "1200 entries")):
        d = e5[f"r{r}_{name}"]
        rows.append(f"{int(100 * r)}\\% & {lab} ({d['library']}) & {d['rho_c']:.3f} & {d['rho_omega']:.3f} & "
                    f"{d['rho_mean']:.3f}\\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tabS_library.tex", "w").write("\n".join(rows) + "\n")
c3 = e5["c3_residual_conditioning"]
with open(f"{OUT}/valS_conditioning.tex", "w") as fh:
    fh.write(f"\\newcommand{{\\SGgain}}{{{c3['sg_second_derivative_noise_gain']:.2f}}}\n"
             f"\\newcommand{{\\dcdd}}{{{c3['delta_cdd']:.3f}}}\n"
             f"\\newcommand{{\\thrsc}}{{{c3['threshold_abs_s_cos']:.3f}}}\n"
             f"\\newcommand{{\\fracbelow}}{{{100 * c3['fraction_below_threshold']:.1f}}}\n")

e6 = json.load(open("results/e6_fullgp.json"))
rows = [r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Mask & Training points & $\rho$ & NRMSE & CRPS & cov. & sign\\", r"\midrule"]
for k in sorted(e6["runs"]):
    d = e6["runs"][k]
    rows.append(f"{int(k.split('_s')[1]) + 1} & {d['n_train']} & {d['rho']:.3f} & {d['nrmse']:.3f} & "
                f"{d['crps']:.4f} & {d['coverage90']:.3f} & {d['p_correct_sign']:.3f}\\\\")
m = e6["mean"]
rows.append(r"\midrule")
rows.append(f"mean & -- & {m['rho']:.3f} & {m['nrmse']:.3f} & {m['crps']:.4f} & {m['coverage90']:.3f} & "
            f"{m['p_correct_sign']:.3f}\\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tabS_fullgp.tex", "w").write("\n".join(rows) + "\n")

# ------------------------------------------------------------------ cost benchmark
e7 = json.load(open("results/e7_engine_cost.json"))
t1, t2 = e7["tasks"]["C1"], e7["tasks"]["C2"]
spec = [("Target dimension", "dim", "{:d}"), ("Conditioning dimension", "ctx_dim", "{:d}"),
        ("Parameters, NSF", "params_nsf", "{:,d}"), ("Parameters, DDPM", "params_ddpm", "{:,d}"),
        ("Network evaluations per sample, NSF$^{a}$", "evals_per_sample_nsf", "{:d}"),
        ("Network evaluations per sample, DDPM", "evals_per_sample_ddpm", "{:d}"),
        ("Network evaluations per log density, NSF", "evals_per_logprob_nsf", "{:d}"),
        ("Sampling time (s), NSF", "sample_seconds_nsf", "{:.3f}"),
        ("Sampling time (s), DDPM", "sample_seconds_ddpm", "{:.1f}"),
        ("Sampling-time ratio DDPM/NSF", "sample_ratio", "{:.0f}"),
        ("Gradient of the penalty (s), NSF", "phys_step_seconds_nsf", "{:.4f}"),
        ("Gradient of the penalty (s), DDPM", "phys_step_seconds_ddpm", "{:.2f}"),
        ("Gradient-time ratio DDPM/NSF", "phys_step_ratio", "{:.0f}")]
THIN = "\\,"
rows = [r"\begin{tabular}{lcc}", r"\toprule", r"Quantity & C1 & C2\\", r"\midrule"]
for lab, key, f in spec:
    v1, v2 = (f.format(t[key]).replace(",", THIN) for t in (t1, t2))
    rows.append(f"{lab} & {v1} & {v2}\\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tabS_cost.tex", "w").write("\n".join(rows) + "\n")
with open(f"{OUT}/valS_cost.tex", "w") as fh:
    ghz = float(e7['cpu'].split('@')[1].strip().replace('GHz', '')) if '@' in e7['cpu'] else None
    cpu = e7['cpu'].split('@')[0].replace('(R)', '').replace('Processor', 'processor').strip()
    cpu = f"{cpu} at {ghz:g}~GHz" if ghz else cpu
    fh.write(f"\\newcommand{{\\torchver}}{{{e7['torch'].split('+')[0]}}}\n"
             f"\\newcommand{{\\costcpu}}{{{cpu}}}\n"
             f"\\newcommand{{\\costrows}}{{{e7['n_rows']}}}\n\\newcommand{{\\costsamples}}{{{e7['n_samples']}}}\n")

# ------------------------------------------------------------------ symmetry check
e8 = json.load(open("results/e8_symmetry.json"))
rows = [r"\begin{tabular}{lccc}", r"\toprule",
        r"Initial condition $(\theta_0,\dot\theta_0)$ & $\langle\sin\theta\rangle$ & $\langle\dot\theta\rangle$ & "
        r"$D_{\mathrm{TV}}(\mathcal P_0,\,R\mathcal P_{\pi})$\\", r"\midrule"]


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"


for d in e8["runs"]:
    rows.append(f"$({d['ic'][0]:g},\\,{d['ic'][1]:g})$ & {sci(d['mean_sin_theta'])} & {sci(d['mean_omega'])} & "
                f"{d['tv_section0_vs_mirrored_pi']:.4f}\\\\")
rows.append(r"\midrule")
rows.append(r"\multicolumn{3}{l}{Independent records, $D_{\mathrm{TV}}(\mathcal P_0,\,\mathcal P_0')$, pairs "
            + ", ".join(f"({a + 1},{b + 1})" for a, b in (p["pair"] for p in e8["baseline_pairs"])) + "} & "
            + ", ".join(f"{p['tv_section0_independent']:.4f}" for p in e8["baseline_pairs"]) + r"\\")
rows += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tabS_symmetry.tex", "w").write("\n".join(rows) + "\n")
with open(f"{OUT}/valS_symmetry.tex", "w") as fh:
    fh.write(f"\\newcommand{{\\symperiods}}{{{e8['params']['periods']:,d}}}\n".replace(",", "\\,"))
print("supplement tables written to", OUT)
