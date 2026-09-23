"""E7 - Computational cost of the generative engines (flow vs. diffusion).

Measures, with the architectures of Table IV and the conditioning dimensions of tasks C1 and C2:
  * network evaluations per sample (forward hooks on the conditioner / denoiser);
  * wall-clock time to draw 64 samples for 1500 missing entries (the evaluation protocol);
  * wall-clock time of one physics-regularized training step, which needs gradients through the
    samples: one reparameterized pass of the flow versus back-propagation through the 100-step
    reverse chain of the DDPM (a step that the DDPM of the benchmark never performs).
Timing does not depend on the trained weights, so untrained networks are used.
Writes results/e7_engine_cost.json.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piar import fprm as F  # noqa: E402

torch.set_num_threads(1)          # same single-thread budget for both engines
REPEATS = 5
N_ROWS, N_SAMPLES = 1500, 64
N_COL, N_DRAW = 256, 8            # collocation rows and samples per row in the physics term (Table IV)

TASKS = {"C1": dict(dim=2, ctx=31), "C2": dict(dim=1, ctx=19)}


def median_time(fn, repeats=REPEATS):
    fn()                                   # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def count_calls(module, fn):
    n = {"calls": 0}

    def hook(*_):
        n["calls"] += 1

    hs = [m.register_forward_hook(hook) for m in module]
    fn()
    for h in hs:
        h.remove()
    return n["calls"]


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    out = {"torch": torch.__version__, "threads": torch.get_num_threads(), "cpu": cpu_model(),
           "n_rows": N_ROWS, "n_samples": N_SAMPLES, "tasks": {}}
    for task, cfg in TASKS.items():
        dim, ctx = cfg["dim"], cfg["ctx"]
        rng = np.random.default_rng(0)
        X = rng.standard_normal((N_ROWS, ctx)).astype(np.float32)
        Y = rng.standard_normal((N_ROWS, dim)).astype(np.float32)

        nsf = F.NSFImputer(dim, ctx, steps=1, seed=0)
        ddpm = F.DDPMImputer(dim, ctx, steps=1, seed=0, schedule="cosine")
        for m in (nsf, ddpm):                      # standardizers only; no training needed for timing
            m.sx, m.sy = F.Standardizer(X), F.Standardizer(Y)

        # --- network evaluations per sample -------------------------------------------------
        Xt = torch.as_tensor(X[:4])
        conditioners = [t.hyper for t in nsf.flow.transform.transforms if hasattr(t, "hyper")]
        with torch.no_grad():
            n_nsf_sample = count_calls(conditioners, lambda: nsf.flow(Xt).sample((1,)))
            n_nsf_logp = count_calls(conditioners, lambda: nsf.flow(Xt).log_prob(torch.zeros(4, dim)))
        n_ddpm_sample = count_calls([ddpm.net], lambda: ddpm.sample(X[:4], n=1))

        # --- sampling time (evaluation protocol) ---------------------------------------------
        t_nsf = median_time(lambda: nsf.sample(X, N_SAMPLES))
        t_ddpm = median_time(lambda: ddpm.sample(X, N_SAMPLES))

        # --- one physics-regularized training step --------------------------------------------
        Xc = torch.as_tensor(X[:N_COL])

        def step_nsf():
            nsf.flow.zero_grad()
            ys = nsf.flow(Xc).rsample((N_DRAW,))                     # (8, 256, d), differentiable
            loss = torch.relu((ys ** 2).sum(-1) - 1.0).mean()        # stand-in residual of the same size
            loss.backward()

        def step_ddpm():
            ddpm.net.zero_grad()
            C = Xc.repeat(N_DRAW, 1)
            y = torch.randn(N_DRAW * N_COL, dim)
            for t in reversed(range(ddpm.T)):                        # reverse chain with gradients
                tt = torch.full((len(y),), t)
                eps = ddpm.net(torch.cat([y, C, ddpm._temb(tt)], 1))
                a, ab, b = ddpm.alphas[t], ddpm.abar[t], ddpm.betas[t]
                y = (y - b / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
                if t > 0:
                    y = y + torch.sqrt(b) * torch.randn_like(y)
            loss = torch.relu((y ** 2).sum(-1) - 1.0).mean()
            loss.backward()

        s_nsf = median_time(step_nsf)
        s_ddpm = median_time(step_ddpm)

        n_par_nsf = sum(p.numel() for p in nsf.flow.parameters())
        n_par_ddpm = sum(p.numel() for p in ddpm.net.parameters())
        out["tasks"][task] = {
            "dim": dim, "ctx_dim": ctx, "params_nsf": n_par_nsf, "params_ddpm": n_par_ddpm,
            "evals_per_sample_nsf": n_nsf_sample, "evals_per_logprob_nsf": n_nsf_logp,
            "evals_per_sample_ddpm": n_ddpm_sample,
            "sample_seconds_nsf": t_nsf, "sample_seconds_ddpm": t_ddpm, "sample_ratio": t_ddpm / t_nsf,
            "phys_step_seconds_nsf": s_nsf, "phys_step_seconds_ddpm": s_ddpm, "phys_step_ratio": s_ddpm / s_nsf,
        }
        print(task, json.dumps(out["tasks"][task], indent=1), flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/e7_engine_cost.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
