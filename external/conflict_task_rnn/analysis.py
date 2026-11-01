"""
Behavioral and neural analyses of trained networks.

- load_net(run_dir)
- simulate(net, cfg, n): trial-level behavior (+ optional rates)
- tachometric(): P(choose big | conflict) vs processing time (rPT), binned
- fit_tachometric(): 4-parameter logistic fit (MLE on single trials), as in Fig 7:
    P(rPT) = lo + (hi - lo) / (1 + exp(-(rPT - mid) / slope))
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from config import Config
from network import BilateralEIRNN
from rollout import run_trials

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_net(run, device="cuda", comm_scale=1.0):
    run_dir = Path(run) if Path(run).is_absolute() else RESULTS / run
    ck = torch.load(run_dir / "model.pt", map_location=device)
    cfg = Config.from_dict(ck["config"])
    cfg.device = device
    net = BilateralEIRNN(cfg).to(device)
    net.load_state_dict(ck["state_dict"])
    net.comm_scale.fill_(comm_scale)
    net.eval()
    return net, cfg


def to_np(d):
    return {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in d.items()}


def simulate(net, cfg, n=20000, seed=0, chunk=4096, return_rates=False, **trial_kw):
    """Simulate n trials; returns dict of numpy arrays (trial params + outcomes)."""
    device = next(net.parameters()).device
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    parts, rates = [], []
    left = n
    while left > 0:
        b = min(chunk, left)
        res = run_trials(net, cfg, b, device, generator=g, return_rates=return_rates, **trial_kw)
        d = {**to_np(res["trials"]), **to_np(res["out"]),
             "act_step": res["act_step"].cpu().numpy(), "act_loc": res["act_loc"].cpu().numpy()}
        parts.append(d)
        if return_rates:
            rates.append(res["rates"].cpu().numpy().astype(np.float32))
        left -= b
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if return_rates:
        out["rates"] = np.concatenate(rates)
    return out


def logistic4(x, lo, hi, mid, slope):
    return lo + (hi - lo) / (1 + np.exp(-(x - mid) / slope))


def fit_tachometric(rpt, y, x0=(0.3, 0.9, 130., 20.)):
    """MLE fit of 4-parameter logistic to single-trial binary outcomes."""
    rpt = np.asarray(rpt, float)
    y = np.asarray(y, float)

    def nll(p):
        lo, hi, mid, slope = p
        pr = np.clip(logistic4(rpt, lo, hi, mid, slope), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    bounds = [(0, 1), (0, 1), (-100, 500), (2, 200)]
    best = None
    for mid0 in (80, 130, 180, 250):
        for lo0 in (0.1, 0.4):
            r = minimize(nll, (lo0, x0[1], mid0, x0[3]), bounds=bounds, method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
    lo, hi, mid, slope = best.x
    return dict(lo=lo, hi=hi, mid=mid, slope=slope, nll=best.fun)


def tachometric(rpt, y, edges=None):
    """Binned P(y) vs rPT with Wilson 95% CI. Returns centers, p, lo, hi, n."""
    if edges is None:
        edges = np.arange(0, 501, 25)
    c, p, lo, hi, nn = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (rpt >= a) & (rpt < b)
        n = m.sum()
        if n < 10:
            continue
        k = y[m].sum()
        ph = k / n
        z = 1.96
        den = 1 + z ** 2 / n
        cen = (ph + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
        c.append((a + b) / 2); p.append(ph); lo.append(cen - half); hi.append(cen + half); nn.append(n)
    return tuple(np.array(v) for v in (c, p, lo, hi, nn))


def behavior_summary(sim, cfg):
    """Key numbers for tuning / reporting."""
    comp = sim["completed"]
    conf = sim["conflict"] & comp
    cong = (~sim["conflict"]) & comp
    fit = fit_tachometric(sim["rpt"][conf], sim["chose_big"][conf])
    s = dict(completed=comp.mean(), pbig_conflict=sim["chose_big"][conf].mean(),
             pbig_congruent=sim["chose_big"][cong].mean(),
             rt_mean=sim["rt"][comp].mean(), rt_sd=sim["rt"][comp].std(),
             fit=fit)
    s["rt_by_dT"] = {int(d): float(sim["rt"][comp & (sim["dT"] == d)].mean()) for d in cfg.delta_ts}
    s["pbig_conf_by_dT"] = {int(d): float(sim["chose_big"][conf & (sim["dT"] == d)].mean()) for d in cfg.delta_ts}
    # capture vs correct RTs on conflict trials
    s["rt_conf_correct"] = sim["rt"][conf & sim["chose_big"]].mean()
    s["rt_conf_error"] = sim["rt"][conf & ~sim["chose_big"]].mean()
    return s


def print_summary(name, s):
    f = s["fit"]
    print(f"{name}: comp {s['completed']:.3f} | Pbig conf {s['pbig_conflict']:.3f} cong {s['pbig_congruent']:.3f} | "
          f"RT {s['rt_mean']:.0f}+-{s['rt_sd']:.0f} | fit lo {f['lo']:.2f} hi {f['hi']:.2f} mid {f['mid']:.0f} "
          f"slope {f['slope']:.1f} | RT corr/err {s['rt_conf_correct']:.0f}/{s['rt_conf_error']:.0f}")
    print("   RT by dT:", {k: round(v) for k, v in s["rt_by_dT"].items()})
    print("   Pbig conf by dT:", {k: round(v, 2) for k, v in s["pbig_conf_by_dT"].items()})


if __name__ == "__main__":
    import sys
    for run in sys.argv[1:]:
        net, cfg = load_net(run)
        sim = simulate(net, cfg, 20000)
        print_summary(run, behavior_summary(sim, cfg))
