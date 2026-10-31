#!/usr/bin/env python3
"""
Urgent-task predictions for resolver (A), priority hub (B) and outcome hub (C) networks
(plans/2026-10-31_urgent_task_plan.md v2). Per trained network (results3 on CRC):
  tachometric curves  P(choose rich) vs processing time rPT = RT(from go) - dt on conflict trials (rich and salient
                      targets on opposite sides) and congruent trials; 4-parameter logistic fit (lower/upper asymptote,
                      midpoint, slope) overall, by value-gap tercile (|block mean T1 - T2|), and at test salience gains
                      0.3 / 1 / 2 (trained at 1)
  value latency       measured value-arrival latency: single-target trials, shown-location pool rate for the rich vs
                      the poor identity, aligned on target onset (first of 3 consecutive bins with t > 3)
  divergence vs dt    conflict trials: time after target onset at which the rich-minus-salient pool difference turns
                      and stays positive, per dt; slope vs dt (0 = target-locked, 1 = go-locked)
  error trials        rich-pool rate 100-150 ms after target onset on capture errors vs correct conflict trials (both
                      with saccades > 150 ms after target onset), as a ratio
  S8 (time-resolved)  fraction of contralateral units with |identity effect beyond value| > 0.2 in target-aligned
                      windows (block-type-averaged log T1/T2 on single-target trials; recorded-like rate mapping)
  S9                  Spearman(single-target RT, belief value of the shown identity)
Usage: python scripts/srs_urgent_analysis.py <results_dir> [name_glob] -> <results_dir>/../analysis3/<name>.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'external' / 'conflict_srs'))
sys.path.insert(0, str(ROOT))
from srs.task import make_urgent_trials, build_inputs  # noqa: E402
from srs.rollout import sample_saccades, evaluate, valid_mask  # noqa: E402
from src.srs_load import load_net  # noqa: E402

torch.set_num_threads(1)
import os
N_TRIALS = int(os.environ.get('URG_N', 6000))
BINS = np.arange(0, 375, 25)


@torch.no_grad()
def run(net, tab, seed):
    cfg = net.cfg
    g = torch.Generator().manual_seed(seed)
    n = len(tab['trial_type'])
    keep = {k: [] for k in ('outcome', 'loc', 'rt', 'step')}
    pools, sc = [], []
    for s in range(0, n, 500):
        sub = {k: v[s:s + 500] for k, v in tab.items()}
        inp = build_inputs(cfg, sub, 'cpu', generator=g)
        T = inp['t'].shape[0]
        out = net(inp, noise=True, generator=g)
        logits = net.hazard_logits(out['pool'], inp)
        _, step, loc, _ = sample_saccades(logits, valid_mask(cfg, inp, T), generator=g)
        ev = evaluate(cfg, inp, step, loc, generator=g)
        keep['outcome'].append(ev['outcome'].numpy()); keep['loc'].append(loc.numpy())
        keep['rt'].append(ev['rt'].numpy()); keep['step'].append(step.numpy())
        pools.append(out['pool'].numpy())
        sc.append(out['sc'].numpy().astype(np.float32))
    Tm = max(p.shape[1] for p in pools)
    pad = lambda a: np.concatenate([a, np.repeat(a[:, -1:], Tm - a.shape[1], 1)], 1) if a.shape[1] < Tm else a
    r = {k: np.concatenate(v) for k, v in keep.items()}
    r['pool'] = np.concatenate([pad(p) for p in pools])
    r['sc'] = np.concatenate([pad(p) for p in sc])
    return r


def aligned(x, onset_step, end_step, pre=5, post=30):
    """x [n, T] -> [n, pre+post] aligned on onset_step, NaN at/after end_step (saccade) and outside the trial."""
    n, T = x.shape
    out = np.full((n, pre + post), np.nan)
    for i in range(n):
        for j, s in enumerate(range(onset_step[i] - pre, onset_step[i] + post)):
            if 0 <= s < T and (end_step[i] < 0 or s < end_step[i]):
                out[i, j] = x[i, s]
    return out


def fit_logistic(x, y):
    """4-parameter logistic MLE on binary y vs x; returns lo, hi, mid, slope (ms)."""
    if len(x) < 40:
        return dict(lo=np.nan, hi=np.nan, mid=np.nan, slope=np.nan)
    def nll(p):
        lo, hi = 1 / (1 + np.exp(-p[0])), 1 / (1 + np.exp(-p[1]))
        q = lo + (hi - lo) / (1 + np.exp(-(x - p[2]) / np.exp(p[3])))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
    best = None
    for mid in (80, 130, 180, 230):
        r = minimize(nll, [-1, 2, mid, np.log(20)], method='Nelder-Mead', options={'maxiter': 3000})
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    return dict(lo=float(1 / (1 + np.exp(-p[0]))), hi=float(1 / (1 + np.exp(-p[1]))), mid=float(p[2]),
                slope=float(np.exp(p[3])))


def tacho(r, tab, m):
    rpt = r['rt'] - tab['dt_go']
    y = (r['loc'] == np.where(tab['mean1'] >= tab['mean2'], np.where(tab['id_R'] == 1, 1, 0),
                              np.where(tab['id_R'] == 2, 1, 0))).astype(float)
    ok = m & (r['outcome'] == 0) & np.isfinite(rpt)
    f = fit_logistic(rpt[ok], y[ok])
    curve = [float(np.mean(y[ok & (rpt >= a) & (rpt < a + 25)])) if np.sum(ok & (rpt >= a) & (rpt < a + 25)) >= 15
             else None for a in BINS]
    f['curve'] = curve
    f['n'] = int(ok.sum())
    return f


def analyse(net, seed=0):
    cfg = net.cfg
    rng = np.random.default_rng(seed)
    tab = make_urgent_trials(cfg, N_TRIALS, rng, 'S3')
    tab['engaged'] = np.ones(N_TRIALS)
    tab['side_bias'] = np.zeros(N_TRIALS)
    out = {}
    sal0 = cfg.sal_gain
    for sg in (1.0, 0.3, 2.0):
        cfg.sal_gain = sg
        r = run(net, tab, seed + 11)
        ch = tab['trial_type'] == 1
        rich_R = np.where(tab['mean1'] >= tab['mean2'], tab['id_R'] == 1, tab['id_R'] == 2)
        sal_R = tab['sal_R'] > tab['sal_L']
        conflict = ch & (rich_R != sal_R)
        congruent = ch & (rich_R == sal_R)
        key = f'sal{sg:g}'
        out[key] = dict(conflict=tacho(r, tab, conflict), congruent=tacho(r, tab, congruent),
                        completed=float(np.mean(r['outcome'] == 0)), inaccurate=float(np.mean(r['outcome'] == 3)),
                        fixbreak=float(np.mean(r['outcome'] == 1)),
                        rt_median=float(np.nanmedian(r['rt'][r['outcome'] == 0])))
        if sg != 1.0:
            continue
        gap = np.abs(tab['mean1'] - tab['mean2'])
        q = np.quantile(gap[conflict], [1 / 3, 2 / 3])
        out['by_gap'] = [dict(gap_lo=float(lo), gap_hi=float(hi), **tacho(r, tab, conflict & (gap >= lo) & (gap < hi)))
                         for lo, hi in ((0, q[0]), (q[0], q[1]), (q[1], np.inf))]
        dt_ms = cfg.dt
        on_step = np.round(tab['fix_dur'] / dt_ms).astype(int)
        end = np.where(r['step'] >= 0, r['step'], -1)
        # measured value-arrival latency (single-target)
        st = tab['trial_type'] == 0
        loc = np.where(tab['id_R'] > 0, 1, 0)
        sid = np.where(tab['id_R'] > 0, tab['id_R'], tab['id_L'])
        rich_id = np.where(tab['mean1'] >= tab['mean2'], 1, 2)
        pool_shown = r['pool'][np.arange(N_TRIALS), :, loc]
        A = aligned(pool_shown, on_step, end)
        isrich = sid == rich_id
        lat = np.nan
        for j in range(5, A.shape[1] - 2):
            ts = []
            for jj in (j, j + 1, j + 2):
                a, b = A[st & isrich, jj], A[st & ~isrich, jj]
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                ts.append((a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b) + 1e-12)
                          if len(a) > 20 and len(b) > 20 else 0)
            if min(ts) > 3:
                lat = (j - 5) * dt_ms
                break
        out['value_latency_ms'] = float(lat)
        # divergence on conflict trials per dt
        rich_loc = rich_R.astype(int)
        d = r['pool'][np.arange(N_TRIALS), :, rich_loc] - r['pool'][np.arange(N_TRIALS), :, 1 - rich_loc]
        D = aligned(d, on_step, end)
        div = {}
        for dtv in np.unique(tab['dt_go']):
            mm = conflict & (tab['dt_go'] == dtv)
            mu = np.nanmean(D[mm], 0)
            cnt = np.sum(np.isfinite(D[mm]), 0)
            t_cross = np.nan
            for j in range(6, D.shape[1] - 2):
                if cnt[j + 2] >= 20 and np.all(mu[j:j + 3] > 0) and np.any(mu[5:j] <= 0):
                    t_cross = (j - 5) * dt_ms
                    break
            div[str(int(dtv))] = float(t_cross)
        out['divergence_ms_by_dt'] = div
        x = np.array([float(k) for k in div]); y = np.array(list(div.values()))
        okd = np.isfinite(y)
        out['divergence_slope_vs_dt'] = float(np.polyfit(x[okd], y[okd], 1)[0]) if okd.sum() >= 3 else np.nan
        # error trials
        sac_after = (r['rt'] - tab['dt_go']) > 150
        err = conflict & (r['outcome'] == 0) & (r['loc'] != rich_loc) & sac_after
        cor = conflict & (r['outcome'] == 0) & (r['loc'] == rich_loc) & sac_after
        Rp = aligned(r['pool'][np.arange(N_TRIALS), :, rich_loc], on_step, end)
        w = slice(5 + 10, 5 + 15)
        out['error_rich_ratio'] = float(np.nanmean(Rp[err, w]) / np.nanmean(Rp[cor, w])) if err.sum() > 20 else np.nan
        out['n_capture_errors'] = int(err.sum())
        # S9
        v = np.where(sid == 1, tab['V1'], tab['V2'])
        ok = st & (r['outcome'] == 0) & np.isfinite(r['rt'])
        out['S9'] = float(spearmanr(v[ok], r['rt'][ok])[0])
        # S8 time-resolved: units contralateral to the shown target (SC side 0 = LEFT SC sees location 1)
        meta = net.idx.meta()
        side = np.where(meta['sc_side'] == 'L', 1, 0)               # location each unit's SC represents
        s8 = {}
        for w0 in (50, 100, 150, 200):
            a, b = 5 + w0 // 10, 5 + (w0 + 50) // 10
            vals = []
            for u in range(r['sc'].shape[2]):
                m_u = st & (loc == side[u])
                X = aligned(r['sc'][:, :, u], on_step, end)[:, a:b]
                rate = np.nanmean(X, 1)
                cells = []
                for rid in (1, 2):
                    m1 = m_u & (rich_id == rid) & (sid == 1) & np.isfinite(rate)
                    m2 = m_u & (rich_id == rid) & (sid == 2) & np.isfinite(rate)
                    if m1.sum() < 10 or m2.sum() < 10:
                        break
                    cells.append((rate[m1].mean(), rate[m2].mean()))
                if len(cells) < 2:
                    continue
                vals.append(cells)
            if not vals:
                s8[str(w0)] = np.nan
                continue
            arr = np.array(vals)                                       # [units, 2 blocks, 2 ids]
            peak = arr.max((1, 2))
            gain = 15.0 / np.maximum(np.median(peak[peak > 0]) if np.any(peak > 0) else 1, 1e-6)
            mapped = 3.0 + gain * arr
            delta = np.log(mapped[:, :, 0] / mapped[:, :, 1]).mean(1)
            s8[str(w0)] = float(np.mean(np.abs(delta) > 0.2))
        out['S8_frac_by_window'] = s8
    cfg.sal_gain = sal0
    return out


def main():
    res_dir = Path(sys.argv[1])
    pat = sys.argv[2] if len(sys.argv) > 2 else '*'
    out_dir = res_dir.parent / 'analysis3'
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in sorted(res_dir.glob(pat)):
        if not (d / 'model.pt').exists() or (out_dir / f'{d.name}.json').exists():
            continue
        log = json.load(open(d / 'log.json'))
        net = load_net(str(d / 'model.pt'))
        o = dict(name=d.name, arch=net.cfg.arch, up_latency=net.cfg.up_latency,
                 id_sc_latency=net.cfg.id_sc_latency, urgency_locus=net.cfg.urgency_locus, final=log[-1],
                 **analyse(net))
        json.dump(o, open(out_dir / f'{d.name}.json', 'w'), indent=1)
        print(d.name, 'done', flush=True)


if __name__ == '__main__':
    main()
