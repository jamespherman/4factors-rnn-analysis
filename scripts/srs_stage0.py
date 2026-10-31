#!/usr/bin/env python3
"""
Stage 0a (plans/2026-10-31_hub_resolver_models_plan.md): hand-built A (resolver), B (priority hub) and C (outcome hub)
networks (conflict-task-RNN code/srs, snapshot in external/conflict_srs), calibrated to the monkey's 08-20 behaviour, then
signature statistics computed on many simulated sessions at the recorded unit and trial counts. Output: per-architecture
distributions and the expected separation (|log LR|) of each signature.

Signatures (Python approximations of bilateral-SC-recordings' analyses; the exact pipeline is run later on exports):
  S1  L-R population signal ~ reward difference (R-L, per 100 ms) + choice + pre-target state, choice trials:
      t of the reward-difference coefficient (delay and pre-saccadic windows).
  S2  per unit (SC contralateral to its target): delay counts ~ own-target reward + competitor reward + chose-own
      (Poisson GLM); population mean competitor coefficient (fractional change per 100 ms) and its t across units.
  S7  single-target trials, units contralateral to the shown target: counts ~ belief value of the shown identity
      (Poisson GLM; per 100 ms); population mean coefficient and t (delay and pre-saccadic windows).
  S8  single-target trials, same units: counts ~ identity + belief value; population mean |identity| effect beyond
      value (z of the identity coefficient, averaged over units) - identity coding in SC.
  S8 (revised after bilateral-SC-recordings' review): GLM counts ~ value + identity + identity x block (rich identity);
      excess of mean |z| over a permutation null (identity shuffled within block x salience) for the identity main effect
      (S8_excess: includes purely sensory shape coding) and for identity x block beyond value (S8int_excess: binding).
      Counts are overdispersed (trial-level lognormal gain, SD OVERDISP) to match recorded variability; z values are
      quasi-Poisson (dispersion from Pearson X2), as bilateral-SC-recordings reports for the recordings (0.54-0.62).
  S9  single-target RT vs belief value of the shown identity (Spearman; behavioural; recorded 08-20: -0.21).
Rates (softplus units) are mapped per unit to recorded-like spike rates: r = base + gain * r_model with base ~ lognormal
(median 3 sp/s) and gain set so the unit's peak trial-averaged modulation ~ lognormal (median 15 sp/s); counts Poisson.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'external' / 'conflict_srs'))
from srs import SRSConfig, build_handbuilt, simulate  # noqa: E402
from srs.task import from_recorded  # noqa: E402
sys.path.insert(0, str(ROOT)); from src.srs_load import load_net  # noqa: E402

torch.set_num_threads(1)
OUT = ROOT / 'results' / 'hub_models' / 'stage0'
BELIEF = ROOT / 'results' / 'hub_models' / 'belief' / '08-20_trials_with_belief.csv'
UNITS = {'L': (19, 6), 'R': (27, 0)}           # 08-20 recorded units per SC and classic (pool-E) count
DT = 10
OVERDISP = 0.4          # SD of log trial gain per unit (Fano ~1.2-1.5 at recorded rates); 0 = Poisson
N_PERM = 10
SHAPE_VIS = 0.25        # block-invariant sensory shape coding in every architecture (hand-builds)


def monkey_targets(tab):
    ch = (tab['trial_type'] == 1) & (tab['monkey_completed'] == 1) & np.isin(tab['monkey_choice_loc'], [0, 1])
    rich = (tab['rew_R'] > tab['rew_L']).astype(int)
    rt = tab['monkey_rt_ms'][ch]
    return {'p_right': float(np.mean(tab['monkey_choice_loc'][ch] == 1)),
            'p_high': float(np.mean(tab['monkey_choice_loc'][ch] == rich[ch])),
            'rt': float(np.nanmedian(rt))}


def behaviour(res):
    t = res['table']
    ch = (t['outcome'] == 0) & (t['trial_type'] == 1)
    rich = (t['rew_R'] > t['rew_L']).astype(int)
    return {'p_right': float(np.mean(t['choice_loc'][ch] == 1)), 'p_high': float(np.mean(t['choice_loc'][ch] == rich[ch])),
            'rt': float(np.nanmedian(t['rt'][ch])), 'completed': float(np.mean(t['outcome'] == 0))}


def with_nuisance(tab, w, bias, rng):
    t = {k: np.array(v, copy=True) for k, v in tab.items()}
    t['engaged'] = (rng.random(len(t['V1'])) < w).astype(float)
    t['side_bias'] = np.full(len(t['V1']), bias)
    return t


def make_net(arch, hb):
    """Hand-built network for arch 'A'|'B'|'C', or a trained network if hb has 'path'."""
    if 'path' in hb:
        return load_net(hb['path'], device='cpu')
    return build_handbuilt(arch, SRSConfig(device='cpu', shape_vis=SHAPE_VIS), **hb)


def calibrate(arch, tab, target, hb_grid, seeds=3):
    best = None
    for hb in hb_grid:
      ws = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6) if 'path' not in hb else (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
      bs = (-0.3, -0.15, 0.0, 0.15, 0.3) if 'path' not in hb else tuple(np.round(np.arange(-0.3, 0.31, 0.075), 3))
      for w in ws:
        for bias in bs:
            b = []
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                net = make_net(arch, hb)
                b.append(behaviour(simulate(net, with_nuisance(tab, w, bias, rng), seed=s)))
            m = {k: np.mean([x[k] for x in b]) for k in b[0]}
            loss = ((m['p_right'] - target['p_right']) ** 2 + (m['p_high'] - target['p_high']) ** 2
                    + 0.5 * (1 - m['completed']) + ((m['rt'] - target['rt']) / 200) ** 2)
            if best is None or loss < best[0]:
                best = (loss, w, bias, m, hb)
    return best


def spike_counts(res, rng):
    """Sample recorded-like units and return per-trial windowed Poisson counts."""
    t, sc, units = res['table'], res['rates']['sc'], res['units']
    side, typ = np.asarray(units['sc_side']), np.asarray(units['type'])
    picks = []
    for s, (n, n_pool) in UNITS.items():
        pool = np.flatnonzero((side == s) & (typ == 'E_pool'))
        other = np.flatnonzero((side == s) & (typ != 'E_pool'))
        picks += [(s, u) for u in rng.choice(pool, n_pool, replace=False)]
        picks += [(s, u) for u in rng.choice(other, n - n_pool, replace=False)]
    idx = np.array([u for _, u in picks])
    sides = np.array([s for s, _ in picks])
    r = sc[:, :, idx].astype(np.float64)
    pad = np.isnan(r)
    tavg = np.nanmean(r, 0)
    base = np.exp(np.log(3) + 0.6 * rng.standard_normal(len(idx)))
    mod = np.exp(np.log(15) + 0.7 * rng.standard_normal(len(idx)))
    gain = mod / np.maximum(np.nanmax(tavg, 0) - np.nanmin(tavg, 0), 1e-3)
    rate = base + gain * (r - np.nanmin(tavg, 0))                           # sp/s
    rate[pad] = 0.0                                                         # after the trial ends
    if OVERDISP > 0:                                                        # trial-level gain noise (overdispersion)
        g = np.exp(OVERDISP * rng.standard_normal((rate.shape[0], 1, rate.shape[2])) - OVERDISP ** 2 / 2)
        rate = rate * g
    tgt = np.where(np.isnan(t['onset_step_L']), t['onset_step_R'], np.fmin(t['onset_step_L'], t['onset_step_R']))
    sac = t['go_step'] + np.round(np.nan_to_num(t['rt'], nan=0) / DT)
    T = r.shape[1]

    def window(ref, a, b):
        out = np.zeros((len(ref), len(idx)))
        for k, x in enumerate(ref):
            lo, hi = int(x + a // DT), int(x + b // DT)
            lo, hi = max(lo, 0), min(hi, T)
            if hi > lo:
                out[k] = rng.poisson(np.clip(rate[k, lo:hi].sum(0), 0, None) * DT / 1000)
            else:
                out[k] = np.nan
        return out
    return {'pre': window(tgt, -300, 0), 'delay': window(tgt, 100, 450), 'presac': window(sac, -100, 0),
            'sides': sides}


def signatures(res, cnt):
    t = res['table']
    out = {}
    comp = t['outcome'] == 0
    ch = comp & (t['trial_type'] == 1)
    L, R = cnt['sides'] == 'L', cnt['sides'] == 'R'
    # S1
    dr = (t['rew_R'] - t['rew_L']) / 100
    chose_r = (t['choice_loc'] == 1).astype(float)
    pre_lr = np.nanmean(cnt['pre'][:, L], 1) - np.nanmean(cnt['pre'][:, R], 1)
    for w in ('delay', 'presac'):
        y = np.nanmean(cnt[w][:, L], 1) - np.nanmean(cnt[w][:, R], 1)
        X = sm.add_constant(np.c_[dr, chose_r, pre_lr])
        ok = ch & ~np.isnan(y)
        f = sm.OLS(y[ok], X[ok]).fit()
        out[f'S1_{w}'] = f.tvalues[1]
    # S2: units on side s see own target at the contralateral location (L SC <- right target, R SC <- left target)
    coefs = []
    for side_name, own, comp_ in (('L', 'rew_R', 'rew_L'), ('R', 'rew_L', 'rew_R')):
        chose_own = (t['choice_loc'] == (1 if side_name == 'L' else 0)).astype(float)
        X = sm.add_constant(np.c_[t[own] / 100, t[comp_] / 100, chose_own])
        for u in np.flatnonzero(cnt['sides'] == side_name):
            y = cnt['delay'][:, u]
            ok = ch & ~np.isnan(y)
            if y[ok].sum() < 20:
                continue
            try:
                f = sm.GLM(y[ok], X[ok], family=sm.families.Poisson()).fit()
                coefs.append(f.params[2])
            except Exception:
                pass
    coefs = np.array(coefs)
    out['S2_mean'] = float(np.mean(coefs)) if len(coefs) else np.nan
    out['S2_t'] = float(np.mean(coefs) / (np.std(coefs, ddof=1) / np.sqrt(len(coefs)))) if len(coefs) > 2 else np.nan
    # S7, S8: single-target trials
    st = comp & (t['trial_type'] == 0)
    shown_loc = np.where(t['id_R'] > 0, 1, 0)
    shown_id = np.where(t['id_R'] > 0, t['id_R'], t['id_L'])
    v_shown = np.where(shown_id == 1, t['V1'], t['V2']) / 100
    rich = np.where(t['block_mean_t1_ms'] > t['block_mean_t2_ms'], 1.0, -1.0) if 'block_mean_t1_ms' in t else np.sign(t['V1'] - t['V2'])
    idv = np.where(shown_id == 1, 1.0, -1.0)
    sal_shown = np.where(t['id_R'] > 0, t['sal_R'], t['sal_L'])
    prng = np.random.default_rng(7)

    def z_identity(y, ok, ident):
        X = sm.add_constant(np.c_[v_shown[ok], ident[ok], ident[ok] * rich[ok]])
        f = sm.GLM(y[ok], X, family=sm.families.Poisson()).fit(scale='X2')     # quasi-Poisson (dispersion from X2)
        return abs(f.tvalues[2]), abs(f.tvalues[3])

    for w in ('delay', 'presac'):
        c7, zm, zi, nm, ni = [], [], [], [], []
        for side_name, loc in (('L', 1), ('R', 0)):
            sel = st & (shown_loc == loc)
            strata = [np.flatnonzero(sel & (rich == r) & (sal_shown == sv)) for r in (-1, 1) for sv in np.unique(sal_shown[sel])]
            for u in np.flatnonzero(cnt['sides'] == side_name):
                y = cnt[w][:, u]
                ok = sel & ~np.isnan(y)
                if y[ok].sum() < 20:
                    continue
                try:
                    f7 = sm.GLM(y[ok], sm.add_constant(v_shown[ok]), family=sm.families.Poisson()).fit()
                    c7.append(f7.params[1])
                    a, b = z_identity(y, ok, idv)
                    zm.append(a)
                    zi.append(b)
                    pa, pb = [], []
                    for _ in range(N_PERM):
                        perm = idv.copy()
                        for g in strata:
                            perm[g] = prng.permutation(perm[g])
                        x1, x2 = z_identity(y, ok, perm)
                        pa.append(x1)
                        pb.append(x2)
                    nm.append(np.mean(pa))
                    ni.append(np.mean(pb))
                except Exception:
                    pass
        c7 = np.array(c7)
        out[f'S7_{w}_t'] = float(np.mean(c7) / (np.std(c7, ddof=1) / np.sqrt(len(c7)))) if len(c7) > 2 else np.nan
        out[f'S8_{w}_absz'] = float(np.mean(zm)) if zm else np.nan
        out[f'S8_{w}_excess'] = float(np.mean(zm) - np.mean(nm)) if zm else np.nan
        out[f'S8int_{w}_excess'] = float(np.mean(zi) - np.mean(ni)) if zi else np.nan
    from scipy.stats import spearmanr
    ok = st & ~np.isnan(t['rt'])
    out['S9_rt_value_rho'] = float(spearmanr(v_shown[ok], t['rt'][ok])[0])
    return out


def main():
    global OUT
    OUT.mkdir(parents=True, exist_ok=True)
    tab = from_recorded(str(BELIEF))
    target = monkey_targets(tab)
    print('monkey 08-20 targets', target, flush=True)
    rows, cal = [], {}
    n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    trained = sys.argv[2:]                       # optional: label=path/to/model.pt ...
    if trained:
        OUT = OUT.parent / 'stage1_trained'
        OUT.mkdir(parents=True, exist_ok=True)
        specs = [(x.split('=')[0], [{'path': x.split('=')[1]}]) for x in trained]
    else:
        specs = [(a, [{}] if a != 'C' else [{'c_gain': g, 'theta': th} for g in (1.0, 1.1, 1.2) for th in (0.75, 0.85)])
                 for a in 'ABC']
    for arch, grid in specs:
        loss, w, bias, m, hb = calibrate(arch, tab, target, grid)
        cal[arch] = {'w': w, 'bias': bias, **m, **{k: v for k, v in hb.items() if k != 'path'}}
        print(arch, 'calibrated', cal[arch], flush=True)
        for d in range(n_draws):
            rng = np.random.default_rng(1000 + d)
            net = make_net(arch, hb)
            res = simulate(net, with_nuisance(tab, w, bias, rng), seed=d)
            sig = signatures(res, spike_counts(res, rng))
            rows.append({'arch': arch, 'draw': d, **behaviour(res), **sig})
        print(arch, 'draws done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'draws.csv', index=False)
    pd.DataFrame(cal).T.to_csv(OUT / 'calibration.csv')
    report(df, cal, target)


def report(df, cal, target):
    from scipy.stats import gaussian_kde
    L = ['# Stage 0a: hand-built architectures at matched behaviour (08-20 sequence, recorded unit/trial counts)\n',
         f'Monkey targets: {target}\n', '## Calibration (engaged fraction w, side bias)']
    for a, c in cal.items():
        L.append(f'- {a}: ' + ', '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in c.items()))
    L.append('\n## Signature distributions (mean +- SD over draws) and separation')
    stats = [c for c in df.columns if c.startswith('S')]
    for s in stats:
        labels = list(dict.fromkeys(df.arch))
        cells = ' | '.join(f'{a}: {df[df.arch == a][s].mean():+.2f} +- {df[df.arch == a][s].std():.2f}' for a in labels)
        sep = []
        fam = df.arch.str[0]
        for a1, a2 in (('B', 'C'), ('A', 'B'), ('A', 'C')):
            x1, x2 = df[fam == a1][s].dropna(), df[fam == a2][s].dropna()
            if len(x1) > 5 and len(x2) > 5 and x1.std() > 0 and x2.std() > 0:
                k1, k2 = gaussian_kde(x1), gaussian_kde(x2)
                ell = np.mean(np.log(k1(x1) + 1e-12) - np.log(k2(x1) + 1e-12))
                sep.append(f'{a1}v{a2} E|logLR| {abs(ell):.1f}')
        L.append(f'- {s}: {cells} || ' + ', '.join(sep))
    text = '\n'.join(L)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
