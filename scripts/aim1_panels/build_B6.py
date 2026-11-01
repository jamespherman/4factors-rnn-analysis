#!/usr/bin/env python3
"""Build the B6 data folder (bilateral Newton srsClassic behaviour, sessions 08-20 and 08-19).

Panels
  a  P(choose right) vs ACTUAL reward difference (right - left, ms), congruent vs conflict, 08-20
  b  same, 08-19 (descriptive)
  c  P(choose right) vs fitted delta-rule belief-value difference, 08-20 (secondary; circular: the belief was fitted to
     these choices, alpha at its 0.01 bound)
  d  single-target RT vs belief value of the shown identity (quintile bins), with the true block-mean value version
     (one point per block x identity) as a drift control, 08-20
  e  within-block RT difference, shown rich minus shown lean identity, one point per block (drift cancels), 08-20

Sources (read-only):
  - results/aim1_panels/_bilateral_source/competition_choice_trials_<session>.csv: MATLAB table B saved by the peer repo's
    code/run_competition_choice.m in data/bilateral/competition_choice.mat (exported by export_bilateral_tables.m);
    the same trial selection as run_competition_behaviour.m (choice trials, chosen side valid, RT >= 80 ms)
  - results/hub_models/belief/<ses>_trials_with_belief.csv (scripts/srs_belief_fit.py)
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B6.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, norm, ttest_1samp

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results' / 'aim1_panels' / '_bilateral_source'
BEL = ROOT / 'results' / 'hub_models' / 'belief'
OUT = ROOT / 'results' / 'aim1_panels' / 'B6_bilateral_behaviour'
OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
SES = {'08-20': 'Newton_08202026', '08-19': 'Newton_08192026'}
N_BOOT = 2000
rng = np.random.default_rng(20261101)
stats = []


def stat(name, value, n, test, window, unit_class, session, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, note=''):
    stats.append(dict(statistic=name, value=value, ci_lo=ci_lo, ci_hi=ci_hi, p=p, n=n, test=test, window=window,
                      unit_class=unit_class, session=session, note=note))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def logistic_curve(x, y, grid):
    """Per-condition logistic fit P(y) ~ x, with a trial-bootstrap 95% band."""
    X = sm.add_constant(x)
    f = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    pred = f.predict(sm.add_constant(grid))
    bs = np.full((N_BOOT, len(grid)), np.nan)
    for b in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        try:
            fb = sm.GLM(y[i], X[i], family=sm.families.Binomial()).fit()
            bs[b] = fb.predict(sm.add_constant(grid))
        except Exception:
            pass
    return pred, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0), f


def psychometric(df, xcol, ses, panel, xlabel_units, nbins=5):
    pts, curves = [], []
    for cond, sub in (('congruent', df[~df.conflict]), ('conflict', df[df.conflict])):
        x, y = sub[xcol].to_numpy(float), sub['choseR'].to_numpy(float)
        edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
        b = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, nbins - 1)
        for k in range(nbins):
            m = b == k
            lo, hi = wilson(y[m].sum(), m.sum())
            pts.append(dict(panel=panel, session=ses, condition=cond, bin=k + 1, **{f'{xcol}_bin_mean': x[m].mean(),
                            f'{xcol}_bin_lo': edges[k], f'{xcol}_bin_hi': edges[k + 1]}, n_trials=int(m.sum()),
                            n_chose_right=int(y[m].sum()), p_chose_right=y[m].mean(), ci95_lo=lo, ci95_hi=hi))
        grid = np.linspace(x.min(), x.max(), 60)
        pred, lo, hi, f = logistic_curve(x, y, grid)
        for g, p_, l_, h_ in zip(grid, pred, lo, hi):
            curves.append(dict(panel=panel, session=ses, condition=cond, **{xcol: g}, p_fit=p_, band95_lo=l_,
                               band95_hi=h_))
        stat(f'{panel}: per-condition logistic slope ({cond}), per {xlabel_units}', f.params[1] * 100, len(x),
             'logistic regression choseR ~ x (display fit only)', 'choice (saccade)', 'behaviour', ses,
             (f.params[1] - 1.96 * f.bse[1]) * 100, (f.params[1] + 1.96 * f.bse[1]) * 100, f.pvalues[1],
             'slope per 100 ms; separate fit per condition, drawn as the curve')
    return pd.DataFrame(pts), pd.DataFrame(curves)


# ---------------- a, b: actual reward difference (exact MATLAB trial tables) ----------------
pts_all, cur_all = [], []
for ses, key in SES.items():
    B = pd.read_csv(SRC / f'competition_choice_trials_{key}.csv')
    B['conflict'] = B['conflict'].astype(bool)
    B['reward_diff_ms'] = 100 * B['rewDiff']
    # joint model of run_competition_behaviour.m: choseR ~ salDiff + rewDiff + prevChoice
    ok = B[['salDiff', 'rewDiff', 'prevChoice']].notna().all(1)
    f = sm.GLM(B.loc[ok, 'choseR'], sm.add_constant(B.loc[ok, ['salDiff', 'rewDiff', 'prevChoice']]),
               family=sm.families.Binomial()).fit()
    for nm, lab in (('rewDiff', 'reward difference (per 100 ms)'), ('salDiff', 'salience difference (per +-1 level)'),
                    ('const', 'side bias (intercept)'), ('prevChoice', 'previous choice')):
        stat(f'choice beta: {lab}', f.params[nm], int(ok.sum()), 'logistic GLM choseR ~ salDiff + rewDiff + prevChoice '
             '(Wald CI, p)', 'choice', 'behaviour', ses, f.params[nm] - 1.96 * f.bse[nm],
             f.params[nm] + 1.96 * f.bse[nm], f.pvalues[nm], 'reproduces run_competition_behaviour.m')
    stat('n choice trials', len(B), len(B), 'count', 'choice', 'behaviour', ses, note=f'{int(B.conflict.sum())} conflict')
    # conflict RT cost (median difference, bootstrap 2,000)
    rc, rn = 1e3 * B.loc[B.conflict, 'rt'].to_numpy(), 1e3 * B.loc[~B.conflict, 'rt'].to_numpy()
    bs = [np.median(rng.choice(rc, len(rc))) - np.median(rng.choice(rn, len(rn))) for _ in range(N_BOOT)]
    stat('conflict RT cost (median conflict - congruent), ms', np.median(rc) - np.median(rn), len(B),
         'trial bootstrap 2,000 (CI)', 'RT = saccade onset - go', 'behaviour', ses, *np.percentile(bs, [2.5, 97.5]),
         note='peer note: +16 ms [+0.1, +25] (08-20)')
    p, c = psychometric(B, 'reward_diff_ms', ses, 'a' if ses == '08-20' else 'b', '100 ms')
    pts_all.append(p); cur_all.append(c)
pd.concat(pts_all).to_csv(OUT / 'B6_ab_psychometric_reward_points.csv', index=False)
pd.concat(cur_all).to_csv(OUT / 'B6_ab_psychometric_reward_fit.csv', index=False)

# ---------------- c: belief-value difference (secondary) ----------------
d = pd.read_csv(BEL / '08-20_trials_with_belief.csv')
ch = ((d.n_targets == 2) & d.outcome.isin(['CHOSE_HIGH_SAL', 'CHOSE_LOW_SAL']) & (d.rt_s >= 0.08)
      & d.chosen_side.isin([1, 2]))
c = d[ch].copy()
rI = c.t1_side == 1
c['belief_diff_ms'] = np.where(rI, c.belief_t1_ms - c.belief_t2_ms, c.belief_t2_ms - c.belief_t1_ms)
c['choseR'] = (c.chosen_side == 1).astype(float)
c['conflict'] = c.is_conflict == 1
p, cu = psychometric(c, 'belief_diff_ms', '08-20', 'c', '100 ms belief')
p.to_csv(OUT / 'B6_c_psychometric_belief_points.csv', index=False)
cu.to_csv(OUT / 'B6_c_psychometric_belief_fit.csv', index=False)
stat('belief fit learning rate alpha (delta model, best by NLL)', 0.010, len(c), 'max likelihood, bounded [0.01, 1]',
     'choice', 'behaviour', '08-20', note='at the lower bound; beta 20 at its upper bound; lapse 0.5 at bound '
     '(results/hub_models/belief/summary.md)')

# ---------------- d, e: single-target RT vs value ----------------
s_rows, bm_rows, e_rows = [], [], []
for ses in SES:
    d = pd.read_csv(BEL / f'{ses}_trials_with_belief.csv')
    st = (d.n_targets == 1) & d.outcome.isin(['SINGLE_T1_CORRECT', 'SINGLE_T2_CORRECT']) & (d.rt_s >= 0.08)
    s = d[st].copy()
    t1 = s.outcome == 'SINGLE_T1_CORRECT'
    s['shown_identity'] = np.where(t1, 'T1', 'T2')
    s['belief_value_ms'] = np.where(t1, s.belief_t1_ms, s.belief_t2_ms)
    s['block_mean_value_ms'] = np.where(t1, s.block_mean_t1_ms, s.block_mean_t2_ms)
    s['other_block_mean_ms'] = np.where(t1, s.block_mean_t2_ms, s.block_mean_t1_ms)
    s['shown_rich'] = (s.block_mean_value_ms > s.other_block_mean_ms).astype(int)
    s['rt_ms'] = 1e3 * s.rt_s
    n = len(s)
    r, pv = spearmanr(s.belief_value_ms, s.rt_ms)
    stat('Spearman rho, single-target RT vs belief value of shown identity', r, n, 'Spearman', 'RT (saccade - go)',
         'behaviour', ses, p=pv, note='write-up: -0.21 (08-20)')
    r2, p2 = spearmanr(s.block_mean_value_ms, s.rt_ms)
    stat('Spearman rho, single-target RT vs TRUE block-mean value of shown identity (drift control)', r2, n,
         'Spearman', 'RT (saccade - go)', 'behaviour', ses, p=p2)
    r3, p3 = spearmanr(s.trial_row, s.belief_value_ms)
    stat('Spearman rho, belief value vs trial index (slow alpha = 0.01 belief drifts)', r3, n, 'Spearman', '-',
         'behaviour', ses, p=p3)
    r4, p4 = spearmanr(s.trial_row, s.rt_ms)
    stat('Spearman rho, single-target RT vs trial index', r4, n, 'Spearman', '-', 'behaviour', ses, p=p4)
    m = smf.ols('rt_ms ~ belief_value_ms + C(conflict_block)', s).fit()
    stat('RT slope on belief value within block (OLS with block fixed effects), ms per 100 ms', 100 *
         m.params['belief_value_ms'], n, 'OLS, block dummies', 'RT', 'behaviour', ses,
         100 * m.conf_int().loc['belief_value_ms', 0], 100 * m.conf_int().loc['belief_value_ms', 1],
         m.pvalues['belief_value_ms'])
    # e: within-block rich - lean
    g = s.groupby(['conflict_block', 'shown_rich']).agg(rt_ms_median=('rt_ms', 'median'), n=('rt_ms', 'size'),
                                                       value=('block_mean_value_ms', 'first')).reset_index()
    for blk, gg in g.groupby('conflict_block'):
        a, b = gg[gg.shown_rich == 1].iloc[0], gg[gg.shown_rich == 0].iloc[0]
        e_rows.append(dict(panel='e', session=ses, conflict_block=int(blk),
                           block_type=int(s.loc[s.conflict_block == blk, 'block_type'].iloc[0]),
                           rich_value_ms=a.value, lean_value_ms=b.value, n_rich=int(a.n), n_lean=int(b.n),
                           rt_median_rich_ms=a.rt_ms_median, rt_median_lean_ms=b.rt_ms_median,
                           rt_diff_rich_minus_lean_ms=a.rt_ms_median - b.rt_ms_median))
    diffs = np.array([x['rt_diff_rich_minus_lean_ms'] for x in e_rows if x['session'] == ses])
    tt = ttest_1samp(diffs, 0)
    bsm = [np.mean(rng.choice(diffs, len(diffs))) for _ in range(N_BOOT)]
    stat('within-block RT difference, shown rich - shown lean identity (mean over blocks), ms', diffs.mean(),
         len(diffs), 'one-sample t over blocks (p); block bootstrap CI', 'RT', 'behaviour', ses,
         *np.percentile(bsm, [2.5, 97.5]), tt.pvalue, 'median RT per block x identity; negative = rich faster')
    # d: belief quintiles and block-mean points
    q = np.quantile(s.belief_value_ms, np.linspace(0, 1, 6))
    b = np.clip(np.searchsorted(q, s.belief_value_ms, side='right') - 1, 0, 4)
    for k in range(5):
        mm = b == k
        x = s.rt_ms[mm]
        s_rows.append(dict(panel='d', session=ses, version='belief value (delta rule, alpha 0.01)', bin=k + 1,
                           value_ms_mean=s.belief_value_ms[mm].mean(), value_ms_lo=q[k], value_ms_hi=q[k + 1],
                           n_trials=int(mm.sum()), rt_ms_mean=x.mean(), rt_ms_sem=x.std(ddof=1) / np.sqrt(len(x))))
    for (blk, ident), gg in s.groupby(['conflict_block', 'shown_identity']):
        x = gg.rt_ms
        bm_rows.append(dict(panel='d', session=ses, version='true block-mean value (drift control)',
                            conflict_block=int(blk), shown_identity=ident, value_ms=gg.block_mean_value_ms.iloc[0],
                            n_trials=len(x), rt_ms_mean=x.mean(), rt_ms_sem=x.std(ddof=1) / np.sqrt(len(x))))
    s[['session' if False else 'trial_row', 'conflict_block', 'block_type', 'shown_identity', 'belief_value_ms',
       'block_mean_value_ms', 'shown_rich', 'rt_ms']].assign(session=ses).to_csv(
        OUT / f'B6_d_single_target_trials_{ses}.csv', index=False)
pd.DataFrame(s_rows).to_csv(OUT / 'B6_d_rt_vs_belief_bins.csv', index=False)
pd.DataFrame(bm_rows).to_csv(OUT / 'B6_d_rt_vs_blockmean_points.csv', index=False)
pd.DataFrame(e_rows).to_csv(OUT / 'B6_e_within_block_rich_minus_lean.csv', index=False)

# hub/resolver S9 values (same statistic in the trained networks; sig_v2), for README context
sig = []
for f in sorted((ROOT / 'results' / 'hub_models' / 'sig_v2').glob('*/draws.csv')):
    dd = pd.read_csv(f)
    sig.append(dict(network=dd.arch.iloc[0], architecture=dd.arch.iloc[0].split('_')[0],
                    S9_rt_value_rho_mean=dd.S9_rt_value_rho.mean(), S9_sd=dd.S9_rt_value_rho.std(), n_draws=len(dd)))
pd.DataFrame(sig).to_csv(OUT / 'B6_model_S9_sig_v2_per_network.csv', index=False)
pd.DataFrame(stats).to_csv(OUT / 'stats.csv', index=False)
print(pd.DataFrame(stats)[['statistic', 'session', 'value', 'ci_lo', 'ci_hi', 'p', 'n']].to_string())
