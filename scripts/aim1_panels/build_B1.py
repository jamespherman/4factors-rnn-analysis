#!/usr/bin/env python3
"""
Build B1 (Aim 1 panels): reward cue x location prior in saccade RT and peak velocity, 4factors task.

Source: results/competition/p1_behaviour/trials.csv (scripts/cp_p1_behaviour.py: successful gSac_4factors trials,
trialEndState 21 from session_data.mat, RT = saccade onset - fixation off > 80 ms; 7 sessions).

Outputs (results/aim1_panels/B1_reward_x_prior/):
  B1_sessions.csv   per session x reward x location class: mean RT, mean peak velocity, n trials, and the change from
                    that session's low-reward expected-location cell (RT in %, PV in deg/s)
  B1_cells.csv      per monkey x reward x location class: mean over sessions of the change, t-based 95% CI over
                    sessions, mean absolute RT / PV, n
  B1_null.csv       no-interaction (additive) null for the high-reward line: low-reward line + additive reward effect
  stats.csv         every statistic quoted in the README / figure
Run: OMP_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B1.py
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / 'results' / 'competition' / 'p1_behaviour' / 'trials.csv'
OUT = REPO / 'results' / 'aim1_panels' / 'B1_reward_x_prior'
CLASSES = ['high', 'neighbour', 'opposite']
LABEL = {'high': 'expected', 'neighbour': 'neighbour_90deg', 'opposite': 'opposite_180deg'}
PROB = 'C(prob, Treatment("high"))'


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
    d = pd.read_csv(SRC)
    d['loc_class'] = d['prob'].map(LABEL)

    # ---- session-level cells -------------------------------------------------------------------------------
    rows = []
    for s, g in d.groupby('session'):
        ref = g[(g.reward == 0) & (g.prob == 'high')]
        rt0, pv0 = ref.rt.mean(), ref.pv.mean()
        for rw in (0, 1):
            for c in CLASSES:
                x = g[(g.reward == rw) & (g.prob == c)]
                rows.append(dict(session=s, monkey=g.monkey.iloc[0], reward_cue='high' if rw else 'low',
                                 loc_class=LABEL[c], n_trials=len(x), rt_ms_mean=x.rt.mean(), pv_deg_s_mean=x.pv.mean(),
                                 rt_pct_change_vs_low_expected=100 * (x.rt.mean() / rt0 - 1),
                                 pv_deg_s_change_vs_low_expected=x.pv.mean() - pv0))
    ses = pd.DataFrame(rows)
    ses.to_csv(OUT / 'B1_sessions.csv', index=False, float_format='%.4f')

    cells = []
    for (m, rw, c), g in ses.groupby(['monkey', 'reward_cue', 'loc_class']):
        for meas, col in [('rt_pct', 'rt_pct_change_vs_low_expected'), ('pv_deg_s', 'pv_deg_s_change_vs_low_expected')]:
            mu, lo, hi = tci(g[col])
            cells.append(dict(monkey=m, measure=meas, reward_cue=rw, loc_class=c, change_mean=mu, ci95_lo=lo,
                              ci95_hi=hi, n_sessions=len(g), n_trials=int(g.n_trials.sum()),
                              abs_mean_over_sessions=g['rt_ms_mean' if meas == 'rt_pct' else 'pv_deg_s_mean'].mean()))
    cells = pd.DataFrame(cells)
    cells.to_csv(OUT / 'B1_cells.csv', index=False, float_format='%.4f')

    # ---- additive null + statistics ------------------------------------------------------------------------
    st, nulls = [], []

    def add(name, value, n, test, p=np.nan, ci_lo=np.nan, ci_hi=np.nan, monkey='', note=''):
        st.append(dict(statistic=name, monkey=monkey, value=value, ci95_lo=ci_lo, ci95_hi=ci_hi, p=p, n=n, test=test,
                       window='RT: saccade onset - fixation off; PV: saccade peak velocity', unit_class='behaviour',
                       session_filter='7 sessions (Feynman 08_12-08_15, Newton 08_13-08_15); successful trials '
                                      '(trialEndState 21), RT > 80 ms', note=note))

    for m in ['Feynman', 'Newton']:
        x = d[d.monkey == m]
        ncl = x.cluster.nunique()
        for y, meas, scale in [('log_rt', 'rt_pct', 100.0), ('pv', 'pv_deg_s', 1.0)]:
            unit = '% (100 x log RT)' if y == 'log_rt' else 'deg/s'
            # additive (no-interaction) model: reward effect common to all location classes
            fa = smf.ols(f'{y} ~ C(session) + reward + {PROB} + C(stim)', x).fit(cov_type='cluster', cov_kwds={'groups': x['cluster']})
            b = fa.params['reward']
            badd = 100 * (np.exp(b) - 1) if y == 'log_rt' else b
            add(f'{meas}: additive reward effect (no-interaction model)', badd, len(x),
                f'OLS {y} ~ session + reward + prob + stim, cluster-robust SE by block run ({ncl} clusters)',
                fa.pvalues['reward'], monkey=m, note=unit)
            low = cells[(cells.monkey == m) & (cells.measure == meas) & (cells.reward_cue == 'low')].set_index('loc_class')
            for c in CLASSES:
                if y == 'log_rt':   # combine multiplicatively: (1 + low%) * (1 + reward%) - 1
                    v = 100 * ((1 + low.loc[LABEL[c], 'change_mean'] / 100) * (1 + badd / 100) - 1)
                else:
                    v = low.loc[LABEL[c], 'change_mean'] + badd
                nulls.append(dict(monkey=m, measure=meas, reward_cue='high', loc_class=LABEL[c], null_change=v,
                                  null_name='no-interaction (additive) null: low-reward cell + additive reward effect'))
            # full interaction model (as cp_p1_behaviour.py)
            ff = smf.ols(f'{y} ~ C(session) + reward * {PROB} + reward * C(stim, Treatment("bull_lo")) '
                         f'+ {PROB} : C(stim, Treatment("bull_lo"))', x).fit(cov_type='cluster', cov_kwds={'groups': x['cluster']})
            for c in ['neighbour', 'opposite']:
                k = f'reward:{PROB}[T.{c}]'
                ci = ff.conf_int().loc[k] * scale
                add(f'{meas}: reward x {LABEL[c]} interaction', ff.params[k] * scale, len(x),
                    'OLS full P1 model (cp_p1_behaviour.py), cluster-robust SE by block run', ff.pvalues[k],
                    ci[0], ci[1], monkey=m, note=unit + ('; log-RT x100' if y == 'log_rt' else ''))
            # simple effects within each reward cue: location class vs expected
            for rw in (0, 1):
                xr = x[x.reward == rw]
                fr = smf.ols(f'{y} ~ C(session) + {PROB} + C(stim)', xr).fit(cov_type='cluster', cov_kwds={'groups': xr['cluster']})
                fo = smf.ols(f'{y} ~ C(session) + {PROB} + C(stim)', xr).fit()
                for c in ['neighbour', 'opposite']:
                    k = f'{PROB}[T.{c}]'
                    v = 100 * (np.exp(fr.params[k]) - 1) if y == 'log_rt' else fr.params[k]
                    ci = fr.conf_int().loc[k]
                    ci = 100 * (np.exp(ci) - 1) if y == 'log_rt' else ci
                    add(f'{meas}: {LABEL[c]} vs expected, {"high" if rw else "low"} reward cue', v, len(xr),
                        'OLS ~ session + prob + stim within reward cue; p cluster-robust by block run '
                        f'(uncorrected OLS p = {fo.pvalues[k]:.2g})', fr.pvalues[k], ci[0], ci[1], monkey=m,
                        note=unit + (f'; exp(b)-1; 100 x log ratio = {100 * fr.params[k]:+.2f}' if y == 'log_rt' else ''))
        # session-level paired test of the crossover (opposite - expected, high minus low reward)
        sx = ses[ses.monkey == m].pivot_table(index='session', columns=['reward_cue', 'loc_class'],
                                              values='rt_pct_change_vs_low_expected')
        dd = (sx[('high', 'opposite_180deg')] - sx[('high', 'expected')]) - (sx[('low', 'opposite_180deg')] - sx[('low', 'expected')])
        mu, lo, hi = tci(dd)
        add('rt_pct: (opposite - expected) high minus low reward, session level', mu, len(dd),
            f'one-sample t over sessions (df={len(dd) - 1})', stats.ttest_1samp(dd, 0).pvalue, lo, hi, monkey=m,
            note='% points; ' + ', '.join(f'{v:+.1f}' for v in dd))
    pd.DataFrame(nulls).to_csv(OUT / 'B1_null.csv', index=False, float_format='%.4f')
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    st = pd.DataFrame(st)
    st['source'] = f'{SRC.relative_to(REPO).as_posix()} (git {commit})'
    st.to_csv(OUT / 'stats.csv', index=False, float_format='%.6g')
    print(cells.to_string())
    print(st[['statistic', 'monkey', 'value', 'ci95_lo', 'ci95_hi', 'p', 'n']].to_string())


if __name__ == '__main__':
    main()
