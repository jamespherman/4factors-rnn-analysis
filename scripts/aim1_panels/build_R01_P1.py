#!/usr/bin/env python3
"""
Build R01 panel P1: reward cue x expectation PSTH, classic (E) units, target at the preferred location.

Unit/trial selection identical to scripts/cp_p3_interactions.py / cp_p4_timing.py: 6 diagonal-target sessions (Newton
08_15 excluded), verified successful trials, unit-stable trials with the target at the unit's preferred contralateral
location P (g1_geometry), valid block (hp > 0), >= 40 such trials, and the unit saw both block types (P = export
loc 1; this is the n = 50 E units of the P3 reward:expected cell).
Cells: reward cue (rewardDuration > 200 = high) x expected (P is the block's high-probability location).
PSTH: 20-ms bins, -300..+300 ms re target onset, rate minus the unit's pre-target baseline (-300..0 ms, mean over the
same selected trials), then mean +- SEM across units.
Statistics are the original ones, recomputed and checked:
  * P3 visual window (50-250 ms): reward:expected standardised OLS coefficient, t-test across units
    (checked against results/competition/p3_interactions/units.csv).
  * P4 time course (results/competition/p4_timing/timecourses.npz): pointwise t-test across units of the
    reward:expected coefficient in 40-ms windows (10-ms steps); windows with p < 0.01 are marked (uncorrected).
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_R01_P1.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, counts  # noqa: E402
from src.competition import trial_table  # noqa: E402

OUT = REPO / 'results' / 'aim1_panels' / 'R01_core' / 'P1_reward_x_expectation_psth'
G1 = REPO / 'results' / 'intercollicular' / 'g1_geometry' / 'rows.npy'
P3 = REPO / 'results' / 'competition' / 'p3_interactions' / 'units.csv'
P4 = REPO / 'results' / 'competition' / 'p4_timing' / 'timecourses.npz'
BIN = 0.02
EDGES = np.round(np.arange(-0.3, 0.3001, BIN), 4)
CELLS = {('high', 'expected'): (1, 1), ('high', 'unexpected'): (1, 0), ('low', 'expected'): (0, 1),
         ('low', 'unexpected'): (0, 0)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
    g1 = {(r['session'], r['id']): r for r in np.load(G1, allow_pickle=True)}
    urows, prow = [], []
    for s in ALL_SESSIONS:
        if s == '08_15':
            continue
        t, tt = session_table(s), trial_table(s)
        n = len(t['loc'])
        hp, rew = np.full(n, -1.0), np.full(n, -1.0)
        m = tt['export_idx'] >= 0
        hp[tt['export_idx'][m]] = tt['high_p_loc'][m]
        rew[tt['export_idx'][m]] = tt['reward_dur'][m] > 200
        on = t['ev']['targetOn']
        for j, i in enumerate(t['ids']):
            if t['type'][j] != 'E':
                continue
            P = g1[(s, i)]['P']
            sel = t['stable'][:, j] & (t['loc'] == P) & (hp > 0)
            if sel.sum() < 40 or len(np.unique(hp[sel] == P)) < 2:
                continue
            spk = t['spikes'][i]
            base = np.nanmean(counts(spk, on[sel] - 0.3, on[sel]) / 0.3)
            vis = counts(spk, on + 0.05, on + 0.25) / 0.2
            tc = np.stack([counts(spk, on + a, on + a + BIN) / BIN for a in EDGES[:-1]], 1)
            row = dict(session=s, unit_id=int(i), unit_type='E', P_loc=int(P), baseline_sp_s=base)
            for (rw, ex), (rv, ev) in CELLS.items():
                c = sel & (rew == rv) & ((hp == P) == bool(ev))
                key = f'{rw}_{ex}'
                row[f'n_trials_{key}'] = int(c.sum())
                row[f'visual_50_250_{key}_sp_s'] = np.nanmean(vis[c]) - base
                mu = np.nanmean(tc[c], 0) - base
                for b, a in enumerate(EDGES[:-1]):
                    prow.append(dict(session=s, unit_id=int(i), reward_cue=rw, target=ex,
                                     bin_center_ms=round(1000 * (a + BIN / 2)), rate_minus_baseline_sp_s=mu[b]))
            urows.append(row)
        print(s, 'done', flush=True)
    units = pd.DataFrame(urows)
    up = pd.DataFrame(prow)

    # merge the original P3 standardised coefficients (identical selection) and check the unit set
    p3 = pd.read_csv(P3)
    p3 = p3[(p3.type == 'E') & p3['visual_reward:expected'].notna()]
    units = units.merge(p3[['session', 'id', 'visual_reward:expected', 'visual_reward:expected_p']].rename(
        columns={'id': 'unit_id', 'visual_reward:expected': 'p3_visual_reward_x_expected_sd',
                 'visual_reward:expected_p': 'p3_visual_reward_x_expected_p_unit'}), on=['session', 'unit_id'],
        how='outer', indicator=True)
    assert (units._merge == 'both').all(), units[units._merge != 'both']
    units = units.drop(columns='_merge')
    units.to_csv(OUT / 'P1_units.csv', index=False, float_format='%.5g')
    up.to_csv(OUT / 'P1_unit_psth.csv', index=False, float_format='%.5g')

    pop = []
    for (rw, ex, b), g in up.groupby(['reward_cue', 'target', 'bin_center_ms']):
        x = g.rate_minus_baseline_sp_s.dropna()
        pop.append(dict(reward_cue=rw, target=ex, bin_center_ms=b, mean_sp_s=x.mean(),
                        sem_sp_s=x.std(ddof=1) / np.sqrt(len(x)), n_units=len(x)))
    pd.DataFrame(pop).to_csv(OUT / 'P1_psth.csv', index=False, float_format='%.5g')

    st, cm = [], []
    for (rw, ex) in CELLS:
        x = units[f'visual_50_250_{rw}_{ex}_sp_s'].dropna()
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        cm.append(dict(reward_cue=rw, target=ex, window='50-250 ms re target onset', mean_sp_s=x.mean(),
                       sem_sp_s=x.std(ddof=1) / np.sqrt(len(x)), ci95_lo=x.mean() - h, ci95_hi=x.mean() + h,
                       n_units=len(x), n_trials_total=int(units[f'n_trials_{rw}_{ex}'].sum())))
        st.append(dict(statistic=f'visual-window rate re baseline, {rw} reward, {ex} target', unit_class='E',
                       value=x.mean(), ci95_lo=x.mean() - h, ci95_hi=x.mean() + h, p=np.nan, n=len(x),
                       test='mean across units (t 95% CI)', window='50-250 ms re target onset'))
    pd.DataFrame(cm).to_csv(OUT / 'P1_cell_means.csv', index=False, float_format='%.5g')
    v = {(r['reward_cue'], r['target']): r['mean_sp_s'] for r in cm}
    # sp/s interaction contrast per unit (descriptive, matches the PSTH)
    d = ((units.visual_50_250_high_expected_sp_s - units.visual_50_250_high_unexpected_sp_s)
         - (units.visual_50_250_low_expected_sp_s - units.visual_50_250_low_unexpected_sp_s)).dropna()
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    st.append(dict(statistic='interaction contrast in sp/s: (high exp - high unexp) - (low exp - low unexp)',
                   unit_class='E', value=d.mean(), ci95_lo=d.mean() - h, ci95_hi=d.mean() + h,
                   p=stats.ttest_1samp(d, 0).pvalue, n=len(d), test='one-sample t across units (raw cell means, '
                   'stimulus types not balanced)', window='50-250 ms re target onset'))
    x = units.p3_visual_reward_x_expected_sd.dropna()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    st.append(dict(statistic='P3 reward x expected, standardised OLS coefficient (SD of sqrt count)', unit_class='E',
                   value=x.mean(), ci95_lo=x.mean() - h, ci95_hi=x.mean() + h, p=stats.ttest_1samp(x, 0).pvalue,
                   n=len(x), test='one-sample t across units (original P3 statistic)', window='50-250 ms re target onset'))
    z = np.load(P4)
    c, tms = z['E_reward:expected'], z['times']
    r = stats.ttest_1samp(c, 0, nan_policy='omit')
    tcrows = []
    for k, tm in enumerate(tms):
        x = c[:, k][~np.isnan(c[:, k])]
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        tcrows.append(dict(window_center_ms=round(tm), window_start_ms=round(tm) - 20, window_end_ms=round(tm) + 20,
                           mean_sd=x.mean(), ci95_lo=x.mean() - h, ci95_hi=x.mean() + h, p=float(r.pvalue[k]),
                           sig_p01=bool(r.pvalue[k] < 0.01), n_units=len(x)))
    tcd = pd.DataFrame(tcrows)
    tcd.to_csv(OUT / 'P1_interaction_timecourse.csv', index=False, float_format='%.5g')
    k = int(tcd.mean_sd.idxmax())
    st.append(dict(statistic='P4 reward x expected time course: peak', unit_class='E', value=tcd.mean_sd[k],
                   ci95_lo=tcd.ci95_lo[k], ci95_hi=tcd.ci95_hi[k], p=tcd.p[k], n=int(tcd.n_units[k]),
                   test='one-sample t across units (pointwise)', window=f'40-ms window centred {tcd.window_center_ms[k]} ms'))
    sig = tcd[tcd.sig_p01]
    st.append(dict(statistic='P4 windows with pointwise p < 0.01 (centres, ms)', unit_class='E',
                   value=';'.join(str(a) for a in sig.window_center_ms), p=np.nan, n=int(tcd.n_units.iloc[0]),
                   test='one-sample t across units, uncorrected', window='40-ms windows, 10-ms steps'))
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                            text=True).stdout.strip()
    st = pd.DataFrame(st)
    st['session_filter'] = ('6 diagonal-target sessions (Newton 08_15 excluded), verified successful trials, unit-stable '
                            'trials, target at P, >= 40 trials, both block types (P3/P4 selection)')
    st['source'] = f'scripts/aim1_panels/build_R01_P1.py (logic of cp_p3_interactions.py, cp_p4_timing.py), git {commit}'
    st.to_csv(OUT / 'stats.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250, 'display.max_colwidth', 80):
        print(st[['statistic', 'value', 'ci95_lo', 'ci95_hi', 'p', 'n']].to_string())
    print(v)


if __name__ == '__main__':
    main()
