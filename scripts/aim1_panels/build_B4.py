#!/usr/bin/env python3
"""
Build B4 (Aim 1 panels): what happens at the expected location when the target appears elsewhere, and
pre-saccadic suppression by target angle.

(a) PSTH re-run of scripts/cp_p2_conflict.py (b) (units.npy stores only the block difference): units whose preferred
contralateral location P (g1_geometry) is a probability location (export loc 1 or 3; in practice loc 1), diagonal-
target sessions (Newton 08_15 excluded), verified successful trials, unit-stable trials, target at one of the two
90-deg neighbours of P (low probability, 12.5%, in both block types). Rate in BIN-ms bins re target onset, per unit
averaged separately over blocks where P is the high-probability location ("expected location = RF") and blocks where
the opposite location is ("expected location = opposite"); minus the unit's pre-target baseline (-300..0 ms, all
stable trials, as in cp_p2_conflict.py). Population mean +- SEM across units, E and I separately; block difference
with 95% t CI across units.
(b) Pre-saccadic suppression by target angle from results/intercollicular/g1_geometry/rows.npy (scripts/
ic_g1_geometry.py): presac window (-50..0 ms re saccade onset) minus pre-target baseline (-200..0 ms), paired t-test
per location, p < 0.01 and negative. Wilson 95% CIs; exact McNemar (binomial on discordant units) opposite vs
ipsi-neighbour, as in the source.
Run: OMP_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B4.py   (~1 min)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, counts  # noqa: E402
from src.competition import trial_table  # noqa: E402

OUT = REPO / 'results' / 'aim1_panels' / 'B4_map_competition'
G1 = REPO / 'results' / 'intercollicular' / 'g1_geometry' / 'rows.npy'
OPP = {1: 3, 3: 1, 2: 4, 4: 2}
BIN = 0.02
EDGES = np.round(np.arange(-0.3, 0.4001, BIN), 4)
W50 = np.round(np.arange(-0.3, 0.4001, 0.05), 4)


def wilson(k, n):
    lo, hi = stats.binomtest(k, n).proportion_ci(method='wilson')
    return lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
    g1 = {(r['session'], r['id']): r for r in np.load(G1, allow_pickle=True)}
    urows, crows = [], []
    for s in ALL_SESSIONS:
        if s == '08_15':
            continue
        t, tt = session_table(s), trial_table(s)
        n = len(t['loc'])
        hp = np.full(n, -1)
        m = tt['export_idx'] >= 0
        hp[tt['export_idx'][m]] = tt['high_p_loc'][m]
        loc, on = t['loc'], t['ev']['targetOn']
        for j, i in enumerate(t['ids']):
            P = g1[(s, i)]['P']
            if P not in (1, 3):
                continue
            spk, stab = t['spikes'][i], t['stable'][:, j]
            nb = (loc != P) & (loc != OPP[P])
            base = np.nanmean((counts(spk, on - 0.3, on) / 0.3)[stab])
            tc = np.stack([counts(spk, on + a, on + a + BIN) / BIN for a in EDGES[:-1]], 1)
            w50 = np.stack([counts(spk, on + a, on + a + 0.05) / 0.05 for a in W50[:-1]], 1)
            rf, away = stab & (hp == P) & nb, stab & (hp == OPP[P]) & nb
            urows.append(dict(session=s, unit_id=int(i), unit_type=t['type'][j], P_loc=int(P), baseline_sp_s=base,
                              n_trials_expected_at_rf=int(rf.sum()), n_trials_expected_opposite=int(away.sum())))
            for blk, sel in (('expected_at_RF', rf), ('expected_opposite', away)):
                mu = np.nanmean(tc[sel], 0)
                for b, a in enumerate(EDGES[:-1]):
                    crows.append(dict(session=s, unit_id=int(i), unit_type=t['type'][j], block=blk,
                                      bin_start_ms=round(1000 * a), bin_center_ms=round(1000 * (a + BIN / 2)),
                                      rate_sp_s=mu[b], rate_minus_baseline_sp_s=mu[b] - base))
            urows[-1].update({f'w50_{round(1000 * a):+d}_diff_sp_s': np.nanmean(w50[rf, b]) - np.nanmean(w50[away, b])
                              for b, a in enumerate(W50[:-1])})
            urows[-1].update({f'w50_{round(1000 * a):+d}_level_sp_s': np.nanmean(w50[stab & nb, b]) - base
                              for b, a in enumerate(W50[:-1])})
        print(s, 'done', flush=True)
    units = pd.DataFrame(urows)
    cur = pd.DataFrame(crows)
    units.to_csv(OUT / 'B4_units.csv', index=False, float_format='%.5g')
    cur.to_csv(OUT / 'B4_unit_psth.csv', index=False, float_format='%.5g')

    # population curves
    pop = []
    for (ty, blk, tc_), g in cur.groupby(['unit_type', 'block', 'bin_center_ms']):
        x = g.rate_minus_baseline_sp_s.dropna()
        pop.append(dict(unit_type=ty, curve=blk, bin_center_ms=tc_, mean_sp_s=x.mean(),
                        sem_sp_s=x.std(ddof=1) / np.sqrt(len(x)), n_units=len(x)))
    wide = cur.pivot_table(index=['session', 'unit_id', 'unit_type', 'bin_center_ms'], columns='block',
                           values='rate_sp_s').reset_index()
    wide['diff'] = wide['expected_at_RF'] - wide['expected_opposite']
    for (ty, tc_), g in wide.groupby(['unit_type', 'bin_center_ms']):
        x = g['diff'].dropna()
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        pop.append(dict(unit_type=ty, curve='difference_RF_minus_opposite', bin_center_ms=tc_, mean_sp_s=x.mean(),
                        sem_sp_s=x.std(ddof=1) / np.sqrt(len(x)), ci95_lo=x.mean() - h, ci95_hi=x.mean() + h,
                        n_units=len(x)))
    pd.DataFrame(pop).to_csv(OUT / 'B4_psth.csv', index=False, float_format='%.5g')

    # suppression bars
    rows = [r for r in np.load(G1, allow_pickle=True) if not r['meridian']]
    bars, st = [], []
    for ty in ('E', 'I'):
        g = [r for r in rows if r['type'] == ty and not np.isnan(r['presac_P'])]
        sup = {k: np.array([r[f'presac_{k}'] < 0 and r[f'presac_{k}_p'] < 0.01 for r in g]) for k in ('cn', 'in', 'op')}
        for k, ang, lab in (('cn', 90, 'contra_neighbour_90deg'), ('in', 90, 'ipsi_neighbour_90deg'),
                            ('op', 180, 'opposite_180deg')):
            lo, hi = wilson(int(sup[k].sum()), len(g))
            bars.append(dict(unit_type=ty, target_location=lab, angle_from_P_deg=ang, n_suppressed=int(sup[k].sum()),
                             n_units=len(g), pct_suppressed=100 * sup[k].mean(), ci95_lo_pct=100 * lo, ci95_hi_pct=100 * hi))
        b2, c2 = int((sup['op'] & ~sup['in']).sum()), int((~sup['op'] & sup['in']).sum())
        st.append(dict(statistic='presac suppression: opposite vs ipsi-neighbour (discordant units op-only:in-only)',
                       unit_class=ty, value=f'{100 * sup["op"].mean():.0f}% vs {100 * sup["in"].mean():.0f}% ({b2}:{c2})',
                       p=binomtest(b2, b2 + c2).pvalue if b2 + c2 else np.nan, n=len(g),
                       test='exact McNemar (binomial on discordant units)',
                       window='presac -50..0 ms re saccade onset vs pre-target -200..0 ms; p<0.01 per location'))
        b3, c3 = int((sup['op'] & ~sup['cn']).sum()), int((~sup['op'] & sup['cn']).sum())
        st.append(dict(statistic='presac suppression: opposite vs contra-neighbour', unit_class=ty,
                       value=f'{100 * sup["op"].mean():.0f}% vs {100 * sup["cn"].mean():.0f}% ({b3}:{c3})',
                       p=binomtest(b3, b3 + c3).pvalue if b3 + c3 else np.nan, n=len(g),
                       test='exact McNemar (binomial on discordant units)',
                       window='presac -50..0 ms re saccade onset vs pre-target -200..0 ms; p<0.01 per location'))
    pd.DataFrame(bars).to_csv(OUT / 'B4_suppression.csv', index=False, float_format='%.5g')

    # stats on the PSTH (50-ms windows, as in cp_p2_conflict.py)
    for ty in ('E', 'I', 'all'):
        g = units if ty == 'all' else units[units.unit_type == ty]
        for a in (-50, 0, 50, 100, 150, 200, 250):
            x = g[f'w50_{a:+d}_diff_sp_s'].dropna()
            st.append(dict(statistic=f'expected-at-RF minus expected-opposite, target at 90-deg neighbour, {a}..{a + 50} ms',
                           unit_class=ty, value=x.mean(), p=stats.ttest_1samp(x, 0).pvalue, n=len(x),
                           test='one-sample t across units', window=f'{a}..{a + 50} ms re target onset'))
            x = g[f'w50_{a:+d}_level_sp_s'].dropna()
            st.append(dict(statistic=f'rate re pre-target baseline, target at 90-deg neighbour (both blocks), {a}..{a + 50} ms',
                           unit_class=ty, value=x.mean(), p=stats.ttest_1samp(x, 0).pvalue, n=len(x),
                           test='one-sample t across units', window=f'{a}..{a + 50} ms re target onset'))
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    st = pd.DataFrame(st)
    st['session_filter'] = ('PSTH: 6 diagonal-target sessions (Newton 08_15 excluded), verified successful trials, units '
                            'with P at a probability location; bars: same 6 sessions, all units (g1_geometry)')
    st['source'] = f'scripts/aim1_panels/build_B4.py (logic of cp_p2_conflict.py, ic_g1_geometry.py), git {commit}'
    st.to_csv(OUT / 'stats.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250, 'display.max_colwidth', 90):
        print(st[['statistic', 'unit_class', 'value', 'p', 'n']].to_string())
        print(pd.DataFrame(bars).round(1).to_string())


if __name__ == '__main__':
    main()
