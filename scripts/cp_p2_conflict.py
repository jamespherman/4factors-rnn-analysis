#!/usr/bin/env python3
"""
P2 (plans/2026-10-31_competition_plan.md): probability conflict in the single-target task.

Units whose preferred (contralateral) location P is a probability location (export loc 1 or 3). In half the blocks
P is the high-probability location (62.5% of targets: prior at the RF), in the other half the opposite location is
(prior away). Targets at the two 90-deg neighbours are low-probability (12.5%) in both blocks, so comparing
prior-at-RF vs prior-away blocks on neighbour-target trials isolates the prior while the target is elsewhere.

(a) Pre-target (-300..0 ms) rate, prior-at-RF minus prior-away, and its interaction with the reward cue.
(b) Time course (50-ms windows, -300..+400 ms re target onset) of that difference on neighbour-target trials
    (target elsewhere): is the prior's advantage suppressed after target onset, and when? Also on trials with the
    target at P (expected vs unexpected target in the RF).
(c) Single-trial RT: on trials with the target at a low-probability location, does pre-target activity of units whose
    RF is the high-probability location (z-scored per unit within block type, averaged per trial) predict RT beyond
    reward/stimulus/location labels? And activity of units whose RF is the target location?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon, ttest_1samp

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, counts  # noqa: E402
from src.competition import trial_table  # noqa: E402

OUT = Path(__file__).parent.parent / 'results' / 'competition' / 'p2_conflict'
G1 = Path(__file__).parent.parent / 'results' / 'intercollicular' / 'g1_geometry' / 'rows.npy'
OPP = {1: 3, 3: 1, 2: 4, 4: 2}
EDGES = np.arange(-0.3, 0.401, 0.05)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g1 = {(r['session'], r['id']): r for r in np.load(G1, allow_pickle=True)}
    unit_rows, trial_rows = [], []
    for s in ALL_SESSIONS:
        t = session_table(s)
        tt = trial_table(s)
        n = len(t['loc'])
        hp = np.full(n, -1)
        rew = np.full(n, -1)
        rt = np.full(n, np.nan)
        stim = np.array([''] * n, dtype=object)
        m = tt['export_idx'] >= 0
        hp[tt['export_idx'][m]] = tt['high_p_loc'][m]
        rew[tt['export_idx'][m]] = (tt['reward_dur'][m] > 200).astype(int)
        rt[tt['export_idx'][m]] = tt['rt'][m] * 1000
        st = tt['stim_type'][m]
        sal = tt['salience'][m]
        stim[tt['export_idx'][m]] = np.where(st == 1, 'face', np.where(st == 2, 'nonface', np.where(sal == 1, 'bull_hi', 'bull_lo')))
        loc, on = t['loc'], t['ev']['targetOn']
        pre = {}
        for j, i in enumerate(t['ids']):
            P = g1[(s, i)]['P']
            if s == '08_15' or P not in (1, 3):
                continue
            spk = t['spikes'][i]
            stab = t['stable'][:, j]
            prior_rf = hp == P
            prior_away = hp == OPP[P]
            nb = (loc != P) & (loc != OPP[P])
            r_pre = counts(spk, on - 0.3, on) / 0.3
            tc = np.stack([counts(spk, on + a, on + a + 0.05) / 0.05 for a in EDGES[:-1]], 1)   # [trials, windows]
            row = {'session': s, 'id': i, 'type': t['type'][j], 'P': P,
                   'pre_diff': np.nanmean(r_pre[stab & prior_rf]) - np.nanmean(r_pre[stab & prior_away]),
                   'pre_base': np.nanmean(r_pre[stab])}
            for rw in (0, 1):
                row[f'pre_diff_rew{rw}'] = (np.nanmean(r_pre[stab & prior_rf & (rew == rw)])
                                           - np.nanmean(r_pre[stab & prior_away & (rew == rw)]))
            row['tc_nb'] = np.nanmean(tc[stab & prior_rf & nb], 0) - np.nanmean(tc[stab & prior_away & nb], 0)
            row['tc_P'] = np.nanmean(tc[stab & prior_rf & (loc == P)], 0) - np.nanmean(tc[stab & prior_away & (loc == P)], 0)
            row['tc_nb_level'] = np.nanmean(tc[stab & nb], 0) - np.nanmean(r_pre[stab])
            unit_rows.append(row)
            # z-scored pre-target activity per unit within block type, for the single-trial RT model
            z = np.full(n, np.nan)
            for blk in (prior_rf, prior_away):
                sel = stab & blk
                if sel.sum() > 10 and np.nanstd(r_pre[sel]) > 0:
                    z[sel] = (r_pre[sel] - np.nanmean(r_pre[sel])) / np.nanstd(r_pre[sel])
            pre[(i, P)] = z
        # per trial: mean z of units whose RF is the high-p location, and of units whose RF is the target location
        for k in range(n):
            if hp[k] < 0 or np.isnan(rt[k]):
                continue
            a = [z[k] for (i, P), z in pre.items() if P == hp[k] and not np.isnan(z[k])]
            b = [z[k] for (i, P), z in pre.items() if P == loc[k] and not np.isnan(z[k])]
            trial_rows.append({'session': s, 'rt': rt[k], 'log_rt': np.log(rt[k]), 'reward': rew[k], 'stim': stim[k],
                               'loc': int(loc[k]), 'cond': 'high' if loc[k] == hp[k] else ('opposite' if loc[k] == OPP[hp[k]] else 'neighbour'),
                               'z_prior': np.mean(a) if a else np.nan, 'z_target': np.mean(b) if b else np.nan})
        print(s, 'done', flush=True)
    np.save(OUT / 'units.npy', unit_rows, allow_pickle=True)
    df = pd.DataFrame(trial_rows)
    df.to_csv(OUT / 'trials.csv', index=False)
    report(unit_rows, df)


def report(units, df):
    L = []
    p = L.append
    p('# P2 probability conflict (diagonal-target sessions; units with RF at a probability location)\n')
    for ty in ('E', 'I', 'all'):
        g = [u for u in units if ty == 'all' or u['type'] == ty]
        d = np.array([u['pre_diff'] for u in g])
        d = d[~np.isnan(d)]
        d0 = np.array([u['pre_diff_rew0'] for u in g]); d1 = np.array([u['pre_diff_rew1'] for u in g])
        ok = ~np.isnan(d0) & ~np.isnan(d1)
        p(f'## {ty} units (n={len(g)})')
        p(f'- (a) pre-target rate, prior at RF minus prior away: mean {d.mean():+.2f} sp/s (median {np.median(d):+.2f}), '
          f'Wilcoxon p={wilcoxon(d).pvalue:.2g}; low-reward cue {np.mean(d0[ok]):+.2f}, high-reward cue {np.mean(d1[ok]):+.2f} '
          f'(difference p={wilcoxon(d1[ok] - d0[ok]).pvalue:.2g})')
        tc = np.array([u['tc_nb'] for u in g])
        tcP = np.array([u['tc_P'] for u in g])
        lvl = np.array([u['tc_nb_level'] for u in g])
        cells = []
        for w in range(tc.shape[1]):
            x = tc[:, w][~np.isnan(tc[:, w])]
            cells.append(f'{1000 * EDGES[w]:+.0f}: {x.mean():+.2f}{"*" if ttest_1samp(x, 0).pvalue < 0.01 else " "}')
        p('- (b) target at a neighbour (elsewhere): prior-at-RF minus prior-away, by 50-ms window (sp/s; * p<0.01):')
        p('    ' + ' | '.join(cells))
        p('    activity level on neighbour trials re pre-target baseline: ' +
          ' | '.join(f'{1000 * EDGES[w]:+.0f}: {np.nanmean(lvl[:, w]):+.1f}' for w in range(lvl.shape[1])))
        cells = []
        for w in range(tcP.shape[1]):
            x = tcP[:, w][~np.isnan(tcP[:, w])]
            cells.append(f'{1000 * EDGES[w]:+.0f}: {x.mean():+.2f}{"*" if ttest_1samp(x, 0).pvalue < 0.01 else " "}')
        p('- (b2) target in the RF: expected minus unexpected:')
        p('    ' + ' | '.join(cells))
        p('')
    p('## (c) Single-trial RT ~ pre-target SC state (OLS, session fixed effects; log RT x100 ~ %)')
    for cond, label in [(('neighbour', 'opposite'), 'target at a low-probability location'), (('high',), 'target at the high-probability location')]:
        d = df[df['cond'].isin(cond)].dropna(subset=['z_prior'])
        f = smf.ols('log_rt ~ C(session) + reward + C(stim) + C(loc) + z_prior', d).fit()
        p(f'- {label}: n={len(d)}; z_prior (units with RF at the high-p location) coef {100 * f.params["z_prior"]:+.2f}% '
          f'per SD, p={f.pvalues["z_prior"]:.2g}')
        d2 = df[df['cond'].isin(cond)].dropna(subset=['z_target'])
        if cond != ('high',) and len(d2) > 50:
            f2 = smf.ols('log_rt ~ C(session) + reward + C(stim) + C(loc) + z_target', d2).fit()
            p(f'  z_target (units with RF at the target location) coef {100 * f2.params["z_target"]:+.2f}% per SD, '
              f'p={f2.pvalues["z_target"]:.2g}, n={len(d2)}')
    text = '\n'.join(L)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
