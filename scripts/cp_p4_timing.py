#!/usr/bin/env python3
"""
P4 (plans/2026-10-31_competition_plan.md): when do the factor effects and the reward x expectation interaction
appear after target onset?

Same per-unit model as P3 (sqrt count, standardised coefficients) in sliding 40-ms windows (10-ms steps) from
-100 to +300 ms re target onset, trials with the target at the unit's preferred location. Population mean
time course per term and class; onset = first time the population mean reaches half its peak (in the direction of
the peak) and stays there >= 30 ms; 95% CI by jackknife over units (Miller, Patterson & Ulrich 1998).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, counts  # noqa: E402
from src.competition import trial_table  # noqa: E402

OUT = Path(__file__).parent.parent / 'results' / 'competition' / 'p4_timing'
G1 = Path(__file__).parent.parent / 'results' / 'intercollicular' / 'g1_geometry' / 'rows.npy'
STARTS = np.arange(-0.1, 0.261, 0.01)
WIDTH = 0.04
TERMS = ['reward', 'expected', 'image', 'salience', 'face', 'reward:expected']


def onset(curve, times):
    k = np.nanargmax(np.abs(curve[times >= 0])) + np.sum(times < 0)
    sign = np.sign(curve[k])
    above = sign * curve >= 0.5 * abs(curve[k])
    for i in np.flatnonzero(times >= 0):
        if above[i:i + 3].all():
            return times[i]
    return np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g1 = {(r['session'], r['id']): r for r in np.load(G1, allow_pickle=True)}
    coefs = {('E', k): [] for k in TERMS} | {('I', k): [] for k in TERMS}
    for s in ALL_SESSIONS:
        if s == '08_15':
            continue
        t, tt = session_table(s), trial_table(s)
        n = len(t['loc'])
        hp, rew, st, sal = (np.full(n, -1.0) for _ in range(4))
        m = tt['export_idx'] >= 0
        hp[tt['export_idx'][m]] = tt['high_p_loc'][m]
        rew[tt['export_idx'][m]] = tt['reward_dur'][m] > 200
        st[tt['export_idx'][m]] = tt['stim_type'][m]
        sal[tt['export_idx'][m]] = tt['salience'][m]
        for j, i in enumerate(t['ids']):
            P = g1[(s, i)]['P']
            sel = t['stable'][:, j] & (t['loc'] == P) & (hp > 0)
            if sel.sum() < 40:
                continue
            X = pd.DataFrame({'reward': rew[sel] - 0.5, 'image': (st[sel] <= 2) - 0.5,
                              'salience': np.where(st[sel] > 2, np.where(sal[sel] == 1, 0.5, -0.5), 0.0),
                              'face': np.where(st[sel] == 1, 0.5, np.where(st[sel] == 2, -0.5, 0.0))})
            has_exp = len(np.unique(hp[sel] == P)) > 1
            if has_exp:
                X['expected'] = (hp[sel] == P) - 0.5
                X['reward:expected'] = X['reward'] * X['expected']
            X = sm.add_constant(X.astype(float))
            on = t['ev']['targetOn'][sel]
            cur = {k: [] for k in TERMS}
            for a in STARTS:
                y = np.sqrt(counts(t['spikes'][i], on + a, on + a + WIDTH))
                f = sm.OLS(y, X).fit()
                sd = np.sqrt(f.scale) if f.scale > 0 else np.nan
                for k in TERMS:
                    cur[k].append(f.params[k] / sd if k in f.params else np.nan)
            for k in TERMS:
                if not np.all(np.isnan(cur[k])):
                    coefs[(t['type'][j], k)].append(cur[k])
        print(s, 'done', flush=True)
    times = (STARTS + WIDTH / 2) * 1000
    L = ['# P4 time course of factor effects at the preferred location (standardised coefficients, 40-ms windows)\n',
         'Onset = population mean reaches half its peak (peak sign) for >= 30 ms; jackknife 95% CI over units.\n']
    save = {}
    for ty in ('E', 'I'):
        L.append(f'## {ty} units')
        for k in TERMS:
            c = np.array(coefs[(ty, k)])
            if len(c) < 5:
                continue
            mean = np.nanmean(c, 0)
            save[f'{ty}_{k}'] = c
            on = onset(mean, times)
            jk = np.array([onset(np.nanmean(np.delete(c, u, 0), 0), times) for u in range(len(c))])
            nj = len(jk)
            se = np.sqrt((nj - 1) / nj * np.nansum((jk - np.nanmean(jk)) ** 2))
            pk = times[np.nanargmax(np.abs(mean))]
            L.append(f'- {k:16s} n={len(c):3d} peak {mean[np.nanargmax(np.abs(mean))]:+.3f} at {pk:.0f} ms; '
                     f'onset {on:.0f} ms (95% CI +-{1.96 * se:.0f})')
        L.append('')
    np.savez(OUT / 'timecourses.npz', times=times, **save)
    text = '\n'.join(L)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
