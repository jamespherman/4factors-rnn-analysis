#!/usr/bin/env python3
"""
P1 (plans/2026-10-31_competition_plan.md): behavioural effects of the four factors and their conflicts.

Successful gSac_4factors trials (end state 21). RT = saccade onset - go (fixation off). Models (OLS, session fixed
effects, cluster-robust SEs by session x block run): log RT and peak velocity ~ reward + probability condition +
stimulus (face / non-face / high-salience bullseye / low-salience bullseye) + reward x stimulus + reward x
probability + probability x stimulus. Probability condition: target at the block's high-probability location, at a
90-deg neighbour of it, or at its opposite (180 deg).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.competition import trial_table  # noqa: E402

OUT = Path(__file__).parent.parent / 'results' / 'competition' / 'p1_behaviour'
OPP = {1: 3, 3: 1, 2: 4, 4: 2}


def frame():
    dfs = []
    for s in ALL_SESSIONS:
        t = trial_table(s)
        ok = (t['end_state'] == 21) & (t['high_p_loc'] > 0) & ~np.isnan(t['rt']) & (t['rt'] > 0.08)
        loc, hp = t['loc'][ok].astype(int), t['high_p_loc'][ok]
        cond = np.where(loc == hp, 'high', np.where(loc == np.vectorize(OPP.get)(hp), 'opposite', 'neighbour'))
        st, sal = t['stim_type'][ok], t['salience'][ok]
        stim = np.where(st == 1, 'face', np.where(st == 2, 'nonface', np.where(sal == 1, 'bull_hi', 'bull_lo')))
        run = np.cumsum(np.r_[0, np.diff(t['block'][ok]) != 0])
        dfs.append(pd.DataFrame({'session': s, 'monkey': 'Feynman' if s.startswith('Feynman') else 'Newton',
                                 'rt': t['rt'][ok] * 1000, 'pv': t['peak_vel'][ok], 'reward': (t['reward_dur'][ok] > 200).astype(int),
                                 'prob': cond, 'stim': stim, 'loc': loc, 'cluster': [f'{s}_{r}' for r in run]}))
    return pd.concat(dfs, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = frame()
    df['log_rt'] = np.log(df['rt'])
    L = []
    p = L.append
    p(f'# P1 behaviour: {len(df)} successful trials, 7 sessions\n')
    p('## Cell means (median RT ms / mean peak velocity deg/s)')
    for by in ['reward', 'prob', 'stim']:
        g = df.groupby(['monkey', by]).agg(rt=('rt', 'median'), pv=('pv', 'mean'), n=('rt', 'size'))
        p(f'\n{by}:\n' + g.round(1).to_string())
    for y in ['log_rt', 'pv']:
        for m in ['Feynman', 'Newton', None]:
            d = df if m is None else df[df['monkey'] == m]
            fit = smf.ols(f'{y} ~ C(session) + reward * C(prob, Treatment("high")) + reward * C(stim, Treatment("bull_lo")) '
                          f'+ C(prob, Treatment("high")) : C(stim, Treatment("bull_lo"))', d).fit(
                cov_type='cluster', cov_kwds={'groups': d['cluster']})
            p(f'\n## {y} ~ factors ({m or "both monkeys"}, n={len(d)}); coefficients with p < 0.1 (log RT: x100 ~ % change)')
            tab = pd.DataFrame({'coef': fit.params, 'se': fit.bse, 'p': fit.pvalues})
            tab = tab[~tab.index.str.startswith('C(session)') & (tab['p'] < 0.1) & (tab.index != 'Intercept')]
            if y == 'log_rt':
                tab[['coef', 'se']] *= 100
            p(tab.round(4).to_string())
    text = '\n'.join(L)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')
    df.to_csv(OUT / 'trials.csv', index=False)
    print(text)


if __name__ == '__main__':
    main()
