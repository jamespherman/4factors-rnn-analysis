#!/usr/bin/env python3
"""
P3 (plans/2026-10-31_competition_plan.md): factor interactions within neurons and a common priority axis.

For each unit and its preferred contralateral location P (G1), trials with the target at P. Per unit and window
(pre-target -300..0, visual 50-250 ms, pre-saccadic -50..0 ms re saccade): OLS of sqrt(spike count) on reward,
expected (P is the block's high-probability location; P = export loc 1 only), stimulus (image vs bullseye;
high vs low salience within bullseyes; face vs non-face within images) and the reward x expected and
reward x image interactions. Coefficients in units of the unit's residual SD (standardised effects).
Population: mean standardised effects (t-test across units) and, for a common "priority" axis, the
correlation across units between factor effects (alignment).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, counts  # noqa: E402
from src.competition import trial_table  # noqa: E402

OUT = Path(__file__).parent.parent / 'results' / 'competition' / 'p3_interactions'
G1 = Path(__file__).parent.parent / 'results' / 'intercollicular' / 'g1_geometry' / 'rows.npy'
WIN = {'pre': ('targetOn', -0.3, 0.0), 'visual': ('targetOn', 0.05, 0.25), 'presac': ('saccadeOnset', -0.05, 0.0)}
TERMS = ['reward', 'expected', 'image', 'salience', 'face', 'reward:expected', 'reward:image']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g1 = {(r['session'], r['id']): r for r in np.load(G1, allow_pickle=True)}
    rows = []
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
            d = pd.DataFrame({'reward': rew[sel], 'expected': (hp[sel] == P).astype(float) - 0.5,
                              'image': (st[sel] <= 2).astype(float) - 0.5,
                              'salience': np.where(st[sel] > 2, np.where(sal[sel] == 1, 0.5, -0.5), 0.0),
                              'face': np.where(st[sel] == 1, 0.5, np.where(st[sel] == 2, -0.5, 0.0))})
            d['reward'] -= 0.5
            row = {'session': s, 'id': i, 'type': t['type'][j], 'P': P, 'has_expected': d['expected'].nunique() > 1}
            for w, (ev, a, b) in WIN.items():
                e = t['ev'][ev][sel]
                d['y'] = np.sqrt(counts(t['spikes'][i], e + a, e + b))
                dd = d.dropna()
                formula = 'y ~ reward + image + salience + face + reward:image' + (
                    ' + expected + reward:expected' if row['has_expected'] else '')
                f = smf.ols(formula, dd).fit()
                sd = np.sqrt(f.scale) if f.scale > 0 else np.nan
                for k in TERMS:
                    row[f'{w}_{k}'] = f.params[k] / sd if k in f.params else np.nan
                    row[f'{w}_{k}_p'] = f.pvalues[k] if k in f.params else np.nan
            rows.append(row)
        print(s, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'units.csv', index=False)
    report(df)


def report(df):
    L = []
    p = L.append
    p('# P3 factor effects and interactions at the preferred location (standardised: coef / residual SD of sqrt count)\n')
    p('Positive: high reward, expected target (P is the high-probability location), image (vs bullseye), high salience,'
      ' face. Cells: mean [fraction p<0.05] t-test p.\n')
    for w in WIN:
        p(f'## {w}')
        p('| units | ' + ' | '.join(TERMS) + ' |')
        p('|---|' + '---|' * len(TERMS))
        for ty in ('E', 'I'):
            g = df[df['type'] == ty]
            cells = []
            for k in TERMS:
                x = g[f'{w}_{k}'].dropna()
                pv = g[f'{w}_{k}_p'].dropna()
                cells.append(f'{x.mean():+.3f} [{(pv < 0.05).mean():.0%}] p={ttest_1samp(x, 0).pvalue:.1g} (n={len(x)})' if len(x) > 3 else '-')
            p(f'| {ty} | ' + ' | '.join(cells) + ' |')
        p('')
    p('## Alignment of factor effects across units (Spearman r; one priority axis predicts positive r)')
    for w in ('visual', 'presac'):
        pairs = [('reward', 'expected'), ('reward', 'image'), ('reward', 'salience'), ('expected', 'image'),
                 ('expected', 'salience'), ('image', 'salience'), ('reward', 'face')]
        cells = []
        for a, b in pairs:
            x = df[[f'{w}_{a}', f'{w}_{b}']].dropna()
            r, pv = spearmanr(x.iloc[:, 0], x.iloc[:, 1])
            cells.append(f'{a}-{b} {r:+.2f} (p={pv:.1g})')
        p(f'- {w}: ' + '; '.join(cells))
    text = '\n'.join(L)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
