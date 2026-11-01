#!/usr/bin/env python3
"""
Build B3 (Aim 1 panels): alignment of goal-directed and stimulus-driven factor effects across SC units (visual window).

Recomputes the per-unit model of scripts/cp_p3_interactions.py (visual window 50-250 ms after target onset, trials
with the target at the unit's preferred contralateral location P, verified trials, unit-stable trials, >= 40 trials;
OLS of sqrt(spike count) on reward, image, salience, face, reward x image [+ expected, reward x expected when the unit
has both block types]; effect = coefficient / residual SD) with plain numpy least squares, then adds:
  - within-class (E, I) Spearman correlations across units between factor effects (full data);
  - split-half reliability: N_SPLIT random halves of each unit's trials; per factor, Spearman r between half-1 and
    half-2 effects across units (reliability); per pair, cross-half r (mean of r(a1, b2) and r(a2, b1); independent
    trial noise) and the ceiling sqrt(rel_a x rel_b) that a perfectly shared axis would reach at this noise level;
  - within-unit trial-shuffle null: N_SHUF permutations of y across each unit's trials (keeps the design and its
    coefficient covariances, removes all true effects), full-data r per pair; 2.5/97.5% quantiles;
  - unit-label permutation null (N_PERM permutations of one effect across units).
Run: OMP_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B3.py   (~2-3 min)
"""
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, counts  # noqa: E402
from src.competition import trial_table  # noqa: E402

OUT = REPO / 'results' / 'aim1_panels' / 'B3_factor_alignment'
G1 = REPO / 'results' / 'intercollicular' / 'g1_geometry' / 'rows.npy'
P3 = REPO / 'results' / 'competition' / 'p3_interactions' / 'units.csv'
FACT = ['reward', 'expected', 'image', 'salience', 'face']
GOAL, STIM = {'reward', 'expected'}, {'image', 'salience', 'face'}
N_SPLIT, N_SHUF, N_PERM = 50, 500, 5000
rng = np.random.default_rng(20261101)


def design(rew, hp, st, sal, P):
    has_exp = len(np.unique(hp == P)) > 1
    cols = {'const': np.ones(len(rew)), 'reward': rew - 0.5, 'image': (st <= 2) - 0.5,
            'salience': np.where(st > 2, np.where(sal == 1, 0.5, -0.5), 0.0),
            'face': np.where(st == 1, 0.5, np.where(st == 2, -0.5, 0.0))}
    cols['reward:image'] = cols['reward'] * cols['image']
    if has_exp:
        cols['expected'] = (hp == P) - 0.5
        cols['reward:expected'] = cols['reward'] * cols['expected']
    return list(cols), np.column_stack([np.asarray(v, float) for v in cols.values()])


def fit(X, Y):
    """Standardised coefficients for each column of Y ([trials, k]); NaN if rank deficient or too few dof."""
    n, p = X.shape
    if n - p < 3 or np.linalg.matrix_rank(X) < p:
        return np.full((p, Y.shape[1]), np.nan)
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ B
    sd = np.sqrt((res ** 2).sum(0) / (n - p))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sd > 0, B / sd, np.nan)


def effects(names, X, Y):
    E = fit(X, Y)
    return {f: (E[names.index(f)] if f in names else np.full(Y.shape[1], np.nan)) for f in FACT}


def sb(r):
    """Spearman-Brown: reliability of the full data from the split-half correlation."""
    return 2 * r / (1 + r)


def sp(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 8:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(a[m], b[m])
    return r, p, int(m.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
    g1 = {(r['session'], r['id']): r for r in np.load(G1, allow_pickle=True)}
    units = []
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
            e = t['ev']['targetOn'][sel]
            y = np.sqrt(counts(t['spikes'][i], e + 0.05, e + 0.25))
            ok = ~np.isnan(y)
            names, X = design(rew[sel][ok], hp[sel][ok], st[sel][ok], sal[sel][ok], P)
            units.append(dict(session=s, id=int(i), type=t['type'][j], P=int(P), names=names, X=X, y=y[ok]))
        print(s, 'loaded', flush=True)

    # ---- full-data effects (check against p3 units.csv) --------------------------------------------------------
    full = pd.DataFrame([dict(session=u['session'], unit_id=u['id'], unit_type=u['type'], P_loc=u['P'],
                              n_trials=len(u['y']),
                              **{f'{f}_sd': v[0] for f, v in effects(u['names'], u['X'], u['y'][:, None]).items()})
                         for u in units])
    p3 = pd.read_csv(P3).rename(columns={'id': 'unit_id'})
    chk = full.merge(p3[['session', 'unit_id'] + [f'visual_{f}' for f in FACT]], on=['session', 'unit_id'])
    maxdiff = max(np.nanmax(np.abs(chk[f'{f}_sd'] - chk[f'visual_{f}'])) for f in FACT)
    print(f'check vs p3 units.csv: {len(chk)}/{len(full)} units matched, max |diff| = {maxdiff:.2e}')
    assert len(chk) == len(full) == len(p3) and maxdiff < 1e-6

    # ---- split halves ------------------------------------------------------------------------------------------
    halves = {f: np.full((N_SPLIT, 2, len(units)), np.nan) for f in FACT}
    for k, u in enumerate(units):
        n = len(u['y'])
        for r in range(N_SPLIT):
            perm = rng.permutation(n)
            for h, idx in enumerate((perm[:n // 2], perm[n // 2:])):
                ef = effects(u['names'], u['X'][idx], u['y'][idx, None])
                for f in FACT:
                    halves[f][r, h, k] = ef[f][0]
    # ---- trial-shuffle null ------------------------------------------------------------------------------------
    shuf = {f: np.full((N_SHUF, len(units)), np.nan) for f in FACT}
    for k, u in enumerate(units):
        Y = np.stack([rng.permutation(u['y']) for _ in range(N_SHUF)], 1)
        ef = effects(u['names'], u['X'], Y)
        for f in FACT:
            shuf[f][:, k] = ef[f]
    print('resampling done', flush=True)

    types = np.array([u['type'] for u in units])
    rows, rel_rows = [], []
    for cls in ('E', 'I', 'all'):
        mk = np.ones(len(units), bool) if cls == 'all' else types == cls
        rel = {}
        for f in FACT:
            rr = [sp(halves[f][r, 0, mk], halves[f][r, 1, mk])[0] for r in range(N_SPLIT)]
            rel[f] = np.nanmean(rr)
            rel_rows.append(dict(unit_class=cls, factor=f, split_half_r=rel[f],
                                 split_half_r_q025=np.nanquantile(rr, 0.025), split_half_r_q975=np.nanquantile(rr, 0.975),
                                 spearman_brown_full=2 * rel[f] / (1 + rel[f]) if rel[f] > -1 else np.nan,
                                 n_units=int((~np.isnan(full[f'{f}_sd'].to_numpy()[mk])).sum()), n_splits=N_SPLIT))
        for a, b in combinations(FACT, 2):
            xa, xb = full[f'{a}_sd'].to_numpy()[mk], full[f'{b}_sd'].to_numpy()[mk]
            r, p, n = sp(xa, xb)
            cross = [np.nanmean([sp(halves[a][q, 0, mk], halves[b][q, 1, mk])[0],
                                 sp(halves[a][q, 1, mk], halves[b][q, 0, mk])[0]]) for q in range(N_SPLIT)]
            within = [np.nanmean([sp(halves[a][q, h, mk], halves[b][q, h, mk])[0] for h in (0, 1)]) for q in range(N_SPLIT)]
            null_ts = np.array([sp(shuf[a][q, mk], shuf[b][q, mk])[0] for q in range(N_SHUF)])
            # unit-label permutation on ranks of the units with both effects
            m2 = ~np.isnan(xa) & ~np.isnan(xb)
            ra, rb = rankdata(xa[m2]), rankdata(xb[m2])
            null_ul = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(N_PERM)])
            ceil = np.sqrt(rel[a] * rel[b]) if rel[a] > 0 and rel[b] > 0 else np.nan
            cr = np.nanmean(cross)
            rows.append(dict(unit_class=cls, factor_a=a, factor_b=b,
                             pair_kind='goal-stim' if ({a, b} & GOAL and {a, b} & STIM) else
                             ('stim-stim' if {a, b} <= STIM else 'goal-goal'),
                             spearman_r_full=r, p_spearman=p, n_units=n,
                             p_trial_shuffle_2sided=(np.sum(np.abs(null_ts - np.nanmean(null_ts)) >= abs(r - np.nanmean(null_ts))) + 1) / (np.sum(~np.isnan(null_ts)) + 1),
                             null_trial_shuffle_mean=np.nanmean(null_ts), null_trial_shuffle_q025=np.nanquantile(null_ts, 0.025),
                             null_trial_shuffle_q975=np.nanquantile(null_ts, 0.975),
                             null_unit_perm_q025=np.quantile(null_ul, 0.025), null_unit_perm_q975=np.quantile(null_ul, 0.975),
                             p_unit_perm_2sided=(np.sum(np.abs(null_ul) >= abs(r)) + 1) / (N_PERM + 1),
                             cross_half_r=cr, cross_half_r_split_q025=np.nanquantile(cross, 0.025),
                             cross_half_r_split_q975=np.nanquantile(cross, 0.975), within_half_r=np.nanmean(within),
                             split_half_rel_a=rel[a], split_half_rel_b=rel[b], ceiling_cross_half=ceil,
                             ceiling_full_data=np.sqrt(sb(rel[a]) * sb(rel[b])) if rel[a] > 0 and rel[b] > 0 else np.nan,
                             cross_half_r_disattenuated=cr / ceil if ceil > 0 else np.nan,
                             n_shuffles=N_SHUF, n_unit_perms=N_PERM, n_splits=N_SPLIT))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / 'B3_pairs.csv', index=False, float_format='%.5g')
    pd.DataFrame(rel_rows).to_csv(OUT / 'B3_reliability.csv', index=False, float_format='%.5g')
    full.to_csv(OUT / 'B3_units.csv', index=False, float_format='%.5g')

    # ---- stats -------------------------------------------------------------------------------------------------
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    st = []
    for _, q in pairs.iterrows():
        st.append(dict(statistic=f'Spearman r {q.factor_a}-{q.factor_b}', unit_class=q.unit_class, value=q.spearman_r_full,
                       p=q.p_spearman, p_trial_shuffle=q.p_trial_shuffle_2sided, p_unit_perm=q.p_unit_perm_2sided,
                       n=q.n_units, test='Spearman across units (scipy p); trial-shuffle and unit-label permutation p'))
        st.append(dict(statistic=f'cross-half r {q.factor_a}-{q.factor_b} (ceiling {q.ceiling_cross_half:.2f})',
                       unit_class=q.unit_class, value=q.cross_half_r, p=np.nan, n=q.n_units,
                       test=f'mean over {N_SPLIT} random trial splits'))
    for q in rel_rows:
        st.append(dict(statistic=f'split-half reliability of {q["factor"]} effect', unit_class=q['unit_class'],
                       value=q['split_half_r'], p=np.nan, n=q['n_units'],
                       test=f'Spearman r half 1 vs half 2 across units, mean over {N_SPLIT} random splits'))
    p3df = pd.read_csv(P3)
    from scipy.stats import ttest_1samp
    for cls in ('E', 'I'):
        x = p3df.loc[p3df.type == cls, 'visual_reward:image'].dropna()
        st.append(dict(statistic='mean reward x image interaction (SD units)', unit_class=cls, value=x.mean(),
                       p=ttest_1samp(x, 0).pvalue, n=len(x), test='one-sample t across units (p3 units.csv)'))
    st = pd.DataFrame(st)
    st['window'] = 'visual 50-250 ms after target onset'
    st['session_filter'] = ('6 diagonal-target sessions (Newton 08_15 excluded); verified successful trials, target at '
                            'the unit\'s preferred contralateral location, unit-stable trials, units with >= 40 trials')
    st['source'] = f'recomputed (scripts/aim1_panels/build_B3.py; matches {P3.relative_to(REPO).as_posix()}), git {commit}'
    st.to_csv(OUT / 'stats.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(pairs[['unit_class', 'factor_a', 'factor_b', 'spearman_r_full', 'p_spearman', 'null_trial_shuffle_q025',
                     'null_trial_shuffle_q975', 'cross_half_r', 'ceiling_cross_half', 'n_units']].round(3).to_string())
        print(pd.DataFrame(rel_rows).round(3).to_string())


if __name__ == '__main__':
    main()
