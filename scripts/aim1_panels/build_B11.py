#!/usr/bin/env python3
"""
Build the B11 panel data (recorded vs data-fitted RNN visual-epoch ROC selectivity) into
results/aim1_panels/B11_rnn_selectivity_scatter/.

Source: results/selectivity_by_type/grid_v13_free_free.npz (scripts/compare_selectivity_by_type.py). Arrays:
  recorded_<factor>  [293 neurons, 2]                        (signed selectivity 2*(AUC-0.5), permutation p)
  rnn_<factor>       [293, 6 restarts (seeds 1-6), 5 Poisson draws, 2]
The existing figure (scripts/plot_selectivity_by_type.py) plots, per neuron, x = recorded[:, 0] / 2 + 0.5 and
y = nanmean(rnn[..., 0], axis=(restarts, draws)) / 2 + 0.5 (ROC area), filled if recorded p < 0.05, and prints
Pearson r per monkey. The summary statistics mirror scripts/summarize_selectivity_by_type.py.
Unit ids: npz rows follow export order; the cluster id of each row is read from the spike cache
(data/spike_cache/<session>.npz 'ids', built from the same v13 verified export).
Cheap (seconds). Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B11.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sessions import session_name  # noqa: E402

SRC = ROOT / 'results' / 'selectivity_by_type' / 'grid_v13_free_free.npz'
OUT = ROOT / 'results' / 'aim1_panels' / 'B11_rnn_selectivity_scatter'
FACTORS = ['location', 'identity', 'salience', 'reward', 'probability']
RESTARTS = [1, 2, 3, 4, 5, 6]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SRC)
    sess = d['session']
    monkey = np.array(['Feynman' if s.startswith('Feynman') else 'Newton' for s in sess])
    nt = d['neuron_type']
    unit_id = np.empty(len(sess), int)
    for s in np.unique(sess):
        rows = np.flatnonzero(sess == s)
        ids = np.load(ROOT / 'data' / 'spike_cache' / f'{session_name(s)}.npz')['ids']
        assert len(ids) == len(rows)
        unit_id[rows] = ids
    unit_rows, draw_rows, stats = [], [], []
    for f in FACTORS:
        rec, rnn = d[f'recorded_{f}'], d[f'rnn_{f}']
        mean_sel = np.nanmean(rnn[..., 0], axis=(1, 2))
        for n in range(len(sess)):
            unit_rows.append({
                'session': sess[n], 'monkey': monkey[n], 'unit_id': unit_id[n], 'export_row': n,
                'unit_type': nt[n], 'factor': f,
                'recorded_selectivity': rec[n, 0], 'recorded_roc_auc': rec[n, 0] / 2 + 0.5, 'recorded_p': rec[n, 1],
                'recorded_significant': bool(rec[n, 1] < 0.05),
                'rnn_selectivity_mean': mean_sel[n], 'rnn_roc_auc_mean': mean_sel[n] / 2 + 0.5,
                'rnn_selectivity_sd_over_restarts': np.nanstd(np.nanmean(rnn[n, :, :, 0], axis=1), ddof=1),
                'rnn_frac_draws_significant': np.nanmean(rnn[n, :, :, 1] < 0.05),
            })
            for k, seed in enumerate(RESTARTS):
                for dr in range(rnn.shape[2]):
                    draw_rows.append({'session': sess[n], 'unit_id': unit_id[n], 'unit_type': nt[n], 'factor': f,
                                      'restart_seed': seed, 'poisson_draw': dr,
                                      'rnn_selectivity': rnn[n, k, dr, 0], 'rnn_p': rnn[n, k, dr, 1]})
        for m in ['Feynman', 'Newton']:
            for t in ['I', 'E']:
                sub = (monkey == m) & (nt == t) & ~np.isnan(rec[:, 0])
                rs, rp = rec[sub, 0], rec[sub, 1]
                ns, np_ = rnn[sub, ..., 0], rnn[sub, ..., 1]
                abs_r = np.nanmean(np.abs(ns), axis=(0, 2))
                sig_r = np.nanmean(np_ < 0.05, axis=(0, 2)) * 100
                sgn_r = np.nanmean(ns, axis=(0, 2))
                ok = ~np.isnan(mean_sel[sub])
                r = pearsonr(rs[ok], mean_sel[sub][ok])[0]
                base = dict(factor=f, monkey=m, unit_class=t, n=int(sub.sum()),
                            window='visual epoch 50-250 ms after target onset, preferred location',
                            session_filter='v13 verified trials, stable trials; RNN grid_v13 free_free, 6 restarts '
                                           'x 5 Poisson draws')
                stats += [
                    dict(statistic='per-neuron Pearson r, recorded vs RNN mean (as printed in figure)', value=r,
                         test='Pearson', **base),
                    dict(statistic='recorded mean |selectivity|', value=np.abs(rs).mean(), test='mean', **base),
                    dict(statistic='RNN mean |selectivity| (mean over restarts)', value=abs_r.mean(),
                         test=f'mean; SD over restarts {abs_r.std(ddof=1):.3f}', **base),
                    dict(statistic='RNN / recorded mean |selectivity|', value=abs_r.mean() / np.abs(rs).mean(),
                         test='ratio', **base),
                    dict(statistic='recorded % significant', value=np.mean(rp < 0.05) * 100,
                         test='500 label permutations, two-sided p < 0.05 (5% expected by chance)', **base),
                    dict(statistic='RNN % significant (mean over restarts)', value=sig_r.mean(),
                         test=f'same test per draw; SD over restarts {sig_r.std(ddof=1):.1f}', **base),
                    dict(statistic='recorded mean signed selectivity', value=rs.mean(),
                         test=f'mean; SEM {rs.std(ddof=1) / np.sqrt(len(rs)):.3f}', **base),
                    dict(statistic='RNN mean signed selectivity (mean over restarts)', value=sgn_r.mean(),
                         test=f'mean; SD over restarts {sgn_r.std(ddof=1):.3f}', **base),
                ]
    pd.DataFrame(unit_rows).to_csv(OUT / 'B11_units.csv', index=False, float_format='%.5g')
    pd.DataFrame(draw_rows).to_csv(OUT / 'B11_rnn_draws.csv', index=False, float_format='%.5g')
    S = pd.DataFrame(stats)[['statistic', 'factor', 'monkey', 'unit_class', 'value', 'n', 'test', 'window',
                             'session_filter']]
    S.to_csv(OUT / 'stats.csv', index=False, float_format='%.4f')
    print(S[S.statistic.str.startswith('per-neuron')][['factor', 'monkey', 'unit_class', 'value', 'n']].to_string())
    shutil.copy(ROOT / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')


if __name__ == '__main__':
    main()
