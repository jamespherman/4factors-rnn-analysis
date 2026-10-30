"""Print the by-monkey, by-type table from compare_selectivity_by_type.py output."""
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

d = np.load(Path(__file__).parent.parent / 'results' / 'selectivity_by_type' / 'grid_v13_free_free.npz')
FACTORS = ['location', 'identity', 'salience', 'reward', 'probability']
monkey = np.array(['Feynman' if s.startswith('Feynman') else 'Newton' for s in d['session']])
nt = d['neuron_type']

print('rows: recorded | RNN (mean over 6 restarts x 5 draws; +- SD over restarts)')
print('mean|sel|, %sig (p<.05), mean signed sel; r = per-neuron corr recorded vs RNN (draw+restart mean)')
for f in FACTORS:
    rec = d[f'recorded_{f}']          # [n, 2]
    rnn = d[f'rnn_{f}']               # [n, restarts, draws, 2]
    print(f'\n== {f}')
    for m in ['Feynman', 'Newton']:
        for t in ['I', 'E']:
            sub = (monkey == m) & (nt == t) & ~np.isnan(rec[:, 0])
            rs, rp = rec[sub, 0], rec[sub, 1]
            ns, np_ = rnn[sub, ..., 0], rnn[sub, ..., 1]
            abs_r = np.nanmean(np.abs(ns), axis=(0, 2))       # per restart
            sig_r = np.nanmean(np_ < 0.05, axis=(0, 2)) * 100
            sgn_r = np.nanmean(ns, axis=(0, 2))
            mean_rnn = np.nanmean(ns, axis=(1, 2))
            ok = ~np.isnan(mean_rnn)
            r = pearsonr(rs[ok], mean_rnn[ok])[0] if ok.sum() > 3 else np.nan
            print(f'  {m:7s} {t} n={sub.sum():3d} | rec |sel| {np.abs(rs).mean():.3f} sig {np.mean(rp < .05)*100:4.0f}% '
                  f'sgn {rs.mean():+.3f}+-{rs.std(ddof=1)/np.sqrt(len(rs)):.3f} | RNN |sel| {abs_r.mean():.3f}+-{abs_r.std(ddof=1):.3f} '
                  f'sig {sig_r.mean():4.0f}+-{sig_r.std(ddof=1):.0f}% sgn {sgn_r.mean():+.3f}+-{sgn_r.std(ddof=1):.3f} | r {r:+.2f}')
