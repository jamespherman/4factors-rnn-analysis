#!/usr/bin/env python3
"""B10: candidate monosynaptic connections (sharp CCG peaks) and factor-selectivity similarity.
Reads only the CSVs in this folder (plus style.py). Run: python plot_B10.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
style.apply()

ccg = pd.read_csv(HERE / 'B10_example_ccg.csv')
pairs = pd.read_csv(HERE / 'B10_pairs_connected.csv')
matched = pd.read_csv(HERE / 'B10_matched_unconnected.csv')
nulls = pd.read_csv(HERE / 'B10_null_medians.csv')
stats = pd.read_csv(HERE / 'stats.csv').set_index('statistic')

peaks = pairs[pairs.pair_kind == 'sharp_peak']
troughs = pairs[(pairs.pair_kind == 'trough') & pairs.powered]
NOLOC = 'without location (identity, salience, reward, probability)'
LOC = 'with location (5 factors)'


def s(name):
    return stats.loc[name]


fig, axes = plt.subplots(2, 2, figsize=(2 * style.PANEL[0] + 0.3, 2 * style.PANEL[1] + 0.3))
(axA, axB), (axC, axD) = axes

# a: example CCG
c = {k: v.to_numpy() for k, v in ccg[(ccg.lag_ms >= -15) & (ccg.lag_ms <= 15)].items()}
axA.axvspan(0.5, 4.5, color=style.NULL, alpha=0.5, lw=0)
axA.fill_between(c['lag_ms'], c['poisson_lo_0.05pct'], c['poisson_hi_99.95pct'], step='mid', color=style.MUTED,
                 alpha=0.35, lw=0, label='jitter-expected,\npointwise Poisson 99.9%')
axA.step(c['lag_ms'], c['jitter_expected_count'], where='mid', color=style.MUTED, lw=0.8)
axA.step(c['lag_ms'], c['observed_count'], where='mid', color=style.DATA, lw=1.0, label='observed')
r0 = ccg.iloc[0]
axA.set_title(f'a  Example sharp peak ({r0.pre_type}→{r0.post_type})', loc='left')
axA.set_xlabel('Lag of post spike (ms)')
axA.set_ylabel('Spike pairs per 1-ms bin')
axA.set_ylim(axA.get_ylim()[0], axA.get_ylim()[1] * 1.08)
axA.text(2.5, axA.get_ylim()[0], '1-4 ms', ha='center', va='bottom', fontsize=5.8, color=style.MUTED)
axA.text(0.98, 0.98, f'STG {r0.stg_1to4ms:.3f}', transform=axA.transAxes,
         ha='right', va='top', fontsize=5.8, color=style.DATA)
axA.legend(loc='upper left', handlelength=1.2, fontsize=5.6)
axA.set_xlim(-15, 15)


def hist(ax, measure, label, title):
    bins = np.linspace(-1, 1, 21)
    m = matched[matched.measure == measure]
    hm, _ = np.histogram(m.r.to_numpy(), bins, weights=m.weight.to_numpy())
    v = peaks[measure].dropna().to_numpy()
    hc, _ = np.histogram(v, bins)
    ax.stairs(hm / hm.sum(), bins, fill=True, color=style.NULL, label='matched unconnected')
    ax.stairs(hc / hc.sum(), bins, color=style.DATA, lw=1.2, label=f'sharp-peak pairs (n = {len(v)})')
    obs = s(f'peak pairs: median profile r, {label}')
    nul = s(f'matched unconnected for peak pairs: median of replicate medians, profile r, {label}')
    top = max((hm / hm.sum()).max(), (hc / hc.sum()).max()) * 1.15
    ax.plot([obs.value] * 2, [0, top], color=style.DATA, lw=0.9, ls='-')
    ax.plot([nul.value] * 2, [0, top], color=style.MUTED, lw=0.9, ls='--')
    ax.set_ylim(0, top * 1.9)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Selectivity-profile similarity (r)')
    ax.set_ylabel('Fraction of pairs')
    ax.set_title(title, loc='left')
    pp = s(f'peak vs matched: p (pair level), {label}').value
    pu = s(f'peak vs matched: p (unit level), {label}').value
    fmt = lambda p: '< 0.001' if p < 0.001 else f'= {p:.3f}'
    ax.text(0.02, 0.98, f'median {obs.value:+.2f} vs {nul.value:+.2f}\npairs: p {fmt(pp)}\n'
                        f'presyn. units (n = {int(obs.n_units)}): p {fmt(pu)}',
            transform=ax.transAxes, va='top', fontsize=5.8)
    return ax


hist(axB, 'r_no_location', NOLOC, 'b  Without location (main)')
axB.legend(loc='upper left', bbox_to_anchor=(0.0, 0.78), fontsize=5.6, handlelength=1.0)
hist(axD, 'r_5factor', LOC, 'd  With location (secondary)')

# c: peaks vs troughs, with the resolver's discriminating prediction for inhibitory pairs
axC.axvspan(-1.08, 0, ymin=0, ymax=0.5, color=style.RESOLVER, alpha=0.12, lw=0)
axC.axvline(0, ymin=0, ymax=0.5, color=style.RESOLVER, ls='--', lw=1.0)
axC.text(-1.04, -0.5, 'resolver: units with\ndissimilar selectivity\ninhibit each other', color=style.RESOLVER, fontsize=5.6,
         va='center')
rng = np.random.default_rng(1)
for y, g, name, lab in [(1, peaks, 'peak', 'sharp peaks\n(excitatory)'), (0, troughs, 'trough', 'troughs\n(inhibitory)')]:
    nm = nulls[(nulls.pair_set == name) & (nulls.measure == 'r_no_location') & (nulls.level == 'pair')].null_median_r
    lo, hi = np.percentile(nm, [2.5, 97.5])
    axC.fill_between([lo, hi], y - 0.28, y + 0.28, color=style.NULL, lw=0)
    v = g.r_no_location.dropna().to_numpy()
    axC.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=5 if y else 12, color=style.DATA,
                alpha=0.5 if y else 1, lw=0, zorder=3, marker='o' if y else 'v')
    axC.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color=style.DATA, lw=1.6, zorder=4)
axC.set_yticks([1, 0], [f'sharp peaks\nn = {peaks.r_no_location.notna().sum()}',
                        f'troughs\nn = {len(troughs)} (I→I)'])
axC.set_ylim(-0.85, 1.6)
axC.set_xlim(-1.08, 1.08)
axC.set_xlabel('Similarity without location (r)')
axC.set_title('c  Inhibitory pairs: too few to test', loc='left')
axC.text(0.98, 0.98, 'grey: matched-null 95% of median', transform=axC.transAxes, ha='right', va='top',
         fontsize=5.6, color=style.MUTED)

fig.tight_layout(w_pad=1.2, h_pad=1.2)
style.save(fig, str(HERE / 'B10_ccg_pair_selectivity'))
print('saved', HERE / 'B10_ccg_pair_selectivity.png')
