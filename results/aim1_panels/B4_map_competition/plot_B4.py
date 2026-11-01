#!/usr/bin/env python3
"""Regenerate B4 from the CSVs in this folder (numpy/pandas/matplotlib only). Edit freely to restyle."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
psth = pd.read_csv(HERE / 'B4_psth.csv')
bars = pd.read_csv(HERE / 'B4_suppression.csv')
st = pd.read_csv(HERE / 'stats.csv')
NAME = {'E': 'classic (E)', 'I': 'interneuron (I)'}
BLK = {'expected_at_RF': dict(color=style.DATA, label='expected location = RF'),
       'expected_opposite': dict(color=style.MUTED, label='expected location = opposite')}

fig = plt.figure(figsize=(6.9, 3.1))
fig.suptitle('Units with RF at the expected location; target appears at a 90° neighbour', x=0.36, y=1.03, fontsize=7)
gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1], width_ratios=[1, 1, 0.85], hspace=0.12, wspace=0.42)
for k, ty in enumerate(('E', 'I')):
    ax = fig.add_subplot(gs[0, k])
    axd = fig.add_subplot(gs[1, k], sharex=ax)
    for blk, kw in BLK.items():
        g = psth[(psth.unit_type == ty) & (psth.curve == blk)].sort_values('bin_center_ms')
        x, m, s = g.bin_center_ms.to_numpy(), g.mean_sp_s.to_numpy(), g.sem_sp_s.to_numpy()
        ax.fill_between(x, m - s, m + s, color=kw['color'], alpha=0.18, lw=0)
        ax.plot(x, m, color=kw['color'], lw=1.3, label=kw['label'])
    n = int(g.n_units.iloc[0])
    ax.axhline(0, color=style.NULL, lw=0.8, zorder=0)
    ax.axvline(0, color=style.MUTED, lw=0.5, ls=':')
    ax.set_title(f'{"ab"[k]}  {NAME[ty]}, n={n}', loc='left')
    ax.set_ylabel('rate re pre-target\nbaseline (sp/s)' if k == 0 else '')
    ax.set_ylim(-5, 5)
    plt.setp(ax.get_xticklabels(), visible=False)
    lv = st[(st.unit_class == ty) & (st.statistic.str.startswith('rate re pre-target')) &
            (st.statistic.str.endswith('200..250 ms'))].iloc[0]
    ax.text(0.99, 0.03, f'200-250 ms: {float(lv.value):+.1f} sp/s (p={lv.p:.2g})', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=5.5, color=style.DATA)
    g = psth[(psth.unit_type == ty) & (psth.curve == 'difference_RF_minus_opposite')].sort_values('bin_center_ms')
    x = g.bin_center_ms.to_numpy()
    axd.fill_between(x, g.ci95_lo.to_numpy(), g.ci95_hi.to_numpy(), color=style.DATA, alpha=0.15, lw=0)
    axd.plot(x, g.mean_sp_s.to_numpy(), color=style.DATA, lw=1.1)
    axd.axhline(0, color=style.NULL, lw=0.8, zorder=0)
    axd.axvline(0, color=style.MUTED, lw=0.5, ls=':')
    axd.set_ylim(-4, 5)
    axd.set_ylabel('RF − opposite\n(sp/s, 95% CI)' if k == 0 else '', fontsize=6)
    axd.set_xlabel('time from target onset (ms)\n(target at a 90° neighbour)')
    axd.set_xlim(-300, 400)
    d0 = st[(st.unit_class == ty) & (st.statistic.str.startswith('expected-at-RF')) & (st.statistic.str.endswith(' 0..50 ms'))].iloc[0]
    axd.text(0.99, 0.97, f'0-50 ms: {float(d0.value):+.2f} (p={d0.p:.2g})', transform=axd.transAxes, ha='right',
             va='top', fontsize=5.5)
    if k == 0:
        ax.legend(loc='upper left', fontsize=5.5, handlelength=1.5, borderaxespad=0.2)

ax = fig.add_subplot(gs[:, 2])
order = ['contra_neighbour_90deg', 'ipsi_neighbour_90deg', 'opposite_180deg']
xl = ['90°\ncontra', '90°\nipsi', '180°\nopposite']
for k, ty in enumerate(('E', 'I')):
    g = bars[bars.unit_type == ty].set_index('target_location').loc[order]
    x = np.arange(3) + (k - 0.5) * 0.38
    ax.bar(x, g.pct_suppressed.to_numpy(), width=0.36, color=style.DATA if ty == 'E' else 'white',
           edgecolor=style.DATA if ty == 'E' else style.MUTED, lw=0.8, label=f'{NAME[ty]} (n={int(g.n_units.iloc[0])})')
    ax.errorbar(x, g.pct_suppressed.to_numpy(), yerr=[(g.pct_suppressed - g.ci95_lo_pct).to_numpy(),
                                                     (g.ci95_hi_pct - g.pct_suppressed).to_numpy()],
                fmt='none', ecolor=style.MUTED if ty == 'E' else style.DATA, elinewidth=0.8, capsize=0)
pv = {ty: st[(st.unit_class == ty) & (st.statistic.str.contains('opposite vs ipsi'))].iloc[0].p for ty in ('E', 'I')}
ax.text(0.02, 0.83, f'180° vs 90° ipsi (McNemar):\nE p={pv["E"]:.2g}, I p={pv["I"]:.2g}', transform=ax.transAxes,
        fontsize=5.5, va='top', color=style.DATA)
ax.set_xticks(range(3))
ax.set_xticklabels(xl, fontsize=6)
ax.set_ylim(0, 55)
ax.set_ylabel('units with pre-saccadic\nsuppression (%)')
ax.set_xlabel('target angle from RF centre')
ax.set_title('c  suppression by target angle', loc='left')
ax.legend(loc='upper left', fontsize=5.5, handlelength=1.2, borderaxespad=0.2)
style.save(fig, str(HERE / 'B4_map_competition'))
