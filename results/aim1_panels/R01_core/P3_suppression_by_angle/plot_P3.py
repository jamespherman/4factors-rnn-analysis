#!/usr/bin/env python3
"""Regenerate P3 from the CSVs in this folder (numpy/pandas/matplotlib only). Edit freely to restyle."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
bars = pd.read_csv(HERE / 'P3_suppression.csv')
st = pd.read_csv(HERE / 'stats.csv')
ORDER = ['contra_neighbour_90deg', 'ipsi_neighbour_90deg', 'opposite_180deg']
XL = ['90°\ncontra', '90°\nipsi', '180°\nopposite']
NAME = {'E': 'classic', 'I': 'interneuron'}

fig, ax = plt.subplots(figsize=style.PANEL)
for k, ty in enumerate(('E', 'I')):
    g = bars[bars.unit_type == ty].set_index('target_location').loc[ORDER]
    x = np.arange(3) + (k - 0.5) * 0.38
    m = g.pct_suppressed.to_numpy()
    ax.bar(x, m, width=0.36, color=style.DATA if ty == 'E' else 'white', edgecolor=style.DATA, lw=0.8,
           label=f'{NAME[ty]} (n = {int(g.n_units.iloc[0])})')
    ax.errorbar(x, m, yerr=[m - g.ci95_lo_pct.to_numpy(), g.ci95_hi_pct.to_numpy() - m], fmt='none',
                ecolor=style.MUTED, elinewidth=0.8, capsize=0)
# McNemar bracket, 180 deg vs 90 deg ipsi, per class
for k, ty, y in ((0, 'E', 29.5), (1, 'I', 45)):
    p = float(st[(st.unit_class == ty) & st.statistic.str.contains('opposite vs ipsi')].iloc[0].p)
    x1, x2 = 1 + (k - 0.5) * 0.38, 2 + (k - 0.5) * 0.38
    ax.plot([x1, x1, x2, x2], [y - 1.2, y, y, y - 1.2], color=style.DATA, lw=0.6)
    ax.text((x1 + x2) / 2, y + 0.5, f'p = {p:.2g}', ha='center', va='bottom', fontsize=5.5)
ax.set_xticks(range(3))
ax.set_xticklabels(XL)
ax.set_ylim(0, 52)
ax.set_ylabel('units with pre-saccadic\nsuppression (%)')
ax.set_xlabel('target angle from RF centre')
ax.legend(loc='upper left', fontsize=5.5, handlelength=1.2, borderaxespad=0.1)
style.save(fig, str(HERE / 'P3_suppression_by_angle'))
