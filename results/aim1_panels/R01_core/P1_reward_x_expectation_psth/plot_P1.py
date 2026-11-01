#!/usr/bin/env python3
"""Regenerate P1 from the CSVs in this folder (numpy/pandas/matplotlib only). Edit freely to restyle."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
psth = pd.read_csv(HERE / 'P1_psth.csv')
tc = pd.read_csv(HERE / 'P1_interaction_timecourse.csv')
st = pd.read_csv(HERE / 'stats.csv')
COL = {'high': '#1f1f1f', 'low': '#9a9a96'}
LS = {'expected': '-', 'unexpected': (0, (3, 1.6))}

fig, ax = plt.subplots(figsize=style.PANEL)
ax.axvspan(-300, 0, color=style.NULL, alpha=0.45, lw=0, zorder=0)
ax.text(-150, -1.6, 'pre-target baseline', ha='center', va='top', fontsize=5.5, color=style.MUTED)
ax.axhline(0, color=style.MUTED, lw=0.5, zorder=1)
ax.axvline(0, color=style.MUTED, lw=0.5, ls=':', zorder=1)
ends = {}
for rw in ('low', 'high'):
    for ex in ('expected', 'unexpected'):
        g = psth[(psth.reward_cue == rw) & (psth.target == ex)].sort_values('bin_center_ms')
        ax.plot(g.bin_center_ms.to_numpy(), g.mean_sp_s.to_numpy(), color=COL[rw], ls=LS[ex], lw=1.2 if rw == 'high' else 1.1, zorder=3)
n = int(psth.n_units.iloc[0])

# significance marks: P3 visual window (grey bar) and P4 pointwise p < 0.01 windows (black segments)
yb = 20.5
ax.plot([50, 250], [yb, yb], color=style.MUTED, lw=2.2, solid_capstyle='butt')
# darker: the contiguous run of P4 40-ms windows with p < 0.01 that contains the peak of the interaction time course
k = int(tc.mean_sd.idxmax())
lo = hi = k
while lo > 0 and tc.sig_p01[lo - 1]:
    lo -= 1
while hi < len(tc) - 1 and tc.sig_p01[hi + 1]:
    hi += 1
ax.plot([tc.window_start_ms[lo], tc.window_end_ms[hi]], [yb, yb], color=style.DATA, lw=2.2, solid_capstyle='butt')
p3 = st[st.statistic.str.startswith('P3 reward x expected')].iloc[0]
ax.text(150, yb + 0.6, f'reward × expectation\n+{float(p3.value):.2f} SD, p = {float(p3.p):.0e}'.replace('e-0', 'e-'),
        ha='center', va='bottom', fontsize=5.5)

# direct labels
lab = [('high', 'expected', 'high reward, expected'), ('high', 'unexpected', 'high reward, unexpected'),
       ('low', 'expected', 'low reward, expected'), ('low', 'unexpected', 'low reward, unexpected')]
y0 = 17.5
for k, (rw, ex, txt) in enumerate(lab):
    y = y0 - 2.1 * k
    ax.plot([-288, -260], [y, y], color=COL[rw], ls=LS[ex], lw=1.2)
    ax.text(-250, y, txt, va='center', fontsize=5, color=COL[rw])
ax.set_xlim(-300, 300)
ax.set_ylim(-3.5, 25)
ax.set_xticks([-300, -150, 0, 150, 300])
ax.set_xlabel('time from target onset (ms)')
ax.set_ylabel('rate re baseline (sp/s)')
ax.set_title(f'Classic SC units, target in RF (n = {n})', loc='left', fontsize=6.5)
style.save(fig, str(HERE / 'P1_reward_x_expectation_psth'))
