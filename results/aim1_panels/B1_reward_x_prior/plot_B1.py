#!/usr/bin/env python3
"""Regenerate B1 from the CSVs in this folder (numpy/pandas/matplotlib only). Edit freely to restyle."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
cells = pd.read_csv(HERE / 'B1_cells.csv')
null = pd.read_csv(HERE / 'B1_null.csv')
st = pd.read_csv(HERE / 'stats.csv')
ORDER = ['expected', 'neighbour_90deg', 'opposite_180deg']
XT = ['expected', '90° nb', 'opposite']
CUE = {'high': dict(color=style.DATA, marker='o', mfc=style.DATA, label='high-reward cue'),
       'low': dict(color=style.MUTED, marker='s', mfc='white', label='low-reward cue')}
YL = {'rt_pct': 'RT change (%)\nfrom low-reward expected', 'pv_deg_s': 'Peak velocity change (deg/s)\nfrom low-reward expected'}


def pfmt(p):
    return 'p<0.001' if p < 0.001 else f'p={p:.2g}'


def simple(m, meas, cue):
    r = st[(st.monkey == m) & (st.statistic == f'{meas}: opposite_180deg vs expected, {cue} reward cue')].iloc[0]
    return r.value, r.p


fig, axs = plt.subplots(2, 2, figsize=(2 * style.PANEL[0] + 0.3, 2 * style.PANEL[1] + 0.2))
for j, m in enumerate(['Feynman', 'Newton']):
    for i, meas in enumerate(['rt_pct', 'pv_deg_s']):
        ax = axs[i, j]
        ax.axhline(0, color=style.NULL, lw=0.8, zorder=0)
        for k, cue in enumerate(['low', 'high']):
            g = cells[(cells.monkey == m) & (cells.measure == meas) & (cells.reward_cue == cue)].set_index('loc_class').loc[ORDER]
            x = [q + (k - 0.5) * 0.12 for q in range(3)]
            kw = CUE[cue]
            ax.errorbar(x, g.change_mean.to_numpy(), yerr=[(g.change_mean - g.ci95_lo).to_numpy(), (g.ci95_hi - g.change_mean).to_numpy()], color=kw['color'],
                        marker=kw['marker'], mfc=kw['mfc'], mec=kw['color'], lw=1.5, elinewidth=0.8, capsize=0,
                        label=kw['label'], zorder=3)
        nl = null[(null.monkey == m) & (null.measure == meas)].set_index('loc_class').loc[ORDER]
        ax.plot([q + 0.06 for q in range(3)], nl.null_change.to_numpy(), ls='--', color=style.DATA, lw=1.0, zorder=2,
                label='no-interaction\n(additive) null')
        ax.set_xticks(range(3))
        ax.set_xticklabels(XT)
        ax.set_xlim(-0.4, 2.4)
        if j == 0:
            ax.set_ylabel(YL[meas])
        if i == 0:
            n_s = int(cells[(cells.monkey == m)].n_sessions.iloc[0])
            ax.set_title(f'{m} ({n_s} sessions)')
            lo, hi = simple(m, meas, 'low'), simple(m, meas, 'high')
            ax.text(0.02, 0.98, f'opposite vs expected:\nlow cue {lo[0]:+.1f}% ({pfmt(lo[1])})\n'
                                f'high cue {hi[0]:+.1f}% ({pfmt(hi[1])})',
                    transform=ax.transAxes, va='top', ha='left', fontsize=6, color=style.DATA)
            ax.set_ylim(-14, 20)
        else:
            ax.set_xlabel('target location re high-probability location')
axs[1, 0].set_ylim(-60, 80)
axs[1, 1].set_ylim(-60, 340)
axs[1, 1].text(2.45, 335, 'neighbour PV differs by\nsaccade direction/amplitude\n(location, not reward)',
               fontsize=5.5, color=style.MUTED, ha='right', va='top')
h, l = axs[0, 0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.0, h_pad=1.0, rect=(0, 0.08, 1, 1))
fig.legend(h, [x.replace('\n', ' ') for x in l], loc='lower center', ncol=3, fontsize=6.5, handlelength=2.4,
           bbox_to_anchor=(0.5, 0.02))
fig.text(0.5, 0.0, 'mean over sessions ± 95% CI (t); p: trial-level OLS, clustered by block',
         fontsize=5.5, color=style.MUTED, ha='center', va='bottom')
style.save(fig, str(HERE / 'B1_reward_x_prior'))
